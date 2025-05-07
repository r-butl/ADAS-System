#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <functional>
#include <algorithm>
#include <cstring>
#include <cctype>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>
#include <cuda_runtime_api.h>

using namespace cv;
using namespace std;
using namespace nvinfer1;

#define ESCAPE_KEY 27
#define SYSTEM_ERROR -1

class SimpleLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << std::endl;
        }
    }
};

SimpleLogger gLogger;

size_t calculateSizeFromDims(const Dims& dims) {
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) return 0;
        size *= dims.d[i];
    }
    return size;
}

void preprocess(const Mat& frame, std::vector<float>& cpu_input_buffer, int input_w, int input_h) {
    Mat resized, rgb;
    resize(frame, resized, Size(input_w, input_h));
    cvtColor(resized, rgb, COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    cpu_input_buffer.resize(3 * input_h * input_w);
    vector<Mat> channels(3);
    split(rgb, channels);

    float* ptr = cpu_input_buffer.data();
    for (int c = 0; c < 3; ++c) {
        memcpy(ptr, channels[c].data, input_h * input_w * sizeof(float));
        ptr += input_h * input_w;
    }
}

void drawBoxes(Mat& frame, float* output_data, int num_detections, 
    int input_width, int input_height, float conf_threshold = 0.5) {
    const int props = 6;
    float scale_x = frame.cols / static_cast<float>(input_width);
    float scale_y = frame.rows / static_cast<float>(input_height);

    for (int i = 0; i < num_detections; ++i) {
    float* det = output_data + i * props;
    float confidence = det[4];
    if (confidence < conf_threshold) continue;

    // Original detection coordinates at input size (640x640)
    int x1 = static_cast<int>(det[0] * scale_x);
    int y1 = static_cast<int>(det[1] * scale_y);
    int x2 = static_cast<int>(det[2] * scale_x);
    int y2 = static_cast<int>(det[3] * scale_y);

    // Clamp to the frame size
    x1 = std::clamp(x1, 0, frame.cols - 1);
    y1 = std::clamp(y1, 0, frame.rows - 1);
    x2 = std::clamp(x2, 0, frame.cols - 1);
    y2 = std::clamp(y2, 0, frame.rows - 1);

    if (x1 >= x2 || y1 >= y2) continue;

    rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);
    putText(frame, "Person " + to_string(int(confidence * 100)) + "%", 
        Point(x1, max(y1 - 5, 15)), FONT_HERSHEY_SIMPLEX, 
        0.6, Scalar(0, 255, 0), 1);
    }
}

class SimplePeopleDetector {
public:

    SimplePeopleDetector(const string& path)
        : runtime_(nullptr), engine_(nullptr), context_(nullptr), stream_(nullptr), network_input_h_(-1), network_input_w_(-1) {

        ifstream engine_file(path, ios::binary);
        if (!engine_file) { cerr << "ERROR: Cannot open engine file." << endl; return; }
        engine_file.seekg(0, ios::end);
        size_t size = engine_file.tellg();
        engine_file.seekg(0, ios::beg);
        vector<char> data(size);
        engine_file.read(data.data(), size);
        engine_file.close();

        runtime_ = createInferRuntime(gLogger);
        engine_ = runtime_->deserializeCudaEngine(data.data(), size);
        context_ = engine_->createExecutionContext();
        cudaStreamCreate(&stream_);

        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) == TensorIOMode::kINPUT && !input_tensor_name_) {
                input_tensor_name_ = name;
                std::cout << "[DEBUG] Input tensor name: " << input_tensor_name_ << std::endl;
            } else if (engine_->getTensorIOMode(name) == TensorIOMode::kOUTPUT && !output_tensor_name_) {
                output_tensor_name_ = name;
                std::cout << "[DEBUG] Output tensor name: " << output_tensor_name_ << std::endl;
            }
        }

        if (!input_tensor_name_ || !output_tensor_name_) {
            cerr << "ERROR: Failed to find input/output tensor names." << endl; return;
        }

        Dims d = engine_->getTensorShape(input_tensor_name_);
        if (d.nbDims == 4 && d.d[0] == 1 && d.d[1] == 3) {
            network_input_h_ = d.d[2];
            network_input_w_ = d.d[3];
        } else {
            cerr << "ERROR: Unexpected input dimensions." << endl;
        }
    }
   
    
    ~SimplePeopleDetector() {
        if (stream_) cudaStreamDestroy(stream_);
        if (context_) delete context_;
        if (engine_) delete engine_;
        if (runtime_) delete runtime_;
    }

    bool isInitialized() const {
        return runtime_ && engine_ && context_ && stream_ && network_input_h_ > 0 && network_input_w_ > 0;
    }

    bool detect(const Mat& frame, vector<float>& host_output_buffer, Dims& output_dims) {
        if (!isInitialized()) return false;

        vector<float> input;
        preprocess(frame, input, network_input_w_, network_input_h_);

        context_->setInputShape(input_tensor_name_, Dims4{1, 3, network_input_h_, network_input_w_});
        Dims in_dims = context_->getTensorShape(input_tensor_name_);
        output_dims = context_->getTensorShape(output_tensor_name_);

        static bool printed = false;
        if (!printed) {
            std::cout << "[DEBUG] Input dims: ";
            for (int i = 0; i < in_dims.nbDims; ++i) std::cout << in_dims.d[i] << " ";
            std::cout << std::endl;
            std::cout << "[DEBUG] Output dims: ";
            for (int i = 0; i < output_dims.nbDims; ++i) std::cout << output_dims.d[i] << " ";
            std::cout << std::endl;
            printed = true;
        }

        size_t in_size = calculateSizeFromDims(in_dims) * sizeof(float);
        size_t out_size = calculateSizeFromDims(output_dims) * sizeof(float);

        if (in_size == 0 || out_size == 0) return false;

        host_output_buffer.resize(out_size / sizeof(float));

        void* d_in = nullptr; void* d_out = nullptr;
        cudaMalloc(&d_in, in_size);
        cudaMalloc(&d_out, out_size);
        cudaMemcpyAsync(d_in, input.data(), in_size, cudaMemcpyHostToDevice, stream_);
        context_->setTensorAddress(input_tensor_name_, d_in);
        context_->setTensorAddress(output_tensor_name_, d_out);

        context_->enqueueV3(stream_);
        cudaMemcpyAsync(host_output_buffer.data(), d_out, out_size, cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        cudaFree(d_in);
        cudaFree(d_out);

        return true;
    }
    int getInputWidth() const { return network_input_w_; }
    
    int getInputHeight() const { return network_input_h_; }

private:
    IRuntime* runtime_;
    ICudaEngine* engine_;
    IExecutionContext* context_;
    cudaStream_t stream_;

    const char* input_tensor_name_ = nullptr;
    const char* output_tensor_name_ = nullptr;
    int network_input_h_;
    int network_input_w_;
};

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "USAGE: ./people_detect <engine_path> <video_or_camera>" << endl;
        return SYSTEM_ERROR;
    }

    string engine_path = argv[1], input_source = argv[2];
    SimplePeopleDetector detector(engine_path);
    if (!detector.isInitialized()) return SYSTEM_ERROR;

    VideoCapture cap;
    if (all_of(input_source.begin(), input_source.end(), ::isdigit)) cap.open(stoi(input_source));
    else cap.open(input_source);

    if (!cap.isOpened()) { cerr << "Failed to open source." << endl; return SYSTEM_ERROR; }

    namedWindow("Pedestrian Detection", WINDOW_AUTOSIZE);
    vector<float> output;
    Dims out_dims;

    while (true) {
        Mat frame;
        if (!cap.read(frame) || frame.empty()) break;

        auto t1 = chrono::high_resolution_clock::now();
        bool ok = detector.detect(frame, output, out_dims);

        if (ok && !output.empty()) {
            int count = 0;
            const int props = 6;
            if (out_dims.nbDims == 3 && out_dims.d[2] == props) count = out_dims.d[1];
            else if (out_dims.nbDims == 2 && out_dims.d[1] == props) count = out_dims.d[0];

            int valid = 0;
            for (int i = 0; i < count; ++i) {
                float* det = output.data() + i * props;
                float confidence = det[4];
                if (confidence < 0.5f) continue;

                if (valid < 5) {
                    std::cout << "[DEBUG] Detection " << i << " - Conf: " << confidence
                              << ", Box: [" << det[0] << ", " << det[1] << ", " << det[2] << ", " << det[3] << "]" << std::endl;
                }
                valid++;
            }
            std::cout << "[DEBUG] Valid detections above threshold: " << valid << std::endl;
            // drawBoxes(frame, output.data(), count);
            drawBoxes(frame, output.data(), count, detector.getInputWidth(), detector.getInputHeight());
        }

        auto t2 = chrono::high_resolution_clock::now();
        float fps = 1000.0f / chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
        putText(frame, "FPS: " + format("%.2f", fps), Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        imshow("Pedestrian Detection", frame);

        if (waitKey(1) == ESCAPE_KEY) break;
    }

    destroyAllWindows();
    cap.release();
    return 0;
}
