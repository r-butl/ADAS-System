/*
 * Pedestrian detection using TensorRT and OpenCV
 * 
 * Compatible with Jetson Orin Nano, JetPack + OpenCV 4.x
 * Based on `capture.cpp` style by Sam Siewert
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

using namespace cv;
using namespace std;
using namespace nvinfer1;

#define ESCAPE_KEY 27
#define SYSTEM_ERROR -1

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            cout << msg << endl;
    }
};

vector<char> loadEngine(const string& filename) {
    ifstream file(filename, ios::binary);
    return vector<char>((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
}

void preprocess(const Mat& frame, float* gpu_input, int input_w, int input_h) {
    Mat resized, rgb;
    resize(frame, resized, Size(input_w, input_h));
    cvtColor(resized, rgb, COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    vector<Mat> channels(3);
    split(rgb, channels);

    float* cpu_input = new float[3 * input_h * input_w];
    for (int c = 0; c < 3; ++c) {
        memcpy(cpu_input + c * input_h * input_w, channels[c].data, input_h * input_w * sizeof(float));
    }

    cudaMemcpy(gpu_input, cpu_input, 3 * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice);
    delete[] cpu_input;
}

void drawBoxes(Mat& frame, float* output, int num_detections, int input_w, int input_h) {
    for (int i = 0; i < num_detections; ++i) {
        float* det = output + i * 6;
        float conf = det[4];
        if (conf < 0.5) continue;

        int x1 = static_cast<int>(det[0] * frame.cols);
        int y1 = static_cast<int>(det[1] * frame.rows);
        int x2 = static_cast<int>(det[2] * frame.cols);
        int y2 = static_cast<int>(det[3] * frame.rows);

        rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);
        putText(frame, "Person", Point(x1, y1 - 5), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 1);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: ./people_detect <video_file>" << endl;
        return SYSTEM_ERROR;
    }

    string video_path = argv[1];
    Logger logger;

    // Load TensorRT engine
    vector<char> engine_data = loadEngine("people_detect.engine");
    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    IExecutionContext* context = engine->createExecutionContext();

    // Engine bindings
    int input_idx = engine->getBindingIndex("input");     // Update this name if needed
    int output_idx = engine->getBindingIndex("output");   // Update this name if needed

    Dims input_dims = engine->getBindingDimensions(input_idx);
    Dims output_dims = engine->getBindingDimensions(output_idx);
    int input_h = input_dims.d[2];
    int input_w = input_dims.d[3];
    size_t input_size = 3 * input_h * input_w * sizeof(float);
    size_t output_size = output_dims.volume() * sizeof(float);

    void* buffers[2];
    cudaMalloc(&buffers[input_idx], input_size);
    cudaMalloc(&buffers[output_idx], output_size);

    float* output_host = new float[output_dims.volume()];

    // Open video
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "Failed to open video: " << video_path << endl;
        return SYSTEM_ERROR;
    }

    namedWindow("Pedestrian Detection");
    char key_input;

    while (true) {
        Mat frame;
        if (!cap.read(frame)) break;

        auto start = chrono::high_resolution_clock::now();

        preprocess(frame, (float*)buffers[input_idx], input_w, input_h);
        context->executeV2(buffers);
        cudaMemcpy(output_host, buffers[output_idx], output_size, cudaMemcpyDeviceToHost);

        drawBoxes(frame, output_host, output_dims.d[1], input_w, input_h);

        auto end = chrono::high_resolution_clock::now();
        float fps = 1000.0f / chrono::duration_cast<chrono::milliseconds>(end - start).count();
        putText(frame, "FPS: " + to_string(fps), Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);

        imshow("Pedestrian Detection", frame);

        if ((key_input = waitKey(1)) == ESCAPE_KEY) {
            break;
        } else if (key_input == 'n') {
            cout << "input " << key_input << " ignored" << endl;
        }
    }

    destroyWindow("Pedestrian Detection");

    delete[] output_host;
    cudaFree(buffers[input_idx]);
    cudaFree(buffers[output_idx]);
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}

