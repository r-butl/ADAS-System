#include <onnxruntime_cxx_api.h>
#include <cuda_provider_factory.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <array>

const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;
const float CONFIDENCE_THRESHOLD = 0.4;
const int PERSON_CLASS_ID = 0;

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo_cuda");

cv::Mat preprocess(const cv::Mat& frame) {
    cv::Mat resized, rgb;
    cv::resize(frame, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    return rgb;
}

std::vector<const char*> getInputNames(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    return { session.GetInputName(0, allocator) };
}

std::vector<const char*> getOutputNames(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    return { session.GetOutputName(0, allocator) };
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path>\n";
        return 1;
    }

    std::string video_path = argv[1];
    std::string model_path = "best.onnx";

    Ort::SessionOptions session_options;
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    Ort::Session session(env, model_path.c_str(), session_options);

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << video_path << "\n";
        return 1;
    }

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter writer("detected_output.avi", cv::VideoWriter::fourcc('M','J','P','G'), fps, cv::Size(frame_width, frame_height));

    auto input_names = getInputNames(session);
    auto output_names = getOutputNames(session);

    cv::Mat frame;
    while (cap.read(frame)) {
        cv::Mat input_img = preprocess(frame);
        std::vector<float> input_tensor_values(INPUT_WIDTH * INPUT_HEIGHT * 3);
        std::memcpy(input_tensor_values.data(), input_img.data, input_tensor_values.size() * sizeof(float));

        std::array<int64_t, 4> input_shape = {1, 3, INPUT_HEIGHT, INPUT_WIDTH};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size());

        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
        float* output = output_tensors[0].GetTensorMutableData<float>();
        auto out_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t num_detections = out_shape[1];

        for (size_t i = 0; i < num_detections; ++i) {
            float x1 = output[i * 6 + 0];
            float y1 = output[i * 6 + 1];
            float x2 = output[i * 6 + 2];
            float y2 = output[i * 6 + 3];
            float conf = output[i * 6 + 4];
            float class_id = output[i * 6 + 5];

            if (conf > CONFIDENCE_THRESHOLD && static_cast<int>(class_id) == PERSON_CLASS_ID) {
                cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
                cv::putText(frame, "Person", cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            }
        }

        writer.write(frame);
        cv::imshow("Person Detection", frame);
        if (cv::waitKey(1) == 27) break; // ESC key to stop
    }

    cap.release();
    writer.release();
    std::cout << "Processing done. Output saved to detected_output.avi\n";
    return 0;
}

