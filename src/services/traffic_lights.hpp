#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <NvInfer.h>

using namespace nvinfer1;
using namespace cv;

class TrafficLights {
public:
    explicit TrafficLights(const std::string& enginePath);
    ~TrafficLights();

    bool isInitialized() const;
    std::vector<cv::Rect> detect(const cv::Mat& frame);
    void preprocess(const cv::Mat& frame, std::vector<float>& cpu_input_buffer, int input_w, int input_h);
    size_t calculateSizeFromDims(const nvinfer1::Dims& dims);
    bool loadEngine(const std::string& filePath);
    void saveEngine(const std::string& filePath, IHostMemory* serializedModel);
    int getInputWidth() const;
    int getInputHeight() const;

private:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    cudaStream_t stream;
    std::string enginePath;

    const char* input_tensor_name_ = nullptr;
    const char* output_tensor_name_ = nullptr;
    int network_input_h_;
    int network_input_w_;
};
