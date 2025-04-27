#ifndef TRAFFIC_LIGHTS_HPP
#define TRAFFIC_LIGHTS_HPP

#include <string>
#include "frame_buffer.hpp"
#include "annotations_buffer.hpp"

#include <NvInfer.h>
using namespace nvinfer1;

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        switch (severity) {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: " << msg << std::endl; break;
        case Severity::kERROR:         std::cerr << "ERROR: " << msg << std::endl; break;
        case Severity::kWARNING:       std::cerr << "WARNING: " << msg << std::endl; break;
        case Severity::kINFO:          std::cout << "INFO: " << msg << std::endl; break;
        case Severity::kVERBOSE:       std::cout << "VERBOSE: " << msg << std::endl; break;
        default:                       std::cout << "UNKNOWN: " << msg << std::endl; break;
        }
    }
} gLogger;

struct Detection {
	float x, y, w, h, conf;
	int class_id;
};

class TrafficLights {
public:
    TrafficLights(const std::string& enginePath);
    ~TrafficLights();

    static void* run(void* arg); // Entry point for pthreads

private:
    void inferenceLoop(cv::Mat frame);
    cv::Mat preprocessImage(const cv::Mat& frame, int input_w, int input_h, float* gpu_input);
    std::vector<Detection> postprocessImage(float* output, int num_detections, float conf_thresh, float nms_thresh);
    float computeIoU(const Detection& a, const Detection& b);
   
    std::string enginePath;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;

    cudaStream_t stream;
    void* buffers[2]; // input , output
    int inputIndex, outputIndex;
    int inputW = 720, inputH = 720;
    int outputSize = 8400;
};

#endif // INFERENCE_SERVICE_HPP

