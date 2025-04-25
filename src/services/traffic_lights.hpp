#ifndef TRAFFIC_LIGHTS_HPP
#define TRAFFIC_LIGHTS_HPP

#include <string>
#include <NvInfer.h>
#include "frame_buffer.hpp"
#include "annotations_buffer.hpp"

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


class TrafficLights {
public:
    TrafficLights(const std::string& enginePath, FrameBuffer* inputBuffer, AnnotationsBuffer* outputBuffer);
    ~TrafficLights();

    static void* run(void* arg); // Entry point for pthreads

    FrameBuffer* inputBuffer;
    AnnotationsBuffer* outputBuffer;

private:
    void inferenceLoop();

    bool newFrame;
    cv::Mat currentFrame;
    uint64_t frameVersion;
    std::string enginePath;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
};

#endif // INFERENCE_SERVICE_HPP

