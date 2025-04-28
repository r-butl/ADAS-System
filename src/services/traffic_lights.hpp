#ifndef TRAFFIC_LIGHTS_HPP
#define TRAFFIC_LIGHTS_HPP

#include <string>
#include "frame_buffer.hpp"
#include "annotations_buffer.hpp"

#include <NvInfer.h>
using namespace nvinfer1;

struct Detection {
	float x, y, w, h, conf;
	int class_id;
};

class TrafficLights {
public:
    TrafficLights(const std::string& onnxPath);
    ~TrafficLights();

    static void* run(void* arg); // Entry point for pthreads

    void inferenceLoop(cv::Mat frame);
private:
    std::vector<Detection> postprocessImage(float* output, int num_detections, float conf_thresh, float nms_thresh);
    float computeIoU(const Detection& a, const Detection& b);
    void saveEngine(const std::string& filePath, IHostMemory* serializedModel);
    bool loadEngine(const std::string& filePath);

    std::string enginePath;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    nvinfer1::IRuntime* runtime;

    cudaStream_t stream;
    void* buffers[2]; // input , output
    int inputIndex, outputIndex;
    int inputW = 720, inputH = 720;
    int outputSize = 8400;
};

#endif // INFERENCE_SERVICE_HPP

