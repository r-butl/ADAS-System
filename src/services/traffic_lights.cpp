#include <fstream>
#include <iostream>
#include <vector>
#include "frame_buffer.hpp"
#include "annotations_buffer.hpp"
#include "traffic_lights.hpp"

using namespace nvinfer1;

TrafficLights::TrafficLights(const std::string& enginePath, FrameBuffer* inputBuffer, AnnotationsBuffer* outputBuffer)
    : inputBuffer(inputBuffer), outputBuffer(outputBuffer), enginePath(enginePath), engine(nullptr), context(nullptr) {

    std::ifstream file(enginePath, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Failed to open engine file: " << enginePath << std::endl;
        return;
    }

    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "Error: Failed to create TensorRT runtime" << std::endl;
        return;
    }

    engine = runtime->deserializeCudaEngine(engineData.data(), size);
    if (runtime) delete runtime;

    if (!engine) {
        std::cerr << "Error: Failed to deserialize engine" << std::endl;
        return;
    }

    context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Error: Failed to create execution context" << std::endl;
    }


    newFrame = false;
    frameVersion = 0;
}

TrafficLights::~TrafficLights() {
    if (context) delete context;
    if (engine) delete engine;
}

void* TrafficLights::run(void* arg) {
    TrafficLights* self = static_cast<TrafficLights*>(arg);
    self->inferenceLoop();
    return nullptr;
}

void TrafficLights::inferenceLoop() {
    	std::cout << "Running inference loop..." << std::endl;

	bool newFrame = false;

	while(inputBuffer->getLatestFrame(currentFrame, frameVersion) == false){
		std::cout << "Traffic lights: waiting for new frame" << std::endl;	
	}
   
	cv::cuda::GpuMat gpu_frame;
      	//gpu_frame.upload(
	//
}

