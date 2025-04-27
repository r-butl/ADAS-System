#include <fstream>
#include "cuda_runtime_api.h"
#include <iostream>
#include <vector>
#include "frame_buffer.hpp"
#include "annotations_buffer.hpp"
#include "traffic_lights.hpp"

#include <NvInfer.h>
#include "NvOnnxParser.h"

using namespace nvonnxparser;
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
};

Logger gLogger;

TrafficLights::TrafficLights(const std::string& onnxPath)
    : enginePath(onnxPath), engine(nullptr), context(nullptr) {

	    // Build the engine
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetworkV2(0);

    	// Parse onnx model
    IParser* parser = createParser(*network, gLogger);
    parser->parseFromFile(onnxPath.c_str(),
		    static_cast<int32_t>(ILogger::Severity::kWARNING));

    for (int32_t i = 0; i < parser->getNbErrors(); ++i){
	std::cout << parser->getError(i)->desc() << std::endl;
    }


    // Builder config
    IBuilderConfig* config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 400U << 20);
    config->setMemoryPoolLimit(MemoryPoolType::kTACTIC_SHARED_MEMORY, 64U << 10);

    // Build deserialized engine
    IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);

    delete parser;
    delete network;
    delete config;
    delete builder;

    	// Runtime
    IRuntime* runtime = createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "Error: Failed to create TensorRT runtime" << std::endl;
        return;
    }

    engine = runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size());
    if (!engine) {
        std::cerr << "Error: Failed to deserialize engine" << std::endl;
        return;
    }

    context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Error: Failed to create execution context" << std::endl;
    }

    delete serializedModel;
    delete runtime;
}

TrafficLights::~TrafficLights() {
    if (context) delete context;
    if (engine) delete engine;
}

cv::Mat TrafficLights::preprocessImage(const cv::Mat& frame, int input_w, int input_h, float* gpu_input){
	cv::Mat resized, rgb, float_img;
	cv::resize(frame, resized, cv::Size(input_w, input_h));
	cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
	rgb.convertTo(float_img, CV_32FC3, 1.0, 255.0);

	std::vector<cv::Mat> chw(3);
	for (int i = 0; i < 3; i++)
		chw[i] = cv::Mat(input_h, input_w, CV_32FC1, gpu_input + i * input_h * input_w);
	cv::split(float_img, chw);

	return resized;
}

std::vector<Detection> TrafficLights::postprocessImage(float* output, int num_detections, float conf_thresh, float nms_thresh){
	std::vector<Detection> detections;

	// confidence thresholding	
	for (int i = 0; i < num_detections; i++) {
		float* det = &output[i * 9];
		float obj_conf = det[4];
		if (obj_conf < conf_thresh) continue;

		float* class_scores = &det[5];
		int class_id = std::max_element(class_scores, class_scores + 4) - class_scores;
		float class_conf = class_scores[class_id];
		float final_conf = obj_conf * class_conf;
		if (final_conf < conf_thresh) continue;
		
		Detection d;
		d.x = det[0];
		d.y = det[1];
		d.w = det[2];
		d.h = det[3];
		d.conf = final_conf;
		d.class_id = class_id;
		detections.push_back(d);
	}


	// NMS
	std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b){
		return a.conf > b.conf;
	});

	std::vector<Detection> results;
	std::vector<bool> suppressed(detections.size(), false);

	for (size_t i = 0; i < detections.size(); ++i){
		if (suppressed[i]) continue;
		results.push_back(detections[i]);
		for (size_t j = i + 1; j < detections.size(); ++j){
			if (suppressed[j]) continue;
			if (computeIoU(detections[i], detections[j]) > nms_thresh) suppressed[j] = true;
		}
	}

	return results;
}

float TrafficLights::computeIoU(const Detection& a, const Detection& b){

	float x1 = std::max(a.x - a.w / 2, b.x - b.w / 2);
	float y1 = std::max(a.y - a.h / 2, b.y - b.h / 2);
	float x2 = std::max(a.x + a.w / 2, b.x + b.w / 2);
	float y2 = std::max(a.x + a.w / 2, b.x + b.w / 2);

	float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
	float union_area = a.w * a.h + b.w * b.h - inter_area;
	return union_area > 0 ? inter_area / union_area : 0.0f;
}

void TrafficLights::inferenceLoop(cv::Mat frame) {
    	std::cout << "Running inference loop..." << std::endl;

	return;
}

