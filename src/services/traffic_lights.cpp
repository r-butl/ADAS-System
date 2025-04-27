#include <fstream>
#include "cuda_runtime_api.h"
#include <iostream>
#include <vector>
#include "frame_buffer.hpp"
#include "annotations_buffer.hpp"
#include "traffic_lights.hpp"

#include <NvInfer.h>
using namespace nvinfer1;

TrafficLights::TrafficLights(const std::string& enginePath)
    : enginePath(enginePath), engine(nullptr), context(nullptr) {


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

    /*
    int nbBindings = engine->getNbBindings();
    for (int i = 0; i < nbBindings; ++i){
	std::cout << "Binding " << i << ": " << engine->getBindingName(i) << std::endl;
    }

    inputIndex = engine->getBindingIndex("images");
    outputIndex = engine->getBindingIndex("output0");

    cudaMalloc(&buffers[inputIndex], 3 * inputW * inputH * sizeof(float));
    cudaMalloc(&buffers[outputIndex], outputSize * 9 * sizeof(float));

    cudaStreamCreate(&stream);
    */

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

	cv::cuda::GpuMat gpu_frame;
      	gpu_frame.upload(frame);
	

}

