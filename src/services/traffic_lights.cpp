#include <fstream>
#include "cuda_runtime_api.h"
#include <iostream>
#include <vector>
#include "traffic_lights.hpp"
#include <numeric>
#include <functional>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

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

void TrafficLights::saveEngine(const std::string& filePath, IHostMemory* serializedModel) {

	std::ofstream outFile(filePath, std::ios::binary);
	if (!outFile) {
		std::cerr << "Error: failed to open file for saving the engine." << std::endl;
		return;
	}

	outFile.write(reinterpret_cast<char*>(serializedModel->data()), serializedModel->size());
	outFile.close();

	std::cout << "Engine saved to: " << filePath << std::endl;
}

bool TrafficLights::loadEngine(const std::string& filePath){

	std::ifstream inFile(filePath, std::ios::binary);
	if (!inFile) {
		std::cerr << "Error: Failed to open engine file for loading." << std::endl;
		return false;
	}

	// Deserialize the model
    	inFile.seekg(0, std::ios::end);
    	size_t modelSize = inFile.tellg();
    	inFile.seekg(0, std::ios::beg);

    	std::vector<char> modelData(modelSize);
    	inFile.read(modelData.data(), modelSize);
    	inFile.close();

	runtime = createInferRuntime(gLogger);
	if (!runtime) {
		std::cerr << "Error: failed to create TensorRT runtime." << std::endl;
		return false;
	}

	engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size());
	if (!engine) {
		std::cerr << "Error: Failed to deserialize engine" << std::endl;
		return false;
	}

	context = engine->createExecutionContext();
	if (!context) {
		std::cerr << "Engine: failed to create execution context" << std::endl;
		return false;
	}

	return true;

}

TrafficLights::TrafficLights(const std::string& onnxPath)
    : enginePath(onnxPath), engine(nullptr), context(nullptr) {

    std::string saveFileName = "tl_detect.engine";
    if(!loadEngine(saveFileName.c_str())){
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

	saveEngine(saveFileName.c_str(), serializedModel);
    	delete serializedModel;
    	delete runtime;
	
	}
}

TrafficLights::~TrafficLights() {
    if (context) delete context;
    if (engine) delete engine;
}


std::vector<Detection> TrafficLights::postprocessImage(float* output, int num_detections, float conf_thresh, float nms_thresh, float rescale_factor_x, float rescale_factor_y){
	std::vector<Detection> detections;

	// confidence thresholding	
	for (int i = 0; i < num_detections; i++) {


		for (int i = 0; i < num_detections; ++i) {
			printf("Detection %d: ", i);
			for (int j = 0; j < 8; ++j) {
				printf("%.3f ", output[i * 8 + j]);
			}
			printf("\n");
		}

		float* det = &output[i * 8]; // x, y, w, h, conf, class0, class1, class2
		float obj_conf = det[4];

		if (obj_conf < conf_thresh) continue;

		float* class_scores = &det[5];
		int class_id = std::max_element(class_scores, class_scores + 3) - class_scores;
		float class_conf = class_scores[class_id];
		float final_conf = obj_conf * class_conf;
		if (final_conf < conf_thresh) continue;
		
		Detection d;
		d.x = det[0] * rescale_factor_x;
		d.y = det[1] * rescale_factor_y;
		d.w = det[2] * rescale_factor_x;
		d.h = det[3] * rescale_factor_y;
		d.conf = final_conf;
		d.class_id = class_id;
		detections.push_back(d);
	}

    printf("Number of detections after confidence thresholding: %zu\n", detections.size());
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

	//printf("Number of detections after NMS: %zu\n", results.size());

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

std::vector<Detection> TrafficLights::inferenceLoop(cv::Mat& frame) {

	// preprocess the image
    cv::Mat resized, resized_rgb, float_img;
	//int input_w = 736, input_h = 736;

	if(frame.empty()) {
		std::cerr << "Error: Empty frame received for inference." << std::endl;
		return {};
	}

	float scale_x = static_cast<float>(frame.cols) / inputW;
	float scale_y = static_cast<float>(frame.rows) / inputH;

	// Yolo Preprcoessing
    cv::resize(frame, resized, cv::Size(inputW, inputH));
    cv::cvtColor(resized, resized_rgb, cv::COLOR_BGR2RGB);
    resized_rgb.convertTo(float_img, CV_32FC3, 1.0f / 255.0);


	// DEBUG
	// cv::Mat gray;
	// cv::cvtColor(float_img, gray, cv::COLOR_RGB2GRAY);
	// double minVal, maxVal;
	// cv::Point minLoc, maxLoc;
	// cv::minMaxLoc(gray, &minVal, &maxVal, &minLoc, &maxLoc);
	// std::cout << "Min: " << minVal << " Max: " << maxVal << std::endl;
	
	std::vector<float> gpu_input(3 * inputH * inputW);
	
	// Split channels
	std::vector<cv::Mat> chw(3);
	for (int i = 0; i < 3; i++)
			chw[i] = cv::Mat(inputH, inputW, CV_32FC1, gpu_input.data() + i * inputH * inputW);
	cv::split(float_img, chw);

	// Set up the execution context input
	char const* input_name = "images";
	assert(engine->getTensorDataType(input_name) == nvinfer1::DataType::kFLOAT);
	auto input_dims = nvinfer1::Dims4{1, /* channels */ 3, inputH, inputW};
	context->setInputShape(input_name, input_dims);
	int input_size = std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1, std::multiplies<int>()) * sizeof(float);

	// set up the output context
	char const* output_name = "output0";

	auto output_dims = context->getTensorShape(output_name);
	int output_size = std::accumulate(output_dims.d, output_dims.d + output_dims.nbDims, 1, std::multiplies<int>()) * sizeof(float);
	
	// Allocate memory on the GPU for the operation
	void* input_mem{nullptr};
	cudaMalloc(&input_mem, input_size);
	void* output_mem{nullptr};
	cudaMalloc(&output_mem, output_size);

	// set up the cuda stream
	cudaStream_t stream;
	cudaStreamCreate(&stream);	
	cudaMemcpyAsync(input_mem, gpu_input.data(), input_size, cudaMemcpyHostToDevice, stream);

	// Run the inference
	context->setTensorAddress(input_name, input_mem);
	context->setTensorAddress(output_name, output_mem);
	bool status = context->enqueueV3(stream);
	auto output_buffer = std::unique_ptr<float[]>{new float[output_size / sizeof(float)]};
	cudaMemcpyAsync(output_buffer.get(), output_mem, output_size, cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	cudaFree(input_mem);
	cudaFree(output_mem);
	
	std::vector<Detection> detections = postprocessImage(output_buffer.get(), 10, 0.3, 0.3, scale_x, scale_y);

	// Show detections on image, make sure that the annotations are rescaled to the input frame image size
	return detections;

}

