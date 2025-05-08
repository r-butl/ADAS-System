#include "traffic_lights.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <functional>
#include <algorithm>
#include <cstring>
#include <cctype>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>
#include <cuda_runtime_api.h>
#include "NvOnnxParser.h"

using namespace cv;
using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;

#define CHECK(status) do { auto ret = (status); if (ret != 0) cerr << "CUDA error: " << ret << endl; } while(0)

class SimpleLogger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << msg << std::endl;
    }
};

static SimpleLogger gLogger;

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

        printf("Creating engine...\n");
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
    cudaStreamCreate(&stream);

    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT && !input_tensor_name_) {
            input_tensor_name_ = name;
            std::cout << "[DEBUG] Input tensor name: " << input_tensor_name_ << std::endl;
        } else if (engine->getTensorIOMode(name) == TensorIOMode::kOUTPUT && !output_tensor_name_) {
            output_tensor_name_ = name;
            std::cout << "[DEBUG] Output tensor name: " << output_tensor_name_ << std::endl;
        }
    }

    if (!input_tensor_name_ || !output_tensor_name_) {
        cerr << "ERROR: Failed to find input/output tensor names." << endl; return;
    }

    Dims d = engine->getTensorShape(input_tensor_name_);
    printf("Input h: %d Input w: %d\n", d.d[2], d.d[3]);

    if (d.nbDims == 4 && d.d[0] == 1 && d.d[1] == 3) {
        network_input_h_ = d.d[2];
        network_input_w_ = d.d[3];
    } else {
        cout << "ERROR: Unexpected input dimensions." << endl;
    }
}

TrafficLights::~TrafficLights() {
    if (context) delete context;
    if (engine) delete engine;
}

bool TrafficLights::isInitialized() const {
    return runtime && engine && context && stream && network_input_h_ > 0 && network_input_w_ > 0;
}

size_t TrafficLights::calculateSizeFromDims(const Dims& dims) {
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) return 0;
        size *= dims.d[i];
    }
    return size;
}

void TrafficLights::preprocess(const Mat& frame, std::vector<float>& cpu_input_buffer, int input_w, int input_h) {
    Mat resized, rgb;
    resize(frame, resized, Size(input_w, input_h));
    cvtColor(resized, rgb, COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    cpu_input_buffer.resize(3 * input_h * input_w);
    vector<Mat> channels(3);
    split(rgb, channels);

    float* ptr = cpu_input_buffer.data();
    for (int c = 0; c < 3; ++c) {
        memcpy(ptr, channels[c].data, input_h * input_w * sizeof(float));
        ptr += input_h * input_w;
    }
}

std::vector<cv::Rect> TrafficLights::detect(const Mat& frame) {

    vector<float> input;
    vector<float> output;
    vector<cv::Rect> detections;
    float conf_threshold = 0.3;

    preprocess(frame, input, network_input_w_, network_input_h_);
    
    context->setInputShape(input_tensor_name_, Dims4{1, 3, network_input_h_, network_input_w_});
    Dims in_dims = context->getTensorShape(input_tensor_name_);
    Dims output_dims = context->getTensorShape(output_tensor_name_);

    size_t in_size = calculateSizeFromDims(in_dims) * sizeof(float);
    size_t out_size = calculateSizeFromDims(output_dims) * sizeof(float);

    output.resize(out_size / sizeof(float));

    void* d_in = nullptr; void* d_out = nullptr;
    CHECK(cudaMalloc(&d_in, in_size));
    CHECK(cudaMalloc(&d_out, out_size));
    CHECK(cudaMemcpyAsync(d_in, input.data(), in_size, cudaMemcpyHostToDevice, stream));

    context->setTensorAddress(input_tensor_name_, d_in);
    context->setTensorAddress(output_tensor_name_, d_out);
    context->enqueueV3(stream);
    CHECK(cudaMemcpyAsync(output.data(), d_out, out_size, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    cudaFree(d_in);
    cudaFree(d_out);

    if (!output.empty()){
        int count = 0;
        const int props = 6;
        if (output_dims.nbDims == 3 && output_dims.d[2] == props) count = output_dims.d[1];
        else if (output_dims.nbDims == 2 && output_dims.d[1] == props) count = output_dims.d[0];

        int valid = 0;
        for (int i = 0; i < count; ++i) {
            float* det = output.data() + i * props;
            float confidence = det[4];
            if (confidence < 0.5f) continue;

            // if (valid < 5) {
            //     std::cout << "[DEBUG] TL Detection " << i << " - Conf: " << confidence
            //                 << ", Box: [" << det[0] << ", " << det[1] << ", " << det[2] << ", " << det[3] << "]" << std::endl;
            // }
            valid++;
        }
       //std::cout << "[DEBUG] TL Valid detections above threshold: " << valid << std::endl;
    
        float scale_x = frame.cols / static_cast<float>(network_input_w_);
        float scale_y = frame.rows / static_cast<float>(network_input_h_);

        for (int i = 0; i < count; ++i) {
            float* det = output.data() + i * props;
            float confidence = det[4];
            if (confidence < conf_threshold) continue;
        
            // Original detection coordinates at input size (640x640)
            int x1 = static_cast<int>(det[0] * scale_x);
            int y1 = static_cast<int>(det[1] * scale_y);
            int x2 = static_cast<int>(det[2] * scale_x);
            int y2 = static_cast<int>(det[3] * scale_y);
        
            // Clamp to the frame size
            x1 = std::clamp(x1, 0, frame.cols - 1);
            y1 = std::clamp(y1, 0, frame.rows - 1);
            x2 = std::clamp(x2, 0, frame.cols - 1);
            y2 = std::clamp(y2, 0, frame.rows - 1);


            detections.push_back(Rect(Point(x1, y1), Point(x2, y2)));
        }

    }
    return detections;
}

int TrafficLights::getInputWidth() const { return network_input_w_; }
int TrafficLights::getInputHeight() const { return network_input_h_; }
