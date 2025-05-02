#include <iostream>
#include <pthread.h>
#include "services/read_frame.hpp"
#include "services/carDet.hpp"
#include "services/traffic_lights.hpp"
#include "service_wrapper.hpp"
#include "services/combine_draw.hpp"
#include <string>

#define ESCAPE_KEY (27)

void setThreadAffinity(pthread_t thread, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);           // Clear the CPU set
    CPU_SET(core_id, &cpuset);   // Add the desired core to the set

    int result = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
        std::cerr << "Error setting thread affinity: " << strerror(result) << std::endl;
    } else {
        std::cout << "Thread bound to core " << core_id << std::endl;
    }
}

int main() {
    std::cout << "Starting frame reader service..." << std::endl;

    // Global variables
    cv::Mat currentFrame;
    std::atomic<bool> stopFlag = false;
    
	// Annotations buffer
	std::vector<Rect> vCarAnnotations;
	std::vector<Detection> vTrafficLightsAnnotations;
	// pedestrians
	// lane lines

    // Sequencer flags
    std::atomic<uint8_t> uFrameReadyFlag(0);
    std::atomic<uint8_t> uProcessingDoneFlag(0);

    // Read frame thread
    cv::Mat frameBuffer;

    FrameReaderArgs frameReaderArgs;
    frameReaderArgs.frameBuffer = &frameBuffer;
    frameReaderArgs.source = "../../video.mp4";
    frameReaderArgs.frameReadyFlag = &uFrameReadyFlag;
    frameReaderArgs.numServices = 3;                             // CRITICAL: needs to be # of annotation services + 1 draw service
    frameReaderArgs.stopFlag = &stopFlag;

    pthread_t frameReaderThreadID;
    pthread_create(&frameReaderThreadID, nullptr, frameReaderThread, &frameReaderArgs);
    setThreadAffinity(frameReaderThreadID, 0); 

    // Traffic lights service
    std::string engine_path = "./services/tl_detect.onnx";
    TrafficLights trafficLights(engine_path);
    serviceWrapperArgs<Detection> trafficLightsArgs;
    trafficLightsArgs.processFunction = [&trafficLights](cv::Mat& frame) { return trafficLights.inferenceLoop(frame); };    
    trafficLightsArgs.frameBuffer = &frameBuffer;
    trafficLightsArgs.outputStore = &vTrafficLightsAnnotations;
    trafficLightsArgs.frameReadyFlag = &uFrameReadyFlag;
    trafficLightsArgs.processingDoneFlag = &uProcessingDoneFlag;
    trafficLightsArgs.activeBit = 0x02;                     // Need to be unique bit for each service   
    trafficLightsArgs.stopFlag = &stopFlag;

    pthread_t trafficLightsThreadID;
    pthread_create(&trafficLightsThreadID, nullptr, ServiceWrapperThread<Detection>, &trafficLightsArgs);
    setThreadAffinity(trafficLightsThreadID, 1); 

    // Car detection service
    serviceWrapperArgs<cv::Rect> carDetectionArgs;
    carDetectionArgs.processFunction = &carDetection;
    carDetectionArgs.frameBuffer = &frameBuffer;
    carDetectionArgs.outputStore = &vCarAnnotations;
    carDetectionArgs.frameReadyFlag = &uFrameReadyFlag;
    carDetectionArgs.processingDoneFlag = &uProcessingDoneFlag;
    carDetectionArgs.activeBit = 0x01;                          // Need to be unique bit for each service
    carDetectionArgs.stopFlag = &stopFlag;

    pthread_t carDetectionThreadID;
    pthread_create(&carDetectionThreadID, nullptr, ServiceWrapperThread<cv::Rect>, &carDetectionArgs);
    setThreadAffinity(carDetectionThreadID, 2);

  // Draw frame service
    DrawFrameArgs drawFrameArgs;
    drawFrameArgs.frameBuffer = &frameBuffer;
    drawFrameArgs.windowName = "Annotated Frame";
    drawFrameArgs.frameReadyFlag = &uFrameReadyFlag;
    drawFrameArgs.processingDoneFlag = &uProcessingDoneFlag;
    drawFrameArgs.activeBit = 0x04;                     // Need to be unique bit for each service    
    drawFrameArgs.numServices = 2;                              // CRITICAL: needs to be # of annotation services
    drawFrameArgs.stopFlag = &stopFlag;

    pthread_t drawFrameThreadID;
    pthread_create(&drawFrameThreadID, nullptr, DrawFrameThread, &drawFrameArgs);
    setThreadAffinity(drawFrameThreadID, 3); // Bind to core 0

    while (true) {
        continue;

    }

    // Wait for the frame reader thread to finish (in a real application, you'd handle this differently)
    pthread_join(frameReaderThreadID, nullptr);
    pthread_join(trafficLightsThreadID, nullptr);
    pthread_join(carDetectionThreadID, nullptr);
    pthread_join(drawFrameThreadID, nullptr);

    return 0;
}
