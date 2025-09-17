#include <iostream>
#include <pthread.h>
#include "services/read_frame.hpp"
#include "services/carDet.hpp"
#include "services/traffic_lights.hpp"
#include "service_wrapper.hpp"
#include "services/combine_draw.hpp"
#include "services/people_detect.hpp"
#include "services/lane_lines.hpp"
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

int main(int argc, char** argv) {

    // Global variables
    cv::Mat currentFrame;
    std::atomic<bool> stopFlag = false;
    
	// Annotations buffer
	std::vector<Rect> vCarAnnotations;
	std::vector<Rect> vTrafficLightsAnnotations;
	std::vector<Rect> vPeopleAnnotations;
    std::vector<cv::Vec4i> vLaneLines;

    // Sequencer flags
    std::atomic<uint8_t> uFrameReadyFlag(0);
    std::atomic<uint8_t> uProcessingDoneFlag(0);

    // Read frame thread
    cv::Mat frameBuffer;

    // Detector classes
    TrafficLights trafficLights("./services/tl_detect.onnx");
    SimplePeopleDetector peopleDetector("./services/people_detect.onnx");
    CarDetector carsDetector("./cars.xml");
//    CarDetector carsDetector("./carLBP.xml"); //lbp version
	      

    // Read Frame Service Initialization
    FrameReaderArgs frameReaderArgs;
    frameReaderArgs.frameBuffer = &frameBuffer;

	if (argc > 1) {
		frameReaderArgs.source = argv[1];
	} else {
		frameReaderArgs.source = "0";
	}

    frameReaderArgs.frameReadyFlag = &uFrameReadyFlag;
    frameReaderArgs.numServices = 5;      // CRITICAL: needs to be # of annotation services + 1 draw service
    frameReaderArgs.stopFlag = &stopFlag;

    pthread_t frameReaderThreadID;
    pthread_create(&frameReaderThreadID, nullptr, frameReaderThread, &frameReaderArgs);
    setThreadAffinity(frameReaderThreadID, 0); 

    // Traffic lights service Initialization
    serviceWrapperArgs<Rect> trafficLightsArgs;
    trafficLightsArgs.processFunction = [&trafficLights](cv::Mat& frame) { return trafficLights.detect(frame); };    
    trafficLightsArgs.frameBuffer = &frameBuffer;
    trafficLightsArgs.outputStore = &vTrafficLightsAnnotations;
    trafficLightsArgs.frameReadyFlag = &uFrameReadyFlag;
    trafficLightsArgs.processingDoneFlag = &uProcessingDoneFlag;
    trafficLightsArgs.activeBit = 0x01;           // CRITICAL: Needs to be a unique bit for each service & farthest left of all annotation services   
    trafficLightsArgs.stopFlag = &stopFlag;

    pthread_t trafficLightsThreadID;
    pthread_create(&trafficLightsThreadID, nullptr, ServiceWrapperThread<Rect>, &trafficLightsArgs);
    setThreadAffinity(trafficLightsThreadID, 1); 

    // Car detection service Initialization
    serviceWrapperArgs<cv::Rect> carDetectionArgs;
    carDetectionArgs.processFunction = [&carsDetector](cv::Mat& frame) { return carsDetector.detectCars(frame); };
    carDetectionArgs.frameBuffer = &frameBuffer;
    carDetectionArgs.outputStore = &vCarAnnotations;
    carDetectionArgs.frameReadyFlag = &uFrameReadyFlag;
    carDetectionArgs.processingDoneFlag = &uProcessingDoneFlag;
    carDetectionArgs.activeBit = 0x02;            // CRITICAL: Needs to be a unique bit for each service & farthest left of all annotation services
    carDetectionArgs.stopFlag = &stopFlag;

    pthread_t carDetectionThreadID;
    pthread_create(&carDetectionThreadID, nullptr, ServiceWrapperThread<cv::Rect>, &carDetectionArgs);
    setThreadAffinity(carDetectionThreadID, 2);

    // Pedestrains detection service Initialization
    serviceWrapperArgs<cv::Rect> peopleDetectionArgs;
    peopleDetectionArgs.processFunction = [&peopleDetector](cv::Mat& frame) { return peopleDetector.detect(frame); };
    peopleDetectionArgs.frameBuffer = &frameBuffer;
    peopleDetectionArgs.outputStore = &vPeopleAnnotations;
    peopleDetectionArgs.frameReadyFlag = &uFrameReadyFlag;
    peopleDetectionArgs.processingDoneFlag = &uProcessingDoneFlag;
    peopleDetectionArgs.activeBit = 0x04;         // CRITICAL: Needs to be a unique bit for each service & farthest left of all annotation services
    peopleDetectionArgs.stopFlag = &stopFlag;

    pthread_t peopleDetectionThreadID;
    pthread_create(&peopleDetectionThreadID, nullptr, ServiceWrapperThread<cv::Rect>, &peopleDetectionArgs);
    setThreadAffinity(peopleDetectionThreadID, 3);

    // Lane lines service Initialization
    serviceWrapperArgs<Vec4i> laneLinesArgs;
    laneLinesArgs.processFunction = &laneDetection;
    laneLinesArgs.frameBuffer = &frameBuffer;
    laneLinesArgs.outputStore = &vLaneLines;
    laneLinesArgs.frameReadyFlag = &uFrameReadyFlag;
    laneLinesArgs.processingDoneFlag = &uProcessingDoneFlag;
    laneLinesArgs.activeBit = 0x08; 
    laneLinesArgs.stopFlag = &stopFlag;

    pthread_t laneLinesDetectionThreadID;
    pthread_create(&laneLinesDetectionThreadID, nullptr, ServiceWrapperThread<Point2f>, &laneLinesArgs);
    setThreadAffinity(laneLinesDetectionThreadID, 4);


    // Draw frame service Initialization
    DrawFrameArgs drawFrameArgs;
    drawFrameArgs.frameBuffer = &frameBuffer;
    drawFrameArgs.windowName = "Annotated Frame";
    drawFrameArgs.frameReadyFlag = &uFrameReadyFlag;
    drawFrameArgs.processingDoneFlag = &uProcessingDoneFlag;
    drawFrameArgs.activeBit = 0x10;        // CRITICAL: Needs to be a unique bit for each service & farthest left of all annotation services
    drawFrameArgs.numServices = 4;         // CRITICAL: needs to be # of annotation services
    drawFrameArgs.stopFlag = &stopFlag;
    drawFrameArgs.trafficLights = &vTrafficLightsAnnotations;
    drawFrameArgs.people = &vPeopleAnnotations;
    drawFrameArgs.cars = &vCarAnnotations;
    drawFrameArgs.laneLinePoints = &vLaneLines;

    pthread_t drawFrameThreadID;
    pthread_create(&drawFrameThreadID, nullptr, DrawFrameThread, &drawFrameArgs);
    setThreadAffinity(drawFrameThreadID, 5); 

    while (true) {
        continue;
    }

    // Wait for the frame reader thread to finish (in a real application, you'd handle this differently)
    pthread_join(frameReaderThreadID, nullptr);
    pthread_join(trafficLightsThreadID, nullptr);
    pthread_join(carDetectionThreadID, nullptr);
    pthread_join(laneLinesDetectionThreadID, nullptr);
    pthread_join(peopleDetectionThreadID, nullptr);
    pthread_join(drawFrameThreadID, nullptr);

    return 0;
}
