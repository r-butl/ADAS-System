#include <iostream>
#include <pthread.h>
#include "services/read_frame.hpp"
#include "services/carDet.hpp"
#include "services/traffic_lights.hpp"
#include "service_wrapper.hpp"
#include "services/combine_draw.hpp"
#include <string>

#define ESCAPE_KEY (27)

// template <typename T>
// void drawAnnotations(cv::Mat& image, const std::vector<T>& annotations) {
//     for (const auto& annotation : annotations) {
//         cv::rectangle(image, cv::Point(annotation.x, annotation.y), 
//                       cv::Point(annotation.x + annotation.w, annotation.y + annotation.h), 
//                       cv::Scalar(0, 255, 0), 2); // green boxes, thickness=2
//     }
// }

// std::vector<cv::Rect> detectionsToRects(const std::vector<Detection>& detections) {
//     std::vector<cv::Rect> rects;
//     rects.reserve(detections.size()); // reserve memory

//     for (const auto& det : detections) {
//         int x = static_cast<int>(det.x);
//         int y = static_cast<int>(det.y);
//         int w = static_cast<int>(det.w);
//         int h = static_cast<int>(det.h);
//         rects.emplace_back(x, y, w, h);
//     }

//     return rects;
// }

// void drawRectangles(cv::Mat& image, const std::vector<cv::Rect>& rects) {
//     for (const auto& rect : rects) {
//         cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2); // green boxes, thickness=2
//     }
// }

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
    
    while (true) {
        continue;

    }

    // Wait for the frame reader thread to finish (in a real application, you'd handle this differently)
    pthread_join(frameReaderThreadID, nullptr);

    return 0;
}
