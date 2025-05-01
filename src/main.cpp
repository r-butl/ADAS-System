#include <iostream>
#include <pthread.h>
#include "services/read_frame.hpp"
#include "datastructures/frame_buffer.hpp"
#include "services/carDet.hpp"
#include "services/traffic_lights.hpp"
#include "service_wrapper.hpp"
#include <string>

#define ESCAPE_KEY (27)

std::vector<cv::Rect> detectionsToRects(const std::vector<Detection>& detections) {
    std::vector<cv::Rect> rects;
    rects.reserve(detections.size()); // reserve memory

    for (const auto& det : detections) {
        int x = static_cast<int>(det.x);
        int y = static_cast<int>(det.y);
        int w = static_cast<int>(det.w);
        int h = static_cast<int>(det.h);
        rects.emplace_back(x, y, w, h);
    }

    return rects;
}

void drawRectangles(cv::Mat& image, const std::vector<cv::Rect>& rects) {
    for (const auto& rect : rects) {
        cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2); // green boxes, thickness=2
    }
}

int main() {
    std::cout << "Starting frame reader service..." << std::endl;

    // Global variables
    cv::Mat currentFrame;
    uint64_t lastFrameVersion = 0;
    char winInput;
    bool stopFlag = false;
    
	// Annotations buffer
	std::vector<Rect> vCarAnnotations;
	std::vector<Detection> vTrafficLightsAnnotations;
	// pedestrians
	// lane lines

    // Sequencer flags
    std::atomic<uint8_t> uFrameReadyFlag(0);
    std::atomic<uint8_t> uProcessingDoneFlag(0);

    // Read frame thread
    FrameBuffer frameBuffer;

    FrameReaderArgs args;
    args.frameBuffer = &frameBuffer;
    args.source = "../../video.mp4";
    pthread_t frameReaderThreadID;
    pthread_create(&frameReaderThreadID, nullptr, frameReaderThread, &args);

    // Traffic lights service
    std::string engine_path = "./services/tl_detect.onnx";
    TrafficLights trafficLights(engine_path);
    serviceWrapperArgs trafficLightsArgs;
    trafficLightsArgs.processFunction = &traffic_lights.inferenceLoop;    
    trafficLightsArgs.frameBuffer = &frameBuffer;
    trafficLightsArgs.outputStore = &vTrafficLightsAnnotations;
    trafficLightsArgs.frameReadyFlag = &uFrameReadyFlag;
    trafficLightsArgs.processingDoneFlag = &uProcessingDoneFlag;
    trafficLightsArgs.activeBit = 0x01;
    trafficLightsArgs.stopFlag = &stopFlag;

    pthread_t trafficLightsThreadID;
    pthread_create(&trafficLightsThreadID, nullptr, ServiceWrapperProcess<Detection>, &trafficLightsArgs);

    // Car detection service
    serviceWrapperArgs carDetectionArgs;
    carDetectionArgs.processFunction = &carDetection;
    carDetectionArgs.frameBuffer = &frameBuffer;
    carDetectionArgs.outputStore = &vCarAnnotations;
    carDetectionArgs.frameReadyFlag = &uFrameReadyFlag;
    carDetectionArgs.processingDoneFlag = &uProcessingDoneFlag;
    carDetectionArgs.activeBit = 0x02;
    carDetectionArgs.stopFlag = &stopFlag;

    pthread_t carDetectionThreadID;
    pthread_create(&carDetectionThreadID, nullptr, ServiceWrapperProcess<Rect>, &carDetectionArgs);

    // fps
    double fps;
    int64 start, end;
    
    while (true) {
    start = cv::getTickCount();
        bool newFrame = frameBuffer.getLatestFrame(currentFrame, lastFrameVersion);

	if (newFrame && !frame.empty()){
		trafficLights.inferenceLoop(frame, traffic_lights);
		lastFrameVersion++;
		carDetection(frame, cars);

		drawRectangles(frame, cars);
		drawRectangles(frame, detectionsToRects(traffic_lights));
		
		// FPS
		end = cv::getTickCount();
		double duration = (end - start) / cv::getTickFrequency();
		fps = 1.0/duration;
		cv::putText(frame, "FPS: " + std::to_string(int(fps)), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,255,0), 2);								// ==================
		cv::imshow("video", frame);
		if ((winInput = waitKey(10)) == ESCAPE_KEY){
		  break;
		}
		
	}
	
    }
    cv::destroyWindow("video_display");
    

    // Wait for the frame reader thread to finish (in a real application, you'd handle this differently)
    pthread_join(frameReaderThreadID, nullptr);

    return 0;
}
