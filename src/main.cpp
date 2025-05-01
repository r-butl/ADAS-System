#include <iostream>
#include <pthread.h>
#include "services/read_frame.hpp"
#include "datastructures/frame_buffer.hpp"
#include "services/carDet.hpp"
#include "services/traffic_lights.hpp"
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
    
    // Read frame thread
    FrameBuffer frameBuffer;

    FrameReaderArgs args;
    args.frameBuffer = &frameBuffer;
    args.source = "../../video.mp4";

    pthread_t frameReaderThreadID;
    pthread_create(&frameReaderThreadID, nullptr, frameReaderThread, &args);


    std::string engine_path = "./services/tl_detect.onnx";
    TrafficLights trafficLights(engine_path);


	// Annotations buffer
	std::vector<Rect> cars;
	std::vector<Detection> traffic_lights;
	// pedestrians
	// lane lines
	
    // Simulate other services grabbing frames
    cv::Mat frame;
    uint64_t lastFrameVersion = 0;
    char winInput;
    
    // fps
    double fps;
    int64 start, end;
    
    vector<Rect> test;  // for cars ;;;; testing

    while (true) {
    start = cv::getTickCount();
        bool newFrame = frameBuffer.getLatestFrame(frame, lastFrameVersion);

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
