// This service reads the input frame and distributes it to whatever service needs it

#include "read_frame.hpp"
#include <iostream>
#include <string.h>
#include "frame_buffer.hpp"
#include <opencv2/opencv.hpp>

void* frameReaderThread(void* arg) {
    FrameReaderArgs* args = static_cast<FrameReaderArgs*>(arg);
    FrameBuffer* frameBuffer = args->frameBuffer;

    cv::VideoCapture cap; // Open the default camera (or replace with a video file path)

   	if (args->source == "0"){
	   	cap.open(0);
	} else {
		cap.open(args->source);
	}

    	if (!cap.isOpened()) {
       		std::cerr << "Error: Unable to open video source." << std::endl;
        	return nullptr;
    	}

    cv::Mat frame;
    while (true) {
        cap >> frame; // Capture a frame
        if (frame.empty()) {
            std::cerr << "Error: Empty frame captured." << std::endl;
            break;
        }
        frameBuffer->updateFrame(frame); // Add the frame to the buffer
    }

    cap.release();
    return nullptr;
}
