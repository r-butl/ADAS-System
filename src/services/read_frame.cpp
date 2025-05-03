// This service reads the input frame and distributes it to whatever service needs it

#include "read_frame.hpp"
#include <iostream>
#include <string.h>
#include "frame_buffer.hpp"
#include <opencv2/opencv.hpp>

void* frameReaderThread(void* arg) {
    FrameReaderArgs* args = static_cast<FrameReaderArgs*>(arg);
    cv::Mat* frameBuffer = args->frameBuffer;
    std::atomic<uint8_t>* frameReadyFlag = args->frameReadyFlag;

    if (args->numServices > 8) {
        std::cerr << "Error: Number of services exceeds 8 bits." << std::endl;
        std::abort();
    }

    uint8_t bitmask = (1 << args->numServices) - 1; // Create a bitmask for the number of services

    cv::VideoCapture cap; 

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
    while (!args->stopFlag){
        
        cap >> frame; // Capture a frame
        if (frame.empty()) {
            std::cerr << "Error: Empty frame captured." << std::endl;
            continue;
        }

        // Check if all frameReadyFlags are 0
        while ((*frameReadyFlag & bitmask) != 0) {
            sched_yield(); // Yield the CPU to other threads
        }

        // Copy the frame to the frame buffer
        frame.copyTo(*frameBuffer);

        // Set the frameReadyFlag to indicate that a new frame is ready
        *frameReadyFlag |= bitmask;
    }

    cap.release();
    return nullptr;
}
