// This service reads the input frame and distributes it to whatever service needs it

#include "read_frame.hpp"
#include <iostream>
#include <string.h>

FrameBuffer::FrameBuffer(size_t size)
    : bufferSize(size), latestIndex(0), frameReady(false) {
    buffer.resize(size);
}

void FrameBuffer::addFrame(const cv::Mat& frame) {
    std::unique_lock<std::mutex> lock(bufferMutex);
    buffer[latestIndex] = frame.clone(); // Store a copy of the frame
    latestIndex = (latestIndex + 1) % bufferSize;
    frameReady = true;
    frameAvailable.notify_all(); // Notify waiting threads
}

cv::Mat FrameBuffer::getLatestFrame() {
    std::unique_lock<std::mutex> lock(bufferMutex);
    while (!frameReady) {
        frameAvailable.wait(lock); // Wait for a new frame
    }
    return buffer[(latestIndex + bufferSize - 1) % bufferSize].clone(); // Return the latest frame
}

bool FrameBuffer::isFrameAvailable() {
    std::lock_guard<std::mutex> lock(bufferMutex);
    return frameReady;
}

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
        frameBuffer->addFrame(frame); // Add the frame to the buffer
    }

    cap.release();
    return nullptr;
}
