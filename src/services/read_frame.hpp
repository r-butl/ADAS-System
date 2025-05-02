#ifndef READ_FRAME_HPP
#define READ_FRAME_HPP

#include <iostream>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <sched.h>
#include <atomic>
#include <thread>
#include <chrono>

// Thread function for reading frames
void* frameReaderThread(void* arg);

struct FrameReaderArgs {
        cv::Mat* frameBuffer;
        std::string source;
        std::atomic<uint8_t>* frameReadyFlag;
        std::atomic<bool>* stopFlag;
        int numServices;
};

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

        std::cout << "Read frame service is alive." << std::endl;
        cv::Mat frame;
        while (!args->stopFlag->load()){
                
                cap >> frame; // Capture a frame
                if (frame.empty()) {
                        std::cerr << "Error: Empty frame captured." << std::endl;
                        continue;
                }

                std::cout << "Frame captured. Waiting to write to the buffer." << std::endl;

                // Check if all frameReadyFlags are 0
                while ((frameReadyFlag->load() & bitmask) != 0) {
                        //sched_yield(); // Yield the CPU to other threads
                        printf("Frame Ready Flag: %d\n", frameReadyFlag->load());
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                std::cout << "Writing frame." << std::endl;
                std::cout.flush();

                // Copy the frame to the frame buffer
                frame.copyTo(*frameBuffer);

                // Set the frameReadyFlag to indicate that a new frame is ready
                frameReadyFlag->fetch_or(bitmask);
        }

        std::cout << "Read frame service is stopping." << std::endl;

        cap.release();
        return nullptr;
}
#endif // READ_FRAME_HPP
