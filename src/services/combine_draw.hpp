#ifndef DRAW_FRAME_HPP
#define DRAW_FRAME_HPP

#define ESCAPE_KEY 27 // ASCII code for the ESC key

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <sched.h>
#include <atomic>
#include <vector>
#include <thread>
#include <chrono>

// Thread function for reading frames
void* drawFrameThread(void* arg);

struct DrawFrameArgs {
    cv::Mat* frameBuffer;
    std::string windowName;
    std::atomic<uint8_t>* frameReadyFlag;
    std::atomic<uint8_t>* processingDoneFlag;
    uint8_t activeBit;
    int numServices;
    std::atomic<bool>* stopFlag;

};

void* DrawFrameThread(void* arg) {
    DrawFrameArgs* args = static_cast<DrawFrameArgs*>(arg);
    cv::Mat* frameBuffer = args->frameBuffer;
    std::string windowName = args->windowName;
    std::atomic<uint8_t>* frameReadyFlag = args->frameReadyFlag;
    std::atomic<uint8_t>* processingDoneFlag = args->processingDoneFlag;
    std::atomic<bool>* stopFlag = args->stopFlag;
    uint8_t activeBit = args->activeBit;

    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    if (args->numServices > 8) {
        std::cerr << "Error: Number of services exceeds 8 bits." << std::endl;
        std::abort();
    }

    uint8_t bitmask = (1 << args->numServices) - 1; // Create a bitmask for the number of services

    cv::Mat frame;
    while (!stopFlag->load()) {


        std::cout << "Draw: Waiting for frame..." << std::endl;
        std::cout.flush();
        // Wait for the frameReadyFlag to be set to 1
        while ((frameReadyFlag->load() & activeBit) == 0 && !stopFlag->load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Sleep for 1 ms

        }

        // Copy the frame from the frameBuffer to localFrame
        frame = *frameBuffer;

        std::cout << "Draw: Frame pulled. Waiting for services to finish previous frame." << std::endl;

        // Wait for the processingDoneFlag to be set to 0
        while ((*processingDoneFlag & bitmask) != bitmask && !stopFlag->load()) {
            printf("Processing Done Flag: %d\n", processingDoneFlag->load());
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Sleep for 1 ms
        }

        // Pull in annoations

        // Flip the processingDoneFlag to 0
        processingDoneFlag->store(0);

        // Draw rectangles on the frame (example)

        std::cout << "Draw: Drawing." << std::endl;

        // Display the frame
        if (!frame.empty()) {
            cv::imshow(windowName, frame);
        } else {
            std::cerr << "Error: Empty frame." << std::endl;
        }

        // Check for ESC key press
        if (cv::waitKey(10) == ESCAPE_KEY) {
            *stopFlag = true;
        }

        // Reset the frameReadyFlag to 0
        *frameReadyFlag &= ~activeBit;
    }

    cv::destroyWindow(windowName);

    return nullptr;
}
#endif
