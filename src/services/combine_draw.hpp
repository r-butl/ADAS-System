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


        //std::cout << "Draw: Waiting for frame..." << std::endl;
        //std::cout.flush();
        // Wait for the frameReadyFlag to be set to 1
        while ((frameReadyFlag->load() & activeBit) == 0 && !stopFlag->load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Sleep for 1 ms

        }

        // Copy the frame from the frameBuffer to localFrame
        frame = *frameBuffer;
        // Reset the frameReadyFlag to 0
        *frameReadyFlag &= ~activeBit;
        //std::cout << "Draw: Frame pulled. Waiting for services to finish previous frame." << std::endl;

        // Wait for the processingDoneFlag to be set to 0
        while ((*processingDoneFlag & bitmask) != bitmask && !stopFlag->load()) {
            //printf("Processing Done Flag: %d\n", processingDoneFlag->load());
            //std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Sleep for 1 ms
        }

        // Pull in annoations

        // Flip the processingDoneFlag to 0
        processingDoneFlag->store(0);

        // Draw rectangles on the frame (example)

        //std::cout << "Draw: Drawing." << std::endl;

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


    }

    cv::destroyWindow(windowName);

    return nullptr;
}
#endif


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