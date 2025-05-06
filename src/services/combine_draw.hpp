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
#include <traffic_lights.hpp>

// Thread function for reading frames
void* drawFrameThread(void* arg);

struct DrawFrameArgs {
    cv::Mat* frameBuffer;
    std::string windowName;
    std::atomic<uint8_t>* frameReadyFlag;
    std::atomic<uint8_t>* processingDoneFlag;
    uint8_t activeBit;
    int numServices;
    // annotations
    std::atomic<bool>* stopFlag;
    std::vector<Detection>* trafficLights;
    std::vector<Rect>* cars;
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
    auto lastTime = std::chrono::high_resolution_clock::now(); // Initialize the timer
    double fps = 0.0;
 
    while (!stopFlag->load()) {

        // Wait for the frameReadyFlag to be set to 1
        while ((frameReadyFlag->load() & activeBit) == 0 && !stopFlag->load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Sleep for 1 ms
        }

        // Copy the frame from the frameBuffer to localFrame
        frame = *frameBuffer;
        *frameReadyFlag &= ~activeBit;

        // Wait for the processingDoneFlag to be set to 0
        while ((*processingDoneFlag & bitmask) != bitmask && !stopFlag->load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Sleep for 1 ms
        }

        /////////////////////////// Annotations ///////////////////////////

        // Draw traffic lights annotations
        for (const auto& annotation : *args->trafficLights) {
            cv::rectangle(frame, cv::Point(annotation.x, annotation.y), 
                          cv::Point(annotation.x + annotation.w, annotation.y + annotation.h), 
                          cv::Scalar(0, 255, 0), 2); // green boxes, thickness=2
        }


        // draw cars annotations
        for (const auto& rect: *args->cars) {
            cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 2); // red boxes, thickness=2
        }

        ////////////////////////// End Annotations ///////////////////////////

        // Flip the processingDoneFlag to 0
        processingDoneFlag->store(0);

        // Calculate FPS
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = currentTime - lastTime;
        lastTime = currentTime;
        fps = 1.0 / elapsed.count();

        // Draw FPS on the frame
        std::string fpsText = "FPS: " + std::to_string(static_cast<int>(fps));
        cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        
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

        std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Sleep for 1 ms


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