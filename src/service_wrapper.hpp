#ifndef SERVICE_WRAPPER_HPP
#define SERVICE_WRAPPER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>
#include <atomic>
#include <sched.h>
#include <thread>
#include <chrono>

template <typename T>
struct serviceWrapperArgs {
    std::function<std::vector<T>(cv::Mat&)> processFunction;
    cv::Mat* frameBuffer;
    std::vector<T>* outputStore;
    std::atomic<uint8_t>* frameReadyFlag;
    std::atomic<uint8_t>* processingDoneFlag;
    uint8_t activeBit;
    std::atomic<bool>* stopFlag;
};

template <typename T>
void* ServiceWrapperThread(void* args) {
    serviceWrapperArgs<T>* argsStruct = static_cast<serviceWrapperArgs<T>*>(args);
    std::function<std::vector<T>(cv::Mat&)> processFunction = argsStruct->processFunction;
    cv::Mat* frameBuffer = argsStruct->frameBuffer;
    std::vector<T>* outputStore = argsStruct->outputStore;
    std::atomic<uint8_t>* frameReadyFlag = argsStruct->frameReadyFlag;
    std::atomic<uint8_t>* processingDoneFlag = argsStruct->processingDoneFlag;
    uint8_t activeBit = argsStruct->activeBit;
    std::atomic<bool>* stopFlag= argsStruct->stopFlag;

    cv::Mat localFrame;

    while (!stopFlag->load()) {

        // Wait for the frameReadyFlag to be set to 1
        while ((frameReadyFlag->load() & activeBit) == 0 && !stopFlag->load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Copy the frame from the frameBuffer to localFrame
        localFrame = *frameBuffer;

        // Reset the frameReadyFlag to 0
        *frameReadyFlag &= ~activeBit;
        
        // Process the frame
        std::vector<T> result = processFunction(*frameBuffer);

        // Wait for the processingDoneFlag to be set to 0
        while ((*processingDoneFlag & activeBit) != 0 && !stopFlag->load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Write the result to the output store
        *outputStore = result;

        // Flip the processingDoneFlag to 1
        *processingDoneFlag |= activeBit;
    }
}

#endif // SERVICE_WRAPPER_HPP