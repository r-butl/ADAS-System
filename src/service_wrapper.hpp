#ifndef SERVICE_WRAPPER_HPP
#define SERVICE_WRAPPER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>
#include <atomic>
#include <sched.h>

template <typename T>
struct serviceWrapperArgs {
    std::function<std::vector<T>(cv::Mat&)> processFunction;
    cv::Mat* frameBuffer;
    std::vector<T>* outputStore;
    std::atomic<uint8_t>* frameReadyFlag;
    std::atomic<uint8_t>* processingDoneFlag;
    uint8_t activeBit;
    bool stopFlag*;
};

template <typename T>
void ServiceWrapperProcess(void *args) {

    std::function<std::vector<T>(cv::Mat&)> processFunction = args->processFunction;
    cv::Mat* frameBuffer = args->frameBuffer;
    std::vector<T>* outputStore = args->outputStore;
    std::atomic<uint8_t>* frameReadyFlag = args->frameReadyFlag;
    std::atomic<uint8_t>* processingDoneFlag = args->processingDoneFlag;
    uint8_t activeBit = args->activeBit;
    bool stopFlag* = args->stopFlag;

    cv::Mat localFrame;

    while (!stopFlag*) {
        // Wait for the frameReadyFlag to be set to 1
        while ((*frameReadyFlag & activeBit) == 0 && !*stopFlag) {
            sched_yield();
        }

        if (*stopFlag) break;

        // Copy the frame from the frameBuffer to localFrame
        localFrame = *frameBuffer;

        // Reset the frameReadyFlag to 0
        *frameReadyFlag &= ~activeBit;
        
        // Process the frame
        std::vector<T> result = processFunction(*frameBuffer);

        // Wait for the processingDoneFlag to be set to 0
        while ((*processingDoneFlag & activeBit) != 0 && !*stopFlag) {
            sched_yield();
        }

        if (*stopFlag) break;

        // Write the result to the output store
        *outputStore = result;

        // Flip the processingDoneFlag to 1
        *processingDoneFlag |= activeBit;
    }
}

#endif // SERVICE_WRAPPER_HPP