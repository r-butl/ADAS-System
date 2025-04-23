#ifndef READ_FRAME_HPP
#define READ_FRAME_HPP

#include <opencv2/opencv.hpp>
#include <mutex>
#include <condition_variable>
#include <vector>

class FrameBuffer {
public:
    FrameBuffer(size_t size);
    void addFrame(const cv::Mat& frame);
    cv::Mat getLatestFrame();
    bool isFrameAvailable();

private:
    std::vector<cv::Mat> buffer;
    size_t bufferSize;
    size_t latestIndex;
    std::mutex bufferMutex;
    std::condition_variable frameAvailable;
    bool frameReady;
};

struct FrameReaderArgs {
	FrameBuffer* frameBuffer;
	std::string source;
};

// Thread function for reading frames
void* frameReaderThread(void* arg);

#endif // READ_FRAME_HPP
