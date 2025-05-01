#ifndef FRAME_BUFFER_HPP
#define FRAME_BUFFER_HPP

#include <atomic>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <condition_variable>
#include <vector>

// Double buffer with atomic copies and a frame versioning system to ensure each time 
// 	a frame is requested, only fresh ones are allowed
class FrameBuffer {
public:
	FrameBuffer() {
		currentFrame.store(&buffer1);
	}

	void updateFrame(const cv::Mat& newFrame) {
		newFrame.copyTo(*backBuffer());	// copy the new frame into the older buffer
		currentFrame.store(backBuffer(), std::memory_order_release);	// point the current buffer to the unseen buffer
		version.fetch_add(1, std::memory_order_release); // add 1 to the frame version counter
		swapBuffers();	// set the old current buffer as the hidden buffer
	}

	bool getLatestFrame(cv::Mat& outFrame, uint64_t& lastSeenVersion) {
		uint64_t currentVersion = version.load(std::memory_order_acquire); // load the current frame version
		if (currentVersion != lastSeenVersion) {
			lastSeenVersion = currentVersion;	// Check frame version
			outFrame = currentFrame.load(std::memory_order_acquire)->clone(); // atomic copy of current frame
			return true;
		}
		return false;
	}

private:
	cv::Mat buffer1, buffer2;		// double buffer
	std::atomic<cv::Mat*> currentFrame;	// atomic pointer to current buffer
	std::atomic<uint64_t> version{0};	// frame version counter
	bool backIsBuffer1 = false;		// keeps track of which buffer is the back buffer
	
	cv::Mat* backBuffer() {
		return backIsBuffer1 ? &buffer1 : &buffer2;	// returns a pointer to the back buffer
	}

	void swapBuffers() {
		backIsBuffer1 = !backIsBuffer1;		// Toggle the buffer
	}
};


#endif
