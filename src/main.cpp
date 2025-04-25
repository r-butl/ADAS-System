#include <iostream>
#include <pthread.h>
#include "services/read_frame.hpp"
#include "datastructures/frame_buffer.hpp"

int main() {
    std::cout << "Starting frame reader service..." << std::endl;

    FrameBuffer frameBuffer; // Double buffer

    FrameReaderArgs args;
    args.frameBuffer = &frameBuffer;
    args.source = "../../video.mp4";

    pthread_t frameReaderThreadID;
    pthread_create(&frameReaderThreadID, nullptr, frameReaderThread, &args);

    // Simulate other services grabbing frames
    cv::Mat frame;
    uint64_t lastFrameVersion = 0;

    while (true) {
        bool newFrame = frameBuffer.getLatestFrame(frame, lastFrameVersion);

	if (newFrame){
		cv::imshow("video", frame);
        	std::cout << "Service grabbed a frame of size: " << frame.rows << "x" << frame.cols << std::endl;
	}else{
		std::cout << "Frame is old" << std::endl;	
	}
	
	cv::waitKey(1);
    }

    // Wait for the frame reader thread to finish (in a real application, you'd handle this differently)
    pthread_join(frameReaderThreadID, nullptr);

    return 0;
}
