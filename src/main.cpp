#include <iostream>
#include <pthread.h>
#include "services/read_frame.hpp"
#include "datastructures/frame_buffer.hpp"
#include "services/carDet.hpp"

#define ESCAPE_KEY (27)

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
    char winInput;
    
    
    vector<Rect> test;  // for cars ;;;; testing

    while (true) {
        bool newFrame = frameBuffer.getLatestFrame(frame, lastFrameVersion);

	if (newFrame && !frame.empty()){
		lastFrameVersion++;
		test = carDetection(frame);					// for cars;; testing
		for(auto& car: test){
			cv::rectangle(frame, car, Scalar(0,255,0), 2);
		}								// ==================
		cv::imshow("video", frame);
		if ((winInput = waitKey(10)) == ESCAPE_KEY){
		  break;
		}
	}
	
    }
    cv::destroyWindow("video_display");
    

    // Wait for the frame reader thread to finish (in a real application, you'd handle this differently)
    pthread_join(frameReaderThreadID, nullptr);

    return 0;
}
