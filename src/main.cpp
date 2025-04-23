#include <iostream>
#include <pthread.h>
#include "services/read_frame.hpp"
#include "unimplemented/car/carDet.cpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"

#define ESCAPE_KEY (27)

int main() {
    std::cout << "Starting frame reader service..." << std::endl;

    // Create the frame buffer
    FrameBuffer frameBuffer(10); // Circular buffer with 10 slots

    // Create the frame reader thread
    pthread_t frameReaderThreadID;
    pthread_create(&frameReaderThreadID, nullptr, frameReaderThread, &frameBuffer);

    // Simulate other services grabbing frames
    //for (int i = 0; i < 5; ++i) {
      //  cv::Mat frame = frameBuffer.getLatestFrame();
        //std::cout << "Service grabbed a frame of size: " << frame.rows << "x" << frame.cols << std::endl;
    //}
    
    cv::namedWindow("video_display");
    cv::imshow("video_display", frameBuffer.getLatestFrame());
    cv::waitKey();
    cv::destroyWindow("video_display");
    

    // Wait for the frame reader thread to finish (in a real application, you'd handle this differently)
    pthread_join(frameReaderThreadID, nullptr);

    return 0;
}
