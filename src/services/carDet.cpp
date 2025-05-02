
#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>


using namespace std;
using namespace cv;

vector<Rect> carDetection(Mat Frame){

    vector<Rect> annotations_buffer;
    String cascadename = "./cars.xml";
    CascadeClassifier carCascade;

    if (!carCascade.load(cascadename)) {
        cerr << "Error loading cascade";
        std::abort();
    }
    
    int frameHeight = Frame.rows;
        int frameWidth = Frame.cols;

        // Define Region of Interest: exclude top 1/4 and bottom 1/5
        Rect roi(0, frameHeight / 4, frameWidth, frameHeight - (frameHeight / 5) - (frameHeight / 4));
        Mat roiFrame = Frame(roi);

        Mat grayFrame;
        cvtColor(roiFrame, grayFrame, COLOR_BGR2GRAY);
        equalizeHist(grayFrame, grayFrame);

        vector<Rect> cars;
        carCascade.detectMultiScale(
            grayFrame,
            cars,
            1.25,              // Scale factor
            2,                // Increase minNeighbors to reduce false positives
            0,
            Size(20, 20)      // Minimum size
        );
	

        for (auto& car : cars) {
            // Filter false positives using aspect ratio and size
            float aspectRatio = static_cast<float>(car.width) / car.height;
            int area = car.width * car.height;

            if (aspectRatio < 0.75 || aspectRatio > 3.0) continue; // Likely not a car
            if (area < 1000) continue; // Skip tiny detections

            // Offset to place detection in original frame
            car.x += roi.x;
            car.y += roi.y;
	
            annotations_buffer.push_back(car);
        }

    return annotations_buffer;
}

/*
#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ESCAPE_KEY (27)

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: { <input file> }";
        return -1;
    }
    string inputVideo = argv[1];
    
    struct timespec start, end, now;
    int frame_count = 0;
    float fps = 0;
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 1.0;
    cv::Scalar text_color(255, 255, 255); // White color
    int thickness = 2;

    VideoCapture cap(inputVideo);
    namedWindow("video_display");

    if (!cap.isOpened()) {
        cerr << "Error opening video";
        return -1;
    }

    char winInput;
    String cascadename = "./cars.xml";
    CascadeClassifier carCascade;

    if (!carCascade.load(cascadename)) {
        cerr << "Error loading cascade";
        return -1;
    }

    clock_gettime(CLOCK_MONOTONIC, &start);
    Mat frame;
    while (cap.read(frame)) {
    
    	frame_count++;	
    	
    
        int frameHeight = frame.rows;
        int frameWidth = frame.cols;

        // Define Region of Interest: exclude top 1/4 and bottom 1/5
        Rect roi(0, frameHeight / 4, frameWidth, frameHeight - (frameHeight / 5) - (frameHeight / 4));
        Mat roiFrame = frame(roi);

        Mat grayFrame;
        cvtColor(roiFrame, grayFrame, COLOR_BGR2GRAY);
        equalizeHist(grayFrame, grayFrame);

        vector<Rect> cars;
        carCascade.detectMultiScale(
            grayFrame,
            cars,
            1.25,              // Scale factor
            2,                // Increase minNeighbors to reduce false positives
            0,
            Size(20, 20)      // Minimum size
        );

        for (auto& car : cars) {
            // Filter false positives using aspect ratio and size
            float aspectRatio = static_cast<float>(car.width) / car.height;
            int area = car.width * car.height;

            if (aspectRatio < 0.75 || aspectRatio > 3.0) continue; // Likely not a car
            if (area < 1000) continue; // Skip tiny detections

            // Offset to place detection in original frame
            car.x += roi.x;
            car.y += roi.y;

            rectangle(frame, car, Scalar(0, 255, 0), 2);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &now);
    	
    	// If the elapsed time is greater than 1 second
    	if (now.tv_sec - start.tv_sec >= 1){
    		fps = static_cast<float>(frame_count) / (now.tv_sec - start.tv_sec);
    		frame_count = 0;
    		clock_gettime(CLOCK_MONOTONIC, &start);
    	}
    	
    	string frameText = "FPS: " + to_string(fps);
    	
    	putText(frame, frameText, Point(50,250), font_face, font_scale, text_color, thickness);

        imshow("video_display", frame);

        if ((winInput = waitKey(10)) == ESCAPE_KEY)
            break;
    }

    cap.release();
    destroyWindow("video_display");
    return 0;
}
*/

