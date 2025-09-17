// Taiga Odonnel

#include "carDet.hpp"
#include <iostream>

using namespace std;
using namespace cv;

CarDetector::CarDetector(const String& cascadeFilePath) : cascadePath(cascadeFilePath) {
    // Load the cascade file
    if (!carCascade.load(cascadePath)) {
        cerr << "Error loading cascade" << endl;
        exit(-1);
    }
}

vector<Rect> CarDetector::detectCars(const Mat& Frame) {
    vector<Rect> annotations_buffer;

    int frameHeight = Frame.rows;
    int frameWidth = Frame.cols;

    // Define Region of Interest: exclude top 1/4 and bottom 1/5
    //Rect roi(0, frameHeight / 4, frameWidth, frameHeight - (frameHeight / 5) - (frameHeight / 4));
    Rect roi(0, frameHeight /2, frameWidth, frameHeight/2);
    Mat roiFrame = Frame(roi);

    Mat grayFrame(roiFrame.size(), CV_8UC1);

    #pragma omp parallel for
    for (int i = 0; i < roiFrame.rows; i++) {
        cvtColor(roiFrame.row(i), grayFrame.row(i), COLOR_BGR2GRAY);
    }
    //cvtColor(roiFrame, grayFrame, COLOR_BGR2GRAY);
    
    // #pragma omp parallel for
    // for (int i = 0; i < grayFrame.rows; i++) {
    //     equalizeHist(grayFrame.row(i), grayFrame.row(i));
    // }
    //equalizeHist(grayFrame, grayFrame);

    // resize the frame
    Mat smallFrame;
    resize(grayFrame, smallFrame, Size(grayFrame.cols / 2, grayFrame.rows / 2));

    vector<Rect> cars;
    carCascade.detectMultiScale(
        smallFrame,
        cars,
        1.15,              // Scale factor, 1.25
        3,
	0,	// Increase minNeighbors to reduce false positives
        Size(20, 20)       // Minimum size
    );

    #pragma omp parallel for
    for (size_t i = 0; i < cars.size(); ++i) {
        Rect& car = cars[i];
        car.x *= 2;  
        car.y *= 2;  
        car.width *= 2;
        car.height *= 2;

        // Filter false positives using aspect ratio and size
        float aspectRatio = static_cast<float>(car.width) / car.height;
        int area = car.width * car.height;

        if (aspectRatio < 0.75 || aspectRatio > 3.0) continue; // Likely not a car
        if (area < 1000) continue; // Skip tiny detections

        // Offset to place detection in original frame
        car.x += roi.x;
        car.y += roi.y;

        #pragma omp critical
        {
            annotations_buffer.push_back(car);
        }
        
    }

    return annotations_buffer;
}
