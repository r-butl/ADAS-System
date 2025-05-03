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
        2,                 // Increase minNeighbors to reduce false positives
        0,
        Size(20, 20)       // Minimum size
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