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

    Mat grayFrame(roiFrame.size(), CV_8UC1);

    #pragma omp parallel for
    for (int i = 0; i < roiFrame.rows; i++) {
        cvtColor(roiFrame.row(i), grayFrame.row(i), COLOR_BGR2GRAY);
    }
    //cvtColor(roiFrame, grayFrame, COLOR_BGR2GRAY);
    
    #pragma omp parallel for
    for (int i = 0; i < grayFrame.rows; i++) {
        equalizeHist(grayFrame.row(i), grayFrame.row(i));
    }
    //equalizeHist(grayFrame, grayFrame);

    // resize the frame
    Mat smallFrame;
    resize(grayFrame, smallFrame, Size(grayFrame.cols / 2, grayFrame.rows / 2));

    vector<Rect> cars;
    carCascade.detectMultiScale(
        smallFrame,
        cars,
        1.25,              // Scale factor
        2,                 // Increase minNeighbors to reduce false positives
        0,
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



/*
#include <omp.h>

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
        1.25,
        2,
        0,
        Size(20, 20)
    );

    // Create per-thread buffers
    int n = cars.size();
    vector<vector<Rect>> thread_buffers;

    #pragma omp parallel
    {
        vector<Rect> local_buffer;

        #pragma omp for nowait
        for (int i = 0; i < n; ++i) {
            float aspectRatio = static_cast<float>(cars[i].width) / cars[i].height;
            int area = cars[i].width * cars[i].height;

            if (aspectRatio < 0.75 || aspectRatio > 3.0) continue;
            if (area < 1000) continue;

            Rect adjustedCar = cars[i];
            adjustedCar.x += roi.x;
            adjustedCar.y += roi.y;

            local_buffer.push_back(adjustedCar);
        }

        #pragma omp critical
        thread_buffers.push_back(local_buffer);
    }

    // Merge all thread-local buffers
    for (const auto& buf : thread_buffers) {
        annotations_buffer.insert(annotations_buffer.end(), buf.begin(), buf.end());
    }

    return annotations_buffer;
}

*/