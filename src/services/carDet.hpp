#ifndef CAR_DETECTOR_HPP
#define CAR_DETECTOR_HPP

#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include <vector>
#include <string>

using namespace cv;
using namespace std;

class CarDetector {
private:
    CascadeClassifier carCascade;
    String cascadePath;

public:
    // Constructor to load the cascade classifier
    CarDetector(const String& cascadeFilePath);

    // Function to detect cars in the given frame
    vector<Rect> detectCars(const Mat& Frame);
};

#endif // CAR_DETECTOR_HPP