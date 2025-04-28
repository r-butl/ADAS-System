#ifndef CARDET_HPP
#define CARDET_HPP


#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void carDetection(Mat Frame, vector<Rect> &annotations_buffer);

#endif 
