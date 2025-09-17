// Akshay Arali

#ifndef LANE_LINES
#define LANE_LINES

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;

std::vector<cv::Vec4i> laneDetection(cv::Mat& frame);

#endif