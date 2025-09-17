// Akshay Arali

#include "lane_lines.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;


std::vector<cv::Vec4i> laneDetection(Mat& frame) {
    static Mat gray, edges, mask, roiEdges;

    // 1) Preprocess
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(5,5), 0);
    Canny(gray, edges, 50, 150);

    // 2) Region of interest (trapezoid)
    int h = frame.rows, w = frame.cols;
    mask.create(edges.size(), edges.type());
    mask.setTo(Scalar(0));

    Point pts[4] = {
        Point(0, h), Point(w, h),
        Point(int(w * 0.6), int(h * 0.6)),
        Point(int(w * 0.4), int(h * 0.6))
    };
    fillPoly(mask, vector<vector<Point>>{vector<Point>(pts, pts + 4)}, Scalar(255));
    bitwise_and(edges, mask, roiEdges);

    // 3) Hough lines
    vector<Vec4i> lines;
    HoughLinesP(roiEdges, lines, 2, CV_PI/180, 50, 50, 10);

    // 4) Separate left/right by slope
    vector<Vec4i> leftL, rightL;
    #pragma omp parallel
    {
        vector<Vec4i> localLeft, localRight;
        #pragma omp for nowait
        for (int i = 0; i < lines.size(); ++i) {
            Vec4i l = lines[i];
            float dx = float(l[2] - l[0]), dy = float(l[3] - l[1]);
            if (fabs(dx) < 1e-2f) continue; // avoid near-vertical
            float slope = dy / dx;
            if (fabs(slope) < 0.5f) continue;
            if (slope < 0 && l[0] < w / 2)
                localLeft.push_back(l);
            else if (slope > 0 && l[0] > w / 2)
                localRight.push_back(l);
        }
        #pragma omp critical
        {
            leftL.insert(leftL.end(), localLeft.begin(), localLeft.end());
            rightL.insert(rightL.end(), localRight.begin(), localRight.end());
        }
    }

    // 5) Fit line on each side
    auto fitSide = [&](vector<Vec4i>& side, Point2f& bot, Point2f& top, float yTopFrac) {
        if (side.empty()) return false;

        vector<Point2f> pts;
        #pragma omp parallel
        {
            vector<Point2f> localPts;
            #pragma omp for nowait
            for (int i = 0; i < side.size(); ++i) {
                localPts.emplace_back(side[i][0], side[i][1]);
                localPts.emplace_back(side[i][2], side[i][3]);
            }
            #pragma omp critical
            pts.insert(pts.end(), localPts.begin(), localPts.end());
        }

        if (pts.size() < 4) return false; // not enough points for stable fitting

        Vec4f f;
        fitLine(pts, f, DIST_L2, 0, 0.01, 0.01);
        float vx = f[0], vy = f[1], x0 = f[2], y0 = f[3];
        bot = Point2f(x0 + vx * (h - y0) / vy, float(h));
        top = Point2f(x0 + vx * (h * yTopFrac - y0) / vy, h * yTopFrac);
        return true;
    };

    Point2f lb, lt, rb, rt;
    bool gotLeft = fitSide(leftL, lb, lt, 0.6f);
    bool gotRight = fitSide(rightL, rb, rt, 0.6f);

    if (!gotLeft) {
        lb = Point2f(0, h);
        lt = Point2f(w * 0.4f, h * 0.6f);
    }
    if (!gotRight) {
        rb = Point2f(w, h);
        rt = Point2f(w * 0.6f, h * 0.6f);
    }

    // 6) Temporal smoothing
    static Point2f p_lb, p_lt, p_rb, p_rt;
    auto blend = [](Point2f in, Point2f& prev) -> Point2f {
        if (prev == Point2f()) prev = in;
        prev = prev * 0.8f + in * 0.2f;
        return prev;
    };
    Point2f s_lb = blend(lb, p_lb);
    Point2f s_lt = blend(lt, p_lt);
    Point2f s_rb = blend(rb, p_rb);
    Point2f s_rt = blend(rt, p_rt);

    std::vector<cv::Vec4i> result;
    result.emplace_back(cvRound(s_lb.x), cvRound(s_lb.y), cvRound(s_lt.x), cvRound(s_lt.y));
    result.emplace_back(cvRound(s_rb.x), cvRound(s_rb.y), cvRound(s_rt.x), cvRound(s_rt.y));
    return result;
}