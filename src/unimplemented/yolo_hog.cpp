#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace std;

bool overlapsAny(const Rect& box, const vector<Rect>& rois) {
    for (const Rect& roi : rois) {
        if ((box & roi).area() > 0) return true;
    }
    return false;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <video_file>" << endl;
        return -1;
    }

    string videoPath = argv[1];
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video file." << endl;
        return -1;
    }

    // Load YOLOv8 ONNX model
    dnn::Net net = dnn::readNetFromONNX("best.onnx");
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);  // Use DNN_TARGET_CUDA if GPU available

    // Setup HOG
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    int64 prevTick = getTickCount();
    double fps = 0.0;

    Mat frame;
    while (cap.read(frame)) {
        resize(frame, frame, Size(), 0.4, 0.4);

        // Step 1: HOG detection
        vector<Rect> hogROIs;
        vector<double> weights;
        hog.detectMultiScale(frame, hogROIs, weights, 0, Size(16, 16), Size(0, 0), 1.1, 2.0);

        // Step 2: Mask frame using HOG ROIs
        Mat maskedFrame = Mat::zeros(frame.size(), frame.type());
        for (const Rect& roi : hogROIs) {
            frame(roi).copyTo(maskedFrame(roi));
        }

        // Step 3: Run YOLO on masked frame
        Mat blob = dnn::blobFromImage(maskedFrame, 1.0 / 255.0, Size(640, 640), Scalar(), true, false);
        net.setInput(blob);
        Mat output = net.forward();

        // Step 4: Process YOLO output
        const int numDetections = output.size[1];
        const int dimensions = output.size[2];
        const float* data = (float*)output.data;

        for (int i = 0; i < numDetections; ++i) {
            float conf = data[i * dimensions + 4];  // objectness
            if (conf < 0.5) continue;

            float* scores = (float*)&data[i * dimensions + 5];
            Point classIdPoint;
            double classScore;
            minMaxLoc(Mat(1, dimensions - 5, CV_32F, scores), 0, &classScore, 0, &classIdPoint);

            if (classScore * conf < 0.5) continue;

            float cx = data[i * dimensions + 0];
            float cy = data[i * dimensions + 1];
            float w  = data[i * dimensions + 2];
            float h  = data[i * dimensions + 3];

            // YOLOv8 normalizes coords (cx, cy, w, h) relative to input image size (640x640)
            int inputW = 640, inputH = 640;
            int x = static_cast<int>((cx - w / 2) * frame.cols);
            int y = static_cast<int>((cy - h / 2) * frame.rows);
            int width = static_cast<int>(w * frame.cols);
            int height = static_cast<int>(h * frame.rows);

            Rect box(Point(x, y), Size(width, height));

            // Keep only if overlaps HOG ROIs
            if (overlapsAny(box, hogROIs)) {
                rectangle(frame, box, Scalar(0, 255, 0), 2);
            }
        }

        // FPS display
        int64 currentTick = getTickCount();
        fps = getTickFrequency() / (currentTick - prevTick);
        prevTick = currentTick;

        putText(frame, "FPS: " + to_string(static_cast<int>(fps)), Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);

        imshow("YOLOv8 + HOG Pedestrian Detection", frame);
        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

