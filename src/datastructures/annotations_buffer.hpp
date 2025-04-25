#ifndef ANNOTATIONS_BUFFER_HPP
#define ANNOTATIONS_BUFFER_HPP

#include <vector>
#include <string>
#include <mutex>
#include <opencv2/core.hpp>

struct Annotation {
    cv::Rect bbox;
    std::string label;
    float confidence;
    int frame_id;
};

class AnnotationsBuffer {
public:
    // Add a single annotation to the buffer
    void addAnnotation(const Annotation& annotation) {
        std::lock_guard<std::mutex> lock(mutex_);
        annotations_.push_back(annotation);
    }

    // Add a batch of annotations to the buffer
    void addAnnotations(const std::vector<Annotation>& batch) {
        std::lock_guard<std::mutex> lock(mutex_);
        annotations_.insert(annotations_.end(), batch.begin(), batch.end());
    }

    // Get a copy of all annotations in the buffer
    std::vector<Annotation> getAnnotations() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return annotations_;
    }

    // Clear all annotations from the buffer
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        annotations_.clear();
    }

private:
    std::vector<Annotation> annotations_;
    mutable std::mutex mutex_; // Mutex for thread safety
};

#endif // ANNOTATIONS_BUFFER_HPP

