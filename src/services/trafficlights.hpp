#ifndef TRAFFIC_LIGHTS_DETECT_SERVICE
#define TRAFFIC_LIGHTS_DETECT_SERVICE

#include <frame_buffer.hpp>
#include <pthread.h>
#include <NvInfer.h>

class InferenceService {
public:
    InferenceService(const std::string& enginePath, FrameBuffer* inputBuffer, FrameBuffer* outputBuffer);
    ~InferenceService();
    void start();
    void stop();

private:
    static void* run(void* arg);
    void inferenceLoop();

    FrameBuffer* inputBuffer;
    FrameBuffer* outputBuffer;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    std::string enginePath;
    pthread_t thread;
    bool running;
};


#endif
