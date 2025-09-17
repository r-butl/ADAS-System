# Automated Driver Assistance System  
This project implements an ADAS system with four detection services and 2 system services. These services include:  
- Read frame - *Lucas* (me) - Responsible for reading new frames from the input video/camera and writing them to a global buffer.
- Draw frame - *Lucas* (me) - Responsible for taking all annations made by each service, drawing them on the frame, and writing the frame to the GUI.
- Traffic light detection (YOLO model) – *Lucas* (me)
- Pedestrian detection (YOLO model) – *Himanshu*  
- Lane line detection (first principles) – *Akshay*  
- Car detection (first principles) – *Taiga*  

## System Optimizations  
Our objective was to run all four services at a target of 25+ FPS. We achieved this by:  
- Applying the Blackboard architecture and using atomic flags and synchronization techniques for service synchronization.
- Threading each individual service with PThreads, utilizing core affinity, and applying compile-time optimizations via OpenMP. 
- Deploying optimized YOLO models using the TensorRT platform on the Jetson Nano 4G.

## Traffic Lights Service
The traffic lights detection system is based on the [Yolo11n](https://docs.ultralytics.com/models/yolo11/#overview) model and was trained on the [Lisa Traffic Lights](https://universe.roboflow.com/ithb-5ka4m/lisa-traffic-light-detection-8vuch) dataset and finetuned on the [Bosch Small Traffic Lights](https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset) dataset. The model was train to detect the presence of lights in general, proving feasibility.

## Demonstration
External video of the system was captured via cellphone since we maxed out the compute resources of the Jetson Nano 4G and wanted to capture the raw speed of the system without hinderance of system screen recording software. Each system uses a unique color for it's annotations:
- Green - Traffic lights
- Dark Blue - Pedestrains
- Light Blue - Lane Lines
- Orange - Cars

![demo_video](resources/adas-gif.gif)

## Synchronization System
At a high level, the system has three main services:

1. **ReadFrame** – reads frames and writes them to a shared frame buffer.
2. **ServiceWrapper** – reads frames from the buffer, runs them through its assigned annotation service (e.g., detection model), and writes results to an annotations buffer.
3. **DrawFrame** – reads the frame, applies annotations from all annotation buffers, and renders the final output.

Two sets of 8-bit flags—**FrameReady** and **ProcessingFinished**—manage synchronization. At initialization, each service (except ReadFrame) is assigned a unique bit. After ReadFrame writes a frame, it sets all bits in the FrameReady flag to 1. Each ServiceWrapper polls its assigned bit; once it reads the frame, it resets its bit to 0. ReadFrame waits for all bits to be 0 before writing the next frame.

Each ServiceWrapper uses a function pointer to call its detection service and writes results to its dedicated annotations buffer. It then sets its bit in the ProcessingFinished flag to signal DrawFrame that annotations are ready. DrawFrame waits for a new frame, then polls the ProcessingFinished flag until all bits are set, retrieves the annotations, applies them, and draws the frame. This architecture maintains synchronization at critical points while allowing parallel work for performance.

![Alt text](resources/sequence-diagram.png)

## Running the project

Make the project:
```
make clean && make
```
Run on a target video:
```
./adas_app <target-video>
```
Run using the camera:
```
./adas_app
```
