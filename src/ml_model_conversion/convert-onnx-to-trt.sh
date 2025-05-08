#!/bin/bash

# https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html#export-from-pytorch

/usr/src/tensorrt/bin/trtexec --onnx=best_detect.onnx --saveEngine=people_detect.engine --useDLACore=-1

