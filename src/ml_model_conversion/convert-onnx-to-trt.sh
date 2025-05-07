#!/bin/bash

# https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html#export-from-pytorch

/usr/src/tensorrt/bin/trtexec --onnx=tl_detect.onnx --saveEngine=tl_detect.engine --useDLACore=-1

/usr/src/tensorrt/bin/trtexec --onnx=best_old.onnx --saveEngine=person_detect.engine --useDLACore=-1
