#!/bin/bash

# https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html#export-from-pytorch

trtexec --onnx=tl_detect.onnx --saveEngine=tl_detect.engine --useDLACore=-1
