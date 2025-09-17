# https://docs.ultralytics.com/integrations/onnx/#installation
from ultralytics import YOLO

model = YOLO('tl_detect.pt')

model.export(format='onnx', opset=12, simplify=True, dynamic=False, nms=True)

