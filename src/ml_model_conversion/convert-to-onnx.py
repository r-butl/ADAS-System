# https://docs.ultralytics.com/integrations/onnx/#installation
from ultralytics import YOLO

model = YOLO('best.pt')

model.export(format='.onnx')

