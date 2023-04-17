from ultralytics import YOLO
import pdb 
import cv2

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
filename = "/home/robesafe/sort_gitt/mot_benchmark/train/KITTI-17/img1/000001.jpg"
results = model(filename)  # predict on an image
res_plotted = results[0].plot()
pdb.set_trace()
cv2.imwrite("results_yolo",res_plotted)
pdb.set_trace()
# success = model.export(format="onnx")  # export the model to ONNX format