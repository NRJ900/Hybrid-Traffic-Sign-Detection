# Project Python Scripts Overview

This file documents the purpose of the primary Python scripts used in the Hybrid Traffic Sign Detection Framework.

## Root Directory

### `train_rcnn.py`
A standalone PyTorch script written to custom-train a Faster R-CNN model on a local object detection dataset. 
* Parses YOLO-formatted `.txt` bounding box coordinates and corresponding images.
* Prepares a generic `resnet50_fpn` PyTorch model for transfer learning by replacing the classification head.
* Executes the GPU training loop and saves the resulting model state dictionary to `Model/weights/best_rcnn.pth`.

### `Dataset/getting-full-path.py`
A small utility script likely used for dataset preparation, such as generating absolute file paths for training or testing sets.

## `Codes/` Directory

### `Codes/app_hybrid.py`
**The core application file for the Hybrid Framework.**
* Contains the Streamlit web interface and WebRTC video processing logic.
* Initializes and manages three concurrent neural networks: YOLOv5, Custom Faster R-CNN, and Vision Transformer (ViT).
* Implements the **Probabilistic Fusion Engine** to analyze intersections over union (IoU) across detector predictions and apply semantic verification multipliers to render the final bounding boxes.

### `Codes/app.py`
The original Streamlit application from the legacy project.
* Runs solely the YOLOv5 object detector.
* Provides a baseline comparison for the hybrid multi-stage approach.

### `Codes/detect.py`
A standalone inference script (standard in YOLO repositories) that can be run from the command line to process images, videos, or webcam feeds without the Streamlit UI.

### `Codes/test.py`
A testing/validation script designed to evaluate the trained model's accuracy, precision, recall, and mAP (mean Average Precision) against a testing dataset.

### `Codes/hubconf.py`
A PyTorch Hub configuration file that defines how models in this repository can be loaded via `torch.hub.load()` by defining entry points.

## Helper Modules (`Codes/models/` and `Codes/utils/`)

These directories contain the foundational code for the YOLOv5 architecture:

### `Codes/models/yolo.py` & `Codes/models/common.py`
Define the structural layers and network architecture for the YOLO model.

### `Codes/models/experimental.py`
Contains the `attempt_load` function used to cleanly load the custom `.pt` weights into the PyTorch environment.

### `Codes/utils/general.py`, `Codes/utils/datasets.py`, `Codes/utils/plots.py`
Core utility scripts that handle non-maximum suppression (NMS), bounding box scaling, image loading/transformations, and drawing results onto frames.
