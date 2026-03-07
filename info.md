# 🚦 Hybrid Traffic Sign Detection Framework: Technical Script Overview

This document provides a highly detailed breakdown of the Python source files running the Hybrid AI framework. 

---

## 💻 1. The Application Interface Layer (`Codes/`)

This directory contains the Streamlit User Interfaces and the execution pipelines that the end-user interacts with.

### `Codes/app_hybrid_v8.py` (Modern Architecture)
**The state-of-the-art implementation of the Hybrid Architecture.**
*   **YOLOv8 Backpack:** Uses the highly optimized `ultralytics` package to process the First-Stage "Fast Scan", dramatically improving bounding box tightness and CPU/GPU memory inference compared to v5.
*   **Dynamic Weight Loader:** Features an automated scanning engine that reads the `Model/weights/` directory for both PyTorch `.pt` and Open Neural Network Exchange `.onnx` files, auto-populating the UI UI dropdown without code changes.
*   **Triple-Model Probability Engine:** Instantiates the YOLOv8, Custom ResNet-50 Faster R-CNN, and Google's OWL Vision Transformer simultaneously. It actively runs Intersection over Union (IoU) algorithms against all 3 model outputs, applying Semantic Verifier Confidence Multiplier math to suppress False Positives.
*   **Hardware Interface:** Manages asynchronous frame streams from Images, Local `.mp4` storage, and live WebRTC PC Webcams.

### `Codes/app_hybrid.py` (Legacy YOLOv5 Architecture)
**The original Hybrid Architecture system.**
*   Runs the exact same Tri-Model Probability Engine as above but mathematically processes Stage 1 purely through the older YOLOv5 PyTorch Hub implementation. 
*   Because the codebase has been disconnected from the Ultralytics API server, this relies on a local loader script (`Codes/models/experimental.py:attempt_load()`) to inject the `best.pt` file directly into memory.

### `Codes/app.py`
*   The legacy, single-model Streamlit playground.
*   Runs solely the `best.pt` YOLO detector and renders green boxes immediately. Has no Verification Layer.

### `Codes/detect.py` & `Codes/test.py`
*   Command-line inference tools standard to YOLO environments. 
*   `detect.py` allows developers to process batches of images natively without Streamlit.
*   `test.py` validates the exact mAP (Mean Average Precision) math against a holdout testing dataset.

---

## 🧠 2. The Training Layer (Root Folder)

### `train_rcnn.py`
**A completely custom Transfer-Learning loop written natively in PyTorch.**
To verify YOLO's detections, the Hybrid Framework requires a slower, more deliberate two-stage detector. This script builds it:
1.  **Dataset Ingestion:** Creates a custom PyTorch `TrafficSignDataset` loader that intelligently parses the 1-indexed YOLO `.txt` coordinate files and converts the normalized (Center-X, Center-Y, W, H) values into the absolute pixel (X_Min, Y_Min, X_Max, Y_Max) bounds demanded by Faster R-CNN.
2.  **Model Amputation:** Downloads a `torchvision` ResNet-50 FPN model trained originally on 80 generalized MS COCO classes. It severs the final classification node and injects a blank `FastRCNNPredictor` configured strictly for 5 channels (4 traffic signs + 1 background).
3.  **The Loop:** Executes Stochastic Gradient Descent to update the model weights across `N` epochs, dumping the highly trained result to `Model/weights/best_rcnn.pth`.

---

## ⚙️ 3. The YOLO Mechanics Factory (`Codes/models/` & `Codes/utils/`)

These directories contain the foundational Deep Learning architecture required for the legacy `app_hybrid.py` file to parse `.pt` files without relying on external packages.

### `Codes/models/yolo.py` & `Codes/models/common.py`
*   The literal blueprint of the YOLO brain. This defines the PyTorch structural sequence: `Conv2d` -> `BatchNorm2d` -> `SiLU Activation` -> `C3 Bottleneck` -> `SPPF Spatial Pooling`. 

### `Codes/utils/general.py`
*   The mathematical heavy lifter. Performs Non-Maximum Suppression (deleting redundant, overlapping AI guesses) and scales tiny 640x640 AI tensors back up into massive 1080p imagery without losing coordinate mapping.

### `Codes/utils/datasets.py`
*   The data ingestion tool. Its most famous piece is the `letterbox` function, which automatically resizes any weirdly shaped photo or video into a perfect mathematical square padded heavily with grey pixels, protecting the aspect ratio of the traffic sign from stretching and destroying the AI's understanding of geometry.

### `Codes/utils/torch_utils.py`
*   Hardware abstraction. Silently forces heavy AI matrix math over to `cuda:0` if an NVIDIA GPU is detected during runtime, drastically boosting Frames Per Second.
