# 🚦 Hybrid Traffic Sign Detection Framework

 **Hybrid Traffic Sign Detection Framework**, an advanced multi-stage AI pipeline for real-time and high-accuracy traffic sign recognition.

## 🧠 Architecture Overview

This project implements a novel **Adaptive Fusion Intelligence Framework** that runs multiple Deep Learning models in parallel to aggressively eliminate false positives and ensure extreme accuracy in detections.

The pipeline consists of three core AI stages:
1. **The Fast Scanner (YOLOv5):** A highly optimized, custom-trained YOLOv5 backbone performs initial high-speed sweeps across the video frame, locking onto immediate region proposals.
2. **The Detail Verifier (Faster R-CNN):** A custom-trained native PyTorch Faster R-CNN model running a deep ResNet-50 backbone. This two-stage detector independently evaluates the frame, providing precise secondary cross-verification.
3. **The Semantic Judge (Vision Transformer/ViT):** Generated bounding boxes from the earlier stages are cropped and passed into a Vision Transformer for final semantic verification, calculating a weighted confidence multiplier.

These three models communicate via a dynamic **Probabilistic Fusion Engine** that calculates the final visual Master Bounding Box overlays.

## 🚀 Features
* **Real-Time Video & Webcam Processing:** Process videos or stream live feeds directly from your browser via WebRTC.
* **Intelligent Frame Skipping:** A built-in optimization UI allows you to run slow, heavy mathematical models alongside fast video playback by dynamically interpolating frame data.
* **Component Toggling:** Visually enable or disable the YOLO, Faster R-CNN, or ViT stages in real-time to see how the mathematical fusion adapts.

## 🛠️ Installation & Setup (One-Click)

1. Double-click the **`TrafficSign_Installer.bat`** file on Windows.
2. The interactive script will automatically create a Python virtual environment, install all required dependencies (including PyTorch, Streamlit, and Transformers), and boot up the Web UI server for you.
3. Once running, open your web browser to `http://localhost:8502`.

*(For Manual Installation, see the `requirements.txt` file and initialize a `venv`).*

## 📈 Training Custom Architectures
The framework supports native transfer learning if you wish to custom train the Stage 2 Faster R-CNN model on your own dataset for 30+ Epochs:
1. Extract your YOLO-formatted dataset into the `Dataset/` folder.
2. Run `python train_rcnn.py` to initiate the PyTorch transfer learning loop.
3. The script will automatically save your new custom weights to `Model/weights/best_rcnn.pth` for immediate use in the hybrid app.

Note:
  Install Models and extract it into the project folder : https://drive.google.com/file/d/1ByZSaihm0UGUfBqIBdywLw5-IH9zjxc1/view?usp=drive_link
---

