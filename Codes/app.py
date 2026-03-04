import streamlit as st
import cv2
import numpy as np
import torch
import tempfile
import time
from pathlib import Path
from numpy import random
import av

from streamlit_webrtc import webrtc_streamer, RTCConfiguration

import sys
sys.path.append(str(Path(__file__).parent))

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device

st.set_page_config(page_title="Real-Time Traffic Sign Detection", page_icon="🚦", layout="wide")

# Cached model loader
@st.cache_resource
def load_model(weights, device_str):
    set_logging()
    device = select_device(device_str)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    half = device.type != 'cpu'
    if half:
        model.half()  # to FP16
    return model, device, half

# Core Inference Function
def infer_frame(img_bgr, model, device, half, imgsz, conf_thres, iou_thres, classes=None, agnostic_nms=False):
    # Padded resize
    img = letterbox(img_bgr, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img_t = torch.from_numpy(img).to(device)
    img_t = img_t.half() if half else img_t.float()  # uint8 to fp16/32
    img_t /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img_t.ndimension() == 3:
        img_t = img_t.unsqueeze(0)

    # Inference
    pred = model(img_t, augment=False)[0]

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    im0 = img_bgr.copy()

    # Process detections
    for i, det in enumerate(pred):
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img_t.shape[2:], det[:, :4], im0.shape).round()

            # Plot the bounding boxes
            for *xyxy, conf, cls in reversed(det):
                label = f"{names[int(cls)]} {conf:.2f}"
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

    return im0

# Streamlit App UI
st.title("🚦 Real-Time Traffic Sign Detection")
st.markdown("Upload images, videos, or use your live webcam to detect traffic signs! 🚀")

st.sidebar.header("⚙️ Configuration")
source_type = st.sidebar.radio("Select Source", ("Live Webcam", "Image", "Video"))
weights = st.sidebar.text_input("Model Weights path", "../Model/weights/best.pt")
device_str = st.sidebar.selectbox("Device", ("cpu", "0"))
conf_thres = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.40)
iou_thres = st.sidebar.slider("NMS IoU Threshold", 0.0, 1.0, 0.45)
imgsz = 640

# Load Model
try:
    with st.spinner("Loading Model..."):
        model, device, half = load_model(weights, device_str)
    st.sidebar.success("Model Loaded Successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()


# Handling different Sources
if source_type == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)

        with col2:
            if st.button("Detect Traffic Signs", type="primary"):
                with st.spinner("Analyzing..."):
                    t1 = time.time()
                    res_img = infer_frame(img_bgr, model, device, half, imgsz, conf_thres, iou_thres)
                    t2 = time.time()
                    st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption=f"Result (Inference: {t2-t1:.3f}s)", use_container_width=True)

elif source_type == "Video":
    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())

        vf = cv2.VideoCapture(tfile.name)

        st.markdown("### Processed Video Feed")
        stframe = st.empty()

        col1, col2 = st.columns([1, 4])
        stop_button = col1.button("Stop Processing")

        while vf.isOpened():
            if stop_button:
                break
            ret, frame = vf.read()
            if not ret:
                break

            res_img = infer_frame(frame, model, device, half, imgsz, conf_thres, iou_thres)
            stframe.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        vf.release()
        try:
            Path(tfile.name).unlink()
        except:
            pass

elif source_type == "Live Webcam":
    st.markdown("### Live Webcam Detection 🔴")
    st.markdown("This uses `streamlit-webrtc` to establish a direct P2P connection between your browser's camera and the detection backend.")

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        res_img = infer_frame(img_bgr, model, device, half, imgsz, conf_thres, iou_thres)
        return av.VideoFrame.from_ndarray(res_img, format="bgr24")

    webrtc_streamer(
        key="traffic-sign-detector",
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
