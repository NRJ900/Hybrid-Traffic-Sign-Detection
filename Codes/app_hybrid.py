import streamlit as st
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from transformers import ViTForImageClassification, ViTImageProcessor
import time
import tempfile
from pathlib import Path
import sys
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

sys.path.append(str(Path(__file__).parent))

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device

st.set_page_config(page_title="Hybrid Traffic Sign Detection", page_icon="🚥", layout="wide")

@st.cache_resource
def load_models(yolo_weights, device_str):
    set_logging()
    
    # Custom YOLO loader mechanics bypassing PyTorch Hub
    device = select_device(device_str)
    yolo_model = attempt_load(yolo_weights, map_location=device)
    yolo_half = device.type != 'cpu'
    if yolo_half:
        yolo_model.half()
        
    yolo_names = yolo_model.module.names if hasattr(yolo_model, 'module') else yolo_model.names

    # 2. Custom Faster R-CNN Transfer Learned Model
    num_classes = 5 # 4 signs + 1 background
    rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = rcnn_model.roi_heads.box_predictor.cls_score.in_features
    rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    rcnn_weights_path = Path(__file__).parent.parent / "Model" / "weights" / "best_rcnn.pth"
    rcnn_model.load_state_dict(torch.load(rcnn_weights_path, map_location=device))
    
    if device_str != "cpu":
        rcnn_model = rcnn_model.cuda()
    rcnn_model.eval()

    # 3. Transformer Verifier (ViTImageProcessor)
    extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    if device_str != "cpu":
        vit_model = vit_model.cuda()
    vit_model.eval()

    return yolo_model, yolo_names, device, yolo_half, rcnn_model, vit_model, extractor

def process_hybrid_frame(img_bgr, yolo_model, yolo_names, device, half, rcnn, vit, extractor, conf_thresh, show_yolo, show_rcnn, show_vit, show_master):
    # Prepare YOLO tensor
    imgsz = 640
    img = letterbox(img_bgr, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img_t = torch.from_numpy(img).to(device)
    img_t = img_t.half() if half else img_t.float()
    img_t /= 255.0
    if img_t.ndimension() == 3:
        img_t = img_t.unsqueeze(0)

    # 1. YOLO Inference
    pred = yolo_model(img_t, augment=False)[0]
    pred = non_max_suppression(pred, conf_thresh, 0.45, classes=None, agnostic=False)

    yolo_bboxes = []
    im0_shape = img_bgr.shape
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img_t.shape[2:], det[:, :4], im0_shape).round()
            for *xyxy, conf, cls in reversed(det):
                xmin, ymin, xmax, ymax = map(int, xyxy)
                yolo_bboxes.append((xmin, ymin, xmax, ymax, conf.item(), yolo_names[int(cls)]))

    # 2. Custom Faster R-CNN Inference
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    rcnn_bboxes = []
    if show_rcnn:
        rcnn_tensor = T.ToTensor()(img_rgb).unsqueeze(0).to("cuda:0" if device.type != "cpu" else "cpu")
        with torch.no_grad():
            rcnn_res = rcnn(rcnn_tensor)[0]
        
        # Our custom classes map: 1=prohibitory, 2=danger, 3=mandatory, 4=other
        custom_labels = {1: "Prohib", 2: "Danger", 3: "Mandatory"}# 4: "Other"}
        
        for box, score, label in zip(rcnn_res['boxes'], rcnn_res['scores'], rcnn_res['labels']):
            if score > conf_thresh:
                xmin, ymin, xmax, ymax = map(int, box.tolist())
                lbl_str = custom_labels.get(label.item(), str(label.item()))
                rcnn_bboxes.append((xmin, ymin, xmax, ymax, score.item(), lbl_str))

    
    # 3. Fusion & ViT Evaluation
    final_img = img_bgr.copy()
    
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        if (box1Area + box2Area - interArea) <= 0: return 0
        return interArea / float(box1Area + box2Area - interArea)

    matched_yolo = set()
    fused_proposals = []

    # Correlate bounding boxes
    for b1 in rcnn_bboxes:
        for j, b2 in enumerate(yolo_bboxes):
            if calculate_iou(b1[:4], b2[:4]) > 0.3:
                x1, y1 = int((b1[0] + b2[0])/2), int((b1[1] + b2[1])/2)
                x2, y2 = int((b1[2] + b2[2])/2), int((b1[3] + b2[3])/2)
                
                # Run ViT Semantic context
                vit_conf = 0
                crop = img_rgb[y1:y2, x1:x2]
                if crop.size != 0 and show_vit:
                    inputs = extractor(images=crop, return_tensors="pt").to("cuda:0" if device.type != "cpu" else "cpu")
                    with torch.no_grad():
                        vit_outputs = vit(**inputs)
                        vit_conf = torch.nn.functional.softmax(vit_outputs.logits, dim=-1)[0].max().item()
                elif not show_vit:
                    vit_conf = 1.0 # bypass ViT confidence hit
                
                fused_conf = (b2[4] * 0.5) + (b1[4] * 0.3) + (vit_conf * 0.2)
                if fused_conf > conf_thresh:
                    fused_proposals.append((x1, y1, x2, y2, fused_conf, b2[5], vit_conf))
                matched_yolo.add(j)
                break

    # Assume YOLO expertise bounds over remaining RCNN generic misses
    for j, b2 in enumerate(yolo_bboxes):
        if j not in matched_yolo and b2[4] > conf_thresh:
            fused_proposals.append((b2[0], b2[1], b2[2], b2[3], b2[4], b2[5], "N/A"))


    # Rendering Flags
    if show_yolo:
        for b in yolo_bboxes:
            cv2.rectangle(final_img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.putText(final_img, f"YOLO: {b[4]:.2f}", (b[0], b[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if show_rcnn:
        for b in rcnn_bboxes:
            cv2.rectangle(final_img, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
            cv2.putText(final_img, f"RCNN: {b[5]} {b[4]:.2f}", (b[0], b[3]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    if show_master:
        for box in fused_proposals:
            cv2.rectangle(final_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
            v_str = f" ViT:{box[6]:.2f}" if type(box[6]) is float and show_vit else ""
            cv2.putText(final_img, f"{box[5]} MASTER: {box[4]:.2f}{v_str}", (box[0], max(15, box[1]-20)), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)

    return final_img

# --- Core Streamlit UI ---

st.title("🚥 Hybrid Master Evaluator: Adaptive Fusion Intelligence Framework")
st.markdown("This application runs **Three Parallel Neural Networks** (YOLOv5, Faster R-CNN, ViT) and dynamically arbitrates collisions.")

st.sidebar.header("⚙️ Settings")
# Dynamically find available YOLO weights in Model/weights/ (excluding V8 and RCNN files)
weights_dir = Path(__file__).parent.parent / "Model" / "weights"
available_weights = [f.name for f in weights_dir.iterdir() if f.suffix == '.pt' and 'v8' not in f.name and 'rcnn' not in f.name]

if not available_weights:
    st.sidebar.error(f"No YOLOv5 .pt weights found in {weights_dir}")
    st.stop()

selected_weight = st.sidebar.selectbox("Select YOLOv5 Weights Model", available_weights, index=0)
weights = str(weights_dir / selected_weight)
device_str = st.sidebar.selectbox("Device", ("0", "cpu") if torch.cuda.is_available() else ("cpu",))
source_type = st.sidebar.radio("Select Input Source", ("Image", "Video", "Live Webcam"))

st.sidebar.markdown("---")
st.sidebar.markdown("### Neural Architectures 🧠")
st.sidebar.markdown("*Tip: Unchecking heavy models restores the stream to fast YOLO real-time speed. Checking them activates slow multi-model inference for maximum accuracy.*")
show_master = st.sidebar.checkbox("Show Final Master Fusion Bounds (Red)", value=True, help="Draws the final intelligently-merged box. If you turn off all 3 AIs above, but leave this on, it will still draw the default YOLO boxes because YOLO acts as the default Master.")
show_yolo = st.sidebar.checkbox("Compute custom YOLOv5 (Green)", value=True)
show_rcnn = st.sidebar.checkbox("Compute custom Faster R-CNN (Blue)", value=False, help="Replaces the generic Zero-Shot model with our newly trained native PyTorch Faster R-CNN Transfer Learned model.")
show_vit = st.sidebar.checkbox("Compute Vision Transformer (ViT)", value=False, help="Extracts cropped proposals and passes into ViT for Semantic Verification probability multipliers.")

conf_thres = st.sidebar.slider("Framework Confidence Threshold", 0.0, 1.0, 0.35)
st.sidebar.markdown("---")
st.sidebar.markdown("### Performance ⚡")
frame_skip = st.sidebar.slider("Video Frame Skip (Interpolation)", 1, 15, 5, help="When parsing Videos or Webcams, only process 1 out of every N frames through the heavy AI models to maintain real-time speeds. The bounding boxes will graphically persist on the skipped frames.")

try:
    with st.spinner("Downloading and caching massive architectures internally... (YOLO, R-CNN, ViT)"):
        yolo_m, yolo_n, yolo_device, yolo_half, rcnn_m, vit_m, vit_ext = load_models(weights, device_str)
    st.sidebar.success("All 3 AIs Active and Locked.")
except Exception as e:
    st.sidebar.error(f"Framework Error Loader: {e}")
    st.stop()


# Handlers
if source_type == "Image":
    uploaded_file = st.file_uploader("Upload an Image:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        
        if st.button("Run Adaptive Verification Fusion", type="primary"):
            with st.spinner("Pipeline Processing... Synthesizing YOLO + RCNN + ViT"):
                t1 = time.time()
                fused_out = process_hybrid_frame(img_bgr, yolo_m, yolo_n, yolo_device, yolo_half, rcnn_m, vit_m, vit_ext, conf_thres, show_yolo, show_rcnn, show_vit, show_master)
                t2 = time.time()
                
                st.success(f"Tri-Stage Analysis complete in {t2-t1:.3f} seconds")
                st.image(cv2.cvtColor(fused_out, cv2.COLOR_BGR2RGB), use_container_width=True)

elif source_type == "Video":
    uploaded_file = st.file_uploader("Upload a Video (MP4, AVI)", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        vf = cv2.VideoCapture(tfile.name)
        
        stframe = st.empty()
        stop_btn = st.button("Stop Processing")
        
        frame_idx = 0
        last_fused_out = None
        
        while vf.isOpened():
            if stop_btn: break
            ret, frame = vf.read()
            if not ret: break
            
            if frame_idx % frame_skip == 0:
                last_fused_out = process_hybrid_frame(frame, yolo_m, yolo_n, yolo_device, yolo_half, rcnn_m, vit_m, vit_ext, conf_thres, show_yolo, show_rcnn, show_vit, show_master)
            
            stframe.image(last_fused_out if last_fused_out is not None else frame, channels="BGR", use_container_width=True)
            frame_idx += 1
            
        vf.release()

elif source_type == "Live Webcam":
    st.markdown("### WebRTC P2P Webcam Pipeline 🔴")
    st.info("Passes local web browser frames into the heavily threaded Tri-Model pipeline.")
    
    class VideoProcessor:
        def __init__(self):
            self.frame_idx = 0
            self.last_img = None

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img_bgr = frame.to_ndarray(format="bgr24")
            
            if self.frame_idx % frame_skip == 0:
                self.last_img = process_hybrid_frame(img_bgr, yolo_m, yolo_n, yolo_device, yolo_half, rcnn_m, vit_m, vit_ext, conf_thres, show_yolo, show_rcnn, show_vit, show_master)
                
            self.frame_idx += 1
            return av.VideoFrame.from_ndarray(self.last_img if self.last_img is not None else img_bgr, format="bgr24")

    webrtc_streamer(
        key="hybrid-detector",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
