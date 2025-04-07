import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

st.set_page_config(page_title="Manual Validation Pipeline", layout="wide")

# Cache the model loading function
@st.cache_resource
def load_model(checkpoint_path, model):
    if not os.path.exists(checkpoint_path):
        st.error(f"Model ckpt not found at: {checkpoint_path}")
        return None
    else:
        return model(checkpoint_path)

def initialize_resources(checkpoint_path, model, video_path):
    """initialize session state variables"""
    if not os.path.exists(video_path) or not os.path.exists(checkpoint_path):
        return False
    try:
        st.session_state_model = load_model(checkpoint_path, model)
        st.session_state.cap = cv2.VideoCapture(video_path)
        st.session_state.total_frames = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return True
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        return False

checkpoints = {}
# yolo checkpoints
checkpoints["yolo"] = ["/home/owendu/dslab25/obj_detection/yolo/yolov9_finetune2/weights/best.pt"]
models = {}
models["yolo"] = YOLO

# paths
video_paths = [f"/home/owendu/dslab25/training/vacuum_pump/videos/01_run{run}_cam_{cam}_1024x1024_15fps_3mbps.mp4" \
    for run in (1,2,3) for cam in (2,3)]

st.session_state.model = None
st.session_state.cap = None # video capture
st.session_state.total_frames = 0
st.session_state.current_frame = 0

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    model_option = st.selectbox('Select Model', ['yolo'])
    checkpoint_path = st.selectbox('Select Checkpoint', checkpoints["model_option"])
    video_path = st.selectbox('Select video', video_paths)
    md = models[model_option]
    if initialize_resources(checkpoint_path, md, video_path):
        st.success("video loaded")
    else:
        st.error("video load failed")


# Main content area
st.title("Video Validation Pipeline")

if st.session_state.cap is not None and st.session_state.model is not None:

    new_frame = st.slider(
        "Select frame",
        0,
        st.session_state.total_frames-1,
        value=st.session_state.current_frame,
        key="frame_slider"
    )

    if new_frame != st.session_state.current_frame:
        st.session_state.current_frame = new_frame
        st.session_state.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)

    ret, frame = st.session_state.cap.read()
    if ret:
        # Convert and process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = st.session_state.model(frame_rgb)
        
        # Draw detections
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        
        for box, cls, score in zip(boxes, classes, scores):
            x1, y1, x2, y2 = map(int, box)
            label = st.session_state.model.model.names[int(cls)]
            
            # Draw bounding box
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            (text_width, text_height), _ = cv2.getTextSize(
                f"{label} {score:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_rgb, (x1, y1 - text_height - 10),
                        (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(frame_rgb, f"{label} {score:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 0), 1)
        
        # Display frame and info
        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(frame_rgb, use_column_width=True, channels="RGB")
        with col2:
            st.write(f"**Frame:** {st.session_state.current_frame + 1}/{st.session_state.total_frames}")
            st.write(f"**Detections:** {len(boxes)} objects")
            st.write(f"**Model:** {os.path.basename(checkpoint_path)}")
else:
    st.info("Please initialize resources in the sidebar to begin")

# Cleanup when done
if st.session_state.cap is not None:
    st.session_state.cap.release()