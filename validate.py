import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

st.set_page_config(page_title="Manual Validation Pipeline", layout="wide")

# Cache the model loading function
@st.cache_resource
def load_model(model_name):
    if model_name == 'YOLO':
        return YOLO('yolov8n.pt')  # Load pretrained YOLOv8n model
    else:
        raise ValueError("Selected model not supported")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    model_option = st.selectbox('Select Model', ['YOLO'])
    uploaded_video = st.file_uploader("Upload MP4 Video", type=["mp4"])

# Main content area
st.title("Video Validation Pipeline")
col1, col2 = st.columns([3, 1])

# Initialize session state variables
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0

if uploaded_video is not None:
    # Save uploaded video to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_video.read())
        video_path = tfile.name

    # Process video button
    if st.sidebar.button('Process Video'):
        model = load_model(model_option)
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        processed_frames = []
        processed_detections = []
        
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        for frame_num in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to RGB and process with YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)
            
            # Store frame and detections
            processed_frames.append(frame_rgb)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            processed_detections.append((boxes, classes, scores))
            
            # Update progress
            progress = (frame_num + 1) / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing Frame {frame_num + 1}/{total_frames}")
        
        cap.release()
        os.unlink(video_path)  # Delete temporary file
        
        # Save results to session state
        st.session_state.processed_data = {
            'frames': processed_frames,
            'detections': processed_detections,
            'fps': fps,
            'total_frames': total_frames
        }
        
        progress_bar.empty()
        status_text.success("Video processing completed!")

# Display processed video
if st.session_state.processed_data:
    frames = st.session_state.processed_data['frames']
    detections = st.session_state.processed_data['detections']
    total_frames = st.session_state.processed_data['total_frames']
    
    # Frame slider
    current_frame = col2.slider(
        "Select Frame",
        0, total_frames - 1,
        value=st.session_state.current_frame,
        key="frame_slider"
    )
    
    # Navigation buttons
    col1, col2, col3 = col2.columns([1, 3, 1])
    if col1.button("Previous") and current_frame > 0:
        st.session_state.current_frame -= 1
    if col3.button("Next") and current_frame < total_frames - 1:
        st.session_state.current_frame += 1
    
    # Get current frame data
    frame = frames[current_frame]
    boxes, classes, scores = detections[current_frame]
    
    # Draw bounding boxes and labels
    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = map(int, box)
        label = model.model.names[int(cls)]
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background
        (text_width, text_height), _ = cv2.getTextSize(
            f"{label} {score:.2f}", 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 1
        )
        cv2.rectangle(
            frame, 
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            (0, 255, 0), 
            -1
        )
        
        # Draw text
        cv2.putText(
            frame, 
            f"{label} {score:.2f}",
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 0), 
            1
        )
    
    # Display frame
    col1.image(frame, use_column_width=True, channels="RGB")
    
    # Display frame info
    col2.write(f"**Current Frame:** {current_frame + 1}/{total_frames}")
    col2.write(f"**Detections:** {len(boxes)} objects found")
else:
    col1.write("Upload and process a video to begin validation")

# Add some styling
st.markdown("""
    <style>
    .stSlider>div>div>div>div {
        background: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)