import os
import json
import torch
import cv2
from ultralytics import YOLO

# CONFIGURATION
repo_dir = os.getcwd().split('dslab25')[0] + 'dslab25/'
video_path = os.path.join(repo_dir, "assets/vacuum_pump/videos/01_run1_cam_2_1024x1024_15fps_3mbps.mp4")
labels_path = os.path.join(repo_dir, "assets/vacuum_pump/videos/output.txt")
coco_path = os.path.join(repo_dir, "training/vacuum_pump/coco_annotations.json")
# Path to your trained YOLOv12 weights (adjust as needed)
yolo_model_path = os.path.join(repo_dir, "obj_detection/dino/yolo_runs/yolov12_boundingbox11", "weights", "best.pt")

def load_labels(labels_path):
    """Load ground truth labels from file."""
    frame_to_class = {}
    with open(labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                state_class, start_frame, end_frame = int(parts[0]), int(parts[1]), int(parts[2])
                for frame_idx in range(start_frame, end_frame + 1):
                    frame_to_class[frame_idx] = state_class
    return frame_to_class

def main():
    # Load ground truth labels.
    print(f"Loading labels from: {labels_path}")
    frame_to_class = load_labels(labels_path)
    
    # Load COCO annotations to map category IDs to names.
    print(f"Loading COCO annotations from: {coco_path}")
    try:
        with open(coco_path, 'r') as f:
            coco_data = json.load(f)
        category_id_to_name = {cat['id']: cat.get('name', f'category_{cat["id"]}')
                                 for cat in coco_data.get('categories', [])}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading COCO annotations: {e}")
        category_id_to_name = {}

    # Open the video.
    print(f"Loading video from: {video_path}")
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video file.")
        return
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Video info: {total_frames} frames, {fps} fps")
    
    # Load the YOLO model.
    print("Loading YOLO model...")
    yolo_model = YOLO(yolo_model_path)
    
    print("\n--- Starting YOLO Evaluation ---")
    frame_idx = 0
    frames_to_process = []
    # Process every 5th frame that has a ground truth label.
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_idx % 5 == 0 and frame_idx in frame_to_class:
            frames_to_process.append((frame_idx, frame))
        frame_idx += 1
    video.release()
    print(f"Total frames to evaluate: {len(frames_to_process)}")
    
    correct_predictions = 0
    total_predictions = 0
    
    for frame_idx, frame in frames_to_process:
        true_label = frame_to_class[frame_idx]
        
        # Run YOLO detection on the frame.
        yolo_results = yolo_model(frame)
        if len(yolo_results) == 0 or len(yolo_results[0].boxes) == 0:
            print(f"Frame {frame_idx}: No detection found. Skipping frame.")
            continue
        
        # Retrieve detections and select the one with the highest confidence.
        boxes = yolo_results[0].boxes.data  # Each row: [x1, y1, x2, y2, conf, cls]
        idx = torch.argmax(boxes[:, 4])
        box = boxes[idx]
        
        # YOLO prediction: class is at index 5.
        predicted_label = int(box[5].item())
        
        # Map numeric labels to names (if available).
        true_label_name = category_id_to_name.get(true_label, f"Class_{true_label}")
        predicted_label_name = category_id_to_name.get(predicted_label, f"Class_{predicted_label}")
        
        is_correct = predicted_label == true_label
        if is_correct:
            correct_predictions += 1
        total_predictions += 1
        
        print(f"Frame {frame_idx}:")
        print(f"  True label: {true_label_name} (ID: {true_label})")
        print(f"  YOLO Predicted: {predicted_label_name} (ID: {predicted_label})")
        print(f"  Correct: {'Yes' if is_correct else 'No'}")
        print("-" * 20)
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print("\nEvaluation Summary:")
    print(f"  Total frames evaluated: {total_predictions}")
    print(f"  Correct predictions: {correct_predictions}")
    print(f"  Accuracy: {accuracy:.2f} ({correct_predictions}/{total_predictions})")

if __name__ == "__main__":
    main()