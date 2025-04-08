import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor
from train import DINOv2Classifier  # Your defined classifier
from safetensors.torch import load_file as load_safetensors
from IPython.display import display
# CONFIGURATION
repo_dir = os.getcwd().split('dslab25')[0] + 'dslab25/'
base_dir = os.path.join(repo_dir, "training/vacuum_pump")
video_path = os.path.join(repo_dir, "assets/vacuum_pump/videos/01_run1_cam_2_1024x1024_15fps_3mbps.mp4")
labels_path = os.path.join(repo_dir, "assets/vacuum_pump/videos/output.txt")
model_dir = os.path.join(repo_dir, "obj_detection/dino/dinov2_finetune/final_model/")
coco_path = os.path.join(base_dir, "coco_annotations.json")

# YOLO model path – adjust according to where your trained YOLO weights are saved.
yolo_model_path = os.path.join(repo_dir, "object_detection/dino/yolo_runs/yolov12_boundingbox", "weights", "best.pt")

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
	
	# Load COCO annotations to get class names.
	print(f"Loading COCO annotations from: {coco_path}")
	try:
		with open(coco_path, 'r') as f:
			coco_data = json.load(f)
		category_id_to_name = {cat['id']: cat.get('name', f'category_{cat["id"]}')
								 for cat in coco_data.get('categories', [])}
	except (FileNotFoundError, json.JSONDecodeError) as e:
		print(f"Error loading COCO file: {e}")
		category_id_to_name = {}
	
	# Open video file.
	print(f"Loading video from: {video_path}")
	video = cv2.VideoCapture(video_path)
	if not video.isOpened():
		print(f"Error: Could not open video at {video_path}")
		return
	total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = video.get(cv2.CAP_PROP_FPS)
	print(f"Video info: {total_frames} frames, {fps} fps")
	
	# Load image processor for DinoV2.
	print("Loading image processor for DinoV2...")
	processor = AutoImageProcessor.from_pretrained("facebook/dinov2-with-registers-base")
	
	# Set device.
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	
	# Determine number of classes.
	num_labels = max(frame_to_class.values()) + 1 if frame_to_class else 8
	print(f"Number of classes: {num_labels}")
	
	# Load the DinoV2 classifier.
	print("Loading DinoV2 classifier model...")
	dino_model = DINOv2Classifier(num_labels=num_labels)
	
	# Load DinoV2 model weights.
	safetensors_path = os.path.join(model_dir, "model.safetensors")
	bin_path = os.path.join(model_dir, "pytorch_model.bin")
	model_weights_path = None
	if os.path.exists(safetensors_path):
		model_weights_path = safetensors_path
	elif os.path.exists(bin_path):
		model_weights_path = bin_path
		
	if model_weights_path:
		print(f"Loading DinoV2 model weights from: {model_weights_path}")
		try:
			if model_weights_path.endswith(".safetensors"):
				state_dict = load_safetensors(model_weights_path, device=str(device))
			else:
				state_dict = torch.load(model_weights_path, map_location=str(device), weights_only=True)
			# Remove potential DDP prefix
			if next(iter(state_dict)).startswith('module.'):
				state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}
			dino_model.load_state_dict(state_dict)
		except Exception as e:
			print(f"Error loading DinoV2 model weights: {e}")
			return
	else:
		print(f"Error: DinoV2 model weights not found in {model_dir}")
		return
	
	dino_model.to(device)
	dino_model.eval()
	
	# Load the YOLOv12 model.
	print("Loading YOLOv12 model for bounding box extraction...")
	yolo_model = YOLO(yolo_model_path)
	# If supported, you can move the YOLO model to the same device:
	# yolo_model.to(device)
	
	print("\n--- Starting Evaluation ---")
	frame_idx = 0
	frames_to_process = []
	while True:
		ret, frame = video.read()
		if not ret:
			break
		# Process every 5th frame that has a label.
		if frame_idx % 5 == 0 and frame_idx in frame_to_class:
			frames_to_process.append((frame_idx, frame))
		frame_idx += 1
	video.release()
	print(f"Total frames to evaluate: {len(frames_to_process)}")
	
	correct_predictions = 0
	total_predictions = 0
	
	# For demonstration, we process a subset (frames 20 to 40).
	for frame_idx, frame in frames_to_process[20:40]:
		true_label = frame_to_class[frame_idx]
		
		# Run YOLO on the frame to extract bounding boxes.
		yolo_results = yolo_model(frame)
		if len(yolo_results) == 0 or len(yolo_results[0].boxes) == 0:
			print(f"Frame {frame_idx}: No bounding box detected. Skipping frame.")
			continue
		
		# Choose the bounding box with the highest confidence.
		boxes = yolo_results[0].boxes
		box_data = boxes.data  # Assumes tensor with columns [x1, y1, x2, y2, conf, cls]
		idx = torch.argmax(box_data[:, 4])
		box = box_data[idx]
		x1, y1, x2, y2 = map(int, box[:4].tolist())
		# Clamp coordinates to frame dimensions.
		h, w, _ = frame.shape
		x1, y1 = max(0, x1), max(0, y1)
		x2, y2 = min(w, x2), min(h, y2)
		
		# Crop the detected region.
		cropped = frame[y1:y2, x1:x2]
		if cropped.size == 0:
			print(f"Frame {frame_idx}: Cropped region is empty. Skipping frame.")
			continue
		cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
		cropped_image = Image.fromarray(cropped_rgb)
		
		true_label = frame_to_class[frame_idx]
		
		# Preprocess the cropped image and classify using DinoV2.
		inputs = processor(images=cropped_image, return_tensors="pt")
		pixel_values = inputs['pixel_values'].to(device)
		with torch.no_grad():
			outputs = dino_model(pixel_values=pixel_values)
		logits = outputs['logits']
		predicted_label = logits.argmax(-1).item()
		
		# Map numerical labels to names.
		true_label_name = category_id_to_name.get(true_label, f"Class_{true_label}")
		predicted_label_name = category_id_to_name.get(predicted_label, f"Class_{predicted_label}")
		
		is_correct = predicted_label == true_label
		if is_correct:
			correct_predictions += 1
		total_predictions += 1
		
		
		# Convert frame from BGR to RGB and then to PIL Image
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(frame_rgb)
		# Show the image frame in notebook
		display(image)
		print(f"Frame {frame_idx}:")
		print(f"  True label: {true_label_name} (ID: {true_label})")
		print(f"  Predicted: {predicted_label_name} (ID: {predicted_label})")
		print(f"  Correct: {'Yes' if is_correct else 'No'}")
		print("-" * 20)
	
	accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
	print(f"\nEvaluation Summary:")
	print(f"  Total frames evaluated: {total_predictions}")
	print(f"  Correct predictions: {correct_predictions}")
	print(f"  Accuracy: {accuracy:.2f} ({correct_predictions}/{total_predictions})")

if __name__ == "__main__":
	main()