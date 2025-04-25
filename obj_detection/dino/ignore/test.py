import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor
from train import DINOv2Classifier
from safetensors.torch import load_file as load_safetensors

# CONFIG
repo_dir = os.getcwd().split('dslab25')[0] + 'dslab25/'
base_dir = os.path.join(repo_dir, "training/vacuum_pump")
video_path = os.path.join(repo_dir, "assets/vacuum_pump/videos/01_run1_cam_2_1024x1024_15fps_3mbps.mp4")
labels_path = os.path.join(repo_dir, "assets/vacuum_pump/videos/output.txt")
model_dir = os.path.join(repo_dir, "obj_detection/dino/dinov2_finetune/final_model/")
coco_path = os.path.join(base_dir, "coco_annotations.json")

def load_labels(labels_path):
	"""Load ground truth labels from file"""
	frame_to_class = {}
	with open(labels_path, 'r') as f:
		for line in f:
			parts = line.strip().split()
			if len(parts) == 3:
				state_class, start_frame, end_frame = int(parts[0]), int(parts[1]), int(parts[2])
				for frame_idx in range(start_frame, end_frame + 1):  # +1 to include end_frame
					frame_to_class[frame_idx] = state_class
	return frame_to_class

def main():
	# Load ground truth labels
	print(f"Loading labels from: {labels_path}")
	frame_to_class = load_labels(labels_path)
	
	# Load categories from COCO file for class names
	print(f"Loading COCO annotations from: {coco_path}")
	try:
		with open(coco_path, 'r') as f:
			coco_data = json.load(f)
		category_id_to_name = {cat['id']: cat.get('name', f'category_{cat["id"]}') 
							  for cat in coco_data.get('categories', [])}
	except (FileNotFoundError, json.JSONDecodeError) as e:
		print(f"Error loading COCO file: {e}")
		# Default to numeric IDs if COCO file not available
		category_id_to_name = {}
	
	# Load video
	print(f"Loading video from: {video_path}")
	video = cv2.VideoCapture(video_path)
	if not video.isOpened():
		print(f"Error: Could not open video at {video_path}")
		return
	
	total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = video.get(cv2.CAP_PROP_FPS)
	print(f"Video info: {total_frames} frames, {fps} fps")
	
	# Load image processor and model
	print("Loading image processor...")
	processor = AutoImageProcessor.from_pretrained("facebook/dinov2-with-registers-base")

	# Determine device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	
	# Get number of classes from frame_to_class
	num_labels = max(frame_to_class.values()) + 1 if frame_to_class else 8
	print(f"Number of classes: {num_labels}")
	
	# Load model
	print("Loading model...")
	model = DINOv2Classifier(num_labels=num_labels)
	
	# Load model weights
	safetensors_path = os.path.join(model_dir, "model.safetensors")
	bin_path = os.path.join(model_dir, "pytorch_model.bin")
	
	model_weights_path = None
	if os.path.exists(safetensors_path):
		model_weights_path = safetensors_path
	elif os.path.exists(bin_path):
		model_weights_path = bin_path
		
	if model_weights_path:
		print(f"Loading model weights from: {model_weights_path}")
		try:
			if model_weights_path.endswith(".safetensors"):
				state_dict = load_safetensors(model_weights_path, device=str(device))
			else:
				state_dict = torch.load(model_weights_path, map_location=str(device), weights_only=True)
				
			# Handle potential DDP prefix
			if next(iter(state_dict)).startswith('module.'):
				state_dict = {k.partition('module.')[2]: v for k,v in state_dict.items()}
				 
			model.load_state_dict(state_dict)
		except Exception as e:
			print(f"Error loading model weights: {e}")
			return
	else:
		print(f"Error: Model weights not found in {model_dir}")
		return
	
	model.to(device)
	model.eval()
	
	# Process frames
	print("\n--- Starting Evaluation ---")
	frame_idx = 0
	frames_to_process = []
	
	# Collect frames to process (every 5th frame)
	while True:
		ret, frame = video.read()
		if not ret:
			break
		
		if frame_idx % 5 == 0:  # Process every 5th frame
			if frame_idx in frame_to_class:  # Only process frames with labels
				frames_to_process.append((frame_idx, frame))
		
		frame_idx += 1
	
	video.release()
	print(f"Total frames to evaluate: {len(frames_to_process)}")
	
	# Evaluate model on selected frames
	correct_predictions = 0
	total_predictions = 0
	
	for frame_idx, frame in frames_to_process[20:40]:
		true_label = frame_to_class[frame_idx]
		
		# Convert frame from BGR to RGB and then to PIL Image
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(frame_rgb)
		
		# Process image
		inputs = processor(images=image, return_tensors="pt")
		pixel_values = inputs['pixel_values'].to(device)
		
		# Make prediction
		with torch.no_grad():
			outputs = model(pixel_values=pixel_values)
		
		logits = outputs['logits']
		predicted_label = logits.argmax(-1).item()
		
		# Get class names
		true_label_name = category_id_to_name.get(true_label, f"Class_{true_label}")
		predicted_label_name = category_id_to_name.get(predicted_label, f"Class_{predicted_label}")
		
		# Log result
		is_correct = predicted_label == true_label
		if is_correct:
			correct_predictions += 1
		total_predictions += 1
		
		print(f"Frame {frame_idx}:")
		print(f"  True label: {true_label_name} (ID: {true_label})")
		print(f"  Predicted: {predicted_label_name} (ID: {predicted_label})")
		print(f"  Correct: {'Yes' if is_correct else 'No'}")
		print("-" * 20)
		
	# Print summary
	accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
	print(f"\nEvaluation Summary:")
	print(f"  Total frames evaluated: {total_predictions}")
	print(f"  Correct predictions: {correct_predictions}")
	print(f"  Accuracy: {accuracy:.2f} ({correct_predictions}/{total_predictions})")

if __name__ == "__main__":
	main()
