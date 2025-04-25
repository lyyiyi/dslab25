from ultralytics import YOLO
import os
import yaml
import shutil
import random
import torch
# Define paths
repo_dir = os.getcwd().split('dslab25')[0] + 'dslab25/'
dino_dir = os.path.join(repo_dir, "obj_detection/dino/")
images = os.path.join(repo_dir, "training/vacuum_pump/images/augmented/")
labels = os.path.join(repo_dir, "training/vacuum_pump/annotation/augmented/")

yolo_train_images = os.path.join(repo_dir, "yolo_dataset/images/train")
yolo_val_images = os.path.join(repo_dir, "yolo_dataset/images/val")
yolo_train_labels = os.path.join(repo_dir, "yolo_dataset/labels/train")
yolo_val_labels = os.path.join(repo_dir, "yolo_dataset/labels/val")

def create_yolo_dataset():
	os.makedirs(yolo_train_images, exist_ok=True)
	os.makedirs(yolo_val_images, exist_ok=True)
	os.makedirs(yolo_train_labels, exist_ok=True)
	os.makedirs(yolo_val_labels, exist_ok=True)
	for folder in os.listdir(images):
			
		image_files = os.listdir(os.path.join(images, folder))
		random.shuffle(image_files)

		# Split into 90% train, 10% validation
		split_idx = int(len(image_files) * 0.9)
		train_images = image_files[:split_idx]
		val_images = image_files[split_idx:]

		# Copy training images and labels
		print(len(train_images))
		for image in train_images:
			src_image = os.path.join(images, folder, image)
			dst_image = os.path.join(yolo_train_images, image)
			src_label = os.path.join(labels, folder, image.replace(".jpg", ".txt"))
			dst_label = os.path.join(yolo_train_labels, image.replace(".jpg", ".txt"))
			
			if os.path.exists(src_image) and os.path.exists(src_label):
				shutil.copy(src_image, dst_image)
				shutil.copy(src_label, dst_label)

		# Copy validation images and labels
		for image in val_images:
			src_image = os.path.join(images, folder, image)
			dst_image = os.path.join(yolo_val_images, image)
			src_label = os.path.join(labels, folder, image.replace(".jpg", ".txt"))
			dst_label = os.path.join(yolo_val_labels, image.replace(".jpg", ".txt"))
			if os.path.exists(src_image) and os.path.exists(src_label):
				shutil.copy(src_image, dst_image)
				shutil.copy(src_label, dst_label)

	
create_yolo_dataset()
# YOLOv12 model (you can use "yolov12n.pt", "yolov12s.pt", "yolov12m.pt" etc.)
model_path = "yolo12m.pt"  # pretrained weights from Ultralytics
yaml_path = os.path.join(repo_dir, "yolo_dataset/yolo_dataset.yaml")

# Create dataset YAML
dataset_yaml = {
	"path": os.path.join(repo_dir, "yolo_dataset"),
	"train": "images/train",
	"val": "images/val",
	"nc": 8,  # Number of classes, set to 1 if you only care about bounding boxes
	"names": ["stage_0", "stage_1", "stage_2", "stage_3", "stage_4", "stage_5", "stage_6", "stage_7"]  # name of your single class
}

with open(yaml_path, "w") as f:
    yaml.dump(dataset_yaml, f)

# Load and train model
model = YOLO(model_path)  # Load YOLOv12 model (Ultralytics must support it)

model.train(
	data=yaml_path,
	epochs=1,
	imgsz=640,
	batch=32,
	name="yolov12_boundingbox",
	project=os.path.join(dino_dir, "yolo_runs"),
	device=0 if torch.cuda.is_available() else "cpu"
)