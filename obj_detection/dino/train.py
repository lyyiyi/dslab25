import os
import json
import numpy as np
import evaluate
import torch
from transformers import AutoImageProcessor, TrainingArguments, Trainer
from datasets import Dataset
from utils import DINOv2Dataset, DINOv2Classifier

# CONFIG
repo_dir = os.getcwd().split('dslab25')[0] + 'dslab25/'
base_dir = repo_dir + "obj_detection/dino/"
root_dir = base_dir + "training/vacuum_pump"
image_dir = os.path.join(root_dir, "images/augmented")
label_dir = os.path.join(root_dir, "annotation/augmented")
coco_path = os.path.join(root_dir, "coco_annotations.json")

# Note: predict_image would also need refactoring to use the DINOv2Classifier if used post-training
# For simplicity, it's removed from this training script focus.

def main():
	# Determine device (Trainer will handle distribution, but good practice)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Initial device check: {device}")

	# Load COCO annotations and build the dataset dictionary
	with open(coco_path, 'r') as f:
		coco_data = json.load(f)
	
	image_to_category = {ann["image_id"]: ann["category_id"] for ann in coco_data["annotations"]}
	dataset_dict = {"image_path": [], "label": []}
	for image_info in coco_data["images"]:
		image_id = image_info["id"]
		file_name = image_info["file_name"]
		full_path = os.path.join(image_dir, file_name)
		if image_id in image_to_category:
			dataset_dict["image_path"].append(full_path)
			dataset_dict["label"].append(image_to_category[image_id])
	
	# Split the dataset into train and validation (80/20)
	dataset = Dataset.from_dict(dataset_dict)
	dataset = dataset.train_test_split(test_size=0.2, seed=42)
	
	print("Initializing Image Processor...")
	# Initialize the image processor
	processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
	
	print("Preparing datasets...")
	train_dataset = DINOv2Dataset(dataset["train"], processor)
	eval_dataset = DINOv2Dataset(dataset["test"], processor)
	
	print("Initializing DINOv2 Classifier model...")
	# Initialize the combined classifier model
	# Trainer will handle moving the model to the correct device(s)
	model = DINOv2Classifier(num_labels=len(set(dataset_dict["label"]))) 
	
	training_args = TrainingArguments(
		output_dir=os.path.join(root_dir, "dinov2_register_classifier_multi_gpu"), # New output dir
		learning_rate=3e-4, # Adjust as needed
		per_device_train_batch_size=64, # Batch size PER GPU
		per_device_eval_batch_size=64,  # Batch size PER GPU
		num_train_epochs=10, # Adjust as needed
		weight_decay=0.01, # Adjust as needed
		eval_strategy="epoch",
		save_strategy="epoch",
		load_best_model_at_end=True,
		dataloader_num_workers=4,  # Re-enable workers (adjust based on your system)
		logging_steps=10,         # Log more frequently
		# fp16=torch.cuda.is_available(), # Optional: Enable mixed precision if desired
	)
	
	metric = evaluate.load("accuracy")
	def compute_metrics(eval_pred):
		logits, labels = eval_pred
		predictions = np.argmax(logits, axis=1)
		return metric.compute(predictions=predictions, references=labels)
	
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		compute_metrics=compute_metrics,
		# data_collator=default_data_collator # Use default collator
	)
	
	print("Starting training...")
	trainer.train()
	
	model_save_path = os.path.join(training_args.output_dir, "final_model")
	trainer.save_model(model_save_path)
	print(f"Model saved to {model_save_path}")
	
	print("Evaluating final model...")
	eval_results = trainer.evaluate()
	print(f"Evaluation results: {eval_results}")

if __name__ == "__main__":
	# No specific multiprocessing start method needed here,
	# use torchrun for distributed training.
	main()
