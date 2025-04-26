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
dino_dir = os.path.join(repo_dir, "obj_detection/dino") 
training_dir = os.path.join(repo_dir, "training/vacuum_pump")
image_dir = os.path.join(training_dir, "images/augmented")
label_dir = os.path.join(training_dir, "annotation/augmented")
coco_path = os.path.join(training_dir, "coco_annotations.json")
pretrained_model = "facebook/dinov2-with-registers-base"

def main():
	# Determine device
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
	dataset = dataset.train_test_split(test_size=0.1, seed=42)
	
	print("Initializing Image Processor...")
	processor = AutoImageProcessor.from_pretrained(pretrained_model)
	
	print("Preparing datasets...")
	train_dataset = DINOv2Dataset(dataset["train"], processor)
	eval_dataset = DINOv2Dataset(dataset["test"], processor)
	
	print("Train dataset size:", len(train_dataset))
	print("Eval dataset size:", len(eval_dataset))
	
	print("Initializing DINOv2 Classifier model for fine-tuning...")
	num_labels = len(set(dataset_dict["label"]))
	model = DINOv2Classifier(num_labels=num_labels, pretrained_model=pretrained_model)
	
	training_args = TrainingArguments(
		output_dir=os.path.join(dino_dir, "dinov2_finetune_base"),
		learning_rate=1e-5,  # Lower learning rate for fine-tuning
		per_device_train_batch_size=16,  # Adjust batch size to your GPU memory
		per_device_eval_batch_size=16,
		num_train_epochs=10,  # Fewer epochs may suffice for fine-tuning
		weight_decay=0.01,
		eval_strategy="epoch",
		save_strategy="epoch",
		load_best_model_at_end=True,
		dataloader_num_workers=4,
		logging_steps=10,
		fp16=torch.cuda.is_available(),
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
	main()