import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForImageClassification
# Dataset: Returns processed images and labels
class DINOv2Dataset(torch.utils.data.Dataset):
	def __init__(self, dataset_dict, processor):
		self.dataset = dataset_dict
		self.processor = processor
		
	def __len__(self):
		return len(self.dataset)
	
	def __getitem__(self, idx):
		item = self.dataset[idx]
		image = Image.open(item["image_path"]).convert("RGB")
		inputs = self.processor(images=image, return_tensors="pt")
		return {
			"pixel_values": inputs['pixel_values'].squeeze(0), 
			"labels": torch.tensor(item["label"], dtype=torch.long)
		}

# Fine-tuning model: load DINOv2 for image classification and update the classifier layer.
class DINOv2Classifier(nn.Module):
	def __init__(self, num_labels=8, pretrained_model="facebook/dinov2-with-registers-base"):
		super().__init__()
		# Load the pre-trained model without overriding the classifier head
		self.model = AutoModelForImageClassification.from_pretrained(pretrained_model, num_labels=num_labels)
		
	def forward(self, pixel_values, labels=None):
		# The model returns a dict with loss (if labels are provided) and logits.
		outputs = self.model(pixel_values, labels=labels)
		return outputs
