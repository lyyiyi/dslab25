import torch
import torch.nn as nn
from PIL import Image

class DINOv2Dataset(torch.utils.data.Dataset):
	"""Dataset returning processed images"""
	def __init__(self, dataset_dict, processor):
		self.dataset = dataset_dict
		self.processor = processor
		
	def __len__(self):
		return len(self.dataset)
	
	def __getitem__(self, idx):
		item = self.dataset[idx]
		image = Image.open(item["image_path"]).convert("RGB")
		
		# Process the image and return pixel values
		inputs = self.processor(images=image, return_tensors="pt")
		
		# Squeeze the batch dimension added by the processor
		return {
			"pixel_values": inputs['pixel_values'].squeeze(0), 
			"labels": torch.tensor(item["label"], dtype=torch.long)
		}

class DINOv2Classifier(nn.Module):
	"""Combines DINOv2 feature extraction and a linear classifier"""
	def __init__(self, num_labels=8):
		super().__init__()
		self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', verbose=False) # Load DINOv2 here
		# Freeze DINOv2 parameters
		for param in self.dinov2.parameters():
			param.requires_grad = False
			
		self.classifier = nn.Linear(self.dinov2.embed_dim, num_labels) # Use embed_dim from DINOv2
		self.num_labels = num_labels

	def forward(self, pixel_values, labels=None):
		# Ensure DINOv2 is in eval mode for feature extraction consistency
		self.dinov2.eval() 
		
		# Extract features
		# DINOv2 from torch.hub returns features directly
		features = self.dinov2(pixel_values) 
		
		# Classify
		logits = self.classifier(features)
		
		loss = None
		if labels is not None:
			loss_fct = nn.CrossEntropyLoss()
			loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			
		return {"loss": loss, "logits": logits}
