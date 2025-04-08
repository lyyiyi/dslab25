import torch
from PIL import Image

class DINOv2Dataset(torch.utils.data.Dataset):
	def __init__(self, dataset_dict, processor, feature_extractor, device):
		self.dataset = dataset_dict
		self.processor = processor
		self.feature_extractor = feature_extractor
		self.device = device
		
	def __len__(self):
		return len(self.dataset)
	
	def __getitem__(self, idx):
		item = self.dataset[idx]
		image = Image.open(item["image_path"]).convert("RGB")
		
		# Process image for DINOv2
		inputs = self.processor(images=image, return_tensors="pt")
		inputs = {k: v.to(self.device) for k, v in inputs.items()}
		
		# Extract features - use the class token ([CLS])
		with torch.no_grad():
			features = self.feature_extractor(**inputs).last_hidden_state[:, 0].squeeze().cpu()
		
		return {
			"features": features,
			"labels": torch.tensor(item["label"], dtype=torch.long)
		}
