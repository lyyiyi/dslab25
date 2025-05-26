import os
from sam2.sam2_image_predictor  import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import numpy as np

data_dir = "/work/courses/dslab/team14/"

img_pred = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-tiny")

refs_hand_dir = os.path.join(data_dir, "training/hand/cropped/")
refs_hand = [os.path.join(refs_hand_dir, p) for p in os.listdir(refs_hand_dir)]

def visualize_mask(mask, mask_output_path):
	"""Visualize the mask and save it to a file."""
	try:
		# Convert boolean tensor to an 8-bit grayscale NumPy array
		# True -> 255 (white), False -> 0 (black)
		mask_image = mask.astype(np.uint8) * 255
		
        # invert mask (special case for hands)
		mask_image = cv2.bitwise_not(mask_image)
		# Save the image
		cv2.imwrite(mask_output_path, mask_image)
		print(f"Initial mask saved to: {mask_output_path}", flush=True)
	except Exception as e:
		print(f"Error saving mask visualization: {e}", flush=True)
		
mask_gen = SAM2AutomaticMaskGenerator(
	img_pred.model, # Pass the SAM model directly
	points_per_side=8,
	pred_iou_thresh=0.5,		  # only keep masks with IoU-pred confidence ≥ 0.7
	stability_score_thresh=0.9,   # only keep very stable masks
	box_nms_thresh=0.5,		   # merge overlapping boxes more aggressively
	min_mask_region_area=1000	 # drop very small regions
)
		
masks_hand = []
for ref in refs_hand:
	ref = cv2.imread(ref)
	ref = cv2.resize(ref, (512, 512))
	ref = cv2.GaussianBlur(ref, (5, 5), 0)
	cur_masks = mask_gen.generate(ref)
	print(f"current amount of masks: {len(cur_masks)}", flush=True)
	masks_hand.extend(cur_masks)
cc = 0
for m in masks_hand:
	visualize_mask(m["segmentation"], f"masks/mask_hand_{cc}.png")
	cc+=1