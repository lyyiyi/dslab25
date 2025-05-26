import os
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor
from utils import *
from safetensors.torch import load_file as load_safetensors
from IPython.display import display, HTML, Video
import imageio
import torch.nn.functional as F
from sam2.sam2_image_predictor  import SAM2ImagePredictor
from sam2.sam2_video_predictor  import SAM2VideoPredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from transformers import AutoModel, AutoImageProcessor
import gc
import sys
from multiprocessing import Process

# CONFIGURATION
repo_dir = os.getcwd().split('dslab25')[0] + 'dslab25/'
data_dir = "/work/courses/dslab/team14/"
video_path = os.path.join(data_dir, "assets/vacuum_pump/videos/01_run1_cam_2_1024x1024_5fps.mp4")
labels_path = os.path.join(data_dir, "videos/01_run1_simplified_5fps.txt")


pretrained_model_cls = "facebook/dinov2-with-registers-base"
pretrained_model_sam = "facebook/dinov2-with-registers-small"
model_dir = os.path.join(data_dir, "dino/final_model/")

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

# Load ground truth labels.
print(f"Loading labels from: {labels_path}", flush=True)
frame_to_class = load_labels(labels_path)

# Load image processor for classifier
print("Loading image processor for classifier...", flush=True)
processor_cls = AutoImageProcessor.from_pretrained(pretrained_model_cls)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# Get number of classes from frame_to_class
num_labels = max(frame_to_class.values()) + 1 if frame_to_class else 8
print(f"Number of classes: {num_labels}", flush=True)

# Load classifier model
print("Loading classifier model...", flush=True)
model = DINOv2Classifier(num_labels=num_labels, pretrained_model=pretrained_model_cls)

# Load model weights
safetensors_path = os.path.join(model_dir, "model.safetensors")
bin_path = os.path.join(model_dir, "pytorch_model.bin")

model_weights_path = None
if os.path.exists(safetensors_path):
	model_weights_path = safetensors_path
elif os.path.exists(bin_path):
	model_weights_path = bin_path
	
if model_weights_path:
	print(f"Loading model weights from: {model_weights_path}", flush=True)
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
		raise e
else:
	raise Exception(f"Error: Model weights not found in {model_dir}")


model.to(device)
model.eval()

# ── SAM / DINOv2 for SAM Section ─────────────────────────────────────────────

# ── 2.  PATHS & OPTIONS  ─────────────────────────────────────────────────────
ref_dir = os.path.join(data_dir, "training/vacuum_pump/images/cropped/")
print("ref_dir", ref_dir, flush=True)
refs = [
	os.path.join(ref_dir, p) for p in os.listdir(ref_dir)
]
refs_hand_dir = os.path.join(data_dir, "training/hand/cropped/")
refs_hand = [os.path.join(refs_hand_dir, p) for p in os.listdir(refs_hand_dir)]
boxed_out   = "sam2_boxed.mp4"
skip_frames = 31					 # ← skip these many frames for mask initialization

# ── 2. UTILS (Simplified for iterative processing) ───────────────────────────
# Removed read_video_rgb function
c = 0
def get_feat(img_rgb, dinov2_backbone_sam, dinov2_proc_sam, device):
	global c # Declare that we are using the global variable c
	"""Extract DINOv2 features for a single image."""
	# Ensure image is in RGB format (it should be if read with cv2.cvtColor)
	if img_rgb is None or img_rgb.size == 0:
		print("Warning: Trying to get features from an empty image.")
		return None
	try:
		img_rgb = cv2.resize(img_rgb, (512, 512))
		img_rgb = cv2.GaussianBlur(img_rgb, (5, 5), 0)
		cv2.imwrite(f"blur/blur_{c}.png", img_rgb)
		img_rgb = cv2.Canny(img_rgb,0,100)
		cv2.imwrite(f"canny/canny_{c}.png", img_rgb)
		c+=1
		img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
		ipt = dinov2_proc_sam(images=img_rgb, return_tensors="pt").to(device)
		with torch.no_grad():
			out = dinov2_backbone_sam(**ipt).last_hidden_state[:,0] # Use CLS token
		return F.normalize(out.squeeze(0), dim=-1).cpu()
	except Exception as e:
		print(f"Error during feature extraction: {e}")
		# Consider adding more diagnostics, e.g., image shape
		# print(f"Image shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
		return None

def get_ref_feats(refs):
	ref_feats = []
	for p in refs:
		bgr = cv2.imread(p)
		if bgr is None:
			raise IOError(f"cannot open reference image {p}")
		# rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
		rgb = bgr
		feat = get_feat(rgb, dinov2_backbone_sam, dinov2_proc_sam, device)
		if feat is not None:
			ref_feats.append(feat)
		else:
			print(f"Warning: Failed to get features for reference image {p}", flush=True)

	if not ref_feats:
		cap.release()
		raise RuntimeError("Could not extract features from any reference images!")
	ref_feats_tensor = torch.stack(ref_feats) # Stack features for efficient comparison
	return ref_feats_tensor

def find_best_mask(masks, ref_feats_tensor, seed_frame_rgb):
	best_m_info, best_sim = None, -1.0
	c=0
	with torch.no_grad(): # Ensure no gradients are calculated here
		for m in masks:
			# visualize_mask(torch.from_numpy(m["segmentation"]).bool(), f"masks/mask_{c}.png")
			x_f, y_f, w_f, h_f = m["bbox"]
			x0 = max(0, int(round(x_f))); y0 = max(0, int(round(y_f)))
			x1 = min(W, int(round(x_f + w_f))); y1 = min(H, int(round(y_f + h_f)))
			if x1 <= x0 or y1 <= y0: continue

			crop_rgb = seed_frame_rgb[y0:y1, x0:x1]
			if crop_rgb.size == 0:
				print(f"Warning: Empty crop at bbox [{x0},{y0},{x1},{y1}]")
				continue
			feat = get_feat(crop_rgb, dinov2_backbone_sam, dinov2_proc_sam, device)

			if feat is not None:
				sims = ref_feats_tensor @ feat # [n_refs] x [feat_dim] @ [feat_dim]
				sim = sims.mean().item()
				print(f"sim {c}: {sim}")
				# print(f"with bounding box: {x0}, {y0}, {x1}, {y1}")
				c+=1
				if sim > best_sim:
					# Store necessary info: mask tensor and bbox
					best_m_info = {
						"segmentation": torch.from_numpy(m["segmentation"]).to(device).bool(),
						"bbox": (x0, y0, x1, y1) # Store integer bbox
					}
					best_sim = sim
			else:
				print(f"Warning: Could not get features for a mask crop at bbox [{x0},{y0},{x1},{y1}]", flush=True)

	if best_m_info is None:
		cap.release()
		raise RuntimeError("No mask matched the reference images sufficiently!")
	return best_m_info["segmentation"], best_sim

# ── 3. LOAD VIDEO INFO (Not all frames) ──────────────────────────────────────
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
	raise IOError(f"Cannot open video {video_path}")

FPS = int(cap.get(cv2.CAP_PROP_FPS) or 30)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video Info: {W}x{H} @ {FPS}fps, {TOTAL_FRAMES} frames", flush=True)

if TOTAL_FRAMES <= skip_frames:
	cap.release()
	raise ValueError(f"Video has {TOTAL_FRAMES} frames, which is not more than skip_frames={skip_frames}")

# ── 4. INITIALISE MODELS ────────────────────────────────────────────────────
print("Loading DINOv2 backbone for SAM...", flush=True)
dinov2_backbone_sam = AutoModel.from_pretrained(pretrained_model_sam).to(device).eval()
print("Loading DINOv2 processor for SAM...", flush=True)
dinov2_proc_sam = AutoImageProcessor.from_pretrained(pretrained_model_sam)

print("Loading SAM2 Image Predictor...", flush=True)
img_pred = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-tiny")

# ── 5. REFERENCE EMBEDDINGS ─────────────────────────────────────────────────
print("Calculating reference embeddings...", flush=True)

# from reference images extract features
ref_feats_tensor = get_ref_feats(refs)
ref_feats_tensor_hand = get_ref_feats(refs_hand)

# print("rft", len(ref_feats_tensor), flush=True)
# print("rft_hand", len(ref_feats_tensor_hand), flush=True)


# ── 6. GET SEED FRAME & AUTO MASKS ───────────────────────────────────────────
print(f"Seeking to seed frame {skip_frames}...", flush=True)
cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
ret, seed_frame_bgr = cap.read()
if not ret or seed_frame_bgr is None:
	cap.release()
	raise IOError(f"Could not read seed frame {skip_frames} from video.")
# seed_frame_rgb = cv2.cvtColor(seed_frame_bgr, cv2.COLOR_BGR2RGB)
seed_frame_rgb = seed_frame_bgr
cv2.imwrite("seed_frame.png", seed_frame_rgb)

print("Generating masks on seed frame...", flush=True)

# Initialize SAM2 Mask Generator (ensure model is on the correct device)
mask_gen = SAM2AutomaticMaskGenerator(
	img_pred.model, # Pass the SAM model directly
	points_per_side=8,
	pred_iou_thresh=0.5,		  # only keep masks with IoU-pred confidence ≥ 0.7
	stability_score_thresh=0.9,   # only keep very stable masks
	box_nms_thresh=0.5,		   # merge overlapping boxes more aggressively
	min_mask_region_area=1000	 # drop very small regions
)

# Generate masks requires the image in RGB format
masks = mask_gen.generate(seed_frame_rgb)
print(f"Generated {len(masks)} masks.", flush=True)

# ── 7. PICK BEST MASK BY COS-SIM ────────────────────────────────────────────
print("Finding best initial mask...", flush=True)

mask0, best_sim = find_best_mask(masks, ref_feats_tensor, seed_frame_rgb)
print(f"Best initial mask found with similarity: {best_sim:.4f}", flush=True)

# for hand, compare masks using icp
# masks_hand = [cv2.imread(p).astype(np.bool) for p in refs_hand]

# best_iou = 0
# mask0_hand = None
# mc = 0
# n = 0
# for m in masks:
	
# 	ious, transs = [], []
# 	for m_hand in masks_hand:
# 		iou, trans = get_aligned_iou(m["segmentation"].astype(np.bool), m_hand, n=n)
# 		n += 1
# 		ious.append(iou)
# 		transs.append(trans)
# 	# max over all reference masks
# 	cur_best = max(ious)
# 	# mean over all reference masks
# 	# cur_best = sum(ious) / len(ious)
# 	print(f"best iou at {mc}: {cur_best}")
# 	mc += 1
# 	if cur_best > best_iou:
# 		best_iou = cur_best
# 		mask0_hand = m["segmentation"].astype(np.bool)

# print(f"Best initial mask in seed_frame_hand found with similarity: {best_iou:.4f}", flush=True)

# visualize mask

import clip
clip_model, clip_preprocess = clip.load('ViT-B/32', device)
highest_prob = 0
mask0_hand = None
for i,m in enumerate(masks):
	x_f, y_f, w_f, h_f = m["bbox"]
	x0 = max(0, int(round(x_f))); y0 = max(0, int(round(y_f)))
	x1 = min(W, int(round(x_f + w_f))); y1 = min(H, int(round(y_f + h_f)))
	if x1 <= x0 or y1 <= y0: continue

	# crop_rgb = seed_frame_rgb[y0:y1, x0:x1]
	crop_rgb = m["segmentation"].astype(np.uint8) * 255
	crop_img = clip_preprocess(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
	text = clip.tokenize(["segmentation mask of a pair of hands", "image with no hands visible"]).to(device)

	with torch.no_grad():
		# img_features = clip_model.encode_image(crop_img)
		# text_features = clip_model.encode_text(text)
		logits_per_image, logits_per_text = clip_model(crop_img, text)
		probs = logits_per_image.softmax(dim=-1).cpu().numpy()
		if probs[0][0] > highest_prob:
			highest_prob = probs[0][0]
			mask0_hand = m["segmentation"].astype(np.bool)

	print(f"Label probs for mask {i}:", probs)

visualize_mask(mask0, "initial_mask_visualization.png")
visualize_mask(torch.from_numpy(mask0_hand).bool(), "hand.png")

# Clean up intermediate variables
del masks, seed_frame_rgb, seed_frame_bgr, ref_feats_tensor
gc.collect()
if torch.cuda.is_available():
	torch.cuda.empty_cache()

# Explicitly free CLIP model from memory since we don't need it in the inference loop
print("Freeing memory from CLIP model...", flush=True)
del clip_model, clip_preprocess
# Free reference features for hand as well if you don't need them anymore
del ref_feats_tensor_hand
# Also free any other large tensors or models you don't need
gc.collect()
if torch.cuda.is_available():
	torch.cuda.empty_cache()

# ── 8. TRACK & CLASSIFY FRAME BY FRAME ─────────────────────────────────────
print("Initializing video writer...")
writer = imageio.get_writer(
	boxed_out,
	format="FFMPEG",	# force the FFmpeg plugin
	codec="libx264",	# MP4/H.264 codec
	fps=FPS,
	ffmpeg_params=["-pix_fmt", "yuv420p"]  # ensures broad compatibility
)

print("Initializing SAM2 video state...", flush=True)
# --- Use the video path and total frames for state initialization ---
print("Loading SAM2 Video Predictor...", flush=True)
vid_pred = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-tiny")
state = vid_pred.init_state(video_path=video_path)

# Add the initial mask
vid_pred.add_new_mask(state, frame_idx=skip_frames, mask=mask0, obj_id=0)
vid_pred.add_new_mask(state, frame_idx=skip_frames, mask=mask0_hand, obj_id=1)
print(f"Added initial mask at frame {skip_frames} to tracker.", flush=True)

# Rewind video capture to the beginning to process all frames needed for propagation
cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
current_frame_idx = skip_frames

# Free more memory before starting the inference loop
print("Freeing more memory before inference...", flush=True)
# Free DINOv2 backbone and processor used for SAM - we don't need them in the loop
del dinov2_backbone_sam, dinov2_proc_sam
# Free the mask generator (large model)
del mask_gen, img_pred
# Additional aggressive memory cleanup
gc.collect()
if torch.cuda.is_available():
	torch.cuda.empty_cache()

# sys.exit()

max_frames = 50
video_idx = 1
idx_offset = 0

print("Starting frame-by-frame propagation, classification, and writing...", flush=True)
with torch.inference_mode(), torch.autocast(device_type=str(device), dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
	# Iterate through frames needed by the propagator
	for f_idx, frame_rgb, logits in vid_pred.propagate_in_video(state):

		# Skip processing/writing for frames before skip_frames (or slightly after for buffer)
		# The propagator gives results starting *after* the initial mask frame.
		sam2_video_idx = f_idx + idx_offset
		if sam2_video_idx < skip_frames:
			continue

		assert current_frame_idx == f_idx, f"mismatch betweencurrent_frame_idx: {current_frame_idx}, f_idx: {f_idx}"
		ret, frame_bgr =cap.read()
		current_frame_idx += 1
		if not ret or frame_bgr is None:
			raise ValueError(f"frame {current_frame_idx} cannot be captured")
		frame_output = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
		# Process the mask logits for the current frame f_idx
		masks = []
		# print(f"logits.shape: {logits.shape}", flush=True)
		for obj_id in range(logits.shape[0]):
			# Process on CPU to reduce GPU memory usage
			with torch.no_grad():  # Extra guard against gradient tracking
				mask2d = logits[obj_id].sigmoid().squeeze(0) > 0.5  # BHWC -> HWC -> HW
				# Immediately move to CPU to free GPU memory
				mask_np = mask2d.cpu().numpy()
				masks.append(mask_np)
		
		# Ensure CUDA cache is cleared
		if torch.cuda.is_available() and f_idx % 5 == 0:
			torch.cuda.empty_cache()

		# Calculate IOU between masks for rejection
		rej_threshold = 20
		min_distance = iou_contour(masks)
		# print(f"min_distance: {min_distance}", flush=True)
		# cv2.putText(frame_output, f"min_distance: {min_distance}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
		rej = min_distance < rej_threshold
		mask2d = masks[0] # mask for object for classification

		if mask2d.any():
			ys, xs = np.where(mask2d)
			x0b, y0b, x1b, y1b = xs.min(), ys.min(), xs.max(), ys.max()

			# Ensure bbox coordinates are within frame bounds
			x0b, y0b = max(0, x0b), max(0, y0b)
			x1b, y1b = min(W - 1, x1b), min(H - 1, y1b)

			# Check for valid bbox after clamping
			if x1b > x0b and y1b > y0b:
				
				# ----- classification ------------------------------------------------
				# Crop from the original frame_to_draw (RGB)
				crop_rgb = frame_output[y0b:y1b, x0b:x1b]

				# visualize bbox of object (red if rejected, green otherwise)
				cv2.rectangle(frame_output, (x0b, y0b), (x1b, y1b), (0, 255, 0) if not rej else (255,0,0), 2)

				if crop_rgb.size > 0:
					# Prepare batch for classifier model
					batch = processor_cls(images=[crop_rgb], return_tensors="pt").to(device)
					
					# Run classifier inference
					cls_logits = model(**batch).logits.squeeze(0) # Remove batch dim
					probs = torch.softmax(cls_logits, dim=-1)
					cls_id = probs.argmax().item()
					conf = probs[cls_id].item()

					# Get ground truth label if available
					true_label = frame_to_class.get(f_idx, "N/A") # Use f_idx here
					label_txt = f"class {cls_id} ({conf*100:.1f}%) GT: {true_label}"

					# Add text to frame
					# White text with black outline
					cv2.putText(frame_output, label_txt, (x0b, y0b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
					cv2.putText(frame_output, label_txt, (x0b, y0b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
				else:
					print(f"Warning: Empty crop at frame {f_idx}, bbox [{x0b},{y0b},{x1b},{y1b}]", flush=True)
			else:
				print(f"Warning: Invalid bounding box at frame {f_idx}: ({x0b}, {y0b}) -> ({x1b}, {y1b})", flush=True)


		# Write the frame (with or without bbox/text) to the output video
		# Convert back to BGR for imageio FFMPEG writer if needed (depends on pix_fmt, yuv420p expects RGB-like)
		# Let's keep it RGB as imageio usually handles it. If colors are swapped, change here.
		writer.append_data(frame_output)

		if f_idx > max_frames:
			idx_offset = current_frame_idx

			# Save the current frame index and create a new video starting from here
			print(f"Reached {max_frames} frames, creating new video segment...")
			
			# Create new video path with index
			new_video_path = video_path.replace('.mp4', f'_part{video_idx}.mp4')
			video_idx += 1

			# use cv2 VideoWriter as FFMPEG not installed on cluster
			# ffmpeg will be faster
			out = cv2.VideoWriter(new_video_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W,H))
			r, fr = cap.read()
			while r:
				out.write(fr)
				r, fr = cap.read()
			out.release()

			cap = cv2.VideoCapture(new_video_path)
			cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)

			vid_pred.reset_state(state)
			state = None
			vid_pred = vid_pred.cpu()
			del state
			del vid_pred
			gc.collect()
			torch.cuda.empty_cache()
			vid_pred = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-tiny")
			state = vid_pred.init_state(video_path=new_video_path)
			vid_pred.add_new_mask(state, frame_idx=0, mask=masks[0], obj_id=0)
			vid_pred.add_new_mask(state, frame_idx=0, mask=masks[1], obj_id=1)
			
			print(f"Created new video segment: {new_video_path}")

			

		# Clean up all frame/mask variables
		if 'frame_output' in locals():
			del frame_output
		if 'frame_bgr' in locals():
			del frame_bgr
		if 'crop_rgb' in locals():
			del crop_rgb
		if 'masks' in locals():
			del masks

		# Aggressively clear memory
		gc.collect()
		torch.cuda.empty_cache()


# ── 9. CLEANUP & FINISH ─────────────────────────────────────────────────────
print("Closing video writer...", flush=True)
writer.close()
print("Releasing video capture...", flush=True)
cap.release()
print(f"Processing complete. Output video saved to: {boxed_out}", flush=True)

# ── 9. DISPLAY RESULT ───────────────────────────────────────────────────────
# Video(boxed_out, embed=True, width=min(W, 640)) # Commented out for cluster execution
