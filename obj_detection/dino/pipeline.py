import os
import base64
import io
import json
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor
from utils import DINOv2Classifier
from safetensors.torch import load_file as load_safetensors
from IPython.display import display, HTML

# CONFIGURATION
repo_dir = os.getcwd().split('dslab25')[0] + 'dslab25/'
data_dir = "/work/courses/dslab/team14/"
video_path = os.path.join(data_dir, "assets/vacuum_pump/videos/01_run1_cam_2_1024x1024_15fps_3mbps.mp4")
labels_path = os.path.join(data_dir, "assets/vacuum_pump/videos/output.txt")

# Path to your trained YOLOv12 weights (adjust as needed)
temp_images_dir = os.path.join(repo_dir, "temp_images")
anno_dir = os.path.join(repo_dir, "assets/vacuum_pump/eval/anno")
base_dir = os.path.join(repo_dir, "training/vacuum_pump")

coco_path = os.path.join(base_dir, "coco_annotations.json")
YOL_THRESHOLD = 0.38

os.makedirs(temp_images_dir, exist_ok=True)

pretrained_model = "facebook/dinov2-with-registers-base"
yolo_model_path = os.path.join(repo_dir, "obj_detection/dino/yolo_runs/yolov12_boundingbox2", "weights", "best.pt")
model_dir = os.path.join(repo_dir, "obj_detection/dino/dinov2_finetune/base/final_model/")

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
print(f"Loading labels from: {labels_path}")
frame_to_class = load_labels(labels_path)

# Load COCO annotations to map category IDs to names.
print(f"Loading COCO annotations from: {coco_path}")
try:
	with open(coco_path, 'r') as f:
		coco_data = json.load(f)
	category_id_to_name = {cat['id']: cat.get('name', f'category_{cat["id"]}')
								for cat in coco_data.get('categories', [])}
except (FileNotFoundError, json.JSONDecodeError) as e:
	print(f"Error loading COCO annotations: {e}")
	category_id_to_name = {}

from glob import glob
pretrained_model = "facebook/dinov2-with-registers-base"
image_files = sorted(glob(os.path.join(temp_images_dir, "*.jpg")))
correct_predictions = 0
total_predictions = 0

# Load image processor and model
print("Loading image processor...")
processor = AutoImageProcessor.from_pretrained(pretrained_model)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get number of classes from frame_to_class
num_labels = max(frame_to_class.values()) + 1 if frame_to_class else 8
print(f"Number of classes: {num_labels}")

# Load model
print("Loading model...")
model = DINOv2Classifier(num_labels=num_labels, pretrained_model=pretrained_model)

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
		raise e
else:
	raise Exception(f"Error: Model weights not found in {model_dir}")


model.to(device)
model.eval()

import cv2, imageio, numpy as np, torch, torch.nn.functional as F
from IPython.display import Video
from sam2.sam2_image_predictor  import SAM2ImagePredictor
from sam2.sam2_video_predictor  import SAM2VideoPredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from transformers import AutoModel, AutoImageProcessor

# ── 2.  PATHS & OPTIONS  ─────────────────────────────────────────────────────
refs = [
	os.path.join(base_dir, "images/original/stage_0/stage_0_var_0_case_render_1.jpg"),
	os.path.join(base_dir, "images/original/stage_0/stage_0_var_0_case_render_2.jpg"),
	os.path.join(base_dir, "images/original/stage_0/stage_0_var_0_case_render_3.jpg"),
	os.path.join(base_dir, "images/original/stage_0/stage_0_var_0_case_render_4.jpg"),
	os.path.join(base_dir, "images/original/stage_0/stage_0_var_0_case_render_5.jpg"),
	os.path.join(base_dir, "images/original/stage_0/stage_0_var_0_case_render_6.jpg"),
	os.path.join(base_dir, "images/original/stage_0/stage_0_var_0_case_render_7.jpg"),
	os.path.join(base_dir, "images/original/stage_0/stage_0_var_0_case_render_8.jpg"),
	os.path.join(base_dir, "images/original/stage_0/stage_0_var_0_case_render_9.jpg")
]
boxed_out   = "sam2_boxed.mp4"
device	  = "cuda" if torch.cuda.is_available() else "cpu"
skip_frames = 70							 # ← skip these many frames

# ── 2. UTILS ─────────────────────────────────────────────────────────────────
def read_video_rgb(path):
	cap = cv2.VideoCapture(path)
	if not cap.isOpened():
		raise IOError(f"cannot open {path}")
	frames = []
	while True:
		ret, frm = cap.read()
		if not ret: break
		frames.append(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
	fps = cap.get(cv2.CAP_PROP_FPS) or 30
	cap.release()
	return frames, int(fps)

def get_feat(img_rgb):
	ipt = dinov2_proc(images=img_rgb, return_tensors="pt").to(device)
	with torch.no_grad():
		out = dinov2_backbone(**ipt).last_hidden_state[:,0]
	return F.normalize(out.squeeze(0), dim=-1).cpu()

# ── 3. LOAD VIDEO ────────────────────────────────────────────────────────────
frames, FPS = read_video_rgb(video_path)
assert len(frames) > skip_frames, f"Video must have more than {skip_frames} frames!"
H, W = frames[0].shape[:2]

# ── 4. INITIALISE MODELS ────────────────────────────────────────────────────
backbone_name   = "facebook/dinov2-with-registers-small"
dinov2_backbone = AutoModel.from_pretrained(backbone_name).to(device).eval()
dinov2_proc	 = AutoImageProcessor.from_pretrained(backbone_name)

img_pred = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-tiny")
vid_pred = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-tiny")

# ── 5. REFERENCE EMBEDDINGS ─────────────────────────────────────────────────
ref_feats = []
for p in refs:
	bgr = cv2.imread(p)
	if bgr is None:
		raise IOError(f"cannot open reference image {p}")
	rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
	ref_feats.append(get_feat(rgb))

# ── 6. AUTO MASKS ON SEED FRAME ─────────────────────────────────────────────
seed_frame = frames[skip_frames]
mask_gen = SAM2AutomaticMaskGenerator(
	img_pred.model,
	points_per_side=32,
	pred_iou_thresh=0.7,		  # only keep masks with IoU-pred confidence ≥ 0.7
	stability_score_thresh=0.9,   # only keep very stable masks
	box_nms_thresh=0.3,		   # merge overlapping boxes more aggressively
	min_mask_region_area=1000	 # drop very small regions
)
masks = mask_gen.generate(seed_frame)

# ── 7. PICK BEST MASK BY COS-SIM ────────────────────────────────────────────
best_m, best_sim = None, -1.0
for m in masks:
	x_f, y_f, w_f, h_f = m["bbox"]
	x0 = max(0, int(round(x_f)));	 y0 = max(0, int(round(y_f)))
	x1 = min(W, int(round(x_f + w_f))); y1 = min(H, int(round(y_f + h_f)))
	if x1 <= x0 or y1 <= y0:
		continue

	crop = seed_frame[y0:y1, x0:x1]
	if crop.size == 0:
		continue

	feat = get_feat(crop)
	sims = torch.stack(ref_feats) @ feat	 # [n_refs]
	sim  = sims.max().item()
	if sim > best_sim:
		best_m, best_sim = m, sim

if best_m is None:
	raise RuntimeError("No mask matched the reference images!")

mask0 = torch.from_numpy(best_m["segmentation"]).to(device).bool()

# ── 8. TRACK & DRAW BOX (SKIPPING FIRST 70 FRAMES) ──────────────────────────
# --- use the MP4 path rather than a tensor ---
state = vid_pred.init_state(video_path=video_path)

vid_pred.add_new_mask(state, frame_idx=skip_frames, mask=mask0, obj_id=0)
writer = imageio.get_writer(
	boxed_out,
	format="FFMPEG",	# force the FFmpeg plugin
	codec="libx264",	# MP4/H.264 codec
	fps=FPS,
	ffmpeg_params=["-pix_fmt", "yuv420p"]  # ensures broad compatibility
)
with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
	for f_idx, _, logits in vid_pred.propagate_in_video(state):
		if f_idx < 76:
			continue

		mask2d = logits.sigmoid()[0].squeeze(0) > 0.5
		frame   = frames[f_idx].copy()

		if mask2d.any():
			ys, xs = np.where(mask2d.cpu().numpy())
			x0b, y0b, x1b, y1b = xs.min(), ys.min(), xs.max(), ys.max()
			cv2.rectangle(frame, (x0b, y0b), (x1b, y1b), (0,255,0), 2)

			# ----- classification ------------------------------------------------
			crop_rgb = frames[f_idx][y0b:y1b, x0b:x1b]         # use pristine frame
			batch    = processor(images=[crop_rgb], return_tensors="pt").to(device)
			cls_logits = model(**batch).logits.squeeze(0)
			probs   = torch.softmax(cls_logits, dim=-1)
			cls_id  = probs.argmax().item()
			conf    = probs[cls_id].item()

			label_txt = f"class {cls_id}  {conf*100:.1f}% true: {frame_to_class[f_idx]}"
			# outline for readability
			cv2.putText(frame, label_txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
			cv2.putText(frame, label_txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)

		writer.append_data(frame)
writer.close()

# ── 9. DISPLAY RESULT ───────────────────────────────────────────────────────
Video(boxed_out, embed=True, width=min(W, 640))
