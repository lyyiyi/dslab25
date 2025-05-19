import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
from skimage.draw import polygon
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageClassification


def set_seed(seed=42):
    """
    Sets seeds and configuration to ensure (as much as possible) deterministic behavior in PyTorch.

    Args:
        seed (int): Seed value to use. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def read_video_rgb(path, output_dir="video_frames", use_cached=True):
    if use_cached and os.path.isdir(output_dir):
        # Collect all .png files in directory, assuming they're frames
        frame_paths = sorted([
            os.path.join(output_dir, fname)
            for fname in os.listdir(output_dir)
            if fname.endswith(".png")
        ])
        # Optional: return an estimated FPS if you skip actual decoding
        print("Debug")
        return frame_paths, 30

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {path}")

    os.makedirs(output_dir, exist_ok=True)

    frame_paths = []
    frame_idx = 0

    while True:
        ret, frm = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        frame_filename = f"frame_{frame_idx:05d}.png"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        frame_paths.append(frame_path)

        frame_idx += 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    print("FPS:", fps)

    return frame_paths, int(fps)


def load_frame_rgb(frame_id, base_path="video_frames"):
    filename = f"frame_{frame_id:05d}.png"
    frame_path = os.path.join(base_path, filename)

    bgr_img = cv2.imread(frame_path)
    if bgr_img is None:
        raise IOError(f"Cannot read image at {frame_path}")

    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img


def get_feat(img_rgb, dinov2_proc, dinov2_backbone, device='cuda'):
	ipt = dinov2_proc(images=img_rgb, return_tensors="pt").to(device)
	with torch.no_grad():
		out = dinov2_backbone(**ipt).last_hidden_state[:,0]
	return F.normalize(out.squeeze(0), dim=-1).cpu()


def crop_mask(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros((0, 0), dtype=bool)  # or return the original mask if preferred

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return mask[rmin:rmax+1, cmin:cmax+1]


def euclidean_distance(point1, point2):
    a = np.array(point1)
    b = np.array(point2)
    return np.linalg.norm(a - b, ord=2)


def point_based_matching(point_pairs):
    x_mean = y_mean = xp_mean = yp_mean = 0
    n = len(point_pairs)

    if n == 0:
        return None, None, None, None

    for (x, y), (xp, yp) in point_pairs:
        x_mean += x
        y_mean += y
        xp_mean += xp
        yp_mean += yp

    x_mean /= n
    y_mean /= n
    xp_mean /= n
    yp_mean /= n

    s_x_xp = s_y_yp = s_x_yp = s_y_xp = 0
    sigma_xx = sigma_yy = 0
    for (x, y), (xp, yp) in point_pairs:
        dx = x - x_mean
        dy = y - y_mean
        dxp = xp - xp_mean
        dyp = yp - yp_mean

        s_x_xp += dx * dxp
        s_y_yp += dy * dyp
        s_x_yp += dx * dyp
        s_y_xp += dy * dxp

        sigma_xx += dx * dx
        sigma_yy += dy * dy

    rot_angle = math.atan2(s_x_yp - s_y_xp, s_x_xp + s_y_yp)
    c, s = math.cos(rot_angle), math.sin(rot_angle)

    numerator = s_x_xp + s_y_yp
    denominator = sigma_xx + sigma_yy
    scale = numerator / denominator if denominator != 0 else 1.0

    tx = xp_mean - scale * (x_mean * c - y_mean * s)
    ty = yp_mean - scale * (x_mean * s + y_mean * c)

    return rot_angle, tx, ty, scale


def icp(reference_points, points, max_iterations=200, distance_threshold=20.0,
        convergence_translation_threshold=1e-3, convergence_rotation_threshold=1e-4,
        convergence_scale_threshold=1e-4, point_pairs_threshold=5, verbose=False):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(reference_points)
    src = points.copy()

    # Cumulative transformation: initialize
    total_rotation = 0.0
    total_scale = 1.0
    total_translation = np.zeros(2)

    for iter_num in range(max_iterations):
        if verbose:
            print(f'--- ICP Iteration {iter_num} ---')

        closest_point_pairs = []

        distances, indices = nbrs.kneighbors(src)
        for i in range(len(distances)):
            if distances[i][0] < distance_threshold:
                closest_point_pairs.append((src[i], reference_points[indices[i][0]]))

        if verbose:
            print(f'Point pairs found: {len(closest_point_pairs)}')
        if len(closest_point_pairs) < point_pairs_threshold:
            if verbose:
                print("Too few point pairs — stopping.")
            break

        angle, tx, ty, scale = point_based_matching(closest_point_pairs)
        if angle is None:
            break

        # Compose this iteration's transform
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s], [s, c]])
        src = scale * (src @ R.T)
        src[:, 0] += tx
        src[:, 1] += ty

        # Update cumulative transform
        total_rotation += angle
        total_scale *= scale
        total_translation = scale * (total_translation @ R.T) + [tx, ty]

        if abs(angle) < convergence_rotation_threshold and \
            abs(tx) < convergence_translation_threshold and \
            abs(ty) < convergence_translation_threshold and \
            abs(scale-1) < convergence_scale_threshold:
            if verbose:
                print(f"Converged after {iter_num+1} iterations.")
            break

    # Return final aligned points and full transformation
    transform = {
        "rotation": total_rotation,                 # angle in radians
        "scale": total_scale,                       # scalar
        "translation": total_translation.tolist(),  # [tx, ty]
    }

    return transform, src


def get_largest_contour(mask):
    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float32)


def contour_to_mask(shape, contour):
    rr, cc = polygon(contour[:, 1], contour[:, 0], shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def get_aligned_iou(mask, mask_ref, max_points=200):
    """
    Deprecated: use get_best_iou instead
    mask: mask of the current object to be matched to the mask of reference object (mask_ref)
    """
    mask1 = crop_mask(mask_ref)
    mask2 = crop_mask(mask)

    contour1 = get_largest_contour(mask1)
    contour2 = get_largest_contour(mask2)
    if contour1 is None or contour2 is None:
        return 0.0

    N = min(len(contour1), len(contour2), max_points)
    idx1 = np.linspace(0, len(contour1) - 1, N).astype(int)
    idx2 = np.linspace(0, len(contour2) - 1, N).astype(int)
    points1 = contour1[idx1]
    points2 = contour2[idx2]

    trans, aligned_points2 = icp(
		reference_points=points1,
		points=points2,
		max_iterations=200,
        point_pairs_threshold=N//10,
		distance_threshold=150,
		convergence_translation_threshold=1e-3,
		convergence_rotation_threshold=1e-4,
		convergence_scale_threshold=1e-3,
		verbose=False
    )

    aligned_mask = contour_to_mask(mask1.shape, aligned_points2)

    intersection = np.logical_and(mask1, aligned_mask).sum()
    union = np.logical_or(mask1, aligned_mask).sum()
    iou = intersection / union if union > 0 else 0.0
    return iou, trans


def plot_contours(*contours, colors=None, labels=None, figsize=(6, 6), title="Contours"):
    plt.figure(figsize=figsize)
    for i, contour in enumerate(contours):
        # Default color cycle
        color = colors[i] if colors and i < len(colors) else None
        label = labels[i] if labels and i < len(labels) else f"Mask {i+1}"
        plt.plot(contour[:, 0], contour[:, 1], label=label, color=color)

    plt.gca().invert_yaxis()  # image coordinates: origin at top-left
    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def compute_metrics(results, merge_classes=None):
    """
    Computes accuracy, precision, recall, and F1-score for multi-class classification,
    with optional merging of specific classes.

    Args:
        results (dict): Dictionary of frame results keyed by frame ID (e.g., '25').
        merge_classes (list of tuples, optional): List of class pairs to merge. 
            For example, [(3, 4), (6, 7)] will merge classes 3 & 4 and 6 & 7.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1-score.
    """
    y_true = []
    y_pred = []

    for _, data in results.items():
        if not data.get('rejected', False):  # Only include if 'rejected' == False
            y_true.append(data['gt'])
            y_pred.append(data['pred'])

    if not y_true:
        return {"accuracy": None, "precision": None, "recall": None, "f1": None}

    # Apply class merging if specified
    if merge_classes:
        merge_map = {}
        for pair in merge_classes:
            for cls in pair:
                merge_map[cls] = pair[0]  # Map all classes in the pair to the first class

        y_true = [merge_map.get(cls, cls) for cls in y_true]
        y_pred = [merge_map.get(cls, cls) for cls in y_pred]

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def load_occlusion_labels(filepath, num_frames=None):
    """
    Load occlusion labels from a file and create a boolean occlusion mask.
    Args:
        filepath (str): Path to the file with occlusion intervals (start and end frames per line).
        num_frames (int, optional): Total number of frames. Defaults to the highest end frame + 1.
    Returns:
        numpy.ndarray: Boolean array where `True` indicates occluded frames.
    """
    occlusion_intervals = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                start_frame, end_frame = int(parts[0]), int(parts[1])
                occlusion_intervals.append((start_frame, end_frame))

    # Case when no frames are occluded
    if not occlusion_intervals:
        return np.zeros(num_frames or 0, dtype=bool)

    # Determine the number of frames if not provided
    if num_frames is None:
        num_frames = occlusion_intervals[-1][1] + 1

    # Create a boolean mask for occluded frames
    occlusion_mask = np.zeros(num_frames, dtype=bool)
    for start_frame, end_frame in occlusion_intervals:
        occlusion_mask[start_frame:end_frame + 1] = True

    return occlusion_mask

def rotate_contour(contour, angle):
    """Rotate contour by a given angle (in radians) around its centroid."""
    angle = np.radians(angle)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    contour = contour @ rotation_matrix.T

    # Translate the contour so that the top-left corner of its bounding box is at (0, 0)
    min_x, min_y = np.min(contour, axis=0)
    contour -= [min_x, min_y]

    return contour

def get_best_iou(mask1, mask2):
    """
    mask1: mask of the current object to be matched to the mask of reference object (mask_ref)
    rotate mask1 to find the best iou with mask2 (ref_mask)

    Reinitailizes ICP for each rotation which is beneficial for performance -> avoids falling into local minima.
    """
    mask1 = crop_mask(mask1)
    mask2 = crop_mask(mask2)

    contour1 = get_largest_contour(mask1)
    contour2 = get_largest_contour(mask2)
    if contour1 is None or contour2 is None:
        return 0.0

    iou = []
    trans = []
    for angle in range(0,360,90):
        rot_contour = rotate_contour(contour2, angle)
        N = min(len(contour1), len(rot_contour), 200)
        idx1 = np.linspace(0, len(contour1) - 1, N).astype(int)
        idx2 = np.linspace(0, len(rot_contour) - 1, N).astype(int)
        points1 = contour1[idx1]
        points2 = rot_contour[idx2]

        tran, aligned_points2 = icp(
            reference_points=points1,
            points=points2,
            max_iterations=200,
            point_pairs_threshold=N//10,
            distance_threshold=150,
            convergence_translation_threshold=1e-3,
            convergence_rotation_threshold=1e-4,
            convergence_scale_threshold=1e-3,
            verbose=False
        )

        aligned_mask2 = contour_to_mask(mask1.shape, aligned_points2)

        intersection = np.logical_and(mask1, aligned_mask2).sum()
        union = np.logical_or(mask1, aligned_mask2).sum()
        res = intersection / union if union > 0 else 0.0
        iou.append(res)
        trans.append(tran)
    #return the highest iou
    return max(iou), trans[iou.index(max(iou))]