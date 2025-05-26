import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.draw import polygon
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
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
     
def visualize_mask(mask, mask_output_path):
	"""Visualize the mask and save it to a file."""
	try:
		# Convert boolean tensor to an 8-bit grayscale NumPy array
		# True -> 255 (white), False -> 0 (black)
		mask_image = mask.cpu().numpy().astype(np.uint8) * 255
		
		# Save the image
		cv2.imwrite(mask_output_path, mask_image)
		print(f"Initial mask saved to: {mask_output_path}", flush=True)
	except Exception as e:
		print(f"Error saving mask visualization: {e}", flush=True)
	
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


def get_top_n_contours(mask, n=1, min_area = 5000):
    mask_u8 = (mask.astype(np.uint8)) * 255
    if len(mask_u8.shape) == 3:
         mask_u8 = cv2.cvtColor(mask_u8, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    # Filter contours by area and sort
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    valid_contours.sort(key=cv2.contourArea, reverse=True)

    # Return top N contours (or fewer if not enough valid ones exist)
    return [cnt[:, 0, :].astype(np.float32) for cnt in valid_contours[:n]]


def contour_to_mask(shape, contour, boundary2=None):
    rr, cc = polygon(contour[:boundary2, 1], contour[:boundary2, 0], shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    if boundary2 is not None:
        rr, cc = polygon(contour[boundary2:, 1], contour[boundary2:, 0], shape)
        mask[rr, cc] = True
    return mask


def get_aligned_iou(mask, mask_ref, max_points=200, n = 0):
    """
    mask: mask of the current object to be matched to the mask of reference object (mask_ref)
    """
    mask1 = crop_mask(mask_ref)
    mask2 = crop_mask(mask)

    contour1 = get_top_n_contours(mask1, n=2)
    contour2 = get_top_n_contours(mask2, n=2)
    
    plot_contours(*contour1, *contour2, title=f"contours_{n}")
    if contour1 is None or contour2 is None:
        return 0.0

    N = min(*[len(c) for c in contour1], *[len(c) for c in contour2], max_points)
    idx1 = [np.linspace(0, len(c) - 1, N).astype(int) for c in contour1]    
    idx2 = [np.linspace(0, len(c) - 1, N).astype(int) for c in contour2]

    points1 = [c[i] for c,i in zip(contour1, idx1)]
    points2 = [c[i] for c,i in zip(contour2, idx2)]
    points1 = np.concatenate(points1,axis=0)
    points2 = np.concatenate(points2,axis=0)

    boundary2 = None
    if points2.shape[0] == 2 * N: # for top 2 contours
        boundary2 = N

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

    aligned_mask = contour_to_mask(mask1.shape, aligned_points2, boundary2)

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
    plt.savefig(f"contours/{title}.png")


def compute_metrics(results):
    """
    Computes accuracy, precision, recall, and F1-score for multi-class classification,
    considering only frames with 'rejected' == False.

    Args:
        results (dict): Dictionary of frame results keyed by frame ID (e.g., '25').

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

def iou(masks):
    """
    Calculate Intersection Over Union (IoU) for a list of masks.
    
    Args:
        masks (list): List of 2 binary masks (object + hands)
        threshold (float): IoU threshold for considering a match
        
    Returns:
        list: List of IoU values for each mask pair
    """
    assert len(masks) == 2, "should provide 2 masks (object + hands)"

    mask1 = masks[0]
    mask2 = masks[1]

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union > 0 else 0.0
    return iou

def iou_contour(masks):
    assert len(masks) == 2, "should provide 2 masks (object + hands)"

    mask1 = masks[0]
    mask2 = masks[1]

    contour1 = get_top_n_contours(mask1, n=1, min_area=1000)
    contour2 = get_top_n_contours(mask2, n=2, min_area=1000)

    if contour1 is None or contour2 is None:
        return 0.0
    if len(contour1) == 0 or len(contour2) == 0:
        return 0.0
    
    # Test all pairs of contours (object vs hand)
    proximity_threshold = 10  # pixels (adjust based on your image size)

    min_d = float('inf')
    
    for hand_contour in contour2:
        # Calculate minimum distance between contours
        is_close, min_distance = are_contours_close(contour1[0], hand_contour, proximity_threshold)
        if min_distance < min_d:
             min_d = min_distance
    return min_d


def are_contours_close(contour1, contour2, threshold):
    """
    Check if two contours are within a certain distance of each other.
    
    Args:
        contour1: First contour points as numpy array
        contour2: Second contour points as numpy array
        threshold: Distance threshold in pixels
        
    Returns:
        tuple: (is_close, min_distance) - Boolean indicating if contours are close,
               and the minimum distance between them
    """
    # For efficiency, first check bounding box proximity
    x1_min, y1_min = np.min(contour1, axis=0)
    x1_max, y1_max = np.max(contour1, axis=0)
    
    x2_min, y2_min = np.min(contour2, axis=0)
    x2_max, y2_max = np.max(contour2, axis=0)
    
    # Quick rejection: if bounding boxes are far apart
    if (x1_min > x2_max + threshold or x2_min > x1_max + threshold or
        y1_min > y2_max + threshold or y2_min > y1_max + threshold):
        # Calculate approximate distance between boxes
        dx = max(0, max(x1_min - x2_max, x2_min - x1_max))
        dy = max(0, max(y1_min - y2_max, y2_min - y1_max))
        bb_distance = np.sqrt(dx*dx + dy*dy)
        return False, bb_distance
    
    # If bounding boxes are close or overlapping, compute point-wise distances
    # Use a subset of points for efficiency (can be adjusted)
    points_sample_rate = 50
    sample_rate = max(1, len(contour1) // points_sample_rate)  # Sample at most 20 points
    contour1_sampled = contour1[::sample_rate]
    
    sample_rate = max(1, len(contour2) // points_sample_rate)
    contour2_sampled = contour2[::sample_rate]
    
    min_distance = float('inf')
    for pt1 in contour1_sampled:
        for pt2 in contour2_sampled:
            distance = np.sqrt(np.sum((pt1 - pt2)**2))
            min_distance = min(min_distance, distance)
            if min_distance <= threshold:
                return True, min_distance
    
    return min_distance <= threshold, min_distance

def check_bbox_overlap(contour1, contour2, margin=0):
    """
    Check if the bounding boxes of two contours overlap (with optional margin).
    
    Args:
        contour1: First contour
        contour2: Second contour
        margin: Extra margin to add around bounding boxes
        
    Returns:
        bool: True if bounding boxes overlap
    """
    x1_min, y1_min = np.min(contour1, axis=0)
    x1_max, y1_max = np.max(contour1, axis=0)
    
    x2_min, y2_min = np.min(contour2, axis=0)
    x2_max, y2_max = np.max(contour2, axis=0)
    
    # Add margin
    x1_min -= margin
    y1_min -= margin
    x1_max += margin
    y1_max += margin
    
    # Check for overlap
    return (x1_min <= x2_max and x1_max >= x2_min and
            y1_min <= y2_max and y1_max >= y2_min)

def check_hands_above(object_contour, hand_contour):
    """
    Check if the hand contour is positioned above the object contour
    (specific to vacuum pump assembly scenario).
    
    Args:
        object_contour: Contour of the main object
        hand_contour: Contour of the hand
        
    Returns:
        bool: True if hands are positioned above the object
    """
    # Get bounding boxes
    obj_y_min = np.min(object_contour[:, 1])
    obj_y_max = np.max(object_contour[:, 1])
    obj_x_min = np.min(object_contour[:, 0])
    obj_x_max = np.max(object_contour[:, 0])
    
    hand_y_min = np.min(hand_contour[:, 1])
    hand_y_max = np.max(hand_contour[:, 1])
    hand_x_min = np.min(hand_contour[:, 0])
    hand_x_max = np.max(hand_contour[:, 0])
    
    # Calculate centroids
    obj_center_x = (obj_x_min + obj_x_max) / 2
    hand_center_x = (hand_x_min + hand_x_max) / 2
    
    # Check if hands are above and aligned with the object
    # For vacuum pump assembly, hands coming from above are likely to be occluding
    hands_above = (
        hand_y_max >= obj_y_min and    # Hand extends below the top of the object
        hand_y_min < obj_y_min and     # Hand starts above the object
        abs(hand_center_x - obj_center_x) < (obj_x_max - obj_x_min) / 2  # Hand is somewhat aligned with object
    )
    
    return hands_above

def detect_contour_interaction(masks):
    """
    Main function to detect various types of interactions between contours.
    Particularly focused on hand-object interactions in vacuum pump assembly.
    
    Args:
        masks (list): List of 2 binary masks (object + hands)
        
    Returns:
        tuple: (occlusion_score, interaction_info) where:
            - occlusion_score is a float from 0.0 to 1.0
            - interaction_info is a dict with detailed information
    """
    contour_score = iou_contour(masks)
    
    # Get detailed interaction information
    interaction_info = analyze_contour_interaction(masks)
    
    # Return score and detailed information
    return contour_score, interaction_info

def analyze_contour_interaction(masks):
    """
    Analyze the interaction between object and hand contours
    to identify specific types of interactions.
    
    Args:
        masks (list): List of 2 binary masks (object + hands)
        
    Returns:
        dict: Dictionary with detailed interaction information
    """
    if len(masks) != 2:
        return {}
        
    obj_mask = masks[0]
    hand_mask = masks[1]
    
    # Get top contours
    obj_contours = get_top_n_contours(obj_mask, n=1, min_area=500)
    hand_contours = get_top_n_contours(hand_mask, n=2, min_area=300)
    
    if obj_contours is None or not obj_contours:
        return {"error": "No object contours found"}
        
    if hand_contours is None or not hand_contours:
        return {"error": "No hand contours found"}
    
    obj_contour = obj_contours[0]
    hand_contour = hand_contours[0]
    
    # Get bounding boxes
    obj_bbox = get_contour_bbox(obj_contour)
    hand_bbox = get_contour_bbox(hand_contour)
    
    # 1. Check for hands above the object
    hands_above = is_hands_above(obj_bbox, hand_bbox)
    
    # 2. Check for hands approaching from sides
    hands_from_side = is_hands_from_side(obj_bbox, hand_bbox)
    
    # 3. Check for proximity
    _, min_distance = are_contours_close(obj_contour, hand_contour, threshold=50)
    
    # 4. Check for direct overlap
    is_overlapping = check_bbox_overlap(obj_contour, hand_contour)
    
    # 5. Check positional relationships specific to vacuum pump assembly
    assembly_relationships = check_assembly_relationships(obj_bbox, hand_bbox)
    
    return {
        "hands_above": hands_above,
        "hands_from_side": hands_from_side,
        "min_distance": min_distance,
        "is_overlapping": is_overlapping,
        "assembly_context": assembly_relationships
    }

def get_contour_bbox(contour):
    """
    Get bounding box from contour.
    
    Args:
        contour: Contour points as numpy array
        
    Returns:
        tuple: (x_min, y_min, x_max, y_max)
    """
    x_min, y_min = np.min(contour, axis=0)
    x_max, y_max = np.max(contour, axis=0)
    return (x_min, y_min, x_max, y_max)

def is_hands_above(obj_bbox, hand_bbox):
    """
    Check if hands are above the object.
    
    Args:
        obj_bbox: Object bounding box (x_min, y_min, x_max, y_max)
        hand_bbox: Hand bounding box (x_min, y_min, x_max, y_max)
        
    Returns:
        bool: True if hands are above the object
    """
    obj_x_min, obj_y_min, obj_x_max, obj_y_max = obj_bbox
    hand_x_min, hand_y_min, hand_x_max, hand_y_max = hand_bbox
    
    obj_width = obj_x_max - obj_x_min
    obj_center_x = (obj_x_min + obj_x_max) / 2
    hand_center_x = (hand_x_min + hand_x_max) / 2
    
    # Hands are above if they're vertically above and horizontally aligned
    return (
        hand_y_max >= obj_y_min and   # Hand extends to or below the top of the object
        hand_y_min < obj_y_min and    # Hand starts above the object
        abs(hand_center_x - obj_center_x) < obj_width / 2  # Hand is aligned with object
    )

def is_hands_from_side(obj_bbox, hand_bbox):
    """
    Check if hands are approaching from the side.
    
    Args:
        obj_bbox: Object bounding box (x_min, y_min, x_max, y_max)
        hand_bbox: Hand bounding box (x_min, y_min, x_max, y_max)
        
    Returns:
        bool: True if hands are approaching from the side
    """
    obj_x_min, obj_y_min, obj_x_max, obj_y_max = obj_bbox
    hand_x_min, hand_y_min, hand_x_max, hand_y_max = hand_bbox
    
    obj_height = obj_y_max - obj_y_min
    obj_center_y = (obj_y_min + obj_y_max) / 2
    hand_center_y = (hand_y_min + hand_y_max) / 2
    
    # Hands from left
    from_left = (
        hand_x_max >= obj_x_min and
        hand_x_min < obj_x_min and
        abs(hand_center_y - obj_center_y) < obj_height / 2
    )
    
    # Hands from right
    from_right = (
        hand_x_min <= obj_x_max and
        hand_x_max > obj_x_max and
        abs(hand_center_y - obj_center_y) < obj_height / 2
    )
    
    return from_left or from_right

def check_assembly_relationships(obj_bbox, hand_bbox, context="vacuum_pump"):
    """
    Check for spatial relationships specific to the given context.
    For vacuum pump assembly, we're interested in detecting when
    hands are interacting with the pump during assembly.
    
    Args:
        obj_bbox: Object bounding box (x_min, y_min, x_max, y_max)
        hand_bbox: Hand bounding box (x_min, y_min, x_max, y_max)
        context: Assembly context (default: "vacuum_pump")
        
    Returns:
        dict: Contextual relationship information
    """
    obj_x_min, obj_y_min, obj_x_max, obj_y_max = obj_bbox
    hand_x_min, hand_y_min, hand_x_max, hand_y_max = hand_bbox
    
    obj_width = obj_x_max - obj_x_min
    obj_height = obj_y_max - obj_y_min
    obj_area = obj_width * obj_height
    
    hand_width = hand_x_max - hand_x_min
    hand_height = hand_y_max - hand_y_min
    hand_area = hand_width * hand_height
    
    # For vacuum pump, we're often assembling from the top
    if context == "vacuum_pump":
        # Hand to object size ratio - if hands are much larger, likely handling the object
        hand_obj_size_ratio = hand_area / obj_area if obj_area > 0 else 0
        
        # Vertical alignment - for placing components from above
        vertical_alignment = is_hands_above(obj_bbox, hand_bbox)
        
        # Check if hand width is greater than object width (enveloping the object)
        hand_enveloping = hand_width > obj_width * 1.2
        
        return {
            "hand_obj_size_ratio": hand_obj_size_ratio,
            "vertical_alignment": vertical_alignment,
            "hand_enveloping": hand_enveloping,
            "likely_handling": hand_enveloping or hand_obj_size_ratio > 0.8 or vertical_alignment
        }
    
    # Default return for unknown context
    return {"context_error": "Unknown assembly context"}

def draw_contour_analysis(frame, masks, score, info):
    """
    Draw analysis results on the frame for visualization.
    
    Args:
        frame: The input frame (in BGR format for OpenCV)
        masks: List of masks [object_mask, hand_mask]
        score: Occlusion score from detect_contour_interaction
        info: Interaction info dictionary
        
    Returns:
        frame: The annotated frame
    """
    frame_with_vis = frame.copy()
    h, w = frame.shape[:2]
    
    # Draw banner at the top with occlusion score
    banner_height = 40
    cv2.rectangle(frame_with_vis, (0, 0), (w, banner_height), (0, 0, 0), -1)
    
    # Determine color based on occlusion score (green -> yellow -> red)
    if score < 0.3:
        color = (0, 255, 0)  # Green - no occlusion
        status = "CLEAR VIEW"
    elif score < 0.7:
        color = (0, 255, 255)  # Yellow - potential occlusion
        status = "POTENTIAL OCCLUSION"
    else:
        color = (0, 0, 255)  # Red - likely occlusion
        status = "OCCLUSION DETECTED"
    
    # Draw status text
    cv2.putText(frame_with_vis, f"{status} - Score: {score:.2f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Draw contours if masks are available
    if len(masks) == 2:
        obj_mask = masks[0]
        hand_mask = masks[1]
        
        # Draw object contour
        obj_contours = get_top_n_contours(obj_mask, n=1)
        if obj_contours:
            cv2.drawContours(frame_with_vis, [obj_contours[0].astype(np.int32)], 0, (0, 255, 0), 2)
        
        # Draw hand contour
        hand_contours = get_top_n_contours(hand_mask, n=2)
        if hand_contours:
            for i, contour in enumerate(hand_contours):
                cv2.drawContours(frame_with_vis, [contour.astype(np.int32)], 0, (0, 0, 255), 2)
    
    # Add detailed info at the bottom
    if info:
        y_pos = h - 120
        cv2.rectangle(frame_with_vis, (0, h - 130), (w, h), (0, 0, 0), -1)
        
        if "hands_above" in info:
            cv2.putText(frame_with_vis, f"Hands above: {info['hands_above']}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 20
            
        if "hands_from_side" in info:
            cv2.putText(frame_with_vis, f"Hands from side: {info['hands_from_side']}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 20
            
        if "min_distance" in info:
            cv2.putText(frame_with_vis, f"Min distance: {info['min_distance']:.1f}px", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 20
            
        if "assembly_context" in info and "likely_handling" in info["assembly_context"]:
            cv2.putText(frame_with_vis, f"Likely handling: {info['assembly_context']['likely_handling']}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame_with_vis


    
    

