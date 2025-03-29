#!/usr/bin/env python3
import os
import re
import argparse

# -------------------------------
# Manual Annotations (pixel coordinates)
# -------------------------------
# These are your manually labeled bounding boxes (format: [x, y, width, height])
bbox_render1 = [4, 241, 255.22, 261.14]	# render_1 (bottom left)
bbox_render5 = [124, 125, 253.32, 256.51]	# render_5 (middle middle)
bbox_render7 = [5, 8, 254.63, 250.72]		# render_7 (top left)

# Compute average width and height
bbox_width = (bbox_render1[2] + bbox_render5[2] + bbox_render7[2]) / 3.0
bbox_height = (bbox_render1[3] + bbox_render5[3] + bbox_render7[3]) / 3.0

# Reference bounding box (from render_5)
ref_x, ref_y = bbox_render5[0], bbox_render5[1]

# Compute horizontal offsets for left column (average from render_1 and render_7)
offset_left = ((ref_x - bbox_render1[0]) + (ref_x - bbox_render7[0])) / 2.0
# For right column, assume symmetric offset:
offset_right = offset_left

# Compute vertical offsets: bottom (from render_1) and top (from render_7)
offset_bottom = (bbox_render1[1] - ref_y)
offset_top = (ref_y - bbox_render7[1])

print(f"Computed offsets (in pixels):")
print(f"  Left: {offset_left}   Right: {offset_right}")
print(f"  Top: {offset_top}	Bottom: {offset_bottom}")
print(f"Averaged bbox size (pixels): width = {bbox_width}, height = {bbox_height}")

# -------------------------------
# Grid Mapping (relative to the middle cell)
# -------------------------------
# Mapping for render numbers (1 to 9) to grid offsets:
#   render 1: bottom left	-> (col: -1, row: +1)
#   render 2: bottom middle  -> (col:  0, row: +1)
#   render 3: bottom right   -> (col: +1, row: +1)
#   render 4: middle left	-> (col: -1, row:  0)
#   render 5: middle middle  -> (col:  0, row:  0)
#   render 6: middle right   -> (col: +1, row:  0)
#   render 7: top left	   -> (col: -1, row: -1)
#   render 8: top middle	 -> (col:  0, row: -1)
#   render 9: top right	  -> (col: +1, row: -1)
grid_mapping = {
	1: (-1, 1),
	2: (0, 1),
	3: (1, 1),
	4: (-1, 0),
	5: (0, 0),
	6: (1, 0),
	7: (-1, -1),
	8: (0, -1),
	9: (1, -1)
}

# Define the pixel offsets for each grid column and row
col_offsets = {-1: -offset_left, 0: 0, 1: offset_right}
row_offsets = {-1: -offset_top, 0: 0, 1: offset_bottom}

# -------------------------------
# YOLO Conversion Helper
# -------------------------------
def convert_bbox_to_yolo(bbox, img_width=512, img_height=512):
	"""
	Convert a bounding box from [x, y, width, height] (pixel coordinates)
	to YOLO format: [center_x_norm, center_y_norm, width_norm, height_norm].
	"""
	x, y, w, h = bbox
	center_x = x + w / 2.0
	center_y = y + h / 2.0
	return (center_x / img_width, center_y / img_height, w / img_width, h / img_height)

# -------------------------------
# Processing function
# -------------------------------
def process_images(root_dir, output_dir=None):
	"""
	Process each image inside stage_0 to stage_7 folders, compute its bounding box
	based on grid offsets, convert to YOLO format, and write a label file.
	"""
	# Walk through subfolders in root_dir
	for stage in sorted(os.listdir(root_dir)):
		stage_path = os.path.join(root_dir, stage)
		if not os.path.isdir(stage_path) or not stage.startswith("stage_"):
			continue

		# Get list of image files (accepting .jpg, .jpeg, .png)
		image_files = [f for f in os.listdir(stage_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
		for img_file in sorted(image_files):
			# Extract render number from filename using regex (e.g., "render_1")
			match = re.search(r'render_(\d+)', img_file)
			if not match:
				print(f"Warning: Could not extract render number from {img_file}; skipping.")
				continue

			render_num = int(match.group(1))
			if render_num not in grid_mapping:
				print(f"Warning: Render number {render_num} not in grid mapping for {img_file}; skipping.")
				continue

			# Determine grid offset for this image
			col, row = grid_mapping[render_num]
			# Compute top-left corner of bounding box in pixels
			bbox_x = ref_x + col_offsets[col]
			bbox_y = ref_y + row_offsets[row]
			bbox = [bbox_x, bbox_y, bbox_width, bbox_height]

			# Convert bbox to YOLO format (normalized)
			yolo_bbox = convert_bbox_to_yolo(bbox, img_width=512, img_height=512)
			# Format the annotation line (class "0")
			annotation_line = "0 {:.9f} {:.9f} {:.9f} {:.9f}".format(*yolo_bbox)

			# Determine where to save the label file
			# If an output directory is provided, mirror the stage folder structure.
			if output_dir:
				stage_out_dir = os.path.join(output_dir, stage)
				os.makedirs(stage_out_dir, exist_ok=True)
				label_path = os.path.join(stage_out_dir, os.path.splitext(img_file)[0] + ".txt")
			else:
				# Save label file in the same folder as the image
				label_path = os.path.join(stage_path, os.path.splitext(img_file)[0] + ".txt")

			# Write the annotation to the file
			with open(label_path, "w") as f:
				f.write(annotation_line + "\n")
			print(f"Saved YOLO annotation to {label_path}")

# -------------------------------
# Argument Parsing and Entry Point
# -------------------------------
if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Generate YOLOv12 annotation files for images using fixed bbox sizes and grid shifts."
	)
	parser.add_argument("--root", type=str, required=True,
						help="Root directory containing stage_0, stage_1, ... stage_7 folders with images.")
	parser.add_argument("--output-dir", type=str, default=None,
						help="Optional output directory for YOLO annotation files. "
							 "If not provided, annotations are saved next to the images.")
	args = parser.parse_args()

	process_images(args.root, args.output_dir)