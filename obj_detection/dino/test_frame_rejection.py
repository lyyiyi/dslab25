import os
import random
import cv2
import imageio
from IPython.display import Video
import matplotlib.pyplot as plt
import numpy as np
import pickle
from safetensors.torch import load_file as load_safetensors
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor  import SAM2ImagePredictor
from sam2.sam2_video_predictor  import SAM2VideoPredictor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm

from utils import compute_metrics, crop_mask, get_aligned_iou, get_feat, load_frame_rgb, \
    load_occlusion_labels, read_video_rgb, set_seed

if __name__ == "__main__":
    # Set a fixed seed for deterministic behavior
    set_seed(42)

    # 1. CONFIGURATION
    print("\n---------------- 1. CONFIGURATION ----------------")
    video_nr = 1
    iou_threshold = 0.6 
    expected_scale = 0.65 # obtained from reference-5-render vs. clean frame 70
    scale_tolerance = None

    use_cached_frames = False
    fps_video = 5
    skip_frames_dict = {
        1: 31,
        2: 24
    }
    skip_frames = skip_frames_dict[video_nr]
    threshold = skip_frames

    print(f"Config for video: 0{video_nr}_run1_cam_2_1024x1024_{fps_video}fps.mp4")
    DATA_DIR = '/work/courses/dslab/team14/'
    repo_dir = os.getcwd().split('dslab25')[0] + '/dslab25/'
    video_path = os.path.join(DATA_DIR, f'videos/0{video_nr}_run1_cam_2_1024x1024_{fps_video}fps.mp4')
    frames_dir = os.path.join(DATA_DIR, 'videos/frames')
    boxed_out = os.path.join(DATA_DIR, f"videos/0{video_nr}_occlusion_boxed_{fps_video}fps_{iou_threshold}iouthresh_{scale_tolerance}scaletol.mp4")
    mask_ref_path = os.path.join(repo_dir, "obj_detection/dino/", "ref_mask.pkl")
    OUT_PATH = os.path.join(DATA_DIR, f"videos/0{video_nr}_out_occlusions_5fps.pkl")
    occlusion_file = os.path.join(DATA_DIR, f"videos/0{video_nr}_run1_occlusion_5fps.txt")
    out_confusion_matrix = os.path.join(DATA_DIR, f"videos/0{video_nr}_confusion_matrix.png")


    # -- 2.  PATHS & OPTIONS  -----------------------------------------------------
    print("\n---------------- 2. PATHS & OPTIONS ----------------")
    refs = [
        os.path.join(repo_dir, "training/vacuum_pump/images/original/stage_0/stage_0_var_0_case_render_1.jpg"),
        os.path.join(repo_dir, "training/vacuum_pump/images/original/stage_0/stage_0_var_0_case_render_2.jpg"),
        os.path.join(repo_dir, "training/vacuum_pump/images/original/stage_0/stage_0_var_0_case_render_3.jpg"),
        os.path.join(repo_dir, "training/vacuum_pump/images/original/stage_0/stage_0_var_0_case_render_4.jpg"),
        os.path.join(repo_dir, "training/vacuum_pump/images/original/stage_0/stage_0_var_0_case_render_5.jpg"),
        os.path.join(repo_dir, "training/vacuum_pump/images/original/stage_0/stage_0_var_0_case_render_6.jpg"),
        os.path.join(repo_dir, "training/vacuum_pump/images/original/stage_0/stage_0_var_0_case_render_7.jpg"),
        os.path.join(repo_dir, "training/vacuum_pump/images/original/stage_0/stage_0_var_0_case_render_8.jpg"),
        os.path.join(repo_dir, "training/vacuum_pump/images/original/stage_0/stage_0_var_0_case_render_9.jpg")
    ]

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device being used: {device}")
    

    # ── 3. EXTRACT FRAMES FROM THE VIDEO AND LOAD OCCLUSION LABELS ──────────────
    print("\n---------------- 3. EXTRACT FRAMES FROM THE VIDEO ----------------")
    frame_paths, FPS = read_video_rgb(video_path, output_dir=frames_dir, use_cached=use_cached_frames)
    assert len(frame_paths) > skip_frames, f"Video must have more than {skip_frames} frames!"
    H, W = load_frame_rgb(0, base_path=frames_dir).shape[:2]
    # Load occlusion labels
    occluded_gt = load_occlusion_labels(occlusion_file, num_frames=len(frame_paths))


    # ── 4. INITIALISE MODELS ────────────────────────────────────────────────────
    print("\n---------------- 4. INITIALISE MODELS ----------------")
    backbone_name = "facebook/dinov2-with-registers-small"
    dinov2_backbone = AutoModel.from_pretrained(backbone_name).to(device).eval()
    dinov2_proc = AutoImageProcessor.from_pretrained(backbone_name)
    img_pred = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-tiny")
    vid_pred = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-tiny")

    # ── 5. REFERENCE EMBEDDINGS ─────────────────────────────────────────────────
    print("\n---------------- 5. REFERENCE EMBEDDINGS ----------------")
    ref_feats = []
    for p in tqdm(refs):
        bgr = cv2.imread(p)
        if bgr is None:
            raise IOError(f"cannot open reference image {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ref_feat = get_feat(rgb, dinov2_proc, dinov2_backbone, device=device)
        ref_feats.append(ref_feat)
    print("Shape ref_feats:", ref_feats[0].shape)

    # ── 6. AUTO MASKS ON SEED FRAME ─────────────────────────────────────────────
    print("\n---------------- 6. AUTO MASKS ON SEED FRAME ----------------")
    seed_frame = load_frame_rgb(skip_frames, base_path=frames_dir)
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
    print("\n---------------- 7. PICK BEST MASK BY COS-SIM ----------------")
    best_m, best_sim = None, -1.0
    for m in tqdm(masks):
        x_f, y_f, w_f, h_f = m["bbox"]
        print(f"bbox size (x,y,w,h): ({x_f}, {y_f}, {w_f}, {h_f})")
        x0 = max(0, int(round(x_f)))
        y0 = max(0, int(round(y_f)))
        x1 = min(W, int(round(x_f + w_f)))
        y1 = min(H, int(round(y_f + h_f)))
        if x1 <= x0 or y1 <= y0:
            print("[Warning] No bounding box found.")
            continue

        crop = seed_frame[y0:y1, x0:x1]
        if crop.size == 0:
            print("[Warning] Crop.size == 0")
            continue

        feat = get_feat(crop, dinov2_proc, dinov2_backbone, device=device)
        sims = torch.stack(ref_feats) @ feat	 # [n_refs]
        sim  = sims.max().item()
        if sim > best_sim:
            best_m, best_sim = m, sim

    if best_m is None:
        raise RuntimeError("No mask matched the reference images!")

    mask0 = torch.from_numpy(best_m["segmentation"]).to(device).bool()
    print("Shape of mask0:", mask0.shape)

    # ── 8. TRACK & DRAW BOX (SKIPPING FIRST 70 FRAMES) ──────────────────────────
    # --- use the MP4 path rather than a tensor ---
    print("\n---------------- 8. TRACK & DRAW BOX (SKIPPING FIRST 70 FRAMES) ----------------")

    del masks
    del dinov2_backbone
    del dinov2_proc
    del img_pred
    del ref_feats
    del mask_gen
    del seed_frame
    del best_m

    with open(mask_ref_path, 'rb') as f:
        mask_ref = crop_mask(pickle.load(f))
    state = vid_pred.init_state(video_path=video_path, offload_video_to_cpu=False, async_loading_frames=False)
    print("Initialized video predictor.")
    vid_pred.add_new_mask(state, frame_idx=skip_frames, mask=mask0, obj_id=0)
    print("Video predictor done.")
    writer = imageio.get_writer(
        boxed_out,
        format="FFMPEG", # force the FFmpeg plugin
        codec="libx264", # MP4/H.264 codec
        fps=fps_video,
        ffmpeg_params=[
            "-pix_fmt", "yuv420p", # ensures broad compatibility
            "-r", str(fps_video),
            "-vsync", "cfr" # Force constant frame rate
        ]
    )

    frame_count = {
        'total': 0,
        'accepted': 0,
        'rejected': 0
    }

    # ---- CONFIGS FOR HEADERS IN VIDEO OUTPUT ----------------
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    label_pad = 4  # padding around the label text

    # ----------------------------------------------------------
    preds = {} # store all predictions
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        for f_idx, _, logits in tqdm(vid_pred.propagate_in_video(state)):
            if f_idx < threshold: # 76 for 15fps video
                continue

            frame_count['total'] = frame_count['total'] + 1

            mask2d = logits.sigmoid()[0].squeeze(0) > 0.5
            frame = load_frame_rgb(f_idx, base_path=frames_dir)
            frame_with_bbox = frame.copy()
            
            if mask2d.any():
                # Get bounding box
                ys, xs = np.where(mask2d.cpu().numpy())
                x0b, y0b, x1b, y1b = xs.min(), ys.min(), xs.max(), ys.max()
                
                # ----- FRAME REJECTION ------------------------------------------------
                # Perform frame rejection based on boolean masks IoU
                mask_cropped = crop_mask(mask2d.cpu().numpy())
                iou, trans = get_aligned_iou(mask_cropped, mask_ref)
                scale = trans['scale']

                iou_reject = bool(iou is None or iou < iou_threshold)
                if scale_tolerance is not None:
                    scale_reject = bool(scale is None or abs(float(scale) - expected_scale) > scale_tolerance)
                else:
                    scale_reject = False
                rejected = iou_reject or scale_reject

                if rejected:
                    frame_count['rejected'] = frame_count['rejected'] + 1
                else:
                    frame_count['accepted'] = frame_count['accepted'] + 1

                # Draw bounding box
                color = (0, 255, 0) if not rejected else (255, 0, 0)
                cv2.rectangle(frame_with_bbox, (x0b, y0b), (x1b, y1b), color, 2)

                # Occlusion prediction
                predicted_label = 1 if rejected else 0
                true_label = 1 if occluded_gt[f_idx] else 0
                preds[str(f_idx)] = {
                    "pred": predicted_label,
                    "gt": true_label,
                    "iou": iou,
                    "trans": trans
                }

                # ------ ADD HEADERS FOR VISUALIZATION ----------
                header_text = f"Pred: {'occluded' if predicted_label else 'not occluded'}, GT: {'occluded' if true_label else 'not occluded'}"
                header_color = (0, 255, 0) if predicted_label == true_label else (255, 0, 0)
                ((text_w, text_h), _) = cv2.getTextSize(header_text, font, font_scale, font_thickness)
                header_x0 = x0b
                header_y0 = max(0, y0b - text_h - 2 * label_pad)
                header_x1 = x0b + text_w + 2 * label_pad
                header_y1 = y0b

                cv2.rectangle(frame_with_bbox, (header_x0, header_y0), (header_x1, header_y1), (0, 0, 0), -1)
                cv2.rectangle(frame_with_bbox, (header_x0, header_y0), (header_x1, header_y1), header_color, 2)
                cv2.putText(
                    frame_with_bbox,
                    header_text,
                    (header_x0 + label_pad, header_y1 - label_pad),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    lineType=cv2.LINE_AA
                )

                # --- ADD CLASS PROBABILITIES  TO TOP---
                cv2.putText(
                    frame_with_bbox,
                    f"IoU: {iou:.2f} | Scale: {scale:.2f} (expected: {expected_scale:.2f})",
                    (10, 40),
                    font,
                    0.8,
                    (0, 255, 255),
                    1,
                    lineType=cv2.LINE_AA
                )
                
            else:
                print(f"[Warning]: No mask2d found for frame: {f_idx}")

            writer.append_data(frame_with_bbox)

    writer.close()
    with open(OUT_PATH, "wb") as f:
        pickle.dump(preds, f)

    # ── 9. COMPUTE METRICS ───────────────────────────────────────────────────────
    print("\n---------------- 9. COMPUTE METRICS ----------------")
    metrics = compute_metrics(preds)
    print(
        str(
            f"Total frames: {frame_count['total']} | "
            f"Accepted: {frame_count['accepted']}({frame_count['accepted']/frame_count['total']*100:.2f}%) | "
            f"Rejected: {frame_count['rejected']}({frame_count['rejected']/frame_count['total']*100:.2f}%)"
        )
    )
    print("Metrics for ACCEPTED frames only:")
    print(f"F1: {metrics['f1'] * 100:.3f}%")
    print(f"Precision: {metrics['precision'] * 100:.3f}%")
    print(f"Recall: {metrics['recall'] * 100:.3f}%")
    print(f"Accuracy: {metrics['accuracy'] * 100:.3f}%")

    # ── 10. SAVE VIDEO ───────────────────────────────────────────────────────
    print("\n---------------- 10. SAVE VIDEO ----------------")
    Video(boxed_out, embed=True, width=min(W, 640))

    # ── 11. CREATE AND SAVE CONFUSION MATRIX ────────────────────────────────────
    
    print("\n---------------- 11. CREATE AND SAVE CONFUSION MATRIX ----------------")
    y_true = [v["gt"] for v in preds.values()]
    y_pred = [v["pred"] for v in preds.values()]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Occluded", "Occluded"]
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix for Video #{video_nr}")
    plt.tight_layout()  
    plt.savefig(out_confusion_matrix, dpi=300)  
    plt.close()