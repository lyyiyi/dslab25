import os
import random
import cv2
import imageio
from IPython.display import Video
import numpy as np
import pickle
from safetensors.torch import load_file as load_safetensors
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor  import SAM2ImagePredictor
from sam2.sam2_video_predictor  import SAM2VideoPredictor
import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm

from utils import compute_metrics, crop_mask, DINOv2Classifier, get_aligned_iou, \
    get_feat, load_frame_rgb, load_labels, read_video_rgb, set_seed

if __name__ == "__main__":
    # Set a fixed seed for deterministic behavior
    set_seed(42)

    # 1. CONFIGURATION
    print("\n---------------- 1. CONFIGURATION ----------------")
    fps_video = 5
    skip_frames = 31  # ← skip these many frames (70 from Georg with 15fps)
    threshold = skip_frames
    iou_threshold = 0.6 
    expected_scale = 0.65 # obtained from reference-5-render vs. clean frame 70
    scale_tolerance = 0.1

    print(f"Config for video: 01_run1_cam_2_1024x1024_{fps_video}fps.mp4")
    DATA_DIR = '/work/courses/dslab/team14/'
    repo_dir = os.getcwd().split('dslab25')[0] + '/dslab25/'
    video_path = os.path.join(DATA_DIR, f'videos/01_run1_cam_2_1024x1024_{fps_video}fps.mp4')
    labels_file = os.path.join(DATA_DIR, "videos/01_run1_labels_5fps.txt")
    # temp_images_dir = os.path.join(DATA_DIR, "videos/temp_images/")
    model_dir = os.path.join(DATA_DIR, 'ckpt/dino/')
    frames_dir = os.path.join(DATA_DIR, 'videos/frames')
    boxed_out = os.path.join(DATA_DIR, f"videos/sam_boxed_{fps_video}fps_{iou_threshold}iouthresh_{scale_tolerance}scaletol_with_headers.mp4")
    temp_results_dir = os.path.join(repo_dir, 'obj_detection/dino/temp_results') # TODO: delete this
    mask_ref_path = os.path.join(repo_dir, "obj_detection/dino/", "ref_mask.pkl")
    safetensors_path = os.path.join(model_dir, "dino_fine_tuned.safetensors")
    bin_path = os.path.join(model_dir, "training_args.bin") 
    OUT_PATH = os.path.join(DATA_DIR, "videos/predictions_sam_5fps.pkl")


    # os.makedirs(temp_images_dir, exist_ok=True)

    # 2.  PATHS & OPTIONS  ---------------------------------------------------------
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
    

    # ── 3. EXTRACT FRAMES FROM THE VIDEO ────────────────────────────────────────────────────────────
    print("\n---------------- 3. EXTRACT FRAMES FROM THE VIDEO ----------------")
    frame_paths, FPS = read_video_rgb(video_path, output_dir=frames_dir, use_cached=False)
    assert len(frame_paths) > skip_frames, f"Video must have more than {skip_frames} frames!"
    H, W = load_frame_rgb(0, base_path=frames_dir).shape[:2]

    # ── 4. INITIALISE MODELS ────────────────────────────────────────────────────
    print("\n---------------- 4. INITIALISE MODELS ----------------")
    backbone_name = "facebook/dinov2-with-registers-small"
    dinov2_backbone = AutoModel.from_pretrained(backbone_name).to(device).eval()
    dinov2_proc = AutoImageProcessor.from_pretrained(backbone_name)
    img_pred = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-tiny")
    vid_pred = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-tiny")

    # DINO
    frame_to_class = load_labels(labels_file)
    num_labels = max(frame_to_class.values()) + 1 if frame_to_class else 8
    pretrained_model = "facebook/dinov2-with-registers-base"
    processor = AutoImageProcessor.from_pretrained(pretrained_model)
    model = DINOv2Classifier(num_labels=num_labels, pretrained_model=pretrained_model)
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
    print(f"Successfully transferred model to device: {device}.")


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

            # # TODO: Delete this
            # import pickle
            # with open(os.path.join(temp_results_dir, f'masks/video/frame_{f_idx}.pkl'), 'wb') as f:
            #     pickle.dump(mask2d.cpu().numpy(), f)
            
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
                scale_reject = bool(scale is None or abs(float(scale) - expected_scale) > scale_tolerance)
                rejected = iou_reject or scale_reject

                if rejected:
                    frame_count['rejected'] = frame_count['rejected'] + 1
                else:
                    frame_count['accepted'] = frame_count['accepted'] + 1

                # Save to file
                # output_path = os.path.join(temp_images_dir, f"frame_{f_idx}.jpg")
                # cv2.imwrite(output_path, crop_bgr)
                color = (0, 255, 0) if not rejected else (255, 0, 0)
                cv2.rectangle(frame_with_bbox, (x0b, y0b), (x1b, y1b), color, 2)

                if not rejected:
                    # ----- CLASSIFICATION ------------------------------------------------
                    crop_rgb = frame[y0b:y1b, x0b:x1b]  # use pristine frame
                    # crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR).copy()  # convert to BGR

                    batch = processor(images=[crop_rgb], return_tensors="pt").to(device)
                    cls_logits = model(**batch).logits.squeeze(0)
                    probs = torch.softmax(cls_logits, dim=-1)
                    predicted_label = probs.argmax().item()

                    true_label = frame_to_class[f_idx]

                    preds[str(f_idx)] = {
                        "logits": probs.cpu().tolist(),
                        "pred": predicted_label,
                        "gt": true_label,
                        "iou": iou,
                        "trans": trans
                    }

                    # ------ ADD HEADERS FOR VISUALIZATION ----------
                    header_text = f"Pred: {predicted_label}, GT: {true_label}"
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
                    probs_str = ", ".join([f"{i} - {p:.2f}" for i, p in enumerate(probs)])
                    cv2.putText(
                        frame_with_bbox,
                        f"Probabilities of every stage: {probs_str}",
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

