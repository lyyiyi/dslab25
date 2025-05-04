import os
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sam2.sam2_video_predictor import SAM2VideoPredictor
from obj_detection.dino.utils import optimize_sam2_video_predictor

def run_optimized_sam2(
    video_path, 
    output_dir=None, 
    model_name="facebook/sam2.1-hiera-tiny", 
    window_size=16, 
    save_visualizations=True
):
    """
    Run SAM2 on a video with optimized memory usage.
    
    Args:
        video_path: Path to the input video
        output_dir: Directory to save output masks (if None, will create a directory based on video name)
        model_name: SAM2 model name to use
        window_size: Number of frames to keep in memory (older frames will be deleted)
        save_visualizations: Whether to save visualization images
        
    Returns:
        List of segmentation masks
    """
    # Create output directory if needed
    if output_dir is None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = f"output_sam2_{video_name}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize video predictor with standard options
    print(f"Loading SAM2 model: {model_name}")
    # Start with these options enabled
    vid_pred = SAM2VideoPredictor.from_pretrained(
        model_name,
        offload_video_to_cpu=True  # Initial memory saving option
    )
    
    # Apply our optimization utilities
    print("Applying memory optimization techniques...")
    vid_pred = optimize_sam2_video_predictor(vid_pred, window_size=window_size)
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    
    # Process video with our memory-optimized predictor
    print("Running segmentation...")
    
    # Set the video
    vid_pred.set_video(video_path=video_path)
    
    # Define points to track
    # In the middle of the video frame
    # You'll want to modify these coordinates for your specific use case
    # Format is (frame_idx, points_coords, point_labels)
    # The SAM2VideoPredictor expects time, xy, and labels as separate arrays
    frame_idx = 0  # Start tracking from the first frame
    # Example coordinate in the middle of the frame
    # You'll want to adjust these based on your video and what you want to track
    point_coords = np.array([[320, 240]])  # Example coordinates - adjust for your video
    point_labels = np.array([1])  # 1 for foreground
    
    # Run object tracking with memory-efficient processing
    masks, scores, logits = vid_pred.predict(
        time_points=np.array([frame_idx]),
        point_coords=point_coords,
        point_labels=point_labels,
    )
    
    print(f"Segmentation complete. Results shape: {masks.shape}")
    
    # Save visualizations if requested
    if save_visualizations:
        print(f"Saving visualizations to {output_dir}...")
        # Read video and save masked frames
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        
        # Create a video writer for masks visualization
        output_video_path = os.path.join(output_dir, "masks_visualization.mp4")
        
        # Check if we have any frames to process
        ret, frame = cap.read()
        if ret:
            # Get frame dimensions
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            # Reset cap to start
            cap.release()
            cap = cv2.VideoCapture(video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Apply mask if available for this frame
            if frame_idx < len(masks):
                mask = masks[frame_idx][0]  # Get the first mask for this frame
                
                # Convert mask to color overlay
                colored_mask = np.zeros_like(frame)
                colored_mask[:, :, 1] = mask * 255  # Green channel
                
                # Blend original frame with mask
                alpha = 0.5
                blended = cv2.addWeighted(frame, 1, colored_mask, alpha, 0)
                
                # Write frame to video
                out.write(blended)
                
                # Save individual frames at intervals
                if frame_idx % 30 == 0:  # Save every 30th frame
                    img_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
                    cv2.imwrite(img_path, blended)
            else:
                # Just write the original frame if no mask is available
                out.write(frame)
                
            frame_idx += 1
            
            # Print progress
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")
        
        # Release resources
        cap.release()
        out.release()
        
    # Clear memory
    del vid_pred
    torch.cuda.empty_cache()
    
    print("Processing complete!")
    return masks

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SAM2 on a video with optimized memory usage")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", default=None, help="Output directory (default: based on video name)")
    parser.add_argument("--model", default="facebook/sam2.1-hiera-tiny", help="SAM2 model name")
    parser.add_argument("--window", type=int, default=16, help="Frame window size to keep in memory")
    
    args = parser.parse_args()
    
    run_optimized_sam2(
        video_path=args.video,
        output_dir=args.output,
        model_name=args.model,
        window_size=args.window
    ) 