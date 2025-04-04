import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

def extract_frames(video_path, output_dir, frame_count=10, frame_size=(224, 224)):
    """
    Extract frames from a video clip for model training
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save extracted frames
        frame_count (int): Number of frames to extract
        frame_size (tuple): Size to resize frames to (width, height)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"Error: Could not determine frame count for {video_path}")
        cap.release()
        return False
    
    # Calculate frame indices to extract (evenly distributed)
    if total_frames <= frame_count:
        # Use all frames if there are fewer than requested
        frame_indices = list(range(total_frames))
    else:
        # Select evenly distributed frames
        frame_indices = np.linspace(0, total_frames - 1, frame_count, dtype=int)
    
    # Extract frames
    frames_saved = 0
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    
    for i, frame_idx in enumerate(frame_indices):
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Failed to read frame {frame_idx} from {video_path}")
            continue
        
        # Resize frame
        frame = cv2.resize(frame, frame_size)
        
        # Save frame
        frame_path = os.path.join(output_dir, f"{video_filename}_frame{i:03d}.jpg")
        cv2.imwrite(frame_path, frame)
        frames_saved += 1
    
    cap.release()
    return frames_saved > 0

def process_dataset(clips_dir, output_dir, class_name, frame_count=10, frame_size=(224, 224)):
    """
    Process video clips in the dataset and extract frames for training
    
    Args:
        clips_dir (str): Directory containing video clips
        output_dir (str): Directory to save processed dataset
        class_name (str): Class name for the clips (e.g., "Normal")
        frame_count (int): Number of frames to extract per clip
        frame_size (tuple): Size to resize frames to (width, height)
    """
    # Create output directory structure
    processed_dir = os.path.join(output_dir, class_name)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Find all video clips
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(list(Path(clips_dir).glob(f"*{ext}")))
    
    if not video_files:
        print(f"No video files found in {clips_dir}")
        return
    
    print(f"Found {len(video_files)} video clips in {clips_dir}")
    print(f"Extracting {frame_count} frames per clip...")
    
    # Process each video
    successful = 0
    for video_path in tqdm(video_files):
        if extract_frames(str(video_path), processed_dir, frame_count, frame_size):
            successful += 1
    
    print(f"Successfully processed {successful}/{len(video_files)} clips")
    print(f"Processed frames saved to {processed_dir}")

def main():
    parser = argparse.ArgumentParser(description='Process recorded video clips for model training')
    parser.add_argument('--input', type=str, required=True,
                        help='Directory containing recorded clips')
    parser.add_argument('--output', type=str, default='processed_dataset',
                        help='Output directory for processed frames')
    parser.add_argument('--class', type=str, dest='class_name', default='Normal',
                        help='Class name for the clips (default: Normal)')
    parser.add_argument('--frames', type=int, default=10,
                        help='Number of frames to extract per clip (default: 10)')
    parser.add_argument('--size', type=int, default=224,
                        help='Size to resize frames to (default: 224)')
    
    args = parser.parse_args()
    
    # Process dataset
    process_dataset(
        clips_dir=args.input,
        output_dir=args.output,
        class_name=args.class_name,
        frame_count=args.frames,
        frame_size=(args.size, args.size)
    )

if __name__ == "__main__":
    main()