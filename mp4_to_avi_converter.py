import os
import argparse
import subprocess
import glob
from tqdm import tqdm
import shutil

def convert_mp4_to_avi(input_file, output_file, codec="mjpeg", crf=18):
    """
    Convert MP4 file to AVI using FFmpeg
    
    Args:
        input_file (str): Path to input MP4 file
        output_file (str): Path to output AVI file
        codec (str): Video codec to use (default: mjpeg)
        crf (int): Constant Rate Factor for quality (lower = better, 18 is visually lossless)
    
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        # Construct FFmpeg command
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-c:v', codec,
            '-q:v', str(crf),
            '-c:a', 'pcm_s16le',  # Uncompressed audio
            '-y',  # Overwrite output files without asking
            output_file
        ]
        
        # Run FFmpeg (hide output unless there's an error)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            print(f"Error converting {input_file}:")
            print(result.stderr.decode())
            return False
        
        return True
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False

def process_directory(input_dir, output_dir, class_name='Normal', split='Train', dry_run=False):
    """
    Process MP4 files in a directory and convert them to AVI in the output directory
    
    Args:
        input_dir (str): Directory containing MP4 files
        output_dir (str): Output directory (should be the existing Train/Normal folder)
        class_name (str): Class label for these clips (Normal, Violence, etc.)
        split (str): Train or Test
        dry_run (bool): If True, only print what would be done, without actual conversion
    
    Returns:
        int: Number of files successfully converted
    """
    # Find all MP4 files
    mp4_files = glob.glob(os.path.join(input_dir, '*.mp4'))
    
    if not mp4_files:
        print(f"No MP4 files found in {input_dir}")
        return 0
    
    print(f"Found {len(mp4_files)} MP4 files in {input_dir}")
    
    # Create destination directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Count existing AVI files for reference
    existing_avi_files = glob.glob(os.path.join(output_dir, '*.avi'))
    print(f"Found {len(existing_avi_files)} existing AVI files in {output_dir}")
    
    # Convert MP4 files to AVI
    success_count = 0
    
    for mp4_file in tqdm(mp4_files, desc=f"Converting {split}/{class_name}"):
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(mp4_file))[0]
        
        # Output AVI path
        avi_file = os.path.join(output_dir, f"{base_name}.avi")
        
        # Skip if output already exists
        if os.path.exists(avi_file):
            print(f"Skipping {mp4_file} (output already exists)")
            continue
        
        if dry_run:
            print(f"Would convert: {mp4_file} -> {avi_file}")
            success_count += 1
        else:
            if convert_mp4_to_avi(mp4_file, avi_file):
                success_count += 1
                print(f"Converted: {mp4_file} -> {avi_file}")
    
    print(f"Successfully processed {success_count}/{len(mp4_files)} files for {split}/{class_name}")
    return success_count

def check_ffmpeg():
    """Check if FFmpeg is installed and available"""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert MP4 video clips to AVI format')
    parser.add_argument('--input', type=str, required=True,
                        help='Directory containing MP4 files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory (existing Train/Normal folder)')
    parser.add_argument('--class', type=str, dest='class_name', default='Normal',
                        help='Class label for the clips (default: Normal)')
    parser.add_argument('--split', type=str, choices=['Train', 'Test'], default='Train',
                        help='Dataset split (Train/Test, default: Train)')
    parser.add_argument('--codec', type=str, default='mjpeg',
                        help='Video codec to use for AVI (default: mjpeg)')
    parser.add_argument('--quality', type=int, default=18,
                        help='Quality setting (CRF, lower is better, default: 18)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only print what would be done without actual conversion')
    
    args = parser.parse_args()
    
    # Check if FFmpeg is installed
    if not check_ffmpeg():
        print("ERROR: FFmpeg is not installed or not in PATH.")
        print("Please install FFmpeg and try again.")
        print("Installation instructions: https://ffmpeg.org/download.html")
        return
    
    # Process directory
    process_directory(
        input_dir=args.input,
        output_dir=args.output,
        class_name=args.class_name,
        split=args.split,
        dry_run=args.dry_run
    )
    
    if not args.dry_run:
        print("\nConversion completed.")
        print(f"AVI files have been placed in: {args.output}")
        print("You can now run your training script.")
    else:
        print("\nDry run completed. No files were actually converted.")

if __name__ == "__main__":
    main()