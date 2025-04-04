import os
import cv2
import time
import argparse
import numpy as np
from datetime import datetime
import threading
from queue import Queue

class DatasetRecorder:
    def __init__(self, output_dir, scene_class="Normal", frame_size=(640, 480), 
                 fps=30, clip_duration=5, camera_id=0, buffer_size=1000):
        """
        Initialize the dataset recorder
        
        Args:
            output_dir (str): Directory to save dataset clips
            scene_class (str): Class label for the recorded scenes
            frame_size (tuple): Output frame size (width, height)
            fps (int): Frames per second to capture
            clip_duration (int): Duration of each clip in seconds
            camera_id (int): Camera device ID
            buffer_size (int): Maximum size of frame buffer
        """
        self.output_dir = output_dir
        self.scene_class = scene_class
        self.frame_size = frame_size
        self.fps = fps
        self.clip_duration = clip_duration
        self.camera_id = camera_id
        self.buffer_size = buffer_size
        
        # Create output directories
        self.clips_dir = os.path.join(output_dir, f"{scene_class}_clips")
        self.frames_dir = os.path.join(output_dir, f"{scene_class}_frames")
        
        os.makedirs(self.clips_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # Initialize camera
        self.cap = None
        
        # Recording variables
        self.recording = False
        self.frame_buffer = Queue(maxsize=buffer_size)
        self.current_clip_path = None
        self.total_clips_recorded = 0
        self.record_start_time = 0
        
        # Performance tracking
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Threading control
        self.running = True
        self.recording_thread = None
    
    def start(self):
        """Start the recording application"""
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)  # Try to set camera FPS
        
        # Get actual camera properties
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera opened successfully (ID: {self.camera_id})")
        print(f"Camera resolution: {actual_width}x{actual_height} (requested: {self.frame_size[0]}x{self.frame_size[1]})")
        print(f"Camera FPS: {actual_fps} (requested: {self.fps})")
        print(f"Saving {self.scene_class} clips to: {self.clips_dir}")
        print(f"Saving {self.scene_class} frames to: {self.frames_dir}")
        print("\nControls:")
        print("  r - Start/stop recording a clip")
        print("  s - Save a single frame")
        print("  q - Quit")
        
        # Main loop
        try:
            while self.running:
                # Capture frame (this is the bottleneck operation)
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # Calculate real FPS
                self.frame_count += 1
                elapsed_time = time.time() - self.fps_start_time
                if elapsed_time > 1.0:
                    self.current_fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.fps_start_time = time.time()
                
                # Create a copy of the display frame so we don't modify the one we might save
                display_frame = frame.copy()
                
                # Display recording status
                if self.recording:
                    elapsed_time = time.time() - self.record_start_time
                    remaining_time = max(0, self.clip_duration - elapsed_time)
                    status_text = f"RECORDING - {remaining_time:.1f}s remaining"
                    cv2.putText(display_frame, status_text, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Add frame to buffer if recording
                    if not self.frame_buffer.full():
                        self.frame_buffer.put(frame.copy())
                    
                    # Check if recording duration is complete
                    if elapsed_time >= self.clip_duration and self.recording_thread is None:
                        # Start a thread to save the clip
                        self.recording_thread = threading.Thread(target=self._save_clip_thread)
                        self.recording_thread.daemon = True
                        self.recording_thread.start()
                else:
                    cv2.putText(display_frame, f"Press 'r' to record {self.clip_duration}s clip", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show total clips recorded and current FPS
                cv2.putText(display_frame, f"Clips: {self.total_clips_recorded}", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"FPS: {self.current_fps:.1f}", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display preview
                cv2.imshow(f"{self.scene_class} Dataset Recorder", display_frame)
                
                # Handle key presses (1ms wait to keep UI responsive)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    # Quit
                    self.running = False
                    break
                elif key == ord('r'):
                    # Toggle recording
                    if not self.recording:
                        self._start_recording()
                    elif self.recording_thread is None:  # Only stop if not already saving
                        self.recording_thread = threading.Thread(target=self._save_clip_thread)
                        self.recording_thread.daemon = True
                        self.recording_thread.start()
                elif key == ord('s'):
                    # Save a single frame
                    threading.Thread(target=self._save_frame, args=(frame.copy(),)).start()
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            # Release resources
            self.running = False
            if self.recording_thread is not None and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=3.0)  # Wait for saving to complete with timeout
                
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            print(f"\nRecorded {self.total_clips_recorded} clips")
    
    def _start_recording(self):
        """Start recording a new clip"""
        # Clear any existing frames in buffer
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except:
                pass
        
        self.recording = True
        self.record_start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_clip_path = os.path.join(self.clips_dir, f"{self.scene_class}_{timestamp}.mp4")
        print(f"Recording started...")
    
    def _save_clip_thread(self):
        """Thread function to save the current clip"""
        try:
            # Set recording flag to false so we stop adding frames
            self.recording = False
            
            # Get all frames from the buffer
            frames = []
            while not self.frame_buffer.empty():
                try:
                    frames.append(self.frame_buffer.get_nowait())
                except:
                    break
            
            if not frames:
                print("No frames to save!")
                return
            
            print(f"Saving clip with {len(frames)} frames to {self.current_clip_path}")
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                self.current_clip_path, 
                fourcc, 
                self.fps, 
                (frames[0].shape[1], frames[0].shape[0])
            )
            
            # Write frames to video
            for frame in frames:
                out.write(frame)
            
            out.release()
            self.total_clips_recorded += 1
            print(f"Clip saved: {self.current_clip_path}")
            
            # Save a thumbnail from the middle of the clip
            if frames:
                middle_idx = len(frames) // 2
                middle_frame = frames[middle_idx]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                frame_path = os.path.join(self.frames_dir, f"{self.scene_class}_thumb_{timestamp}.jpg")
                cv2.imwrite(frame_path, middle_frame)
                print(f"Thumbnail saved: {frame_path}")
        except Exception as e:
            print(f"Error saving clip: {e}")
        finally:
            self.recording_thread = None  # Clear thread reference
    
    def _save_frame(self, frame):
        """Save a single frame"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_path = os.path.join(self.frames_dir, f"{self.scene_class}_{timestamp}.jpg")
        cv2.imwrite(frame_path, frame)
        print(f"Frame saved: {frame_path}")

def main():
    parser = argparse.ArgumentParser(description='Record and save normal scenes for dataset')
    parser.add_argument('--output', type=str, default='dataset',
                        help='Output directory for saved clips and frames')
    parser.add_argument('--class', type=str, dest='scene_class', default='Normal',
                        help='Class label for the recorded scenes (default: Normal)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--width', type=int, default=640,
                        help='Frame width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Frame height (default: 480)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for saved clips (default: 30)')
    parser.add_argument('--duration', type=int, default=5,
                        help='Duration of each clip in seconds (default: 5)')
    parser.add_argument('--buffer-size', type=int, default=1000,
                        help='Maximum number of frames to buffer (default: 1000)')
    
    args = parser.parse_args()
    
    # Create recorder
    recorder = DatasetRecorder(
        output_dir=args.output,
        scene_class=args.scene_class,
        frame_size=(args.width, args.height),
        fps=args.fps,
        clip_duration=args.duration,
        camera_id=args.camera,
        buffer_size=args.buffer_size
    )
    
    # Start recording
    recorder.start()

if __name__ == "__main__":
    main()