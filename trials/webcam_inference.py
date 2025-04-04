import os
import cv2
import torch
import numpy as np
import time
from torchvision import transforms
import torch.nn as nn
import threading
from queue import Queue
import argparse

# Try to import the necessary modules 
try:
    from app.models.two_stage_model_improved import TwoStageClassifier
except ImportError:
    print("WARNING: Could not import TwoStageClassifier from app.models.two_stage_model_improved")
    print("Implementing a basic classifier based on the available information.")
    
    # Define a simple classifier with similar structure to what was likely used
    class TwoStageClassifier(nn.Module):
        def __init__(self, dropout_rate=0.6):
            super(TwoStageClassifier, self).__init__()
            # Base model (likely ResNet or similar)
            self.base_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
            
            # Replace the final fully connected layer
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
            
            # First stage classifier (Normal vs Abnormal)
            self.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 2)  # 2 classes: Normal and Abnormal
            )

        def forward(self, x):
            x = self.base_model(x)
            x = self.classifier(x)
            return x

class FrameProcessor:
    def __init__(self, model_path, frame_size=(160, 160), queue_size=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Set up frame preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Frame dimensions
        self.frame_size = frame_size
        
        # Class names
        self.class_names = ['Normal', 'Abnormal']
        
        # Buffer for frame accumulation
        self.frame_buffer = []
        self.queue_size = queue_size
        
        # Queue for passing frames to processing thread
        self.queue = Queue(maxsize=30)
        self.processing = True
        self.current_prediction = "Waiting..."
        self.confidence = 0.0
        
        # Prediction hysteresis to avoid flickering
        self.hysteresis_counter = 0
        self.hysteresis_threshold = 3  # Number of consistent predictions to change displayed result
        self.last_prediction = None
        self.raw_probs = [0.5, 0.5]
        
        # Start processing thread
        self.thread = threading.Thread(target=self._process_frames)
        self.thread.daemon = True
        self.thread.start()
    
    def _load_model(self, model_path):
        print(f"Loading model from {model_path}")
        
        # Create model instance
        model = TwoStageClassifier()
        
        # Load the saved weights
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure the model path is correct and the model structure matches")
            
        return model
    
    def _process_frames(self):
        """Thread that processes frames from the queue."""
        while self.processing:
            if not self.queue.empty():
                frame = self.queue.get()
                
                try:
                    # Convert frame to tensor
                    frame_tensor = self.transform(frame)
                    
                    # Process single frame for lower latency
                    with torch.no_grad():
                        # Add batch dimension
                        input_tensor = frame_tensor.unsqueeze(0).to(self.device)
                        
                        # Get prediction
                        outputs = self.model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, prediction = torch.max(probabilities, 1)
                        
                        # Store raw probabilities
                        self.raw_probs = [probabilities[0][0].item(), probabilities[0][1].item()]
                        
                        # Apply hysteresis to reduce flickering
                        pred_class = self.class_names[prediction.item()]
                        
                        if self.last_prediction is None:
                            self.last_prediction = pred_class
                            self.hysteresis_counter = 0
                        elif pred_class == self.last_prediction:
                            self.hysteresis_counter += 1
                        else:
                            self.hysteresis_counter -= 1
                            
                        # Change prediction only if consistent for several frames
                        if self.hysteresis_counter >= self.hysteresis_threshold:
                            self.current_prediction = self.last_prediction
                            self.confidence = confidence.item()
                            self.hysteresis_counter = self.hysteresis_threshold  # Cap the counter
                        elif self.hysteresis_counter <= -self.hysteresis_threshold:
                            self.last_prediction = pred_class
                            self.current_prediction = pred_class
                            self.confidence = confidence.item()
                            self.hysteresis_counter = 0
                            
                except Exception as e:
                    print(f"Error during prediction: {e}")
            else:
                # Shorter sleep time to reduce latency
                time.sleep(0.005)
    
    def add_frame(self, frame):
        """Add a frame to the processing queue, replacing old frames if full."""
        if self.queue.full():
            try:
                # Remove oldest frame
                self.queue.get_nowait()
            except:
                pass
        
        # Add new frame
        self.queue.put(frame)
    
    def get_prediction(self):
        """Get the current prediction with confidence."""
        return self.current_prediction, self.confidence
    
    def get_raw_probabilities(self):
        """Get raw probability values for normal and abnormal classes."""
        return self.raw_probs
    
    def stop(self):
        """Stop the processing thread."""
        self.processing = False
        if self.thread.is_alive():
            self.thread.join()


def main():
    parser = argparse.ArgumentParser(description='Run webcam inference with trained model')
    parser.add_argument('--model', type=str, default='models/first_stage_only_classifier_20250401_200740.pth',
                        help='Path to the trained model')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--frame-size', type=int, default=160,
                        help='Frame size for model input (smaller = faster)')
    parser.add_argument('--queue-size', type=int, default=5,
                        help='Number of frames to queue for processing')
    parser.add_argument('--threshold', type=float, default=0.65,
                        help='Confidence threshold for abnormal classification')
    parser.add_argument('--display-width', type=int, default=640,
                        help='Display width for webcam feed')
    parser.add_argument('--display-height', type=int, default=480,
                        help='Display height for webcam feed')
    parser.add_argument('--skip-frames', type=int, default=1,
                        help='Number of frames to skip for processing (higher = faster)')
    
    args = parser.parse_args()
    
    # Initialize frame processor
    processor = FrameProcessor(
        model_path=args.model,
        frame_size=(args.frame_size, args.frame_size),
        queue_size=args.queue_size
    )
    
    # Open webcam
    print(f"Opening webcam (device {args.camera})...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set webcam resolution for display
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.display_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.display_height)
    
    # Try to increase webcam FPS if possible
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Webcam opened successfully")
    print(f"Processing every {args.skip_frames} frame(s)")
    print("Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    fps = 0
    skip_counter = 0
    
    # For FPS calculation
    processing_times = []
    
    # Main loop
    try:
        while True:
            loop_start = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Skip frames to improve performance
            skip_counter += 1
            if skip_counter >= args.skip_frames:
                skip_counter = 0
                
                # Add frame to processor
                processor.add_frame(frame)
            
            # Get current prediction
            prediction, confidence = processor.get_prediction()
            
            # Apply threshold
            threshold = args.threshold
            final_prediction = "Normal"
            if prediction == "Abnormal" and confidence > threshold:
                final_prediction = "Abnormal"
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 0.5:  # Update FPS more frequently
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Get raw probabilities
            normal_prob, abnormal_prob = processor.get_raw_probabilities()
            
            # Create a copy for display to avoid modifying the original processing frame
            display_frame = frame.copy()
            
            # Display prediction on frame
            color = (0, 255, 0) if final_prediction == "Normal" else (0, 0, 255)
            cv2.putText(display_frame, f"{final_prediction}: {confidence:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Display FPS
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Display raw probabilities
            cv2.putText(display_frame, f"Normal prob: {normal_prob:.2f}", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Abnormal prob: {abnormal_prob:.2f}", (10, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('First Stage Classifier', display_frame)
            
            # Calculate processing time for this frame
            loop_time = time.time() - loop_start
            processing_times.append(loop_time)
            if len(processing_times) > 100:
                processing_times.pop(0)
            
            # Check for exit key (1ms wait for key press)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Release resources
        processor.stop()
        cap.release()
        cv2.destroyAllWindows()
        
        # Print performance stats
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            print(f"Average processing time per frame: {avg_time*1000:.2f} ms")
            print(f"Theoretical max FPS: {1/avg_time:.1f}")
            
        print("Resources released")

if __name__ == "__main__":
    main()