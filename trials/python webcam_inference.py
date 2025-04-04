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
    def __init__(self, model_path, frame_size=(224, 224), queue_size=10):
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
        while self.processing:
            if not self.queue.empty():
                frame = self.queue.get()
                self._add_to_buffer(frame)
                
                # Process if we have enough frames
                if len(self.frame_buffer) >= self.queue_size:
                    self._predict()
            else:
                time.sleep(0.01)  # Small sleep to prevent CPU hogging
    
    def _add_to_buffer(self, frame):
        # Convert frame to tensor
        frame_tensor = self.transform(frame)
        
        # Add to buffer, removing oldest frame if needed
        self.frame_buffer.append(frame_tensor)
        if len(self.frame_buffer) > self.queue_size:
            self.frame_buffer.pop(0)
    
    def _predict(self):
        try:
            # Stack frames and create batch
            batch = torch.stack(self.frame_buffer).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(batch)
                # Average predictions over frames
                avg_output = outputs.mean(dim=0, keepdim=True)
                probabilities = torch.nn.functional.softmax(avg_output, dim=1)
                confidence, prediction = torch.max(probabilities, 1)
                
                self.current_prediction = self.class_names[prediction.item()]
                self.confidence = confidence.item()
                
        except Exception as e:
            print(f"Error during prediction: {e}")
    
    def add_frame(self, frame):
        if not self.queue.full():
            self.queue.put(frame)
    
    def get_prediction(self):
        return self.current_prediction, self.confidence
    
    def stop(self):
        self.processing = False
        if self.thread.is_alive():
            self.thread.join()

def main():
    parser = argparse.ArgumentParser(description='Run webcam inference with trained model')
    parser.add_argument('--model', type=str, default='models/first_stage_only_classifier_20250401_200740.pth',
                        help='Path to the trained model')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--frame-size', type=int, default=224,
                        help='Frame size for model input')
    parser.add_argument('--queue-size', type=int, default=10,
                        help='Number of frames to average for prediction')
    
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
    
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Webcam opened successfully")
    print("Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Main loop
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Add frame to processor
            processor.add_frame(frame)
            
            # Get current prediction
            prediction, confidence = processor.get_prediction()
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Display prediction on frame
            color = (0, 255, 0) if prediction == "Normal" else (0, 0, 255)
            cv2.putText(frame, f"{prediction}: {confidence:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('First Stage Classifier', frame)
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Release resources
        processor.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released")

if __name__ == "__main__":
    main()