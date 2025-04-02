import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import warnings
import collections
import time

class FirstStageClassifier:
    def __init__(self, model_path='models/first_stage_best.pth', smoothing_window=10):
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model
        self.model = self._create_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Define image transformation
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
        ])
        
        # Class labels
        self.class_names = ['Normal', 'Not Normal']
        
        # Smoothing buffer
        self.prediction_buffer = collections.deque(maxlen=smoothing_window)
        self.confidence_buffer = collections.deque(maxlen=smoothing_window)
        
        # Debugging information
        self.debug_info = {
            'raw_predictions': [],
            'confidence_scores': []
        }
    
    def _create_model(self):
        """Create the model architecture"""
        model = models.efficientnet_b2(pretrained=False)
        num_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.6, inplace=True),
            torch.nn.Linear(num_features, 2)  # 2 classes
        )
        return model.to(self.device)
    
    def predict(self, frame):
        """Predict class of a single frame with advanced analysis"""
        # Preprocess the frame
        input_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            max_prob, predicted = torch.max(probabilities, 1)
        
        # Get class and probability
        class_idx = predicted.item()
        probability = max_prob.item()
        
        # Store raw prediction data for debugging
        self.debug_info['raw_predictions'].append(class_idx)
        self.debug_info['confidence_scores'].append(probability)
        
        # Add to buffer
        self.prediction_buffer.append(class_idx)
        self.confidence_buffer.append(probability)
        
        # Advanced stability analysis
        prediction_counts = {0: 0, 1: 0}
        for pred in self.prediction_buffer:
            prediction_counts[pred] += 1
        
        # Determine stable prediction
        stable_prediction = max(prediction_counts, key=prediction_counts.get)
        stability_ratio = prediction_counts[stable_prediction] / len(self.prediction_buffer)
        
        # Calculate average confidence
        avg_confidence = sum(self.confidence_buffer) / len(self.confidence_buffer)
        
        # Additional analysis
        if stability_ratio < 0.7:  # If less than 70% stable
            print(f"Warning: Unstable prediction. Stability ratio: {stability_ratio:.2f}")
        
        return self.class_names[stable_prediction], avg_confidence * 100, stability_ratio

def main():
    # Initialize the classifier
    classifier = FirstStageClassifier(smoothing_window=15)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    start_time = time.time()
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Flip the frame (optional, depends on your webcam)
        frame = cv2.flip(frame, 1)
        
        # Make a copy for drawing
        display_frame = frame.copy()
        
        try:
            # Predict with advanced analysis
            prediction, confidence, stability = classifier.predict(frame)
            
            # Prepare display text
            text = f"{prediction}: {confidence:.2f}% (Stability: {stability:.2f})"
            color = (0, 255, 0) if prediction == 'Normal' else (0, 0, 255)
            
            # Draw prediction on frame
            cv2.putText(display_frame, text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except Exception as e:
            print(f"Prediction error: {e}")
        
        # Display the resulting frame
        cv2.imshow('First Stage Classification', display_frame)
        
        # Break loop on 'q' key press or after 2 minutes
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (time.time() - start_time > 120):
            # Print debug information
            print("\nDebug Information:")
            print("Raw Predictions:", classifier.debug_info['raw_predictions'])
            print("Confidence Scores:", classifier.debug_info['confidence_scores'])
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()