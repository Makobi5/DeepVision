import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

# Model Path (from your image)
MODEL_PATH = "F:/main/work/projects/DeepVision2/models/first_stage_best.pth"

# Classes (REPLACE THESE WITH YOUR ACTUAL CLASS NAMES!)
CLASS_NAMES = ['class1', 'class2', 'class3']  # Example: ['person', 'car', 'background']

# Model Input Size (REPLACE THIS WITH THE ACTUAL SIZE!)
INPUT_SIZE = (224, 224)  # (height, width) - common sizes are 224x224, 256x256

# Load the PyTorch model
try:
    model = torch.load(MODEL_PATH)
    model.eval()  # Set to evaluation mode
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# Define image transformations (must match training data)
transform = transforms.Compose([
    transforms.ToPILImage(), # Convert NumPy array to PIL Image
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example: ImageNet normalization
])

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break

    # Preprocess the frame
    try:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        continue


    # Make a prediction (disable gradient calculation for inference)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0) # softmax to get probabilities if the model outputs raw scores (logits).
        predicted_class_index = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class_index].item()
        predicted_class_name = CLASS_NAMES[predicted_class_index]
    # Display the results on the frame
    label = f"{predicted_class_name}: {confidence:.2f}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Webcam Feed", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()