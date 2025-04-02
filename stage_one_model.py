# train_two_stage_model.py (Modified for first-stage-only training)
import os
import sys
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import traceback
import random
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler

# Fix for Windows multiprocessing issues
import platform
if platform.system() == 'Windows':
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

# Set random seeds for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

try:
    from app.utils.dataset_processing import prepare_classification_dataset
    from app.models.two_stage_model_improved import TwoStageModel
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Make sure the following files exist:")
    print("  - app/utils/dataset_processing.py")
    print("  - two_stage_model_improved.py")
    sys.exit(1)

def main():
    """Main function for training the first stage only"""
    try:
        seed_everything(42)
        
        print("Starting FIRST STAGE ONLY training...")
        print(f"Current working directory: {os.getcwd()}")

        # Set up paths
        source_dir = 'data/SCVD/SCVD_converted'
        target_dir = 'data/processed_classification'
        models_dir = 'models'
        
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs('figures', exist_ok=True)
        os.makedirs('evaluations', exist_ok=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if device.type == 'cuda':
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Training parameters (first stage only)
        frame_count = 10
        use_weighted_sampler = True
        epochs_first_stage = 35
        learning_rate_first = 0.0004
        patience_first = 12
        dropout_rate = 0.6
        weight_decay = 4e-4
        label_smoothing = 0.05
        
        print("\nFirst Stage Parameters:")
        print(f"Frame count: {frame_count}")
        print(f"Epochs: {epochs_first_stage}")
        print(f"Learning rate: {learning_rate_first}")
        print(f"Patience: {patience_first}")
        print(f"Dropout rate: {dropout_rate}")
        print(f"Weight decay: {weight_decay}")
        print(f"Label smoothing: {label_smoothing}\n")

        # Prepare dataset
        print("Preparing dataset...")
        video_data = prepare_classification_dataset(source_dir, target_dir)
        classes = ['Normal', 'Violence', 'Weaponized']
        print(f"Dataset prepared with classes: {classes}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"first_stage_only_classifier_{timestamp}"

        # Create and train the model (first stage only)
        print("Initializing model for first stage training...")
        model = TwoStageModel(
            video_data=video_data, 
            dropout_rate=dropout_rate,
            frame_count=frame_count,
            use_weighted_sampler=use_weighted_sampler
        )
        
        # Train only first stage using the existing method
        print("Training FIRST STAGE ONLY...")
        history = model.train_first_stage(
            epochs=epochs_first_stage,
            learning_rate=learning_rate_first,
            weight_decay=weight_decay,
            patience=patience_first,
            label_smoothing=label_smoothing
        )

        # Save first stage model
        model_path = os.path.join(models_dir, f"{model_name}.pth")
        torch.save(model.first_stage_model.state_dict(), model_path)
        print(f"First stage model saved to: {model_path}")

        # Plot training history (first stage only)
        plot_first_stage_history(history, model_name)

        # Evaluate first stage performance
        print("Evaluating first stage model...")
        metrics = evaluate_first_stage(model)
        save_first_stage_metrics(metrics, model_name)

        print("First stage training complete!")
        return model_path
    
    except Exception as e:
        print(f"ERROR: An exception occurred during training: {e}")
        traceback.print_exc()
        return None

def evaluate_first_stage(model):
    """Evaluate the first stage model on the test set"""
    try:
        print("Evaluating first stage model on test set...")
        
        # Initialize metrics
        normal_correct = 0
        normal_total = 0
        not_normal_correct = 0
        not_normal_total = 0
        
        # Set model to evaluation mode
        model.first_stage_model.eval()
        
        with torch.no_grad():
            for inputs, labels in model.first_stage_test_loader:
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                
                outputs = model.first_stage_model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                # Count correct predictions for each class
                normal_mask = (labels == 0)  # Normal class
                normal_total += normal_mask.sum().item()
                normal_correct += ((predicted == labels) & normal_mask).sum().item()
                
                not_normal_mask = (labels == 1)  # Not Normal class
                not_normal_total += not_normal_mask.sum().item()
                not_normal_correct += ((predicted == labels) & not_normal_mask).sum().item()
        
        # Calculate metrics
        normal_acc = normal_correct / normal_total if normal_total > 0 else 0
        not_normal_acc = not_normal_correct / not_normal_total if not_normal_total > 0 else 0
        overall_acc = (normal_correct + not_normal_correct) / (normal_total + not_normal_total)
        
        print(f"First Stage Results:")
        print(f"  Normal Class:     {normal_correct}/{normal_total} ({normal_acc:.4f})")
        print(f"  Not Normal Class: {not_normal_correct}/{not_normal_total} ({not_normal_acc:.4f})")
        print(f"  Overall Accuracy: {overall_acc:.4f}")
        
        return {
            'accuracy': overall_acc,
            'normal_acc': normal_acc,
            'not_normal_acc': not_normal_acc,
            'true_normal': normal_correct,
            'false_normal': normal_total - normal_correct,
            'true_abnormal': not_normal_correct,
            'false_abnormal': not_normal_total - not_normal_correct
        }
        
    except Exception as e:
        print(f"ERROR during evaluation: {e}")
        traceback.print_exc()
        return {
            'error': str(e),
            'accuracy': 0.0,
            'normal_acc': 0.0,
            'not_normal_acc': 0.0
        }

def plot_first_stage_history(history, model_name):
    """Plot first stage training history"""
    try:
        os.makedirs('figures', exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(12, 5), dpi=150)
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'], 'b-', label='Train')
        plt.plot(history['val_acc'], 'r-', label='Validation')
        plt.title('First Stage - Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0.5, 1.0)

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], 'b-', label='Train')
        plt.plot(history['val_loss'], 'r-', label='Validation')
        plt.title('First Stage - Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f'figures/{model_name}_training_history.png', dpi=300)
        plt.close()
        print(f"Training plots saved to figures/{model_name}_training_history.png")
    except Exception as e:
        print(f"WARNING: Failed to plot training history: {e}")

def save_first_stage_metrics(metrics, model_name):
    """Save first stage evaluation metrics"""
    try:
        os.makedirs('evaluations', exist_ok=True)
        
        with open(f'evaluations/{model_name}_metrics.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"Model: {model_name} (First Stage Only)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision (Normal): {metrics['normal_acc']:.4f}\n")
            f.write(f"Recall (Normal): {metrics['normal_acc']:.4f}\n")  # Same as accuracy for binary
            f.write(f"F1-Score (Normal): {metrics['normal_acc']:.4f}\n")  # Same as accuracy for binary
            f.write(f"Precision (Abnormal): {metrics['not_normal_acc']:.4f}\n")
            f.write(f"Recall (Abnormal): {metrics['not_normal_acc']:.4f}\n")  # Same as accuracy for binary
            f.write(f"F1-Score (Abnormal): {metrics['not_normal_acc']:.4f}\n\n")  # Same as accuracy for binary
            
            f.write("Confusion Matrix:\n")
            f.write(f"True Normal: {metrics['true_normal']}\n")
            f.write(f"False Normal: {metrics['false_normal']}\n")
            f.write(f"True Abnormal: {metrics['true_abnormal']}\n")
            f.write(f"False Abnormal: {metrics['false_abnormal']}\n")
        
        print(f"Evaluation metrics saved to evaluations/{model_name}_metrics.txt")
    except Exception as e:
        print(f"WARNING: Failed to save evaluation metrics: {e}")

if __name__ == "__main__":
    main()