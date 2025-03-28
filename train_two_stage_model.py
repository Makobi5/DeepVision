# train_two_stage_model.py
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

# Try to import the necessary modules, with clear error handling
try:
    from app.utils.dataset_processing import prepare_classification_dataset
    #from two_stage_model_improved import TwoStageModel  # Import the improved model
    from app.models.two_stage_model_improved  import TwoStageModel
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Make sure the following files exist:")
    print("  - app/utils/dataset_processing.py")
    print("  - two_stage_model_improved.py")
    sys.exit(1)

def main():
    """Main function for training the improved two-stage model"""
    try:
        # Set random seed for reproducibility
        seed_everything(42)
        
        print("Starting improved two-stage scene classification model training...")
        print(f"Current working directory: {os.getcwd()}")

        # Set up paths
        source_dir = 'data/SCVD/SCVD_converted'
        target_dir = 'data/processed_classification'
        models_dir = 'models'
        
        # Make sure output directories exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs('figures', exist_ok=True)
        os.makedirs('evaluations', exist_ok=True)
        
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if device.type == 'cuda':
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Improved training parameters
        frame_count = 10  # Number of frames to extract per video
        use_weighted_sampler = True
        
        # First stage parameters (binary classification: Normal vs Not Normal)
        epochs_first_stage = 35  # Increased from 30
        learning_rate_first = 0.0004  # Slightly increased
        patience_first = 12  # Increased patience
        
        # Second stage parameters (binary classification: Violence vs Weaponized)
        epochs_second_stage = 90  # Increased for better convergence
        learning_rate_second = 0.0006  # Slightly increased
        patience_second = 25  # Increased patience
        
        # Common parameters
        dropout_rate = 0.6  # Higher dropout for better regularization
        weight_decay = 4e-4  # Increased weight decay
        label_smoothing = 0.05  # Reduced from original
        mixup_alpha = 0.3  # Increased mixup alpha
        
        print(f"Frame count: {frame_count}")
        print(f"Using weighted sampler: {use_weighted_sampler}")
        print(f"Using dropout rate: {dropout_rate}")
        print(f"Using label smoothing: {label_smoothing}")
        print(f"Using mixup with alpha={mixup_alpha}")
        print(f"Training first stage for {epochs_first_stage} epochs with learning rate {learning_rate_first}")
        print(f"Training second stage for {epochs_second_stage} epochs with learning rate {learning_rate_second}")

        # Prepare dataset
        print("Preparing dataset...")
        video_data = prepare_classification_dataset(source_dir, target_dir)
        classes = ['Normal', 'Violence', 'Weaponized']

        print(f"Dataset prepared with classes: {classes}")

        # Create timestamp for model versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"two_stage_classifier_improved_{timestamp}"

        # Create and train the model
        print("Initializing improved two-stage model...")
        model = TwoStageModel(
            video_data=video_data, 
            dropout_rate=dropout_rate,
            frame_count=frame_count,
            use_weighted_sampler=use_weighted_sampler
        )
        
        # Enable mixed precision training using GradScaler
        print("Training with mixed precision for better performance")
        
        print("Training improved two-stage model...")
        history, model_paths = model.train(
            epochs_first=epochs_first_stage,
            epochs_second=epochs_second_stage,
            learning_rate_first=learning_rate_first,
            learning_rate_second=learning_rate_second,
            weight_decay=weight_decay,
            patience_first=patience_first,
            patience_second=patience_second,
            label_smoothing=label_smoothing,
            mixup_alpha=mixup_alpha
        )

        # Plot training history
        plot_training_history(history, model_name)

        # Evaluate model on test set
        print("Evaluating model on test set...")
        metrics = model.evaluate(verbose=True)

        # Save evaluation metrics
        save_evaluation_metrics(metrics, model_name, classes)

        print(f"Training complete! Models saved to: {model_paths}")

        return model_paths
    
    except Exception as e:
        print(f"ERROR: An exception occurred during training: {e}")
        traceback.print_exc()
        return None

def plot_training_history(history, model_name):
    """Plot and save two-stage training history graphs with improved design"""
    try:
        # Create figures directory if it doesn't exist
        os.makedirs('figures', exist_ok=True)
        
        # Set higher DPI and better style
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(16, 12), dpi=150)
        
        # First Stage - Accuracy with smoother plot and moving average
        plt.subplot(2, 3, 1)
        train_acc = history['first_stage']['train_acc']
        val_acc = history['first_stage']['val_acc']
        
        # Apply moving average for smoother plots
        window = min(5, len(train_acc) // 3)
        if window > 1:
            train_acc_smooth = np.convolve(train_acc, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(train_acc)), train_acc_smooth, 'b-', linewidth=2, alpha=0.8, label='Train (MA)')
        
        plt.plot(train_acc, 'b--', linewidth=1, alpha=0.5, label='Train (Raw)')
        plt.plot(val_acc, 'r-', linewidth=2, label='Validation')
        plt.title('First Stage - Model Accuracy', fontsize=14)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(min(0.5, min(train_acc) - 0.05), 1.0)

        # First Stage - Loss
        plt.subplot(2, 3, 2)
        train_loss = history['first_stage']['train_loss']
        val_loss = history['first_stage']['val_loss']
        
        # Apply moving average for smoother plots
        if window > 1:
            train_loss_smooth = np.convolve(train_loss, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(train_loss)), train_loss_smooth, 'b-', linewidth=2, alpha=0.8, label='Train (MA)')
        
        plt.plot(train_loss, 'b--', linewidth=1, alpha=0.5, label='Train (Raw)')
        plt.plot(val_loss, 'r-', linewidth=2, label='Validation')
        plt.title('First Stage - Model Loss', fontsize=14)
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # First Stage - Learning Rate
        plt.subplot(2, 3, 3)
        plt.plot(history['first_stage']['learning_rates'], 'g-', linewidth=2)
        plt.title('First Stage - Learning Rate', fontsize=14)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.yscale('log')
        
        # Second Stage - Accuracy
        plt.subplot(2, 3, 4)
        train_acc = history['second_stage']['train_acc']
        val_acc = history['second_stage']['val_acc']
        
        # Apply moving average for smoother plots
        window = min(5, len(train_acc) // 3)
        if window > 1:
            train_acc_smooth = np.convolve(train_acc, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(train_acc)), train_acc_smooth, 'b-', linewidth=2, alpha=0.8, label='Train (MA)')
        
        plt.plot(train_acc, 'b--', linewidth=1, alpha=0.5, label='Train (Raw)')
        plt.plot(val_acc, 'r-', linewidth=2, label='Validation')
        plt.title('Second Stage - Model Accuracy', fontsize=14)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(min(0.5, min(train_acc) - 0.05), 1.0)

        # Second Stage - Loss
        plt.subplot(2, 3, 5)
        train_loss = history['second_stage']['train_loss']
        val_loss = history['second_stage']['val_loss']
        
        # Apply moving average for smoother plots
        if window > 1:
            train_loss_smooth = np.convolve(train_loss, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(train_loss)), train_loss_smooth, 'b-', linewidth=2, alpha=0.8, label='Train (MA)')
        
        plt.plot(train_loss, 'b--', linewidth=1, alpha=0.5, label='Train (Raw)')
        plt.plot(val_loss, 'r-', linewidth=2, label='Validation')
        plt.title('Second Stage - Model Loss', fontsize=14)
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Second Stage - Learning Rate
        plt.subplot(2, 3, 6)
        plt.plot(history['second_stage']['learning_rates'], 'g-', linewidth=2)
        plt.title('Second Stage - Learning Rate', fontsize=14)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.yscale('log')
        
        # Adjust layout and save combined figure
        plt.tight_layout()
        plt.savefig(f'figures/{model_name}_training_history.png', dpi=300)
        plt.close()

        print(f"Training plots saved to figures/{model_name}_training_history.png")
    except Exception as e:
        print(f"WARNING: Failed to plot training history: {e}")

def save_evaluation_metrics(metrics, model_name, class_names):
    """Save evaluation metrics to a text file with enhanced formatting"""
    try:
        os.makedirs('evaluations', exist_ok=True)
        
        with open(f'evaluations/{model_name}_metrics.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"Model: {model_name} (Improved Two-Stage Classifier)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"First Stage Accuracy: {metrics['first_stage_acc']:.4f}\n")
            f.write(f"Second Stage Accuracy: {metrics['second_stage_acc']:.4f}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("Per-Class Metrics:\n")
            f.write("-"*80 + "\n")
            
            confusion_matrix = metrics['confusion_matrix']
            
            for i, class_name in enumerate(class_names):
                # Calculate true positives, false positives, and false negatives
                true_pos = confusion_matrix[i, i]
                false_pos = np.sum(confusion_matrix[:, i]) - true_pos
                false_neg = np.sum(confusion_matrix[i, :]) - true_pos
                total = np.sum(confusion_matrix[i, :])
                
                # Calculate precision, recall, and F1 score
                precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                f.write(f"{class_name}:\n")
                f.write(f"  Accuracy:  {metrics['class_acc'][class_name]:.4f}\n")
                f.write(f"  Precision: {precision:.4f}\n")
                f.write(f"  Recall:    {recall:.4f}\n")
                f.write(f"  F1-Score:  {f1:.4f}\n")
                f.write(f"  Support:   {total}\n\n")
            
            # Write confusion matrix
            f.write("-"*80 + "\n")
            f.write("Confusion Matrix:\n")
            f.write("-"*80 + "\n")
            
            header = "            " + " ".join(f"{name:>10}" for name in class_names)
            f.write(header + "\n")
            
            for i, name in enumerate(class_names):
                row = f"{name:10}" + " ".join([f"{int(confusion_matrix[i, j]):10d}" for j in range(len(class_names))])
                f.write(row + "\n")
                
            # Calculate and write normalized confusion matrix (percentages)
            f.write("\nNormalized Confusion Matrix (row percentages):\n")
            header = "            " + " ".join(f"{name:>10}" for name in class_names)
            f.write(header + "\n")
            
            for i, name in enumerate(class_names):
                row_sum = np.sum(confusion_matrix[i, :])
                if row_sum > 0:
                    normalized_row = confusion_matrix[i, :] / row_sum * 100
                    row = f"{name:10}" + " ".join([f"{normalized_row[j]:10.2f}%" for j in range(len(class_names))])
                    f.write(row + "\n")
                else:
                    row = f"{name:10}" + " ".join([f"{0:10.2f}%" for _ in range(len(class_names))])
                    f.write(row + "\n")
        
        print(f"Evaluation metrics saved to evaluations/{model_name}_metrics.txt")
    except Exception as e:
        print(f"WARNING: Failed to save evaluation metrics: {e}")

if __name__ == "__main__":
    main()