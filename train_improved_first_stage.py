# train_two_stage_model_improved.py
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
    """Main function for training the improved first stage model"""
    try:
        seed_everything(42)
        
        print("Starting IMPROVED first stage training...")
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

        frame_count = 10
        use_weighted_sampler = True
        epochs_first_stage = 30  # Reduced from 40
        learning_rate_first = 0.0003
        patience_first = 8  # Reduced from 15
        dropout_rate = 0.8  # Increased from 0.7
        weight_decay = 1e-3  # Increased from 5e-4
        label_smoothing = 0.1
        epsilon = 5
        mixup_alpha = 0.2
        
        print("\nImproved First Stage Parameters:")
        print(f"Frame count: {frame_count}")
        print(f"Epochs: {epochs_first_stage}")
        print(f"Learning rate: {learning_rate_first}")
        print(f"Patience: {patience_first}")
        print(f"Dropout rate: {dropout_rate}")
        print(f"Weight decay: {weight_decay}")
        print(f"Label smoothing: {label_smoothing}")
        print(f"Class weight epsilon: {epsilon}\n")

        # Prepare dataset
        print("Preparing dataset...")
        video_data = prepare_classification_dataset(source_dir, target_dir)
        classes = ['Normal', 'Violence', 'Weaponized']
        print(f"Dataset prepared with classes: {classes}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"improved_first_stage_{timestamp}"

        # Create and train the model with improved parameters
        print("Initializing improved first stage model...")
        model = TwoStageModel(
            video_data=video_data, 
            dropout_rate=dropout_rate,
            frame_count=frame_count,
            use_weighted_sampler=use_weighted_sampler
        )
        
        # Train first stage with improved settings
        print("Training improved first stage model...")
        history = model.train_first_stage(
            epochs=epochs_first_stage,
            learning_rate=learning_rate_first,
            weight_decay=weight_decay,
            patience=patience_first,
            label_smoothing=label_smoothing,
            mixup_alpha=mixup_alpha
            #class_weight_epsilon=epsilon  # New parameter to pass
        )

        # Save first stage model
        model_path = os.path.join(models_dir, f"{model_name}.pth")
        torch.save(model.first_stage_model.state_dict(), model_path)
        print(f"Improved first stage model saved to: {model_path}")

        # Plot training history
        try:
            plt.style.use('seaborn')  # Fallback to basic seaborn if v0_8 not available
            plot_training_history(history, model_name)
        except Exception as e:
            print(f"Warning: Plot styling issue - {e}")
            plot_training_history(history, model_name)  # Try without style

        # Evaluate first stage performance
        print("Evaluating improved first stage model...")
        metrics = evaluate_first_stage(model)
        save_first_stage_metrics(metrics, model_name)

        print("Improved first stage training complete!")
        return model_path
    
    except Exception as e:
        print(f"ERROR: An exception occurred during training: {e}")
        traceback.print_exc()
        return None

def plot_training_history(history, model_name):
    """Plot training history with improved visualization"""
    try:
        os.makedirs('figures', exist_ok=True)
        plt.figure(figsize=(16, 6), dpi=150)
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'], 'b-', label='Train', linewidth=2)
        plt.plot(history['val_acc'], 'r-', label='Validation', linewidth=2)
        plt.title('Model Accuracy', fontsize=14)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0.4, 1.0)
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], 'b-', label='Train', linewidth=2)
        plt.plot(history['val_loss'], 'r-', label='Validation', linewidth=2)
        plt.title('Model Loss', fontsize=14)
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'figures/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training plots saved to figures/{model_name}_training_history.png")
    except Exception as e:
        print(f"WARNING: Failed to plot training history: {e}")

def evaluate_first_stage(model):
    """Enhanced evaluation for first stage model"""
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
        
        # Calculate additional metrics
        precision_normal = normal_correct / (normal_correct + (not_normal_total - not_normal_correct))
        recall_normal = normal_acc
        f1_normal = 2 * (precision_normal * recall_normal) / (precision_normal + recall_normal) if (precision_normal + recall_normal) > 0 else 0
        
        precision_abnormal = not_normal_correct / (not_normal_correct + (normal_total - normal_correct))
        recall_abnormal = not_normal_acc
        f1_abnormal = 2 * (precision_abnormal * recall_abnormal) / (precision_abnormal + recall_abnormal) if (precision_abnormal + recall_abnormal) > 0 else 0
        
        print("\nFirst Stage Evaluation Results:")
        print(f"{'Metric':<15}{'Normal':<10}{'Not Normal':<12}{'Overall':<10}")
        print(f"{'Accuracy':<15}{normal_acc:.4f}{not_normal_acc:.4f}{overall_acc:.4f}")
        print(f"{'Precision':<15}{precision_normal:.4f}{precision_abnormal:.4f}{'-':<10}")
        print(f"{'Recall':<15}{recall_normal:.4f}{recall_abnormal:.4f}{'-':<10}")
        print(f"{'F1-Score':<15}{f1_normal:.4f}{f1_abnormal:.4f}{'-':<10}")
        
        return {
            'accuracy': overall_acc,
            'normal_acc': normal_acc,
            'not_normal_acc': not_normal_acc,
            'precision_normal': precision_normal,
            'recall_normal': recall_normal,
            'f1_normal': f1_normal,
            'precision_abnormal': precision_abnormal,
            'recall_abnormal': recall_abnormal,
            'f1_abnormal': f1_abnormal,
            'true_normal': normal_correct,
            'false_normal': normal_total - normal_correct,
            'true_abnormal': not_normal_correct,
            'false_abnormal': not_normal_total - not_normal_correct,
            'support_normal': normal_total,
            'support_abnormal': not_normal_total
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

def save_first_stage_metrics(metrics, model_name):
    """Save enhanced evaluation metrics"""
    try:
        os.makedirs('evaluations', exist_ok=True)
        
        with open(f'evaluations/{model_name}_metrics.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"Model: {model_name} (Improved First Stage)\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"{'Overall Accuracy:':<25}{metrics['accuracy']:.4f}\n\n")
            
            f.write("-"*50 + "\n")
            f.write(f"{'Class':<15}{'Precision':<12}{'Recall':<10}{'F1-Score':<10}{'Support':<10}\n")
            f.write("-"*50 + "\n")
            f.write(f"{'Normal':<15}{metrics['precision_normal']:.4f}{metrics['recall_normal']:.4f}")
            f.write(f"{metrics['f1_normal']:.4f}{metrics['support_normal']:10d}\n")
            f.write(f"{'Not Normal':<15}{metrics['precision_abnormal']:.4f}{metrics['recall_abnormal']:.4f}")
            f.write(f"{metrics['f1_abnormal']:.4f}{metrics['support_abnormal']:10d}\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write(f"{'':<15}{'Pred Normal':<15}{'Pred Abnormal':<15}\n")
            f.write(f"{'Actual Normal':<15}{metrics['true_normal']:<15}{metrics['false_normal']:<15}\n")
            f.write(f"{'Actual Abnormal':<15}{metrics['false_abnormal']:<15}{metrics['true_abnormal']:<15}\n")
        
        print(f"Evaluation metrics saved to evaluations/{model_name}_metrics.txt")
    except Exception as e:
        print(f"WARNING: Failed to save evaluation metrics: {e}")

if __name__ == "__main__":
    main()