# train_two_stage_model.py
import os
import sys
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import traceback

# Try to import the necessary modules, with clear error handling
try:
    from app.utils.dataset_processing import prepare_classification_dataset
    from app.models.two_stage_model import TwoStageModel
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Make sure the following files exist:")
    print("  - app/utils/dataset_processing.py")
    print("  - app/models/two_stage_model.py")
    sys.exit(1)

def main():
    """Main function for training the two-stage model"""
    try:
        print("Starting two-stage scene classification model training...")
        print(f"Current working directory: {os.getcwd()}")

        # Set up paths
        source_dir = 'data/SCVD/SCVD_converted'
        target_dir = 'data/processed_classification'
        models_dir = 'models'

        # Training parameters - first stage remains the same, second stage improved
        frame_count = 10  # Increased frame count for better temporal features
        use_weighted_sampler = True
        epochs_first_stage = 30
        epochs_second_stage = 80  # Increased from 40
        learning_rate_first = 0.0003
        learning_rate_second = 0.0005  # Increased from 0.0002
        dropout_rate = 0.6
        weight_decay = 3e-4  # Increased from 2e-4
        patience_first = 10
        patience_second = 20  # Increased from 12
        label_smoothing = 0.05  # Reduced from 0.1
        mixup_alpha = 0.2  # Added mixup augmentation

        print(f"Frame count: {frame_count}")
        print(f"Using weighted sampler: {use_weighted_sampler}")
        print(f"Training first stage for {epochs_first_stage} epochs with learning rate {learning_rate_first}")
        print(f"Training second stage for {epochs_second_stage} epochs with learning rate {learning_rate_second}")
        print(f"Using mixup with alpha={mixup_alpha} for second stage")

        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)

        # Prepare dataset
        print("Preparing dataset...")
        video_data = prepare_classification_dataset(source_dir, target_dir)
        classes = ['Normal', 'Violence', 'Weaponized']

        print(f"Dataset prepared with classes: {classes}")

        # Create timestamp for model versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"two_stage_classifier_{timestamp}"

        # Create and train the model
        print("Initializing two-stage model...")
        model = TwoStageModel(
            video_data=video_data, 
            dropout_rate=dropout_rate,
            frame_count=frame_count,
            use_weighted_sampler=use_weighted_sampler
        )
        
        print("Training two-stage model...")
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
    """Plot and save two-stage training history graphs"""
    try:
        # Create figure directory if it doesn't exist
        os.makedirs('figures', exist_ok=True)
        
        # Plot first stage
        plt.figure(figsize=(15, 10))
        
        # First Stage - Accuracy
        plt.subplot(2, 2, 1)
        plt.plot(history['first_stage']['train_acc'], label='Train')
        plt.plot(history['first_stage']['val_acc'], label='Validation')
        plt.title('First Stage - Model Accuracy (Normal vs Not Normal)')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)

        # First Stage - Loss
        plt.subplot(2, 2, 2)
        plt.plot(history['first_stage']['train_loss'], label='Train')
        plt.plot(history['first_stage']['val_loss'], label='Validation')
        plt.title('First Stage - Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # First Stage - Learning Rate
        plt.subplot(2, 2, 3)
        plt.plot(history['first_stage']['learning_rates'])
        plt.title('First Stage - Learning Rate')
        plt.ylabel('Learning Rate')
        plt.xlabel('Epoch')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.yscale('log')
        
        # First Stage - Accuracy vs Loss
        plt.subplot(2, 2, 4)
        plt.scatter(history['first_stage']['train_loss'], history['first_stage']['train_acc'], label='Train', alpha=0.5)
        plt.scatter(history['first_stage']['val_loss'], history['first_stage']['val_acc'], label='Validation', alpha=0.5)
        plt.title('First Stage - Accuracy vs Loss')
        plt.xlabel('Loss')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save first stage figure
        plt.tight_layout()
        plt.savefig(f'figures/{model_name}_first_stage_history.png', dpi=300)
        plt.close()
        
        # Plot second stage
        plt.figure(figsize=(15, 10))
        
        # Second Stage - Accuracy
        plt.subplot(2, 2, 1)
        plt.plot(history['second_stage']['train_acc'], label='Train')
        plt.plot(history['second_stage']['val_acc'], label='Validation')
        plt.title('Second Stage - Model Accuracy (Violence vs Weaponized)')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Second Stage - Loss
        plt.subplot(2, 2, 2)
        plt.plot(history['second_stage']['train_loss'], label='Train')
        plt.plot(history['second_stage']['val_loss'], label='Validation')
        plt.title('Second Stage - Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Second Stage - Learning Rate
        plt.subplot(2, 2, 3)
        plt.plot(history['second_stage']['learning_rates'])
        plt.title('Second Stage - Learning Rate')
        plt.ylabel('Learning Rate')
        plt.xlabel('Epoch')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.yscale('log')
        
        # Second Stage - Accuracy vs Loss
        plt.subplot(2, 2, 4)
        plt.scatter(history['second_stage']['train_loss'], history['second_stage']['train_acc'], label='Train', alpha=0.5)
        plt.scatter(history['second_stage']['val_loss'], history['second_stage']['val_acc'], label='Validation', alpha=0.5)
        plt.title('Second Stage - Accuracy vs Loss')
        plt.xlabel('Loss')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save second stage figure
        plt.tight_layout()
        plt.savefig(f'figures/{model_name}_second_stage_history.png', dpi=300)
        plt.close()

        print(f"Training plots saved to figures/{model_name}_first_stage_history.png and {model_name}_second_stage_history.png")
    except Exception as e:
        print(f"WARNING: Failed to plot training history: {e}")

def save_evaluation_metrics(metrics, model_name, class_names):
    """Save evaluation metrics to a text file"""
    try:
        os.makedirs('evaluations', exist_ok=True)
        
        with open(f'evaluations/{model_name}_metrics.txt', 'w') as f:
            f.write(f"Model: {model_name} (Two-Stage Classifier)\n")
            f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n\n")
            
            f.write("Per-Class Metrics:\n")
            confusion_matrix = metrics['confusion_matrix']
            
            for i, class_name in enumerate(class_names):
                # Calculate true positives, false positives, and false negatives
                true_pos = confusion_matrix[i, i]
                false_pos = np.sum(confusion_matrix[:, i]) - true_pos
                false_neg = np.sum(confusion_matrix[i, :]) - true_pos
                
                # Calculate precision, recall, and F1 score
                precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {precision:.4f}\n")
                f.write(f"  Recall:    {recall:.4f}\n")
                f.write(f"  F1-Score:  {f1:.4f}\n\n")
            
            # Write confusion matrix
            f.write("Confusion Matrix:\n")
            header = "            " + " ".join(f"{name:>10}" for name in class_names)
            f.write(header + "\n")
            
            for i, name in enumerate(class_names):
                row = f"{name:10}" + " ".join([f"{int(confusion_matrix[i, j]):10d}" for j in range(len(class_names))])
                f.write(row + "\n")
        
        print(f"Evaluation metrics saved to evaluations/{model_name}_metrics.txt")
    except Exception as e:
        print(f"WARNING: Failed to save evaluation metrics: {e}")

if __name__ == "__main__":
    main()