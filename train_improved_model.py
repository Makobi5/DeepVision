# train_improved_model.py
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
    from app.models.improved_two_stage_model import TwoStageModel  # Import from your improved model file
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Make sure the following files exist:")
    print("  - app/utils/dataset_processing.py")
    print("  - app/models/improved_two_stage_model.py")
    sys.exit(1)

def main():
    """Main function for training the improved two-stage model with ensemble approach"""
    try:
        print("Starting improved scene classification model training...")
        print(f"Current working directory: {os.getcwd()}")

        # Set up paths
        source_dir = 'data/SCVD/SCVD_converted'
        target_dir = 'data/processed_classification'
        models_dir = 'models'

        # Improved training parameters based on recommendations
        frame_count = 10  # Increased frame count for temporal features
        use_weighted_sampler = True
        epochs_first_stage = 30
        epochs_second_stage = 50  # Increased from 40
        epochs_weaponized = 40
        learning_rate_first = 0.0003
        learning_rate_second = 0.0002
        learning_rate_weaponized = 0.0002
        dropout_rate = 0.7  # Increased dropout for better regularization
        weight_decay = 3e-4  # Increased weight decay
        patience_first = 12  # Increased patience as recommended
        patience_second = 15  # Increased patience as recommended
        patience_weaponized = 15
        label_smoothing = 0.1
        augment_minority = True  # Enable additional augmentation for minority classes

        # Manual class weights to balance recognition
        class_weights_manual = {
            'first_stage': {0: 0.8, 1: 1.2},  # Favor detecting Not Normal
            'second_stage': None,  # Use computed weights
            'weaponized_detector': {0: 0.7, 1: 1.3},  # Favor detecting Weaponized
            'first_stage_loss': [0.8, 1.2],  # Similar weights for loss function
            'second_stage_loss': None,  # Use computed weights
            'weaponized_detector_loss': [0.7, 1.3]  # Favor detecting Weaponized
        }

        print(f"Frame count: {frame_count}")
        print(f"Using weighted sampler: {use_weighted_sampler}")
        print(f"Augmenting minority classes: {augment_minority}")
        print(f"Training first stage for {epochs_first_stage} epochs with learning rate {learning_rate_first}")
        print(f"Training second stage for {epochs_second_stage} epochs with learning rate {learning_rate_second}")
        print(f"Training weaponized detector for {epochs_weaponized} epochs with learning rate {learning_rate_weaponized}")
        print(f"Dropout rate: {dropout_rate}")
        print(f"Weight decay: {weight_decay}")
        print(f"Using manual class weights: {class_weights_manual is not None}")

        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)

        # Prepare dataset
        print("Preparing dataset...")
        video_data = prepare_classification_dataset(source_dir, target_dir)
        classes = ['Normal', 'Violence', 'Weaponized']

        print(f"Dataset prepared with classes: {classes}")

        # Create timestamp for model versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"improved_ensemble_classifier_{timestamp}"

        # Create and train the model
        print("Initializing improved two-stage model with ensemble approach...")
        model = TwoStageModel(
            video_data=video_data, 
            dropout_rate=dropout_rate,
            frame_count=frame_count,
            use_weighted_sampler=use_weighted_sampler,
            class_weights_manual=class_weights_manual,
            augment_minority=augment_minority
        )
        
        print("Training improved ensemble model...")
        history, model_paths = model.train(
            epochs_first=epochs_first_stage,
            epochs_second=epochs_second_stage,
            epochs_weaponized=epochs_weaponized,
            learning_rate_first=learning_rate_first,
            learning_rate_second=learning_rate_second,
            learning_rate_weaponized=learning_rate_weaponized,
            weight_decay=weight_decay,
            patience_first=patience_first,
            patience_second=patience_second,
            patience_weaponized=patience_weaponized,
            label_smoothing=label_smoothing
        )

        # Plot training history
        plot_training_history(history, model_name)

        # Evaluate model on test set
        print("Evaluating model on test set...")
        print("\nEvaluating with standard two-stage approach:")
        metrics_standard = model.evaluate(verbose=True, use_ensemble=False)
        
        print("\nEvaluating with ensemble approach:")
        metrics_ensemble = model.evaluate(verbose=True, use_ensemble=True)

        # Save evaluation metrics
        save_evaluation_metrics(metrics_standard, f"{model_name}_standard", classes)
        save_evaluation_metrics(metrics_ensemble, f"{model_name}_ensemble", classes)

        # Create symlinks to latest metrics for easier comparison
        create_latest_symlinks(f"{model_name}_standard", f"{model_name}_ensemble")

        print(f"Training complete! Models saved to: {model_paths}")

        return model_paths
    
    except Exception as e:
        print(f"ERROR: An exception occurred during training: {e}")
        traceback.print_exc()
        return None

def plot_training_history(history, model_name):
    """Plot and save training history graphs for all models in the ensemble"""
    try:
        # Create figure directory if it doesn't exist
        os.makedirs('figures', exist_ok=True)
        
        # Plot each model's training history
        for model_type in ['first_stage', 'second_stage', 'weaponized_detector']:
            if model_type not in history:
                continue
                
            model_history = history[model_type]
            model_title = {
                'first_stage': 'First Stage - Normal vs Not Normal',
                'second_stage': 'Second Stage - Violence vs Weaponized',
                'weaponized_detector': 'Weaponized Detector - Normal vs Weaponized'
            }[model_type]
            
            plt.figure(figsize=(15, 10))
            
            # Accuracy
            plt.subplot(2, 2, 1)
            plt.plot(model_history['train_acc'], label='Train')
            plt.plot(model_history['val_acc'], label='Validation')
            plt.title(f'{model_title} - Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(loc='lower right')
            plt.grid(True, linestyle='--', alpha=0.7)

            # Loss
            plt.subplot(2, 2, 2)
            plt.plot(model_history['train_loss'], label='Train')
            plt.plot(model_history['val_loss'], label='Validation')
            plt.title(f'{model_title} - Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(loc='upper right')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Learning Rate
            plt.subplot(2, 2, 3)
            plt.plot(model_history['learning_rates'])
            plt.title(f'{model_title} - Learning Rate')
            plt.ylabel('Learning Rate')
            plt.xlabel('Epoch')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.yscale('log')
            
            # Accuracy vs Loss
            plt.subplot(2, 2, 4)
            plt.scatter(model_history['train_loss'], model_history['train_acc'], label='Train', alpha=0.5)
            plt.scatter(model_history['val_loss'], model_history['val_acc'], label='Validation', alpha=0.5)
            plt.title(f'{model_title} - Accuracy vs Loss')
            plt.xlabel('Loss')
            plt.ylabel('Accuracy')
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(f'figures/{model_name}_{model_type}_history.png', dpi=300)
            plt.close()
        
        # Create a comparison figure for all models
        plt.figure(figsize=(15, 8))
        
        # Accuracy comparison
        plt.subplot(1, 2, 1)
        for model_type in ['first_stage', 'second_stage', 'weaponized_detector']:
            if model_type in history:
                plt.plot(history[model_type]['val_acc'], 
                        label={
                            'first_stage': 'First Stage',
                            'second_stage': 'Second Stage',
                            'weaponized_detector': 'Weaponized Detector'
                        }[model_type])
        plt.title('Validation Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Loss comparison
        plt.subplot(1, 2, 2)
        for model_type in ['first_stage', 'second_stage', 'weaponized_detector']:
            if model_type in history:
                plt.plot(history[model_type]['val_loss'], 
                        label={
                            'first_stage': 'First Stage',
                            'second_stage': 'Second Stage',
                            'weaponized_detector': 'Weaponized Detector'
                        }[model_type])
        plt.title('Validation Loss Comparison')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save comparison figure
        plt.tight_layout()
        plt.savefig(f'figures/{model_name}_comparison.png', dpi=300)
        plt.close()

        print(f"Training plots saved to figures/{model_name}_*")
    except Exception as e:
        print(f"WARNING: Failed to plot training history: {e}")

def save_evaluation_metrics(metrics, model_name, class_names):
    """Save evaluation metrics to a text file with improved formatting"""
    try:
        os.makedirs('evaluations', exist_ok=True)
        
        with open(f'evaluations/{model_name}_metrics.txt', 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n\n")
            
            f.write("Per-Class Metrics:\n")
            f.write("=====================================\n")
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
                f.write(f"  F1-Score:  {f1:.4f}\n")
                f.write(f"  Support:   {int(np.sum(confusion_matrix[i, :]))}\n\n")
            
            # Write confusion matrix
            f.write("\nConfusion Matrix:\n")
            f.write("=====================================\n")
            header = "            " + " ".join(f"{name:>10}" for name in class_names)
            f.write(header + "\n")
            
            for i, name in enumerate(class_names):
                row = f"{name:10}" + " ".join([f"{int(confusion_matrix[i, j]):10d}" for j in range(len(class_names))])
                f.write(row + "\n")
            
            # Add error analysis
            f.write("\nError Analysis:\n")
            f.write("=====================================\n")
            for i, from_class in enumerate(class_names):
                for j, to_class in enumerate(class_names):
                    if i != j and confusion_matrix[i, j] > 0:
                        error_rate = confusion_matrix[i, j] / np.sum(confusion_matrix[i, :])
                        f.write(f"{from_class} misclassified as {to_class}: {confusion_matrix[i, j]:.0f} instances ({error_rate:.2%})\n")
        
        print(f"Evaluation metrics saved to evaluations/{model_name}_metrics.txt")
    except Exception as e:
        print(f"WARNING: Failed to save evaluation metrics: {e}")

def create_latest_symlinks(standard_name, ensemble_name):
    """Create symlinks to the latest metrics files for easier comparison"""
    try:
        # Create symlinks for standard approach
        standard_source = f"evaluations/{standard_name}_metrics.txt"
        standard_link = "evaluations/improved_ensemble_classifier_latest_standard_metrics.txt"
        
        if os.path.exists(standard_link):
            os.remove(standard_link)
            
        # Create relative symlink
        os.symlink(os.path.basename(standard_source), standard_link)
        
        # Create symlinks for ensemble approach
        ensemble_source = f"evaluations/{ensemble_name}_metrics.txt"
        ensemble_link = "evaluations/improved_ensemble_classifier_latest_ensemble_metrics.txt"
        
        if os.path.exists(ensemble_link):
            os.remove(ensemble_link)
            
        # Create relative symlink
        os.symlink(os.path.basename(ensemble_source), ensemble_link)
        
        print("Created latest metrics symlinks for easy comparison")
    except Exception as e:
        print(f"WARNING: Failed to create symlinks: {e}")

def compare_models(baseline_metrics_path, improved_metrics_path, output_path):
    """Compare the performance of baseline and improved models"""
    try:
        import re
        
        # Function to extract metrics from file
        def extract_metrics(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Extract overall accuracy
            accuracy_match = re.search(r'Overall Accuracy: (\d+\.\d+)', content)
            accuracy = float(accuracy_match.group(1)) if accuracy_match else None
            
            # Extract F1 scores per class
            normal_f1_match = re.search(r'Normal:.*\n.*\n.*\n  F1-Score:  (\d+\.\d+)', content)
            violence_f1_match = re.search(r'Violence:.*\n.*\n.*\n  F1-Score:  (\d+\.\d+)', content)
            weaponized_f1_match = re.search(r'Weaponized:.*\n.*\n.*\n  F1-Score:  (\d+\.\d+)', content)
            
            normal_f1 = float(normal_f1_match.group(1)) if normal_f1_match else None
            violence_f1 = float(violence_f1_match.group(1)) if violence_f1_match else None
            weaponized_f1 = float(weaponized_f1_match.group(1)) if weaponized_f1_match else None
            
            return {
                'accuracy': accuracy,
                'normal_f1': normal_f1,
                'violence_f1': violence_f1,
                'weaponized_f1': weaponized_f1
            }
        
        # Extract metrics from both models
        baseline_metrics = extract_metrics(baseline_metrics_path)
        improved_metrics = extract_metrics(improved_metrics_path)
        
        # Calculate improvements
        accuracy_improvement = improved_metrics['accuracy'] - baseline_metrics['accuracy']
        normal_f1_improvement = improved_metrics['normal_f1'] - baseline_metrics['normal_f1']
        violence_f1_improvement = improved_metrics['violence_f1'] - baseline_metrics['violence_f1']
        weaponized_f1_improvement = improved_metrics['weaponized_f1'] - baseline_metrics['weaponized_f1']
        
        # Write comparison to file
        with open(output_path, 'w') as f:
            f.write("Model Performance Comparison\n")
            f.write("============================\n\n")
            
            f.write("Overall Accuracy:\n")
            f.write(f"  Baseline: {baseline_metrics['accuracy']:.4f}\n")
            f.write(f"  Improved: {improved_metrics['accuracy']:.4f}\n")
            f.write(f"  Change:   {accuracy_improvement:.4f} ({accuracy_improvement/baseline_metrics['accuracy']*100:.1f}%)\n\n")
            
            f.write("Class: Normal (F1-Score)\n")
            f.write(f"  Baseline: {baseline_metrics['normal_f1']:.4f}\n")
            f.write(f"  Improved: {improved_metrics['normal_f1']:.4f}\n")
            f.write(f"  Change:   {normal_f1_improvement:.4f} ({normal_f1_improvement/baseline_metrics['normal_f1']*100:.1f}%)\n\n")
            
            f.write("Class: Violence (F1-Score)\n")
            f.write(f"  Baseline: {baseline_metrics['violence_f1']:.4f}\n")
            f.write(f"  Improved: {improved_metrics['violence_f1']:.4f}\n")
            f.write(f"  Change:   {violence_f1_improvement:.4f} ({violence_f1_improvement/baseline_metrics['violence_f1']*100:.1f}%)\n\n")
            
            f.write("Class: Weaponized (F1-Score)\n")
            f.write(f"  Baseline: {baseline_metrics['weaponized_f1']:.4f}\n")
            f.write(f"  Improved: {improved_metrics['weaponized_f1']:.4f}\n")
            f.write(f"  Change:   {weaponized_f1_improvement:.4f} ({weaponized_f1_improvement/baseline_metrics['weaponized_f1']*100:.1f}%)\n\n")
            
            # Summary of key improvements
            f.write("Summary of Improvements:\n")
            f.write("============================\n")
            f.write("1. Overall model accuracy improved by ")
            f.write(f"{accuracy_improvement:.4f} ({accuracy_improvement/baseline_metrics['accuracy']*100:.1f}%)\n")
            
            # Highlight the biggest improvement
            improvements = [
                ("Normal", normal_f1_improvement/baseline_metrics['normal_f1']*100),
                ("Violence", violence_f1_improvement/baseline_metrics['violence_f1']*100),
                ("Weaponized", weaponized_f1_improvement/baseline_metrics['weaponized_f1']*100)
            ]
            improvements.sort(key=lambda x: x[1], reverse=True)
            f.write(f"2. Biggest improvement in F1-score: {improvements[0][0]} class with {improvements[0][1]:.1f}% increase\n")
            
            # Highlight changes by implementation
            f.write("3. Key implemented improvements:\n")
            f.write("   - Enhanced model architecture with EfficientNet B2 and attention mechanisms\n")
            f.write("   - Ensemble approach combining specialized detectors\n")
            f.write("   - Improved class balance through weighted sampling and augmentation\n")
            f.write("   - Increased frame count and improved temporal feature extraction\n")
            f.write("   - Optimized training hyperparameters with longer patience and cosine annealing\n")
        
        print(f"Model comparison saved to {output_path}")
        
    except Exception as e:
        print(f"WARNING: Failed to compare models: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Train the improved model
    model_paths = main()
    
    # Compare with baseline model if metrics files exist
    try:
        baseline_metrics_path = 'evaluations/two_stage_classifier_latest_metrics.txt'  # Update with your baseline path
        if os.path.exists(baseline_metrics_path):
            improved_metrics_path = 'evaluations/improved_ensemble_classifier_latest_ensemble_metrics.txt'
            if os.path.exists(improved_metrics_path):
                compare_models(
                    baseline_metrics_path, 
                    improved_metrics_path,
                    'evaluations/model_comparison.txt'
                )
    except Exception as e:
        print(f"WARNING: Failed to run model comparison: {e}")
