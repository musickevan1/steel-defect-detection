# final_test_evaluation.py
# Performs final test set evaluation on the best model checkpoint

import os
import argparse
import json
import time
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from datetime import datetime
from PIL import Image

from data_loader import load_data
from lightning_model import SteelDefectClassifier

# --- Parse Arguments ---
parser = argparse.ArgumentParser(description='Final Test Set Evaluation')
parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint. If not provided, will use the best checkpoint from saved_models/')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
parser.add_argument('--results_dir', type=str, default='results/final_test_eval', help='Directory to save results')
parser.add_argument('--num_samples', type=int, default=5, help='Number of misclassified samples to visualize per class')
parser.add_argument('--calc_inference_time', action='store_true', help='Calculate average inference time per image')
args = parser.parse_args()

def find_best_checkpoint():
    """Find the best checkpoint based on validation metrics."""
    print("Finding best checkpoint...")
    
    # Check if there's a best_config.txt file
    best_config_path = os.path.join('results', 'cnn', 'best_config.txt')
    if os.path.exists(best_config_path):
        print(f"Found best_config.txt at {best_config_path}")
        # Parse the file to find the timestamp of the best run
        with open(best_config_path, 'r') as f:
            lines = f.readlines()
            
        # Try to extract the timestamp from the experiment log
        experiments_log_path = os.path.join('results', 'cnn', 'experiments_log.csv')
        if os.path.exists(experiments_log_path):
            df = pd.read_csv(experiments_log_path)
            best_idx = df['val_acc'].idxmax()
            best_row = df.iloc[best_idx]
            timestamp = best_row['timestamp']
            batch_size = best_row['batch_size']
            lr = best_row['learning_rate']
            
            # Construct the path to the best checkpoint
            experiment_dir = f"cnn_bs{batch_size}_lr{lr}_{timestamp}"
            checkpoints_dir = os.path.join('results', 'cnn', experiment_dir, 'checkpoints')
            
            if os.path.exists(checkpoints_dir):
                # Find the checkpoint with the highest val_acc
                checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
                if checkpoints:
                    # Sort by validation accuracy (assuming format like cnn-epoch=02-val_acc=0.9500.ckpt)
                    checkpoints.sort(key=lambda x: float(x.split('val_acc=')[1].split('.ckpt')[0]), reverse=True)
                    best_checkpoint = os.path.join(checkpoints_dir, checkpoints[0])
                    print(f"Found best checkpoint: {best_checkpoint}")
                    return best_checkpoint
    
    # If we couldn't find the best checkpoint from logs, search in saved_models
    saved_models_dir = 'saved_models'
    if os.path.exists(saved_models_dir):
        checkpoints = [f for f in os.listdir(saved_models_dir) if f.endswith('.ckpt')]
        if checkpoints:
            # Sort by validation accuracy if possible
            try:
                checkpoints.sort(key=lambda x: float(x.split('val_acc=')[1].split('.ckpt')[0]), reverse=True)
            except:
                # If not in expected format, just use the most recent
                checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(saved_models_dir, x)), reverse=True)
            
            best_checkpoint = os.path.join(saved_models_dir, checkpoints[0])
            print(f"Found best checkpoint: {best_checkpoint}")
            return best_checkpoint
    
    print("Could not find any checkpoint. Please provide one with --checkpoint.")
    return None

def calculate_roc_curves(all_probs, all_targets, class_names, results_dir):
    """Calculate and plot ROC curves for each class."""
    plt.figure(figsize=(12, 10))
    
    # Convert targets to one-hot encoding
    num_classes = len(class_names)
    all_targets_onehot = np.zeros((len(all_targets), num_classes))
    for i, target in enumerate(all_targets):
        all_targets_onehot[i, target] = 1
    
    # Calculate ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i, class_name in enumerate(class_names):
        fpr[i], tpr[i], _ = roc_curve(all_targets_onehot[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.plot(
            fpr[i], 
            tpr[i], 
            lw=2, 
            label=f'{class_name} (AUC = {roc_auc[i]:.2f})'
        )
    
    # Plot the diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(results_dir, 'roc_curves.png'))
    plt.close()
    
    # Return the AUC values
    return roc_auc

def calculate_pr_curves(all_probs, all_targets, class_names, results_dir):
    """Calculate and plot precision-recall curves for each class."""
    plt.figure(figsize=(12, 10))
    
    # Convert targets to one-hot encoding
    num_classes = len(class_names)
    all_targets_onehot = np.zeros((len(all_targets), num_classes))
    for i, target in enumerate(all_targets):
        all_targets_onehot[i, target] = 1
    
    # Calculate precision-recall curve for each class
    precision = {}
    recall = {}
    avg_precision = {}
    
    for i, class_name in enumerate(class_names):
        precision[i], recall[i], _ = precision_recall_curve(all_targets_onehot[:, i], all_probs[:, i])
        avg_precision[i] = average_precision_score(all_targets_onehot[:, i], all_probs[:, i])
        
        plt.plot(
            recall[i], 
            precision[i], 
            lw=2, 
            label=f'{class_name} (AP = {avg_precision[i]:.2f})'
        )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(results_dir, 'precision_recall_curves.png'))
    plt.close()
    
    # Return the average precision values
    return avg_precision

def visualize_misclassified_samples(test_dataset, all_preds, all_targets, all_probs, class_names, results_dir, num_samples=5):
    """
    Visualize misclassified samples from the test set, focusing on the most confident mistakes.
    
    Args:
        test_dataset: The test dataset
        all_preds: Predicted labels
        all_targets: True labels
        all_probs: Prediction probabilities
        class_names: Names of the classes
        results_dir: Directory to save results
        num_samples: Number of misclassified samples to visualize per class
    """
    # Get image paths from test dataset
    image_paths = [path for path, _ in test_dataset.dataset.samples]
    image_paths = [image_paths[idx] for idx in test_dataset.indices]
    
    # Find misclassified samples for each class
    misclassified = {class_name: [] for class_name in class_names}
    
    for i in range(len(all_targets)):
        if all_preds[i] != all_targets[i]:
            true_class = class_names[all_targets[i]]
            pred_class = class_names[all_preds[i]]
            confidence = all_probs[i, all_preds[i]]  # Confidence in the wrong prediction
            
            misclassified[true_class].append((image_paths[i], pred_class, confidence))
    
    # Sort by confidence and take top samples
    for true_class in class_names:
        if misclassified[true_class]:
            # Sort by confidence (highest first)
            misclassified[true_class].sort(key=lambda x: x[2], reverse=True)
            # Take top samples
            misclassified[true_class] = misclassified[true_class][:num_samples]
    
    # Create a directory for misclassified samples
    misclassified_dir = os.path.join(results_dir, 'misclassified_samples')
    os.makedirs(misclassified_dir, exist_ok=True)
    
    # Visualize misclassified samples for each class
    for true_class, samples in misclassified.items():
        if not samples:
            continue
            
        n_samples = len(samples)
        fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 3, 3))
        
        # Handle case with only one sample
        if n_samples == 1:
            axes = [axes]
            
        for i, (img_path, pred_class, confidence) in enumerate(samples):
            img = Image.open(img_path).convert('RGB')
            axes[i].imshow(img)
            axes[i].set_title(f"Pred: {pred_class}\nConf: {confidence:.2f}")
            axes[i].axis('off')
            
        plt.suptitle(f"Misclassified {true_class} Samples")
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(misclassified_dir, f"misclassified_{true_class}.png")
        plt.savefig(save_path)
        plt.close()
        
        print(f"Saved misclassified {true_class} samples to {save_path}")

def calculate_inference_time(model, test_loader, device):
    """Calculate average inference time per image."""
    print("Calculating average inference time per image...")
    
    model.to(device)
    model.eval()
    
    total_time = 0
    total_images = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            # Warm-up run
            if batch_idx == 0:
                _ = model(inputs)
                torch.cuda.synchronize() if device.type == 'cuda' else None
            
            # Timed run
            start_time = time.time()
            _ = model(inputs)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            batch_time = end_time - start_time
            total_time += batch_time
            total_images += batch_size
    
    avg_time = total_time / total_images
    print(f"Average inference time per image: {avg_time*1000:.2f} ms")
    
    return avg_time

def main():
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Find best checkpoint if not provided
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = find_best_checkpoint()
        if not checkpoint_path:
            return
    
    # Load data
    print("Loading data...")
    _, _, test_loader, test_dataset, class_names = load_data(batch_size=args.batch_size)
    
    if not test_loader:
        print("Failed to load test data. Exiting.")
        return
    
    # Load model from checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = SteelDefectClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Set class names for visualization
    model.class_names = class_names
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Calculate inference time if requested
    inference_time = None
    if args.calc_inference_time:
        inference_time = calculate_inference_time(model, test_loader, device)
    
    # Test model
    print("Evaluating model on test set...")
    trainer = pl.Trainer(accelerator='auto', devices=1)
    results = trainer.test(model, test_loader)
    
    # Get all predictions, targets, and probabilities
    all_preds = torch.cat(model.test_predictions).numpy()
    all_targets = torch.cat(model.test_targets).numpy()
    
    # Get prediction probabilities
    all_probs = []
    model.to(device)
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    
    # Calculate ROC curves
    print("Calculating ROC curves...")
    roc_auc = calculate_roc_curves(all_probs, all_targets, class_names, args.results_dir)
    
    # Calculate precision-recall curves
    print("Calculating precision-recall curves...")
    avg_precision = calculate_pr_curves(all_probs, all_targets, class_names, args.results_dir)
    
    # Visualize misclassified samples
    print("Visualizing misclassified samples...")
    visualize_misclassified_samples(
        test_dataset, 
        all_preds, 
        all_targets, 
        all_probs,
        class_names, 
        args.results_dir,
        args.num_samples
    )
    
    # Create and save detailed confusion matrix with percentages
    print("Generating confusion matrix...")
    cm = model.test_confmat.compute().cpu().numpy()
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_percent, 
        annot=True, 
        fmt='.1f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'confusion_matrix_percent.png'))
    plt.close()
    
    # Save per-class metrics
    print("Calculating per-class metrics...")
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Calculate precision, recall, and F1 for each class
    precision = {}
    recall = {}
    f1_score = {}
    
    for i, class_name in enumerate(class_names):
        # Precision: TP / (TP + FP)
        precision[class_name] = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        
        # Recall: TP / (TP + FN)
        recall[class_name] = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        
        # F1 Score: 2 * (precision * recall) / (precision + recall)
        if precision[class_name] + recall[class_name] > 0:
            f1_score[class_name] = 2 * (precision[class_name] * recall[class_name]) / (precision[class_name] + recall[class_name])
        else:
            f1_score[class_name] = 0
    
    # Create a DataFrame with all metrics
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Accuracy': per_class_acc,
        'Precision': [precision[c] for c in class_names],
        'Recall': [recall[c] for c in class_names],
        'F1-Score': [f1_score[c] for c in class_names],
        'ROC-AUC': [roc_auc[i] for i in range(len(class_names))],
        'PR-AUC': [avg_precision[i] for i in range(len(class_names))],
        'Support': cm.sum(axis=1)
    })
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(args.results_dir, 'per_class_metrics.csv'), index=False)
    
    # Plot per-class metrics
    plt.figure(figsize=(14, 8))
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Reshape data for plotting
    plot_data = []
    for metric in metrics_to_plot:
        for i, class_name in enumerate(class_names):
            plot_data.append({
                'Class': class_name,
                'Metric': metric,
                'Value': metrics_df.loc[i, metric]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create the plot
    sns.barplot(x='Class', y='Value', hue='Metric', data=plot_df)
    plt.title('Per-class Performance Metrics')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'per_class_metrics.png'))
    plt.close()
    
    # Create a summary of results
    summary = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'checkpoint': checkpoint_path,
        'overall_accuracy': results[0]['test_acc'],
        'overall_f1': results[0]['test_f1'],
        'overall_precision': results[0]['test_precision'],
        'overall_recall': results[0]['test_recall'],
        'per_class_metrics': {
            class_name: {
                'accuracy': float(per_class_acc[i]),
                'precision': float(precision[class_name]),
                'recall': float(recall[class_name]),
                'f1_score': float(f1_score[class_name]),
                'roc_auc': float(roc_auc[i]),
                'pr_auc': float(avg_precision[i]),
                'support': int(cm.sum(axis=1)[i])
            } for i, class_name in enumerate(class_names)
        }
    }
    
    if inference_time is not None:
        summary['inference_time_ms'] = inference_time * 1000
    
    # Save summary as JSON
    with open(os.path.join(args.results_dir, 'results_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Save summary as Markdown
    with open(os.path.join(args.results_dir, 'results_summary.md'), 'w') as f:
        f.write(f"# Steel Defect Detection - Final Test Evaluation\n\n")
        f.write(f"**Date:** {summary['timestamp']}\n")
        f.write(f"**Checkpoint:** {summary['checkpoint']}\n\n")
        
        f.write(f"## Overall Metrics\n\n")
        f.write(f"- **Accuracy:** {summary['overall_accuracy']:.4f}\n")
        f.write(f"- **F1 Score:** {summary['overall_f1']:.4f}\n")
        f.write(f"- **Precision:** {summary['overall_precision']:.4f}\n")
        f.write(f"- **Recall:** {summary['overall_recall']:.4f}\n")
        
        if inference_time is not None:
            f.write(f"- **Average Inference Time:** {summary['inference_time_ms']:.2f} ms per image\n")
        
        f.write(f"\n## Per-class Metrics\n\n")
        f.write(f"| Class | Accuracy | Precision | Recall | F1 Score | ROC AUC | PR AUC | Support |\n")
        f.write(f"|-------|----------|-----------|--------|----------|---------|--------|--------|\n")
        
        for class_name in class_names:
            metrics = summary['per_class_metrics'][class_name]
            f.write(f"| {class_name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | ")
            f.write(f"{metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {metrics['roc_auc']:.4f} | ")
            f.write(f"{metrics['pr_auc']:.4f} | {metrics['support']} |\n")
    
    print(f"\nEvaluation complete. Results saved to {args.results_dir}")
    print(f"Overall test accuracy: {results[0]['test_acc']:.4f}")
    print(f"Overall F1 score: {results[0]['test_f1']:.4f}")

if __name__ == '__main__':
    main()