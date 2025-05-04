# evaluate_lightning.py
# Evaluates a trained PyTorch Lightning model on the test set

import os
import argparse
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from data_loader import load_data
from lightning_model import SteelDefectClassifier

# --- Parse Arguments ---
parser = argparse.ArgumentParser(description='Evaluate CNN Model with PyTorch Lightning')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
parser.add_argument('--results_dir', type=str, default='results/cnn', help='Directory to save results')
parser.add_argument('--visualize_misclassified', type=bool, default=True, help='Whether to visualize misclassified samples')
parser.add_argument('--num_samples', type=int, default=10, help='Number of misclassified samples to visualize per class')
args = parser.parse_args()

def visualize_misclassified_samples(test_dataset, all_preds, all_targets, class_names, results_dir, num_samples=10):
    """
    Visualize misclassified samples from the test set.
    
    Args:
        test_dataset: The test dataset
        all_preds: Predicted labels
        all_targets: True labels
        class_names: Names of the classes
        results_dir: Directory to save results
        num_samples: Number of misclassified samples to visualize per class
    """
    # Get image paths from test dataset
    image_paths = [path for path, _ in test_dataset.dataset.samples]
    image_paths = [image_paths[idx] for idx in test_dataset.indices]
    
    # Find misclassified samples for each class
    misclassified = {}
    for i in range(len(all_targets)):
        if all_preds[i] != all_targets[i]:
            true_class = class_names[all_targets[i]]
            pred_class = class_names[all_preds[i]]
            if true_class not in misclassified:
                misclassified[true_class] = []
            
            if len(misclassified[true_class]) < num_samples:
                misclassified[true_class].append((image_paths[i], pred_class))
    
    # Visualize misclassified samples for each class
    for true_class, samples in misclassified.items():
        if not samples:
            continue
            
        n_samples = len(samples)
        fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 3, 3))
        
        # Handle case with only one sample
        if n_samples == 1:
            axes = [axes]
            
        for i, (img_path, pred_class) in enumerate(samples):
            img = Image.open(img_path).convert('RGB')
            axes[i].imshow(img)
            axes[i].set_title(f"Pred: {pred_class}")
            axes[i].axis('off')
            
        plt.suptitle(f"Misclassified {true_class} Samples")
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(results_dir, f"misclassified_{true_class}.png")
        plt.savefig(save_path)
        plt.close()
        
        print(f"Saved misclassified {true_class} samples to {save_path}")

def main():
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    _, _, test_loader, test_dataset, class_names = load_data(batch_size=args.batch_size)
    
    if not test_loader:
        print("Failed to load test data. Exiting.")
        exit()
    
    # Load model from checkpoint
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = SteelDefectClassifier.load_from_checkpoint(args.checkpoint)
    model.eval()
    
    # Set class names for visualization
    model.class_names = class_names
    
    # Test model
    print("Testing model...")
    trainer = pl.Trainer(accelerator='auto', devices=1)
    results = trainer.test(model, test_loader)
    
    # Get all predictions and targets
    all_preds = torch.cat(model.test_predictions).numpy()
    all_targets = torch.cat(model.test_targets).numpy()
    
    # Visualize misclassified samples if requested
    if args.visualize_misclassified:
        print("Visualizing misclassified samples...")
        visualize_misclassified_samples(
            test_dataset, 
            all_preds, 
            all_targets, 
            class_names, 
            args.results_dir,
            args.num_samples
        )
    
    # Create and save detailed confusion matrix with percentages
    cm = confusion_matrix(all_targets, all_preds)
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
    
    # Save per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    acc_df = pd.DataFrame({
        'Class': class_names,
        'Accuracy': per_class_acc,
        'Support': cm.sum(axis=1)
    })
    acc_df.to_csv(os.path.join(args.results_dir, 'per_class_accuracy.csv'), index=False)
    
    # Plot per-class accuracy
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Class', y='Accuracy', data=acc_df)
    plt.title('Per-class Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'per_class_accuracy.png'))
    plt.close()
    
    print(f"\nEvaluation complete. Results saved to {args.results_dir}")
    print(f"Test accuracy: {results[0]['test_acc']:.4f}")

if __name__ == '__main__':
    main()