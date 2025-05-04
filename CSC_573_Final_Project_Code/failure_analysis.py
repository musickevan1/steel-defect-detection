# failure_analysis.py
# Implements failure case analysis and visualization for CNN and traditional ML models

import os
import json
import csv
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import glob

# Import from our other scripts
from data_loader import load_data
from lightning_model import SteelDefectClassifier

# --- Configuration ---
RESULTS_DIR = 'results'
FAILURE_ANALYSIS_DIR = os.path.join(RESULTS_DIR, 'failure_analysis')
CNN_FAILURES_DIR = os.path.join(FAILURE_ANALYSIS_DIR, 'cnn')
ML_FAILURES_DIR = os.path.join(FAILURE_ANALYSIS_DIR, 'ml')
MISCLASSIFIED_SAMPLES_DIR = os.path.join(FAILURE_ANALYSIS_DIR, 'misclassified_samples')

# Create directories
os.makedirs(FAILURE_ANALYSIS_DIR, exist_ok=True)
os.makedirs(CNN_FAILURES_DIR, exist_ok=True)
os.makedirs(ML_FAILURES_DIR, exist_ok=True)
os.makedirs(MISCLASSIFIED_SAMPLES_DIR, exist_ok=True)

# --- CNN Failure Analysis ---

def analyze_cnn_failures(checkpoint_path, batch_size=32):
    """
    Analyze and visualize CNN misclassifications.
    
    Args:
        checkpoint_path: Path to the CNN model checkpoint
        batch_size: Batch size for evaluation
    """
    print("=== Analyzing CNN Failures ===")
    
    # Load data
    _, _, test_loader, test_dataset, class_names = load_data(batch_size=batch_size)
    
    if not test_loader:
        print("Failed to load test data. Exiting.")
        return None
    
    # Load model from checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = SteelDefectClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Set class names for visualization
    model.class_names = class_names
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # Get image paths from test dataset
    image_paths = [path for path, _ in test_dataset.dataset.samples]
    image_paths = [image_paths[idx] for idx in test_dataset.indices]
    
    # Collect misclassified samples
    misclassified = []
    
    # Process test data
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Find misclassified samples in this batch
            for i in range(len(targets)):
                if preds[i] != targets[i]:
                    sample_idx = batch_idx * test_loader.batch_size + i
                    if sample_idx < len(image_paths):
                        true_label = class_names[targets[i].item()]
                        pred_label = class_names[preds[i].item()]
                        confidence = probs[i, preds[i]].item()
                        
                        misclassified.append({
                            'image_path': image_paths[sample_idx],
                            'true_label': true_label,
                            'pred_label': pred_label,
                            'confidence': confidence,
                            'model_type': 'cnn'
                        })
    
    print(f"Found {len(misclassified)} misclassified samples out of {len(test_dataset)} test samples.")
    
    # Save misclassified samples
    save_misclassified_samples(misclassified, CNN_FAILURES_DIR)
    
    # Create confusion grid
    create_confusion_grid(misclassified, class_names, 'cnn')
    
    return misclassified

def analyze_ml_failures(results_dir=os.path.join(RESULTS_DIR, 'traditional_ml')):
    """
    Analyze and visualize traditional ML misclassifications.
    
    Args:
        results_dir: Directory containing traditional ML results
    """
    print("=== Analyzing Traditional ML Failures ===")
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} not found. Exiting.")
        return None
    
    # Look for results files
    hog_svm_results_file = os.path.join(results_dir, 'hog_svm_results.json')
    lbp_rf_results_file = os.path.join(results_dir, 'lbp_rf_results.json')
    
    misclassified = []
    
    # Load test data to get image paths
    _, _, test_loader, test_dataset, class_names = load_data()
    
    if not test_loader:
        print("Failed to load test data. Exiting.")
        return None
    
    # Get image paths from test dataset
    image_paths = [path for path, _ in test_dataset.dataset.samples]
    image_paths = [image_paths[idx] for idx in test_dataset.indices]
    
    # Process HOG+SVM results if available
    if os.path.exists(hog_svm_results_file):
        print("Processing HOG+SVM results...")
        
        # Load predictions from a CSV file if it exists
        predictions_file = os.path.join(results_dir, 'hog_svm_predictions.csv')
        if os.path.exists(predictions_file):
            df = pd.read_csv(predictions_file)
            
            for _, row in df.iterrows():
                if row['true_label'] != row['pred_label']:
                    sample_idx = int(row['sample_idx'])
                    if sample_idx < len(image_paths):
                        misclassified.append({
                            'image_path': image_paths[sample_idx],
                            'true_label': row['true_label'],
                            'pred_label': row['pred_label'],
                            'confidence': row['confidence'] if 'confidence' in row else 0.0,
                            'model_type': 'hog_svm'
                        })
        else:
            print(f"Predictions file {predictions_file} not found.")
    
    # Process LBP+RF results if available
    if os.path.exists(lbp_rf_results_file):
        print("Processing LBP+RF results...")
        
        # Load predictions from a CSV file if it exists
        predictions_file = os.path.join(results_dir, 'lbp_rf_predictions.csv')
        if os.path.exists(predictions_file):
            df = pd.read_csv(predictions_file)
            
            for _, row in df.iterrows():
                if row['true_label'] != row['pred_label']:
                    sample_idx = int(row['sample_idx'])
                    if sample_idx < len(image_paths):
                        misclassified.append({
                            'image_path': image_paths[sample_idx],
                            'true_label': row['true_label'],
                            'pred_label': row['pred_label'],
                            'confidence': row['confidence'] if 'confidence' in row else 0.0,
                            'model_type': 'lbp_rf'
                        })
        else:
            print(f"Predictions file {predictions_file} not found.")
    
    # If no predictions files found, try to extract from the final test evaluation
    if not misclassified:
        print("No prediction files found. Attempting to extract from misclassified samples directory...")
        
        # Look for misclassified sample images
        for model_type in ['hog_svm', 'lbp_rf']:
            model_dir = os.path.join(results_dir, f'{model_type}_misclassified')
            if os.path.exists(model_dir):
                for img_file in os.listdir(model_dir):
                    if img_file.endswith('.png') or img_file.endswith('.jpg'):
                        # Parse filename to extract information
                        parts = img_file.split('_')
                        if len(parts) >= 4:
                            true_label = parts[0]
                            pred_label = parts[2]
                            
                            # Try to extract confidence if available
                            confidence = 0.0
                            for i, part in enumerate(parts):
                                if part == 'confidence' and i+1 < len(parts):
                                    try:
                                        confidence = float(parts[i+1].split('.')[0] + '.' + parts[i+1].split('.')[1])
                                    except:
                                        pass
                            
                            # Find the original image path
                            original_path = None
                            for path in image_paths:
                                if true_label in path and os.path.basename(path).split('.')[0] in img_file:
                                    original_path = path
                                    break
                            
                            if original_path:
                                misclassified.append({
                                    'image_path': original_path,
                                    'true_label': true_label,
                                    'pred_label': pred_label,
                                    'confidence': confidence,
                                    'model_type': model_type
                                })
    
    print(f"Found {len(misclassified)} misclassified samples from traditional ML models.")
    
    # Save misclassified samples
    save_misclassified_samples(misclassified, ML_FAILURES_DIR)
    
    # Create confusion grid
    create_confusion_grid(misclassified, class_names, 'ml')
    
    return misclassified

def save_misclassified_samples(misclassified, output_dir):
    """
    Save misclassified samples with appropriate filenames.
    
    Args:
        misclassified: List of misclassified sample dictionaries
        output_dir: Directory to save samples
    """
    print(f"Saving misclassified samples to {output_dir}...")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for each true class
    true_class_dirs = {}
    for sample in misclassified:
        true_label = sample['true_label']
        if true_label not in true_class_dirs:
            true_class_dir = os.path.join(output_dir, f"true_{true_label}")
            os.makedirs(true_class_dir, exist_ok=True)
            true_class_dirs[true_label] = true_class_dir
    
    # Create subdirectories for each predicted class
    pred_class_dirs = {}
    for sample in misclassified:
        pred_label = sample['pred_label']
        if pred_label not in pred_class_dirs:
            pred_class_dir = os.path.join(output_dir, f"pred_{pred_label}")
            os.makedirs(pred_class_dir, exist_ok=True)
            pred_class_dirs[pred_label] = pred_class_dir
    
    # Save each misclassified sample
    for sample in misclassified:
        image_path = sample['image_path']
        true_label = sample['true_label']
        pred_label = sample['pred_label']
        confidence = sample['confidence']
        model_type = sample['model_type']
        
        # Create filename
        base_filename = f"{true_label}_pred_{pred_label}_confidence_{confidence:.2f}.png"
        
        # Copy to true class directory
        true_class_output_path = os.path.join(true_class_dirs[true_label], base_filename)
        
        # Copy to predicted class directory
        pred_class_output_path = os.path.join(pred_class_dirs[pred_label], base_filename)
        
        # Also save to the misclassified samples directory
        model_type_dir = os.path.join(MISCLASSIFIED_SAMPLES_DIR, model_type)
        os.makedirs(model_type_dir, exist_ok=True)
        misclassified_output_path = os.path.join(model_type_dir, base_filename)
        
        try:
            # Open and save the image
            img = Image.open(image_path)
            img.save(true_class_output_path)
            img.save(pred_class_output_path)
            img.save(misclassified_output_path)
        except Exception as e:
            print(f"Error saving image {image_path}: {e}")

def create_confusion_grid(misclassified, class_names, model_type, grid_size=3):
    """
    Create a grid of misclassified samples for each class.
    
    Args:
        misclassified: List of misclassified sample dictionaries
        class_names: List of class names
        model_type: 'cnn' or 'ml'
        grid_size: Size of the grid (grid_size x grid_size)
    """
    print(f"Creating confusion grid for {model_type}...")
    
    # Group misclassified samples by true class
    samples_by_true_class = {}
    for sample in misclassified:
        true_label = sample['true_label']
        if true_label not in samples_by_true_class:
            samples_by_true_class[true_label] = []
        samples_by_true_class[true_label].append(sample)
    
    # Create a figure for each class
    for true_label, samples in samples_by_true_class.items():
        # Sort by confidence (highest first)
        samples.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Take up to grid_size^2 samples
        samples = samples[:grid_size*grid_size]
        
        # Create figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*3, grid_size*3))
        fig.suptitle(f"{model_type.upper()}: Misclassified {true_label} Samples", fontsize=16)
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Plot each sample
        for i, sample in enumerate(samples):
            if i < len(axes):
                try:
                    img = Image.open(sample['image_path'])
                    axes[i].imshow(img)
                    axes[i].set_title(f"Pred: {sample['pred_label']}\nConf: {sample['confidence']:.2f}")
                    axes[i].axis('off')
                except Exception as e:
                    print(f"Error plotting image {sample['image_path']}: {e}")
                    axes[i].text(0.5, 0.5, "Error loading image", ha='center', va='center')
                    axes[i].axis('off')
        
        # Hide empty subplots
        for i in range(len(samples), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save figure
        output_path = os.path.join(FAILURE_ANALYSIS_DIR, f"{model_type}_{true_label}_confusion_grid.png")
        plt.savefig(output_path)
        plt.close()
    
    # Create a combined confusion grid for all classes
    create_combined_confusion_grid(misclassified, class_names, model_type)

def create_combined_confusion_grid(misclassified, class_names, model_type):
    """
    Create a combined grid of misclassified samples for all classes.
    
    Args:
        misclassified: List of misclassified sample dictionaries
        class_names: List of class names
        model_type: 'cnn' or 'ml'
    """
    print(f"Creating combined confusion grid for {model_type}...")
    
    # Count misclassifications for each true-predicted class pair
    confusion_counts = {}
    for sample in misclassified:
        true_label = sample['true_label']
        pred_label = sample['pred_label']
        key = (true_label, pred_label)
        if key not in confusion_counts:
            confusion_counts[key] = []
        confusion_counts[key].append(sample)
    
    # Sort confusion pairs by count (most confused first)
    sorted_pairs = sorted(confusion_counts.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Take top 9 confusion pairs
    top_pairs = sorted_pairs[:9]
    
    if not top_pairs:
        print("No misclassified samples found.")
        return
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f"{model_type.upper()}: Top Confusion Pairs", fontsize=16)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Plot each confusion pair
    for i, ((true_label, pred_label), samples) in enumerate(top_pairs):
        if i < len(axes):
            # Sort by confidence (highest first)
            samples.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Take the highest confidence sample
            sample = samples[0]
            
            try:
                img = Image.open(sample['image_path'])
                axes[i].imshow(img)
                axes[i].set_title(f"True: {true_label}\nPred: {pred_label}\nCount: {len(samples)}")
                axes[i].axis('off')
            except Exception as e:
                print(f"Error plotting image {sample['image_path']}: {e}")
                axes[i].text(0.5, 0.5, "Error loading image", ha='center', va='center')
                axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(len(top_pairs), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure
    output_path = os.path.join(FAILURE_ANALYSIS_DIR, f"{model_type}_confusion_grid.png")
    plt.savefig(output_path)
    plt.close()

def generate_confusion_summary(cnn_misclassified, ml_misclassified, class_names):
    """
    Generate a summary of confusion rates for CNN and traditional ML models.
    
    Args:
        cnn_misclassified: List of CNN misclassified sample dictionaries
        ml_misclassified: List of traditional ML misclassified sample dictionaries
        class_names: List of class names
    """
    print("Generating confusion summary...")
    
    # Create a DataFrame to store confusion rates
    confusion_data = []
    
    # Process CNN misclassifications
    if cnn_misclassified:
        # Count misclassifications for each true-predicted class pair
        confusion_counts = {}
        for sample in cnn_misclassified:
            true_label = sample['true_label']
            pred_label = sample['pred_label']
            key = (true_label, pred_label)
            if key not in confusion_counts:
                confusion_counts[key] = 0
            confusion_counts[key] += 1
        
        # Calculate total samples for each true class
        true_class_counts = {}
        for sample in cnn_misclassified:
            true_label = sample['true_label']
            if true_label not in true_class_counts:
                true_class_counts[true_label] = 0
            true_class_counts[true_label] += 1
        
        # Add to confusion data
        for (true_label, pred_label), count in confusion_counts.items():
            confusion_data.append({
                'model_type': 'CNN',
                'true_label': true_label,
                'pred_label': pred_label,
                'count': count,
                'confusion_rate': count / true_class_counts[true_label] if true_label in true_class_counts else 0
            })
    
    # Process ML misclassifications
    if ml_misclassified:
        # Group by model type
        ml_by_type = {}
        for sample in ml_misclassified:
            model_type = sample['model_type']
            if model_type not in ml_by_type:
                ml_by_type[model_type] = []
            ml_by_type[model_type].append(sample)
        
        # Process each ML model type
        for model_type, samples in ml_by_type.items():
            # Count misclassifications for each true-predicted class pair
            confusion_counts = {}
            for sample in samples:
                true_label = sample['true_label']
                pred_label = sample['pred_label']
                key = (true_label, pred_label)
                if key not in confusion_counts:
                    confusion_counts[key] = 0
                confusion_counts[key] += 1
            
            # Calculate total samples for each true class
            true_class_counts = {}
            for sample in samples:
                true_label = sample['true_label']
                if true_label not in true_class_counts:
                    true_class_counts[true_label] = 0
                true_class_counts[true_label] += 1
            
            # Add to confusion data
            for (true_label, pred_label), count in confusion_counts.items():
                confusion_data.append({
                    'model_type': model_type.upper(),
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'count': count,
                    'confusion_rate': count / true_class_counts[true_label] if true_label in true_class_counts else 0
                })
    
    # Create DataFrame
    confusion_df = pd.DataFrame(confusion_data)
    
    # Sort by confusion rate (highest first)
    confusion_df = confusion_df.sort_values('confusion_rate', ascending=False)
    
    # Save as CSV
    confusion_df.to_csv(os.path.join(FAILURE_ANALYSIS_DIR, 'confusion_summary.csv'), index=False)
    
    # Generate Markdown summary
    with open(os.path.join(FAILURE_ANALYSIS_DIR, 'summary.md'), 'w') as f:
        f.write("# Steel Defect Detection - Failure Analysis\n\n")
        
        f.write("## Most Confused Classes\n\n")
        f.write("| Model | True Label | Predicted Label | Count | Confusion Rate |\n")
        f.write("|-------|-----------|-----------------|-------|----------------|\n")
        
        # Write top 10 confusion pairs
        for _, row in confusion_df.head(10).iterrows():
            f.write(f"| {row['model_type']} | {row['true_label']} | {row['pred_label']} | ")
            f.write(f"{row['count']} | {row['confusion_rate']:.4f} |\n")
        
        f.write("\n## Per-class Confusion Rates\n\n")
        
        # Group by model type and true label
        for model_type in confusion_df['model_type'].unique():
            f.write(f"### {model_type}\n\n")
            f.write("| True Label | Most Confused With | Count | Confusion Rate |\n")
            f.write("|-----------|-------------------|-------|----------------|\n")
            
            model_df = confusion_df[confusion_df['model_type'] == model_type]
            
            # For each true class, find the most confused predicted class
            for true_label in class_names:
                class_df = model_df[model_df['true_label'] == true_label]
                if not class_df.empty:
                    # Get the row with the highest confusion rate
                    top_row = class_df.iloc[0]
                    f.write(f"| {true_label} | {top_row['pred_label']} | ")
                    f.write(f"{top_row['count']} | {top_row['confusion_rate']:.4f} |\n")
                else:
                    f.write(f"| {true_label} | N/A | 0 | 0.0000 |\n")
            
            f.write("\n")
        
        f.write("\n## Visualization\n\n")
        f.write("Confusion grids for each model type are available in the failure_analysis directory.\n\n")
        
        f.write("### CNN Confusion Grid\n\n")
        f.write("![CNN Confusion Grid](cnn_confusion_grid.png)\n\n")
        
        f.write("### Traditional ML Confusion Grid\n\n")
        f.write("![ML Confusion Grid](ml_confusion_grid.png)\n\n")
        
        f.write("\n## Misclassified Samples\n\n")
        f.write("Misclassified samples are available in the following directories:\n\n")
        f.write("- CNN: `results/failure_analysis/cnn/`\n")
        f.write("- Traditional ML: `results/failure_analysis/ml/`\n")
        f.write("- Combined: `results/failure_analysis/misclassified_samples/`\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Failure Analysis for Steel Defect Detection')
    parser.add_argument('--cnn_checkpoint', type=str, help='Path to CNN model checkpoint')
    parser.add_argument('--ml_results_dir', type=str, default=os.path.join(RESULTS_DIR, 'traditional_ml'),
                        help='Directory containing traditional ML results')
    
    args = parser.parse_args()
    
    # Find CNN checkpoint if not provided
    checkpoint_path = args.cnn_checkpoint
    if not checkpoint_path:
        # Look for checkpoints in saved_models directory
        if os.path.exists('saved_models'):
            checkpoints = [f for f in os.listdir('saved_models') if f.endswith('.ckpt')]
            if checkpoints:
                # Sort by validation accuracy if possible
                try:
                    checkpoints.sort(key=lambda x: float(x.split('val_acc=')[1].split('.ckpt')[0]), reverse=True)
                except:
                    # If not in expected format, just use the most recent
                    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join('saved_models', x)), reverse=True)
                
                checkpoint_path = os.path.join('saved_models', checkpoints[0])
                print(f"Found checkpoint: {checkpoint_path}")
        
        # If still not found, look in results directory
        if not checkpoint_path:
            checkpoint_pattern = os.path.join(RESULTS_DIR, 'cnn', '**', '*.ckpt')
            checkpoints = glob.glob(checkpoint_pattern, recursive=True)
            if checkpoints:
                # Sort by modification time (most recent first)
                checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                checkpoint_path = checkpoints[0]
                print(f"Found checkpoint: {checkpoint_path}")
    
    # Analyze CNN failures if checkpoint is available
    cnn_misclassified = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        cnn_misclassified = analyze_cnn_failures(checkpoint_path)
    else:
        print("No CNN checkpoint found. Skipping CNN failure analysis.")
    
    # Analyze traditional ML failures
    ml_misclassified = analyze_ml_failures(args.ml_results_dir)
    
    # Load class names
    _, _, _, _, class_names = load_data()
    
    # Generate confusion summary
    generate_confusion_summary(cnn_misclassified, ml_misclassified, class_names)
    
    print(f"Failure analysis complete. Results saved to {FAILURE_ANALYSIS_DIR}")

if __name__ == '__main__':
    main()