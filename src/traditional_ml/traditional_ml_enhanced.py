# traditional_ml_enhanced.py
# Enhanced implementation of traditional ML models (HOG+SVM, LBP+RF)
# with proper evaluation, visualization, and result saving.

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from PIL import Image
import joblib

from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Import from our other scripts
from data_loader import load_data, get_image_paths_and_labels, DATASET_PATH, IMG_SIZE
from config import BATCH_SIZES, LEARNING_RATES

# --- Configuration ---
RESULTS_DIR = os.path.join('results', 'traditional_ml')
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Feature Extraction ---

def extract_hog_features(image, size=(128, 128), pixels_per_cell=(16, 16), cells_per_block=(2, 2)):
    """
    Extracts HOG features from a single image.
    
    Args:
        image: PIL Image
        size: Size to resize the image to
        pixels_per_cell: Size of a cell in pixels
        cells_per_block: Number of cells in each block
        
    Returns:
        numpy array: HOG features
    """
    # Ensure image is grayscale and resized appropriately
    if isinstance(image, np.ndarray):
        # If it's a NumPy array, convert to PIL Image first
        # Assuming the array is in HWC format and needs conversion to RGB if it has 3 channels
        if image.ndim == 3 and image.shape[2] == 3:
             # Convert BGR (common in OpenCV) or RGB to PIL Image
             # Check if it might be BGR (if loaded by OpenCV) - needs careful handling
             # Assuming RGB for now based on PyTorch tensor permute
             image_pil = Image.fromarray(image.astype(np.uint8), 'RGB')
        elif image.ndim == 2: # Grayscale array
             image_pil = Image.fromarray(image.astype(np.uint8), 'L')
        else: # Single channel color? Or unexpected format
             # Fallback: try converting first channel if 3D
             if image.ndim == 3:
                  image_pil = Image.fromarray(image[:,:,0].astype(np.uint8), 'L')
             else:
                  raise ValueError("Unsupported NumPy array format for HOG feature extraction.")
    elif isinstance(image, Image.Image):
        image_pil = image
    else:
        raise TypeError("Input must be a PIL Image or a NumPy array.")

    image_gray_pil = image_pil.convert('L').resize(size)
    image_gray_np = np.array(image_gray_pil)

    features = hog(
        image_gray_np,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block, 
        visualize=False
    )
    return features

def extract_lbp_features(image, size=(224, 224), radius=3, n_points=24):
    """
    Extracts LBP features from a single image.
    
    Args:
        image: PIL Image
        size: Size to resize the image to
        radius: Radius of circle for LBP
        n_points: Number of points in the circle
        
    Returns:
        numpy array: LBP histogram features
    """
    if isinstance(image, np.ndarray):
        # If it's a NumPy array, convert to PIL Image first
        # Assuming the array is in HWC format and needs conversion to RGB if it has 3 channels
        if image.ndim == 3 and image.shape[2] == 3:
             # Convert BGR (common in OpenCV) or RGB to PIL Image
             # Check if it might be BGR (if loaded by OpenCV) - needs careful handling
             # Assuming RGB for now based on PyTorch tensor permute
             image_pil = Image.fromarray(image.astype(np.uint8), 'RGB')
        elif image.ndim == 2: # Grayscale array
             image_pil = Image.fromarray(image.astype(np.uint8), 'L')
        else: # Single channel color? Or unexpected format
             # Fallback: try converting first channel if 3D
             if image.ndim == 3:
                  image_pil = Image.fromarray(image[:,:,0].astype(np.uint8), 'L')
             else:
                  raise ValueError("Unsupported NumPy array format for LBP feature extraction.")
    elif isinstance(image, Image.Image):
        image_pil = image
    else:
        raise TypeError("Input must be a PIL Image or a NumPy array.")

    image_gray_pil = image_pil.convert('L').resize(size)
    image_gray_np = np.array(image_gray_pil)

    lbp = local_binary_pattern(
        image_gray_np,
        n_points, 
        radius, 
        method='uniform'
    )
    hist, _ = np.histogram(
        lbp.ravel(), 
        bins=np.arange(0, n_points + 3), 
        range=(0, n_points + 2)
    )
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize
    return hist

def extract_features(image_paths, feature_type='hog', size=None):
    """
    Extracts specified features for a list of image paths.
    
    Args:
        image_paths: List of image paths
        feature_type: 'hog' or 'lbp'
        size: Size to resize images to (optional)
        
    Returns:
        numpy array: Features for all images
    """
    features_list = []
    print(f"Extracting {feature_type.upper()} features...")
    
    # Set default size based on feature type
    if size is None:
        size = (128, 128) if feature_type == 'hog' else (224, 224)
    
    start_time = time.time()
    for i, img_path in enumerate(image_paths):
        try:
            with Image.open(img_path) as img:
                if feature_type == 'hog':
                    features = extract_hog_features(img, size=size)
                elif feature_type == 'lbp':
                    features = extract_lbp_features(img, size=size)
                else:
                    raise ValueError("Unsupported feature type")
                features_list.append(features)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{len(image_paths)} images...")
        except Exception as e:
            print(f"Warning: Could not process image {img_path}: {e}")
            # Skip problematic images
            pass
    
    end_time = time.time()
    print(f"Feature extraction took {end_time - start_time:.2f} seconds.")
    
    return np.array(features_list)

# --- Model Training and Evaluation ---

def train_evaluate_model(
    X_train, y_train, X_test, y_test, 
    model_type='svm', 
    class_names=None,
    use_grid_search=True,
    use_cross_val=True,
    save_model=True
):
    """
    Trains and evaluates a traditional ML model with optional grid search and cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_type: 'svm' or 'rf'
        class_names: List of class names
        use_grid_search: Whether to use grid search for hyperparameter tuning
        use_cross_val: Whether to use cross-validation
        save_model: Whether to save the trained model
        
    Returns:
        dict: Results including model, metrics, and inference time
    """
    print(f"\n--- Training and Evaluating {model_type.upper()} ---")
    
    # Create a pipeline with scaling and model
    if model_type == 'svm':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(probability=True, random_state=42))
        ])
        
        # Define parameter grid for grid search
        param_grid = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__gamma': ['scale', 'auto', 0.01, 0.1],
            'classifier__kernel': ['rbf', 'linear']
        }
    elif model_type == 'rf':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
        ])
        
        # Define parameter grid for grid search
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10]
        }
    else:
        raise ValueError("Unsupported model type")
    
    # Use grid search if requested
    if use_grid_search:
        print(f"Performing grid search for {model_type.upper()}...")
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5, 
            scoring='accuracy', 
            n_jobs=-1,
            verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Use the best model
        model = grid_search.best_estimator_
    else:
        # Train without grid search
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time
        model = pipeline
    
    # Cross-validation if requested
    if use_cross_val:
        print("Performing cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    
    # Measure inference time
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    avg_inference_time = inference_time / len(X_test)
    
    # Get prediction probabilities
    y_proba = model.predict_proba(X_test)
    
    # Save predictions for failure analysis
    if class_names:
        predictions_dir = os.path.join(RESULTS_DIR, f'{model_type}_predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Create a DataFrame with predictions
        predictions_df = pd.DataFrame({
            'sample_idx': range(len(y_test)),
            'true_label': [class_names[y] for y in y_test],
            'pred_label': [class_names[y] for y in y_pred],
            'confidence': [y_proba[i, y_pred[i]] for i in range(len(y_pred))]
        })
        
        # Save to CSV
        predictions_df.to_csv(os.path.join(RESULTS_DIR, f'{model_type}_predictions.csv'), index=False)
        
        # Save misclassified samples
        misclassified_dir = os.path.join(RESULTS_DIR, f'{model_type}_misclassified')
        os.makedirs(misclassified_dir, exist_ok=True)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Generate classification report
    if class_names:
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        report_str = classification_report(y_test, y_pred, target_names=class_names)
    else:
        report = classification_report(y_test, y_pred, output_dict=True)
        report_str = classification_report(y_test, y_pred)
    
    # Print results
    print(f"\n{model_type.upper()} Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Average inference time per image: {avg_inference_time*1000:.2f} ms")
    print("\nClassification Report:")
    print(report_str)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Save model if requested
    if save_model:
        model_dir = os.path.join(RESULTS_DIR, 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = os.path.join(model_dir, f'{model_type}_{timestamp}.joblib')
        joblib.dump(model, model_filename)
        print(f"Model saved to {model_filename}")
    
    # Collect results
    results = {
        'model': model,
        'model_type': model_type,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_test': y_test,
        'y_proba': y_proba,
        'train_time': train_time,
        'inference_time': inference_time,
        'avg_inference_time': avg_inference_time,
    }
    
    if use_grid_search:
        results['best_params'] = grid_search.best_params_
    
    if use_cross_val:
        results['cv_scores'] = cv_scores
        results['cv_mean'] = cv_scores.mean()
        results['cv_std'] = cv_scores.std()
    
    return results

def visualize_results(results, class_names, feature_type, save_dir):
    """
    Visualizes and saves the results of a traditional ML model.
    
    Args:
        results: Results dictionary from train_evaluate_model
        class_names: List of class names
        feature_type: 'hog' or 'lbp'
        save_dir: Directory to save visualizations
    """
    model_type = results['model_type']
    
    # Create directory for visualizations
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = results['confusion_matrix']
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
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
    plt.title(f'Confusion Matrix (%) - {feature_type.upper()} + {model_type.upper()}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{feature_type}_{model_type}_confusion_matrix.png'))
    plt.close()
    
    # Plot per-class metrics
    plt.figure(figsize=(12, 6))
    
    # Extract per-class metrics from classification report
    classes = []
    precision = []
    recall = []
    f1 = []
    
    for cls in class_names:
        if cls in results['report']:
            classes.append(cls)
            precision.append(results['report'][cls]['precision'])
            recall.append(results['report'][cls]['recall'])
            f1.append(results['report'][cls]['f1-score'])
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-score')
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title(f'Per-class Performance - {feature_type.upper()} + {model_type.upper()}')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{feature_type}_{model_type}_per_class_metrics.png'))
    plt.close()
    
    # Save results as JSON
    results_to_save = {
        'model_type': model_type,
        'feature_type': feature_type,
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1'],
        'train_time': results['train_time'],
        'inference_time_ms': results['avg_inference_time'] * 1000,
        'classification_report': results['report'],
        'confusion_matrix': cm.tolist(),
    }
    
    if 'best_params' in results:
        results_to_save['best_params'] = results['best_params']
    
    if 'cv_scores' in results:
        results_to_save['cv_mean'] = results['cv_mean']
        results_to_save['cv_std'] = results['cv_std']
    
    with open(os.path.join(save_dir, f'{feature_type}_{model_type}_results.json'), 'w') as f:
        json.dump(results_to_save, f, indent=4)
    
    # Save results as Markdown
    with open(os.path.join(save_dir, f'{feature_type}_{model_type}_results.md'), 'w') as f:
        f.write(f"# {feature_type.upper()} + {model_type.upper()} Results\n\n")
        f.write(f"## Overall Metrics\n\n")
        f.write(f"- **Accuracy:** {results['accuracy']:.4f}\n")
        f.write(f"- **Precision:** {results['precision']:.4f}\n")
        f.write(f"- **Recall:** {results['recall']:.4f}\n")
        f.write(f"- **F1 Score:** {results['f1']:.4f}\n")
        f.write(f"- **Training Time:** {results['train_time']:.2f} seconds\n")
        f.write(f"- **Average Inference Time:** {results['avg_inference_time']*1000:.2f} ms per image\n\n")
        
        if 'best_params' in results:
            f.write(f"## Best Parameters\n\n")
            for param, value in results['best_params'].items():
                f.write(f"- **{param}:** {value}\n")
            f.write("\n")
        
        if 'cv_scores' in results:
            f.write(f"## Cross-Validation\n\n")
            f.write(f"- **Mean CV Score:** {results['cv_mean']:.4f} ± {results['cv_std']:.4f}\n\n")
        
        f.write(f"## Classification Report\n\n")
        f.write("| Class | Precision | Recall | F1-Score | Support |\n")
        f.write("|-------|-----------|--------|----------|--------|\n")
        
        for cls in class_names:
            if cls in results['report']:
                metrics = results['report'][cls]
                f.write(f"| {cls} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | ")
                f.write(f"{metrics['f1-score']:.4f} | {metrics['support']} |\n")
        
        # Add macro and weighted averages
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in results['report']:
                metrics = results['report'][avg_type]
                f.write(f"| {avg_type} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | ")
                f.write(f"{metrics['f1-score']:.4f} | {metrics['support']} |\n")

def run_traditional_ml_pipeline(use_same_test_split=True, use_grid_search=True, use_cross_val=True):
    """
    Runs the complete traditional ML pipeline with HOG+SVM and LBP+RF.
    
    Args:
        use_same_test_split: Whether to use the same test split as the CNN
        use_grid_search: Whether to use grid search for hyperparameter tuning
        use_cross_val: Whether to use cross-validation
    """
    print("=== Starting Enhanced Traditional ML Pipeline ===")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load data
    if use_same_test_split:
        print("Using the same test split as CNN...")
        train_loader, val_loader, test_loader, test_dataset, class_names = load_data()
        
        # Get image paths and labels from data loaders
        train_images = []
        train_labels = []
        
        for images, labels in train_loader:
            for i in range(len(images)):
                train_images.append(images[i])
                train_labels.append(labels[i].item())
        
        test_images = []
        test_labels = []
        
        for images, labels in test_loader:
            for i in range(len(images)):
                test_images.append(images[i])
                test_labels.append(labels[i].item())
        
        print(f"Loaded {len(train_images)} training images and {len(test_images)} test images.")
        
        # Extract features from images
        print("Extracting features from training images...")
        train_hog_features = np.array([extract_hog_features(img.permute(1, 2, 0).numpy()) for img in train_images])
        train_lbp_features = np.array([extract_lbp_features(img.permute(1, 2, 0).numpy()) for img in train_images])
        
        print("Extracting features from test images...")
        test_hog_features = np.array([extract_hog_features(img.permute(1, 2, 0).numpy()) for img in test_images])
        test_lbp_features = np.array([extract_lbp_features(img.permute(1, 2, 0).numpy()) for img in test_images])
        
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)
    else:
        print("Using a different test split for traditional ML...")
        # Load image paths and labels
        train_image_dir = os.path.join(DATASET_PATH, 'train', 'images')
        val_image_dir = os.path.join(DATASET_PATH, 'validation', 'images')
        
        train_paths, train_labels, class_names, _ = get_image_paths_and_labels(train_image_dir)
        val_paths, val_labels, _, _ = get_image_paths_and_labels(val_image_dir)
        
        all_paths = train_paths + val_paths
        all_labels = train_labels + val_labels
        
        if not all_paths:
            print("Error: No image paths found. Exiting.")
            return
        
        print(f"Total images for traditional ML: {len(all_paths)}")
        
        # Extract features
        hog_features = extract_features(all_paths, feature_type='hog')
        lbp_features = extract_features(all_paths, feature_type='lbp')
        
        # Split data
        train_hog_features, test_hog_features, train_labels, test_labels = train_test_split(
            hog_features, all_labels, test_size=0.25, random_state=42, stratify=all_labels
        )
        
        # Use the same train/test split for LBP features
        _, test_lbp_features, _, _ = train_test_split(
            lbp_features, all_labels, test_size=0.25, random_state=42, stratify=all_labels
        )
        
        train_lbp_features = lbp_features[np.isin(np.arange(len(lbp_features)), np.arange(len(all_labels))[~np.isin(np.arange(len(all_labels)), test_labels)])]
    
    # Train and evaluate HOG + SVM
    print("\n=== HOG + SVM ===")
    hog_svm_results = train_evaluate_model(
        train_hog_features, train_labels, 
        test_hog_features, test_labels,
        model_type='svm',
        class_names=class_names,
        use_grid_search=use_grid_search,
        use_cross_val=use_cross_val
    )
    
    # Visualize and save HOG + SVM results
    visualize_results(hog_svm_results, class_names, 'hog', RESULTS_DIR)
    
    # Train and evaluate LBP + Random Forest
    print("\n=== LBP + Random Forest ===")
    lbp_rf_results = train_evaluate_model(
        train_lbp_features, train_labels, 
        test_lbp_features, test_labels,
        model_type='rf',
        class_names=class_names,
        use_grid_search=use_grid_search,
        use_cross_val=use_cross_val
    )
    
    # Visualize and save LBP + RF results
    visualize_results(lbp_rf_results, class_names, 'lbp', RESULTS_DIR)
    
    # Compare results
    print("\n=== Comparison of Traditional ML Models ===")
    print(f"HOG + SVM Accuracy: {hog_svm_results['accuracy']:.4f}")
    print(f"LBP + RF Accuracy: {lbp_rf_results['accuracy']:.4f}")
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    
    models = ['HOG + SVM', 'LBP + RF']
    accuracies = [hog_svm_results['accuracy'], lbp_rf_results['accuracy']]
    precisions = [hog_svm_results['precision'], lbp_rf_results['precision']]
    recalls = [hog_svm_results['recall'], lbp_rf_results['recall']]
    f1_scores = [hog_svm_results['f1'], lbp_rf_results['f1']]
    
    x = np.arange(len(models))
    width = 0.2
    
    plt.bar(x - 1.5*width, accuracies, width, label='Accuracy')
    plt.bar(x - 0.5*width, precisions, width, label='Precision')
    plt.bar(x + 0.5*width, recalls, width, label='Recall')
    plt.bar(x + 1.5*width, f1_scores, width, label='F1-score')
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Comparison of Traditional ML Models')
    plt.xticks(x, models)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'))
    plt.close()
    
    # Save comparison results
    comparison = {
        'models': models,
        'metrics': {
            'accuracy': accuracies,
            'precision': precisions,
            'recall': recalls,
            'f1': f1_scores,
        },
        'inference_time_ms': [
            hog_svm_results['avg_inference_time'] * 1000,
            lbp_rf_results['avg_inference_time'] * 1000
        ]
    }
    
    with open(os.path.join(RESULTS_DIR, 'model_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=4)
    
    # Save comparison as Markdown
    with open(os.path.join(RESULTS_DIR, 'model_comparison.md'), 'w') as f:
        f.write("# Comparison of Traditional ML Models\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1 Score | Inference Time (ms) |\n")
        f.write("|-------|----------|-----------|--------|----------|--------------------|\n")
        
        for i, model in enumerate(models):
            f.write(f"| {model} | {accuracies[i]:.4f} | {precisions[i]:.4f} | ")
            f.write(f"{recalls[i]:.4f} | {f1_scores[i]:.4f} | {comparison['inference_time_ms'][i]:.2f} |\n")
    
    print(f"\nTraditional ML pipeline completed. Results saved to {RESULTS_DIR}")
    
    return {
        'hog_svm': hog_svm_results,
        'lbp_rf': lbp_rf_results
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Enhanced Traditional ML Pipeline')
    parser.add_argument('--same_split', action='store_true', help='Use the same test split as CNN')
    parser.add_argument('--no_grid_search', action='store_false', dest='grid_search', help='Disable grid search')
    parser.add_argument('--no_cross_val', action='store_false', dest='cross_val', help='Disable cross-validation')
    
    parser.set_defaults(same_split=True, grid_search=True, cross_val=True)
    
    args = parser.parse_args()
    
    run_traditional_ml_pipeline(
        use_same_test_split=args.same_split,
        use_grid_search=args.grid_search,
        use_cross_val=args.cross_val
    )