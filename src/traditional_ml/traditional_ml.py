# traditional_ml.py
# Implements feature extraction and traditional ML models (SVM, Random Forest)
# for comparison with the CNN model.

import os
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from PIL import Image
import joblib # For saving models
import time

# Assuming data_loader.py exists and can provide image paths and labels
# We might need a simplified data loading part here if data_loader is too complex
from data_loader import get_image_paths_and_labels, DATASET_PATH, IMG_SIZE

# --- Feature Extraction ---

def extract_hog_features(image):
    """Extracts HOG features from a single image."""
    # Ensure image is grayscale and resized appropriately
    image_gray = image.convert('L').resize((128, 128)) # HOG often works well on smaller sizes
    features = hog(np.array(image_gray), pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), visualize=False)
    return features

def extract_lbp_features(image, radius=3, n_points=24):
    """Extracts LBP features from a single image."""
    image_gray = image.convert('L').resize((IMG_SIZE, IMG_SIZE))
    lbp = local_binary_pattern(np.array(image_gray), n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6) # Normalize
    return hist

def extract_features(image_paths, feature_type='hog'):
    """Extracts specified features for a list of image paths."""
    features_list = []
    print(f"Extracting {feature_type.upper()} features...")
    start_time = time.time()
    for i, img_path in enumerate(image_paths):
        try:
            with Image.open(img_path) as img:
                if feature_type == 'hog':
                    features = extract_hog_features(img)
                elif feature_type == 'lbp':
                    features = extract_lbp_features(img)
                else:
                    raise ValueError("Unsupported feature type")
                features_list.append(features)
            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{len(image_paths)} images...")
        except Exception as e:
            print(f"Warning: Could not process image {img_path}: {e}")
            # Handle potential errors, maybe append zeros or skip? For now, skipping.
            pass
    end_time = time.time()
    print(f"Feature extraction took {end_time - start_time:.2f} seconds.")
    return np.array(features_list)

# --- Model Training and Evaluation ---

def train_evaluate_traditional_model(features, labels, model_type='svm', test_size=0.25, random_state=42):
    """Trains and evaluates a traditional ML model."""
    print(f"\n--- Training and Evaluating {model_type.upper()} ---")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state, stratify=labels)
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled.")

    # Initialize model
    if model_type == 'svm':
        # Using probability=True for potential later use, might be slower
        model = SVC(kernel='rbf', C=1.0, random_state=random_state, probability=True)
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    else:
        raise ValueError("Unsupported model type")

    # Train model
    print(f"Training {model_type.upper()} model...")
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    end_time = time.time()
    print(f"Training took {end_time - start_time:.2f} seconds.")

    # Evaluate model
    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred) # target_names=class_names if available

    print(f"\n{model_type.upper()} Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Save the model and scaler (optional)
    # model_filename = f'{model_type}_model.joblib'
    # scaler_filename = f'{model_type}_scaler.joblib'
    # joblib.dump(model, model_filename)
    # joblib.dump(scaler, scaler_filename)
    # print(f"Model saved to {model_filename}")
    # print(f"Scaler saved to {scaler_filename}")

    return model, scaler, accuracy

if __name__ == '__main__':
    print("--- Starting Traditional ML Pipeline ---")
    # Load image paths and labels (using combined train+val for traditional ML training)
    train_image_dir = os.path.join(DATASET_PATH, 'train', 'images')
    val_image_dir = os.path.join(DATASET_PATH, 'validation', 'images')

    train_paths, train_labels, class_names, _ = get_image_paths_and_labels(train_image_dir)
    val_paths, val_labels, _, _ = get_image_paths_and_labels(val_image_dir)

    all_paths = train_paths + val_paths
    all_labels = train_labels + val_labels

    if not all_paths:
        print("Error: No image paths found. Exiting.")
    else:
        print(f"Total images for traditional ML: {len(all_paths)}")

        # --- HOG + SVM ---
        hog_features = extract_features(all_paths, feature_type='hog')
        if hog_features.size > 0:
             train_evaluate_traditional_model(hog_features, all_labels, model_type='svm')
        else:
             print("Skipping SVM with HOG due to feature extraction issues.")

        # --- LBP + Random Forest ---
        lbp_features = extract_features(all_paths, feature_type='lbp')
        if lbp_features.size > 0:
            train_evaluate_traditional_model(lbp_features, all_labels, model_type='rf')
        else:
            print("Skipping Random Forest with LBP due to feature extraction issues.")

    print("\n--- Traditional ML Pipeline Finished ---")