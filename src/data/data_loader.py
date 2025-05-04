 # data_loader.py
# This script will contain functions for loading, exploring,
# and preprocessing the NEU-DET dataset using PyTorch.

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import xmltodict
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Define dataset path
# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (parent of the script directory)
project_root = os.path.dirname(script_dir)
# Define the dataset path
DATASET_PATH = os.path.join(project_root, 'NEU-DET')

# TODO: Implement function to parse annotations (if needed for classification)
# The NEU-DET dataset often comes with class folders, but annotations might provide bounding boxes.
# For classification, we primarily need the folder structure.

# Define transformations (resizing, normalization, augmentation)
# Using standard ImageNet stats, resize to 224x224 common for pre-trained models
IMG_SIZE = 224
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(), # Simple augmentation
    transforms.RandomRotation(10),     # Simple augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Implement data loading and splitting logic
def load_data(base_path=DATASET_PATH, batch_size=32, val_split=0.5, random_seed=42):
    """
    Loads the NEU-DET dataset using ImageFolder, splits validation set into validation and test sets,
    and returns DataLoaders.

    Args:
        base_path (str): Path to the NEU-DET directory.
        batch_size (int): Batch size for DataLoaders.
        val_split (float): Proportion of the original validation set to use for the new validation set (rest goes to test).
        random_seed (int): Random seed for reproducible splits.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_dataset, class_names)
               Returns (None, None, None, None, None) if dataset paths are invalid.
    """
    train_dir = os.path.join(base_path, 'train', 'images')
    val_dir = os.path.join(base_path, 'validation', 'images')

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print(f"Error: Train ({train_dir}) or Validation ({val_dir}) directory not found.")
        # Check for alternative structure (classes directly under base_path)
        if os.path.isdir(os.path.join(base_path, 'Crazing')): # Simple check
            print("Attempting to use classes directly under base_path. Manual split required.")
            # TODO: Implement manual split logic if this structure is confirmed
            # For now, we assume the train/validation structure exists
            return None, None, None, None, None
        else:
            return None, None, None, None, None

    print(f"Loading training data from: {train_dir}")
    print(f"Loading validation/test data from: {val_dir}")

    # Load datasets using ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    # Load the original validation dataset (will be split)
    full_val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)

    class_names = train_dataset.classes
    print(f"Classes found: {class_names}")
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation/test images: {len(full_val_dataset)}")

    # Split the original validation set into new validation and test sets
    total_val_size = len(full_val_dataset)
    val_size = int(total_val_size * val_split)
    test_size = total_val_size - val_size

    if val_size == 0 or test_size == 0:
         print(f"Warning: Validation split resulted in zero samples for validation ({val_size}) or test ({test_size}). Adjust val_split.")
         # Defaulting to using the full set as validation and having no test set for now
         val_dataset = full_val_dataset
         test_dataset = None
         print("Using full original validation set as validation. No test set created.")
    else:
        val_dataset, test_dataset = random_split(full_val_dataset, [val_size, test_size],
                                                 generator=torch.Generator().manual_seed(random_seed))
        print(f"Split validation data into: {len(val_dataset)} validation samples, {len(test_dataset)} test samples.")


    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) if test_dataset else None


    return train_loader, val_loader, test_loader, test_dataset, class_names


# Function to get image paths and labels (Used in EDA, keep for reference or future use)
def get_image_paths_and_labels(image_dir):
    image_paths = []
    labels = []
    class_names = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(image_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_to_idx[class_name])
    return image_paths, labels, class_names, class_to_idx

# Implement Exploratory Data Analysis (EDA) function
def perform_eda(base_path=DATASET_PATH):
    print("--- Starting Exploratory Data Analysis (EDA) ---")
    train_image_dir = os.path.join(base_path, 'train', 'images')
    # val_image_dir = os.path.join(base_path, 'validation', 'images') # Assuming similar structure if needed

    if not os.path.isdir(train_image_dir):
        print(f"Error: Training image directory not found at {train_image_dir}")
        # Attempting alternative common structure: NEU-DET/Crazing, NEU-DET/Inclusion, etc.
        if os.path.isdir(os.path.join(base_path, 'Crazing')): # Check if classes are directly under base_path
             train_image_dir = base_path
             print(f"Found class directories directly under {base_path}. Using this path.")
        else:
             print("Could not find image directories. Please check DATASET_PATH and structure.")
             return

    print(f"Using image directory: {train_image_dir}")

    image_paths, labels, class_names, class_to_idx = get_image_paths_and_labels(train_image_dir)

    if not image_paths:
        print("No images found. Please check the dataset directory structure.")
        return

    print(f"\nFound {len(image_paths)} images.")
    print(f"Classes found: {class_names}")
    print(f"Class to index mapping: {class_to_idx}")

    # Class distribution
    label_counts = pd.Series(labels).map({v: k for k, v in class_to_idx.items()}).value_counts().sort_index()
    print("\nClass Distribution:")
    print(label_counts)

    # Image dimensions
    print("\nAnalyzing image dimensions...")
    dims = []
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                dims.append(img.size)
        except Exception as e:
            print(f"Warning: Could not read image {img_path}: {e}")

    if not dims:
        print("Could not read any image dimensions.")
        return

    dims_df = pd.DataFrame(dims, columns=['width', 'height'])
    print("\nImage Dimensions Summary:")
    print(dims_df.describe())

    # Display sample images
    print("\nDisplaying sample images per class...")
    num_samples_to_show = 3
    plt.figure(figsize=(num_samples_to_show * 3, len(class_names) * 3))
    for i, class_name in enumerate(class_names):
        class_indices = [idx for idx, path in enumerate(image_paths) if class_name in path]
        sample_indices = np.random.choice(class_indices, min(num_samples_to_show, len(class_indices)), replace=False)
        for j, img_idx in enumerate(sample_indices):
            plt.subplot(len(class_names), num_samples_to_show, i * num_samples_to_show + j + 1)
            try:
                img = Image.open(image_paths[img_idx]).convert('RGB')
                plt.imshow(img)
                plt.title(f"{class_name} sample {j+1}")
                plt.axis('off')
            except Exception as e:
                 print(f"Warning: Could not display image {image_paths[img_idx]}: {e}")
                 plt.title(f"{class_name} (Error)")
                 plt.axis('off')

    plt.tight_layout()
    plt.show() # This will open a plot window

    print("\n--- EDA Completed ---")


if __name__ == '__main__':
    # Example usage: Test data loading
    print("\n--- Testing Data Loading ---")
    BATCH_SIZE = 16 # Smaller batch size for testing
    train_loader, val_loader, test_loader, class_names = load_data(batch_size=BATCH_SIZE)

    # Perform EDA
    print("\n--- Performing EDA ---")
    perform_eda()

    if train_loader and val_loader and test_loader:
        print(f"\nDataLoaders created successfully.")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        print(f"Number of test batches: {len(test_loader)}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Classes: {class_names}")

        # Optional: Check a sample batch
        print("\nChecking one batch from train_loader...")
        try:
            images, labels = next(iter(train_loader))
            print(f"Sample batch - Image shape: {images.shape}") # Should be [BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE]
            print(f"Sample batch - Labels: {labels}")
        except Exception as e:
            print(f"Error checking sample batch: {e}")
            print("Ensure num_workers is appropriate for your system (try 0 if issues persist).")

    else:
        print("\nData loading failed. Please check dataset paths and structure.")

    # perform_eda() # Can uncomment to run EDA again if needed
