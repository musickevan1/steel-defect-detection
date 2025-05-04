# config.py
# Configuration settings for the steel defect detection project

import os

# Dataset configuration
DATASET_PATH = 'NEU-DET'  # Path to the NEU-DET dataset
IMG_SIZE = 224  # Image size for resizing

# Model configuration
NUM_CLASSES = 6  # Number of defect classes in NEU-DET
PRETRAINED = True  # Whether to use pretrained weights

# Training configuration
BATCH_SIZES = [16, 32, 64]  # Batch sizes to experiment with
LEARNING_RATES = [0.001, 0.0005, 0.0001]  # Learning rates to experiment with
NUM_EPOCHS = 15  # Maximum number of epochs
EARLY_STOPPING_PATIENCE = 5  # Patience for early stopping

# Directories
SAVE_DIR = 'saved_models'  # Directory to save models
RESULTS_DIR = 'results'  # Directory to save results
LOG_DIR = 'logs'  # Directory to save logs

# Ensure directories exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'cnn'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'traditional'), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Training device
DEVICE = 'cuda'  # 'cuda' or 'cpu'

# Experiment tracking
EXPERIMENT_NAME = 'steel_defect_detection'