#!/usr/bin/env python
"""
Path configuration for the Steel Defect Detection project.
This file handles path differences between local and RunPod environments.
"""

import os
import platform

# Determine if we're running on RunPod
ON_RUNPOD = os.environ.get('RUNPOD', '0') == '1'

# Get the project root directory
if ON_RUNPOD:
    # On RunPod, the project is typically extracted to the home directory
    PROJECT_ROOT = os.path.expanduser('~/steel_defect_detection')
else:
    # Locally, use the directory containing this file
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths relative to the project root
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DATASET_DIR = os.path.join(DATA_DIR, 'NEU-DET')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train', 'images')
VALIDATION_DIR = os.path.join(DATASET_DIR, 'validation', 'images')

SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, 'saved_models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
CNN_RESULTS_DIR = os.path.join(RESULTS_DIR, 'cnn')
TRADITIONAL_ML_RESULTS_DIR = os.path.join(RESULTS_DIR, 'traditional_ml')
ANALYSIS_RESULTS_DIR = os.path.join(RESULTS_DIR, 'analysis')
FAILURE_ANALYSIS_DIR = os.path.join(RESULTS_DIR, 'failure_analysis')
DEBUG_RESULTS_DIR = os.path.join(RESULTS_DIR, 'debug')

LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Create directories if they don't exist
def create_directories():
    """Create all necessary directories."""
    directories = [
        DATA_DIR,
        DATASET_DIR,
        SAVED_MODELS_DIR,
        RESULTS_DIR,
        CNN_RESULTS_DIR,
        TRADITIONAL_ML_RESULTS_DIR,
        ANALYSIS_RESULTS_DIR,
        FAILURE_ANALYSIS_DIR,
        DEBUG_RESULTS_DIR,
        LOG_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

# Function to get absolute path from project-relative path
def get_path(relative_path):
    """Get absolute path from project-relative path."""
    return os.path.join(PROJECT_ROOT, relative_path)

# Function to set RunPod environment
def set_runpod_env():
    """Set environment variables for RunPod."""
    os.environ['RUNPOD'] = '1'
    print("Set environment for RunPod")

# Function to print path information
def print_paths():
    """Print path information for debugging."""
    print("\n=== Path Information ===")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Dataset Directory: {DATASET_DIR}")
    print(f"Train Directory: {TRAIN_DIR}")
    print(f"Validation Directory: {VALIDATION_DIR}")
    print(f"Saved Models Directory: {SAVED_MODELS_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"CNN Results Directory: {CNN_RESULTS_DIR}")
    print(f"Traditional ML Results Directory: {TRADITIONAL_ML_RESULTS_DIR}")
    print(f"Analysis Results Directory: {ANALYSIS_RESULTS_DIR}")
    print(f"Failure Analysis Directory: {FAILURE_ANALYSIS_DIR}")
    print(f"Debug Results Directory: {DEBUG_RESULTS_DIR}")
    print(f"Log Directory: {LOG_DIR}")
    print(f"Running on RunPod: {ON_RUNPOD}")
    print(f"Platform: {platform.system()}")
    print("===========================\n")

if __name__ == "__main__":
    # Create directories and print path information
    create_directories()
    print_paths()