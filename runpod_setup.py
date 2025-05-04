#!/usr/bin/env python
"""
Setup script for running on RunPod.io GPU instances.
This script checks the environment, installs dependencies, and verifies GPU availability.
"""

import os
import sys
import subprocess
import platform
import torch

def run_command(command):
    """Run a shell command and return the output."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.stdout

def check_gpu():
    """Check if GPU is available and print information."""
    print("\n=== GPU Information ===")
    
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # Run nvidia-smi for more detailed information
        if platform.system() != "Windows":
            print("\nnvidia-smi output:")
            nvidia_smi = run_command("nvidia-smi")
            print(nvidia_smi)
    else:
        print("CUDA is not available. Training will run on CPU, which will be much slower.")

def install_dependencies():
    """Install required dependencies."""
    print("\n=== Installing Dependencies ===")
    
    # Install requirements
    print("Installing from requirements.txt...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Install tensorboard and tensorboardX
    print("\nInstalling tensorboard and tensorboardX...")
    subprocess.run([sys.executable, "-m", "pip", "install", "tensorboard>=2.5.0", "tensorboardX>=2.1"])
    
    print("\nDependencies installed successfully.")

def setup_environment():
    """Set up environment variables for optimal performance."""
    print("\n=== Setting Up Environment ===")
    
    # Set environment variables for PyTorch
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    print("Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")
    
    # Set number of threads
    if torch.cuda.is_available():
        num_workers = min(4, os.cpu_count())
        print(f"Recommended num_workers for DataLoader: {num_workers}")

def verify_dataset():
    """Verify that the NEU-DET dataset is available."""
    print("\n=== Verifying Dataset ===")
    
    dataset_path = "NEU-DET"
    if os.path.exists(dataset_path):
        train_dir = os.path.join(dataset_path, "train", "images")
        val_dir = os.path.join(dataset_path, "validation", "images")
        
        if os.path.exists(train_dir) and os.path.exists(val_dir):
            print(f"Dataset found at {dataset_path}")
            
            # Count images
            train_count = sum(len(files) for _, _, files in os.walk(train_dir))
            val_count = sum(len(files) for _, _, files in os.walk(val_dir))
            
            print(f"Training images: {train_count}")
            print(f"Validation images: {val_count}")
        else:
            print(f"Dataset structure is incorrect. Expected train and validation directories.")
    else:
        print(f"Dataset not found at {dataset_path}. Please make sure the NEU-DET dataset is available.")

def print_training_commands():
    """Print commands for training."""
    print("\n=== Training Commands ===")
    print("To run a single experiment:")
    print("python debug_experiment.py --batch_size 32 --lr 0.001 --epochs 15")
    
    print("\nTo run multiple experiments:")
    print("python run_experiments.py")
    
    print("\nTo run the training script directly:")
    print("python CSC_573_Final_Project_Code/train_lightning.py --batch_size 32 --lr 0.001 --epochs 15")
    
    print("\nTo monitor with TensorBoard:")
    print("tensorboard --logdir=logs")

def main():
    print("=== RunPod Setup for Steel Defect Detection ===")
    
    # Check GPU
    check_gpu()
    
    # Install dependencies
    install_dependencies()
    
    # Set up environment
    setup_environment()
    
    # Verify dataset
    verify_dataset()
    
    # Print training commands
    print_training_commands()
    
    print("\n=== Setup Complete ===")
    print("You are now ready to run training on RunPod!")

if __name__ == "__main__":
    main()