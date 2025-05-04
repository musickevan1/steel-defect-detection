#!/usr/bin/env python
"""
Setup script for RunPod environment.
This script:
1. Sets the RUNPOD environment variable
2. Creates necessary directories
3. Installs required dependencies
4. Verifies GPU availability
"""

import os
import sys
import subprocess
import platform
import importlib.util

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Set the RUNPOD environment variable
os.environ['RUNPOD'] = '1'

# Import the paths module
from config.paths import create_directories, print_paths

# List of required packages
REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "pytorch_lightning",
    "tensorboard",
    "tensorboardX",
    "numpy",
    "pandas",
    "scikit-learn",
    "scikit-image",
    "matplotlib",
    "seaborn",
    "pillow",
    "joblib",
    "tqdm",
]

def check_package(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package_name):
    """Install a package using pip."""
    print(f"Installing {package_name}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def install_requirements():
    """Install all requirements from requirements.txt."""
    requirements_path = os.path.join(project_root, "requirements.txt")
    if os.path.exists(requirements_path):
        print("Installing all requirements from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
    else:
        print("requirements.txt not found. Installing packages individually...")
        for package in REQUIRED_PACKAGES:
            if not check_package(package):
                install_package(package)

def check_gpu():
    """Check if GPU is available and print information."""
    try:
        import torch
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
                try:
                    nvidia_smi = subprocess.check_output(["nvidia-smi"], text=True)
                    print(nvidia_smi)
                except:
                    print("nvidia-smi command not found")
        else:
            print("CUDA is not available. Training will run on CPU, which will be much slower.")
    except ImportError:
        print("PyTorch not installed. Cannot check GPU availability.")

def setup_environment():
    """Set up environment variables for optimal performance."""
    print("\n=== Setting Up Environment ===")
    
    # Set environment variables for PyTorch
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    print("Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")

def main():
    """Main function."""
    print("=== RunPod Setup for Steel Defect Detection ===")
    
    # Create directories
    create_directories()
    
    # Print path information
    print_paths()
    
    # Install dependencies
    install_requirements()
    
    # Check GPU
    check_gpu()
    
    # Set up environment
    setup_environment()
    
    print("\n=== RunPod Setup Complete ===")
    print("You are now ready to run training on RunPod!")
    print("\nTo run experiments:")
    print("python run_experiments.py")
    
    print("\nTo run a single experiment:")
    print("python debug_experiment.py --batch_size 32 --lr 0.001 --epochs 15")

if __name__ == "__main__":
    main()