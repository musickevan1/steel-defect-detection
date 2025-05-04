#!/usr/bin/env python
"""
Installation script for Steel Defect Detection project.
This script checks for required dependencies and installs them if needed.
"""

import subprocess
import sys
import os
import importlib.util
from pathlib import Path

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
    print("Installing all requirements from requirements.txt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def install_setup():
    """Install using setup.py."""
    print("Installing using setup.py...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])

def main():
    """Main function."""
    print("Steel Defect Detection - Installation Script")
    print("===========================================")
    
    # Check if requirements.txt exists
    if Path("requirements.txt").exists():
        print("\nrequirements.txt found.")
        choice = input("Do you want to install all dependencies from requirements.txt? (y/n): ")
        if choice.lower() == 'y':
            install_requirements()
            print("\nAll dependencies installed successfully!")
            return
    
    # Check if setup.py exists
    if Path("setup.py").exists():
        print("\nsetup.py found.")
        choice = input("Do you want to install using setup.py? (y/n): ")
        if choice.lower() == 'y':
            install_setup()
            print("\nInstallation completed successfully!")
            return
    
    # Manual installation
    print("\nChecking individual packages...")
    missing_packages = []
    
    for package in REQUIRED_PACKAGES:
        if not check_package(package):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nThe following packages are missing: {', '.join(missing_packages)}")
        choice = input("Do you want to install them? (y/n): ")
        if choice.lower() == 'y':
            for package in missing_packages:
                install_package(package)
            print("\nAll missing packages installed successfully!")
        else:
            print("\nPlease install the missing packages manually.")
    else:
        print("\nAll required packages are already installed!")
    
    print("\nInstallation process completed.")

if __name__ == "__main__":
    main()