#!/usr/bin/env python
"""
Script to install tensorboard and tensorboardX dependencies.
"""

import subprocess
import sys

def main():
    print("Installing tensorboard and tensorboardX...")
    
    # Install tensorboard
    print("\nInstalling tensorboard...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard>=2.5.0"])
    
    # Install tensorboardX
    print("\nInstalling tensorboardX...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboardX>=2.1"])
    
    print("\nInstallation complete!")
    print("You should now be able to run the experiments successfully.")

if __name__ == "__main__":
    main()