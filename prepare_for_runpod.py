#!/usr/bin/env python
"""
Main script to prepare the project for RunPod.
This script will:
1. Implement the proposed file structure
2. Create a zip file of the project for RunPod
"""

import os
import sys
import subprocess
import time

def run_script(script_path):
    """Run a Python script."""
    print(f"\n=== Running {script_path} ===\n")
    
    # Run the script
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    
    # Check if the script ran successfully
    if result.returncode != 0:
        print(f"\nError: {script_path} failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"\n=== {script_path} completed successfully ===\n")

def main():
    """Main function."""
    print("=== Preparing Project for RunPod ===")
    
    # Check if the implementation script exists
    if not os.path.exists("implement_structure.py"):
        print("Error: implement_structure.py not found")
        sys.exit(1)
    
    # Check if the zip creation script exists
    if not os.path.exists("create_runpod_zip.py"):
        print("Error: create_runpod_zip.py not found")
        sys.exit(1)
    
    # Run the implementation script
    run_script("implement_structure.py")
    
    # Wait a moment to ensure all files are written
    time.sleep(1)
    
    # Run the zip creation script
    run_script("create_runpod_zip.py")
    
    print("\n=== Project Preparation Complete ===")
    print("You can now upload the zip file to RunPod.")

if __name__ == "__main__":
    main()