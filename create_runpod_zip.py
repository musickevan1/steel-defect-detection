#!/usr/bin/env python
"""
Script to create a zip file of the project for RunPod.
This script will:
1. Create a zip file of the project
2. Exclude unnecessary files and directories
"""

import os
import zipfile
import datetime

def create_zip():
    """Create a zip file of the project."""
    print("Creating zip file for RunPod...")
    
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define zip file name
    zip_filename = f"steel_defect_detection_{timestamp}.zip"
    
    # Define files and directories to exclude
    exclude = [
        "__pycache__",
        ".git",
        ".vscode",
        ".idea",
        ".ipynb_checkpoints",
        "venv",
        ".venv",
        "env",
        ".env",
        ".roomodes",
        "Proposal",
        "Progress_Report",
        zip_filename,  # Exclude the zip file itself
        "create_runpod_zip.py",  # Exclude this script
        "implement_structure.py",  # Exclude the implementation script
        "proposed_file_structure.md",  # Exclude the proposed file structure
    ]
    
    # Create the zip file
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the directory
        for root, dirs, files in os.walk("."):
            # Remove excluded directories from the walk
            dirs[:] = [d for d in dirs if d not in exclude]
            
            # Add files to the zip
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip excluded files
                if any(excl in file_path for excl in exclude):
                    continue
                
                # Add the file to the zip
                arcname = os.path.relpath(file_path, ".")
                zipf.write(file_path, arcname)
                print(f"Added {arcname}")
    
    print(f"\nZip file created: {zip_filename}")
    print(f"Size: {os.path.getsize(zip_filename) / (1024 * 1024):.2f} MB")
    print("\nYou can now upload this zip file to RunPod.")

def main():
    """Main function."""
    print("=== Creating RunPod Zip ===")
    
    # Create the zip file
    create_zip()
    
    print("\n=== RunPod Zip Creation Complete ===")

if __name__ == "__main__":
    main()