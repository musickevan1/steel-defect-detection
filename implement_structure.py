#!/usr/bin/env python
"""
Script to implement the proposed file structure.
This script will:
1. Create the necessary directories
2. Move files to their new locations
3. Create __init__.py files
4. Update import statements in Python files
"""

import os
import shutil
import re
from pathlib import Path

def create_directories():
    """Create the directory structure."""
    print("Creating directories...")
    
    # Main directories
    directories = [
        "src",
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/traditional_ml",
        "src/analysis",
        "config",
        "scripts",
        "runpod",
        "data/NEU-DET",
        "saved_models",
        "results/cnn",
        "results/traditional_ml",
        "results/analysis",
        "results/failure_analysis",
        "results/debug"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create __init__.py files
    init_dirs = [
        "src",
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/traditional_ml",
        "src/analysis"
    ]
    
    for directory in init_dirs:
        init_file = os.path.join(directory, "__init__.py")
        with open(init_file, "w") as f:
            f.write(f"# {os.path.basename(directory)} package\n")
        print(f"Created __init__.py in {directory}")

def move_files():
    """Move files to their new locations."""
    print("\nMoving files...")
    
    # Define file mappings (source -> destination)
    file_mappings = {
        # Source code files
        "CSC_573_Final_Project_Code/data_loader.py": "src/data/data_loader.py",
        "CSC_573_Final_Project_Code/model.py": "src/models/model.py",
        "CSC_573_Final_Project_Code/lightning_model.py": "src/models/lightning_model.py",
        "CSC_573_Final_Project_Code/train_cnn.py": "src/training/train_cnn.py",
        "CSC_573_Final_Project_Code/train_lightning.py": "src/training/train_lightning.py",
        "CSC_573_Final_Project_Code/evaluate_cnn.py": "src/evaluation/evaluate_cnn.py",
        "CSC_573_Final_Project_Code/evaluate_lightning.py": "src/evaluation/evaluate_lightning.py",
        "CSC_573_Final_Project_Code/final_test_evaluation.py": "src/evaluation/final_test_evaluation.py",
        "CSC_573_Final_Project_Code/traditional_ml.py": "src/traditional_ml/traditional_ml.py",
        "CSC_573_Final_Project_Code/traditional_ml_enhanced.py": "src/traditional_ml/traditional_ml_enhanced.py",
        "CSC_573_Final_Project_Code/failure_analysis.py": "src/analysis/failure_analysis.py",
        "analyze_experiments.py": "src/analysis/analyze_experiments.py",
        
        # Configuration
        "CSC_573_Final_Project_Code/config.py": "config/config.py",
        
        # Wrapper scripts
        "run_experiments.py": "scripts/run_experiments.py",
        "run_analysis.py": "scripts/run_analysis.py",
        "run_final_evaluation.py": "scripts/run_final_evaluation.py",
        "run_traditional_ml.py": "scripts/run_traditional_ml.py",
        "run_failure_analysis.py": "scripts/run_failure_analysis.py",
        "debug_experiment.py": "scripts/debug_experiment.py",
        
        # RunPod files
        "runpod_guide.md": "runpod/runpod_guide.md",
        "runpod_setup.py": "runpod/runpod_setup.py",
        
        # Root files remain in place
        # "README.md": "README.md",
        # "requirements.txt": "requirements.txt",
        # "setup.py": "setup.py",
        # "install.py": "install.py",
        # "install_tensorboard.py": "install_tensorboard.py",
    }
    
    # Move files
    for source, destination in file_mappings.items():
        if os.path.exists(source):
            # Create destination directory if it doesn't exist
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            # Copy the file
            shutil.copy2(source, destination)
            print(f"Copied {source} to {destination}")
        else:
            print(f"Warning: Source file {source} not found")

def update_imports():
    """Update import statements in Python files."""
    print("\nUpdating import statements...")
    
    # Define import mappings (old -> new)
    import_mappings = {
        "from src.data.data_loader import": "from src.data.data_loader import",
        "from src.models.model import": "from src.models.model import",
        "from src.models.lightning_model import": "from src.models.lightning_model import",
        "from src.training.train_cnn import": "from src.training.train_cnn import",
        "from src.training.train_lightning import": "from src.training.train_lightning import",
        "from src.evaluation.evaluate_cnn import": "from src.evaluation.evaluate_cnn import",
        "from src.evaluation.evaluate_lightning import": "from src.evaluation.evaluate_lightning import",
        "from src.evaluation.final_test_evaluation import": "from src.evaluation.final_test_evaluation import",
        "from src.traditional_ml.traditional_ml import": "from src.traditional_ml.traditional_ml import",
        "from src.traditional_ml.traditional_ml_enhanced import": "from src.traditional_ml.traditional_ml_enhanced import",
        "from src.analysis.failure_analysis import": "from src.analysis.failure_analysis import",
        "from src.analysis.analyze_experiments import": "from src.analysis.analyze_experiments import",
        "from config.config import": "from config.config import",
    }
    
    # Find all Python files in the project
    python_files = []
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    
    # Update import statements in each file
    for file_path in python_files:
        try:
            # Try to open the file with utf-8 encoding
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                # If utf-8 fails, try with latin-1 encoding (which can read any byte)
                with open(file_path, "r", encoding="latin-1") as f:
                    content = f.read()
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue
        
        # Skip __init__.py files
        if file_path.endswith("__init__.py"):
            continue
        
        # Update import statements
        modified = False
        for old_import, new_import in import_mappings.items():
            if old_import in content:
                content = content.replace(old_import, new_import)
                modified = True
        
        if modified:
            try:
                # Try to write the file with utf-8 encoding
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Updated imports in {file_path}")
            except Exception as e:
                print(f"Warning: Could not write to {file_path}: {e}")

def update_wrapper_scripts():
    """Update file paths in wrapper scripts."""
    print("\nUpdating wrapper scripts...")
    
    # Define wrapper scripts
    wrapper_scripts = [
        "scripts/run_experiments.py",
        "scripts/run_analysis.py",
        "scripts/run_final_evaluation.py",
        "scripts/run_traditional_ml.py",
        "scripts/run_failure_analysis.py",
        "scripts/debug_experiment.py",
    ]
    
    # Update file paths in each script
    for script_path in wrapper_scripts:
        if not os.path.exists(script_path):
            print(f"Warning: Wrapper script {script_path} not found")
            continue
        
        try:
            # Try to open the file with utf-8 encoding
            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                # If utf-8 fails, try with latin-1 encoding
                with open(script_path, "r", encoding="latin-1") as f:
                    content = f.read()
            except Exception as e:
                print(f"Warning: Could not read {script_path}: {e}")
                continue
        
        # Update file paths
        # Map files to their correct subdirectories
        content = re.sub(r'CSC_573_Final_Project_Code/data_loader\.py', r'src/data/data_loader.py', content)
        content = re.sub(r'CSC_573_Final_Project_Code/model\.py', r'src/models/model.py', content)
        content = re.sub(r'CSC_573_Final_Project_Code/lightning_model\.py', r'src/models/lightning_model.py', content)
        content = re.sub(r'CSC_573_Final_Project_Code/train_cnn\.py', r'src/training/train_cnn.py', content)
        content = re.sub(r'CSC_573_Final_Project_Code/train_lightning\.py', r'src/training/train_lightning.py', content)
        content = re.sub(r'CSC_573_Final_Project_Code/evaluate_cnn\.py', r'src/evaluation/evaluate_cnn.py', content)
        content = re.sub(r'CSC_573_Final_Project_Code/evaluate_lightning\.py', r'src/evaluation/evaluate_lightning.py', content)
        content = re.sub(r'CSC_573_Final_Project_Code/final_test_evaluation\.py', r'src/evaluation/final_test_evaluation.py', content)
        content = re.sub(r'CSC_573_Final_Project_Code/traditional_ml\.py', r'src/traditional_ml/traditional_ml.py', content)
        content = re.sub(r'CSC_573_Final_Project_Code/traditional_ml_enhanced\.py', r'src/traditional_ml/traditional_ml_enhanced.py', content)
        content = re.sub(r'CSC_573_Final_Project_Code/failure_analysis\.py', r'src/analysis/failure_analysis.py', content)
        content = re.sub(r'CSC_573_Final_Project_Code/config\.py', r'config/config.py', content)
        
        try:
            # Try to write the file with utf-8 encoding
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            print(f"Warning: Could not write to {script_path}: {e}")
            continue
        
        print(f"Updated file paths in {script_path}")

def create_root_wrappers():
    """Create root wrapper scripts that call the scripts in the scripts directory."""
    print("\nCreating root wrapper scripts...")
    
    # Define wrapper scripts
    wrapper_scripts = [
        "run_experiments.py",
        "run_analysis.py",
        "run_final_evaluation.py",
        "run_traditional_ml.py",
        "run_failure_analysis.py",
        "debug_experiment.py",
    ]
    
    # Create a wrapper script for each script in the scripts directory
    for script_name in wrapper_scripts:
        root_script = script_name
        scripts_script = os.path.join("scripts", script_name)
        
        try:
            # Try to write the file with utf-8 encoding
            with open(root_script, "w", encoding="utf-8") as f:
                # Write the wrapper script content line by line
                f.write("#!/usr/bin/env python\n")
                f.write('"""\n')
                f.write(f"Wrapper script to run {scripts_script}\n")
                f.write('"""\n\n')
                f.write("import os\n")
                f.write("import sys\n")
                f.write("import subprocess\n\n")
                f.write("# Get the directory of this script\n")
                f.write("script_dir = os.path.dirname(os.path.abspath(__file__))\n\n")
                f.write("# Path to the actual script\n")
                f.write(f"script_path = os.path.join(script_dir, \"{scripts_script}\")\n\n")
                f.write("# Check if the script exists\n")
                f.write("if not os.path.exists(script_path):\n")
                f.write("    print(f\"Error: Script not found at {script_path}\")\n")
                f.write("    sys.exit(1)\n\n")
                f.write("# Run the script with any command-line arguments\n")
                f.write("cmd = [sys.executable, script_path] + sys.argv[1:]\n")
                f.write("print(f\"Running: {' '.join(cmd)}\")\n")
                f.write("subprocess.run(cmd)\n")
            
            print(f"Created root wrapper script: {root_script}")
        except Exception as e:
            print(f"Warning: Could not create root wrapper script {root_script}: {e}")

def update_readme():
    """Update README.md with the new file structure."""
    print("\nUpdating README.md...")
    
    # Read the current README.md
    try:
        # Try to open the file with utf-8 encoding
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            # If utf-8 fails, try with latin-1 encoding
            with open("README.md", "r", encoding="latin-1") as f:
                content = f.read()
        except Exception as e:
            print(f"Warning: Could not read README.md: {e}")
            return
    
    # Add a section about the file structure
    file_structure_section = """
## File Structure

```
steel_defect_detection/
├── README.md                      # Main project documentation
├── requirements.txt               # Project dependencies
├── setup.py                       # Setup script for installation
├── install.py                     # Interactive installation script
├── install_tensorboard.py         # Script to install tensorboard dependencies
│
├── scripts/                       # Wrapper scripts for easy execution
│   ├── run_experiments.py         # Script to run multiple experiments
│   ├── run_analysis.py            # Script to analyze experiment results
│   ├── run_final_evaluation.py    # Script to run final test evaluation
│   ├── run_traditional_ml.py      # Script to run traditional ML pipeline
│   ├── run_failure_analysis.py    # Script to run failure analysis
│   └── debug_experiment.py        # Script to run a single experiment for debugging
│
├── runpod/                        # RunPod-specific files
│   ├── runpod_guide.md            # Guide for running on RunPod
│   └── runpod_setup.py            # Setup script for RunPod
│
├── src/                           # Source code
│   ├── data/                      # Data loading and preprocessing
│   │   └── data_loader.py         # Data loading functionality
│   │
│   ├── models/                    # Model definitions
│   │   ├── model.py               # CNN model definition
│   │   └── lightning_model.py     # PyTorch Lightning model
│   │
│   ├── training/                  # Training scripts
│   │   ├── train_cnn.py           # Original training script
│   │   └── train_lightning.py     # PyTorch Lightning training script
│   │
│   ├── evaluation/                # Evaluation scripts
│   │   ├── evaluate_cnn.py        # Original evaluation script
│   │   ├── evaluate_lightning.py  # PyTorch Lightning evaluation script
│   │   └── final_test_evaluation.py # Final test evaluation script
│   │
│   ├── traditional_ml/            # Traditional ML implementation
│   │   ├── traditional_ml.py      # Original traditional ML script
│   │   └── traditional_ml_enhanced.py # Enhanced traditional ML script
│   │
│   └── analysis/                  # Analysis scripts
│       ├── analyze_experiments.py # Script to analyze experiment results
│       └── failure_analysis.py    # Script for failure case analysis
│
├── config/                        # Configuration files
│   └── config.py                  # Configuration settings
│
├── data/                          # Data directory
│   └── NEU-DET/                   # NEU-DET dataset
│
├── saved_models/                  # Directory for saved models
│
└── results/                       # Directory for results
    ├── cnn/                       # CNN results
    ├── traditional_ml/            # Traditional ML results
    ├── analysis/                  # Analysis results
    ├── failure_analysis/          # Failure analysis results
    └── debug/                     # Debug results
```

Note: The root directory contains wrapper scripts that call the scripts in the `scripts` directory, so you can still run commands like `python run_experiments.py` from the root directory.
"""
    
    # Check if the file structure section already exists
    if "## File Structure" not in content:
        # Find the Project Structure section
        project_structure_index = content.find("## Project Structure")
        
        if project_structure_index != -1:
            # Find the end of the Project Structure section
            next_section_index = content.find("##", project_structure_index + 1)
            
            if next_section_index != -1:
                # Insert the file structure section before the next section
                content = content[:next_section_index] + file_structure_section + "\n\n" + content[next_section_index:]
            else:
                # Append the file structure section to the end of the file
                content += "\n" + file_structure_section
        else:
            # Append the file structure section to the end of the file
            content += "\n" + file_structure_section
        
        # Write the updated README.md
        try:
            # Try to write the file with utf-8 encoding
            with open("README.md", "w", encoding="utf-8") as f:
                f.write(content)
            print("Updated README.md with the new file structure")
        except Exception as e:
            print(f"Warning: Could not write to README.md: {e}")
    else:
        print("README.md already contains a File Structure section")

def main():
    """Main function."""
    print("=== Implementing File Structure ===")
    
    # Create directories
    create_directories()
    
    # Move files
    move_files()
    
    # Update import statements
    update_imports()
    
    # Update wrapper scripts
    update_wrapper_scripts()
    
    # Create root wrapper scripts
    create_root_wrappers()
    
    # Update README.md
    update_readme()
    
    print("\n=== File Structure Implementation Complete ===")
    print("You can now zip the project for RunPod.")

if __name__ == "__main__":
    main()