# Steel Defect Detection – Final Project (CSC 537)

This project implements a deep learning and traditional ML pipeline for classifying steel surface defects using the NEU-DET dataset.

## Key Features
- ✅ ResNet18-based CNN with PyTorch and PyTorch Lightning
- ✅ Enhanced traditional ML classifiers with HOG and LBP features
- ✅ Full training, evaluation, and analysis pipelines
- ✅ Visualizations of misclassifications and confusion matrices
- ✅ Failure case analysis and logging
- ✅ Cloud deployment scripts for RunPod

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run model training:
```bash
python src/training/train_lightning.py
```

3. Evaluate:
```bash
python src/evaluation/final_test_evaluation.py
```

## Dataset
Uses NEU-DET (pre-loaded in `NEU-DET/`). Ensure data paths match those in `config/paths.py`.

## Author
Evan Musick  
Missouri State University

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
