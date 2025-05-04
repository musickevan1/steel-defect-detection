# Steel Defect Detection

## Overview
This project implements steel defect detection using the NEU-DET dataset. It includes both CNN-based approaches using ResNet18 with PyTorch Lightning and traditional machine learning approaches using HOG/LBP features with SVM/Random Forest classifiers.

## Project Structure
- `data_loader.py`: Handles dataset loading, preprocessing, and EDA
- `model.py`: Defines ResNet18-based CNN architecture
- `lightning_model.py`: PyTorch Lightning implementation of the CNN model
- `train_cnn.py`: Original training script (without Lightning)
- `train_lightning.py`: Training script using PyTorch Lightning
- `evaluate_cnn.py`: Original evaluation script
- `evaluate_lightning.py`: Evaluation script for Lightning models
- `traditional_ml.py`: Implements HOG/LBP feature extraction with SVM/RF classifiers
- `config.py`: Configuration settings for the project
- `run_experiments.py`: Script to run experiments with different hyperparameters

## Installation
```
pip install torch torchvision pytorch-lightning scikit-learn scikit-image matplotlib seaborn pandas
```

## Usage

### CNN Training with PyTorch Lightning
To train the CNN model using PyTorch Lightning, run:
```
python train_lightning.py [options]
```

#### Options
* `--epochs`: Number of epochs to train (default: 15)
* `--lr`: Learning rate (default: 0.001)
* `--batch_size`: Batch size for training (default: 32)
* `--save_dir`: Directory to save models (default: 'saved_models')
* `--results_dir`: Directory to save results (default: 'results/cnn')
* `--log_dir`: Directory to save logs (default: 'logs')
* `--pretrained`: Whether to use pretrained weights (default: True)
* `--precision`: Precision for training (default: '32-true')
* `--early_stopping`: Whether to use early stopping (default: True)
* `--patience`: Patience for early stopping (default: 5)

### CNN Evaluation
To evaluate a trained CNN model, run:
```
python evaluate_lightning.py --checkpoint [path_to_checkpoint]
```

#### Options
* `--checkpoint`: Path to the model checkpoint (required)
* `--batch_size`: Batch size for evaluation (default: 32)
* `--results_dir`: Directory to save results (default: 'results/cnn')
* `--visualize_misclassified`: Whether to visualize misclassified samples (default: True)
* `--num_samples`: Number of misclassified samples to visualize per class (default: 10)

### Final Test Set Evaluation
To perform a comprehensive final evaluation on the test set, run:
```
python final_test_evaluation.py [options]
```

#### Options
* `--checkpoint`: Path to the model checkpoint (if not provided, will find the best checkpoint automatically)
* `--batch_size`: Batch size for evaluation (default: 32)
* `--results_dir`: Directory to save results (default: 'results/final_test_eval')
* `--num_samples`: Number of misclassified samples to visualize per class (default: 5)
* `--calc_inference_time`: Calculate average inference time per image (optional flag)

This script performs a comprehensive evaluation including:
- Overall accuracy, precision, recall, and F1 score
- Per-class metrics
- Confusion matrix visualization
- ROC curves and Precision-Recall curves
- Visualization of misclassified samples
- Optional inference time calculation
- Detailed results summary in both JSON and Markdown formats

### Traditional ML
To run the basic traditional machine learning pipeline, run:
```
python traditional_ml.py
```

### Enhanced Traditional ML
To run the enhanced traditional ML pipeline with grid search, cross-validation, and comprehensive evaluation, run:
```
python traditional_ml_enhanced.py [options]
```

#### Options
* `--same_split`: Use the same test split as CNN (default: True)
* `--no_grid_search`: Disable grid search for hyperparameter tuning
* `--no_cross_val`: Disable cross-validation

This enhanced pipeline includes:
- HOG feature extraction with SVM classifier
- LBP feature extraction with Random Forest classifier
- Grid search for hyperparameter optimization
- Cross-validation for robust evaluation
- Comprehensive metrics (accuracy, precision, recall, F1)
- Confusion matrix visualization
- Per-class performance metrics
- Inference time measurement
- Results saved in both JSON and Markdown formats

### Failure Analysis
To perform failure case analysis and visualization for both CNN and traditional ML models, run:
```
python failure_analysis.py [options]
```

#### Options
* `--cnn_checkpoint`: Path to CNN model checkpoint (if not provided, will find the best checkpoint automatically)
* `--ml_results_dir`: Directory containing traditional ML results (default: 'results/traditional_ml')

This script performs comprehensive failure analysis:
- Collects all misclassified images from CNN and traditional ML evaluations
- Saves images with filename structure: `{true_label}_pred_{predicted_label}_confidence_{score}.png`
- Organizes by predicted class and true class in separate folders
- Generates visual summary grids using matplotlib
- Creates a CSV and Markdown summary of most confused classes
- Includes per-class confusion rates

Results are saved in:
- `results/failure_analysis/summary.md`: Comprehensive markdown summary
- `results/failure_analysis/cnn/`: CNN misclassified samples
- `results/failure_analysis/ml/`: Traditional ML misclassified samples
- `results/failure_analysis/misclassified_samples/`: Combined misclassified samples
- `results/failure_analysis/cnn_confusion_grid.png`: CNN confusion grid visualization
- `results/failure_analysis/ml_confusion_grid.png`: ML confusion grid visualization

### Running Experiments
To run experiments with different hyperparameters, run:
```
python run_experiments.py
```

This script allows you to:
1. Run CNN experiments with different batch sizes and learning rates
2. Run traditional ML experiments
3. Run both CNN and traditional ML experiments
4. Analyze and visualize experiment results

## Results
Results are saved in the following directories:
- `saved_models/`: Trained model checkpoints
- `results/cnn/`: CNN evaluation results, including confusion matrices and classification reports
- `results/traditional/`: Traditional ML evaluation results
- `logs/`: Training logs for TensorBoard

To view TensorBoard logs, run:
```
tensorboard --logdir=logs
```

## Notes
* The project uses the NEU-DET dataset. Ensure it's properly downloaded and configured.
* The model is saved based on the best validation accuracy.
* For cloud training, consider adjusting the number of epochs, batch size, and potentially using GPU acceleration.
* The PyTorch Lightning implementation includes learning rate scheduling, early stopping, and comprehensive metrics tracking.
