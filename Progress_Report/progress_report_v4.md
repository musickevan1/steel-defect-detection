# CNN-Based Defect Detection in Manufacturing Images
## Progress Report

**Group Member:** Evan Musick

### Introduction
Automated defect detection in manufacturing is crucial for quality control. Traditional methods are often slow and error-prone. This project uses a Convolutional Neural Network (CNN) to classify six types of steel surface defects (crazing, inclusion, patches, pitted surface, rolled-in scale, scratches) from the NEU-DET dataset. The goal is an accurate, automated defect detection model using CNNs. This addresses a key industrial challenge by improving consistency and reducing costs.

### Approach

#### Model Architecture
A transfer learning approach with a pre-trained ResNet18 backbone is used, chosen for its balance of performance and efficiency. Key aspects:

1.  **Base Network**: Pre-trained ResNet18 (ImageNet weights).
2.  **Modifications**: Final fully connected layer replaced for 6 defect classes; input size 224×224 pixels.
3.  **Training**: Only the final classification layer is retrained initially.

Transfer learning proved effective despite the domain difference between ImageNet and steel defects.

```python
# Key model architecture code
def build_model(num_classes=6, pretrained=True):
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)
    
    # Replace the final fully connected layer (classifier)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
```

#### Dataset
The NEU-DET dataset is used, containing images of the six steel surface defect types.
- **Structure**: Organized into training (NEU-DET/train/images) and validation (NEU-DET/validation/images) sets, with XML annotations (currently using only class labels).
- **Observations**: Defect types show varying visual distinctiveness.

#### Preprocessing and Data Augmentation
To improve performance and mitigate overfitting observed in initial runs:
1.  **Resizing**: Images resized to 224×224 for ResNet18 input.
2.  **Augmentation** (Training only): Random horizontal flips, random rotation (max 10 degrees).
3.  **Normalization**: Using ImageNet mean/std values.

Limited rotation helps preserve realistic defect appearance.

```python
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Evaluation Plan

#### Dataset Splitting
- **Training Set**: Used for model training.
- **Validation Set**: Split 50/50 (configurable) into validation (for tuning/early stopping) and test (for final evaluation) subsets.

#### Evaluation Metrics
Due to class imbalance, evaluation includes:
1.  **Accuracy**: Overall performance.
2.  **Classification Report**: Per-class precision, recall, F1-score.
3.  **Loss Curves**: Training/validation loss over epochs (convergence/overfitting analysis).

Per-class metrics are crucial due to varying defect significance.

#### Hyperparameters
Systematic tuning focuses on:
1.  **Learning Rate**: Exploring 0.001, 0.0005, 0.0001 (initial tests showed sensitivity).
2.  **Batch Size**: Exploring 16, 32, 64.
3.  **Epochs**: Target 15, with early stopping based on validation accuracy.
4.  **Optimizer**: Currently Adam, potentially comparing with SGD.

#### Comparison with Traditional Methods
CNN performance will be compared against traditional ML:
1.  **Features**: HOG, LBP.
2.  **Classifiers**: SVM, Random Forest.

Preliminary results show traditional methods (e.g., HOG+SVM ~76% accuracy) struggle with similar defects and are computationally slower for feature extraction.

### Current Progress and Challenges
- **Progress**: CNN pipeline (data loading, model, training, evaluation) is implemented. Initial training runs completed; fine-tuning underway. Traditional ML comparison pipeline also implemented.
- **Challenges Addressed**: CUDA memory management, handling class imbalance in evaluation, noting computational cost of traditional feature extraction.

### Next Steps
1.  Complete hyperparameter optimization.
2.  Perform final evaluation on the test set (focus on per-class metrics).
3.  Finalize CNN vs. traditional methods comparison.
4.  Analyze failure cases for potential improvements.
5.  Document findings and prepare the final report.
