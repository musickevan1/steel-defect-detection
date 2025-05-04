# CNN-Based Defect Detection in Manufacturing Images
## Progress Report

**Group Member:** [Your Name]

### Introduction
Automated defect detection in manufacturing is essential for quality control and reducing production costs. Traditional methods require extensive human intervention, which is time-consuming, subjective, and prone to errors. This project implements a Convolutional Neural Network (CNN) approach to classify and detect defects in steel surface images. The NEU-DET dataset contains images of six different types of steel surface defects: crazing, inclusion, patches, pitted surface, rolled-in scale, and scratches. Through the application of deep learning techniques, specifically CNNs, this project aims to develop a model that can accurately identify these defects with minimal human intervention.

The significance of this work extends beyond academic interest, as manufacturing quality control represents a critical industrial challenge where automation can provide substantial benefits in terms of consistency, throughput, and cost reduction.

### Approach

#### Model Architecture
This project utilizes a transfer learning approach with a pre-trained ResNet18 model as the backbone. After evaluating several potential architectures, ResNet18 was selected for its balance of effectiveness in image classification tasks and computational efficiency. The architecture consists of:

1. **Base Network**: Pre-trained ResNet18 model with weights from ImageNet
2. **Modifications**: 
   - The final fully connected layer has been replaced to output 6 classes (corresponding to the 6 defect types)
   - The input expects RGB images of size 224×224 pixels
3. **Implementation Details**:
   - The model preserves the feature extraction capabilities of ResNet18
   - Only the final classification layer is fully retrained, while earlier layers maintain their pre-trained weights

An interesting observation during implementation was the effectiveness of transfer learning from ImageNet weights to steel defect detection, despite the significant domain difference. This suggests that low-level features learned from natural images remain valuable for industrial surface defect detection.

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
The NEU-DET (Northeastern University Defect Detection) dataset serves as the foundation for this project. Initial exploration of the dataset revealed varying levels of visual distinctiveness among defect types, with some categories like scratches being more visually apparent than others such as inclusions.

The dataset contains:

1. **Content**: Steel surface defect images with six different defect types
2. **Structure**: 
   - Training set: Located in NEU-DET/train/images
   - Validation set: Located in NEU-DET/validation/images
   - Each image is accompanied by XML annotation files

The XML annotations provide valuable information about defect locations, though for the current classification task, only the class labels are utilized.

#### Preprocessing and Data Augmentation
Several preprocessing techniques were implemented to improve model performance. Initial experiments without augmentation demonstrated clear signs of overfitting, with diverging training and validation accuracy curves after approximately 7-8 epochs. 

The preprocessing pipeline includes:

1. **Image Resizing**: All images are resized to 224×224 pixels to match the input requirements of the ResNet18 model
2. **Data Augmentation** (applied only to training data):
   - Random horizontal flipping
   - Random rotation (up to 10 degrees)
3. **Normalization**: Images are normalized using ImageNet mean and standard deviation values (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

Rotation angles were deliberately limited to 10 degrees based on domain knowledge that excessive rotation could distort the natural appearance of defects, potentially creating unrealistic patterns.

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
The dataset is organized as follows:
1. **Training Set**: Used for model training
2. **Validation Set**: Split into:
   - Validation subset: Used for hyperparameter tuning and early stopping
   - Test subset: Used for final model evaluation
3. **Split Ratio**: The validation set is further split with a configurable ratio (default 0.5), meaning 50% for validation and 50% for testing

#### Evaluation Metrics
While accuracy provides a high-level performance indicator, the class imbalance present in the NEU-DET dataset necessitates more nuanced evaluation metrics. The evaluation strategy includes:

1. **Accuracy**: Overall classification accuracy
2. **Classification Report**: Precision, recall, and F1-score for each defect class
3. **Loss Curves**: Training and validation loss over epochs to analyze convergence and potential overfitting

The per-class metrics are particularly important for this application, as certain defect types may have more significant implications for product quality and safety than others.

#### Hyperparameters
Hyperparameter tuning has been approached systematically, with initial experiments revealing sensitivity to learning rate selection. The hyperparameters being tuned include:

1. **Learning Rate**: Default is 0.001, with experiments planned for 0.0005 and 0.0001
2. **Batch Size**: Default is 32, with experiments for 16 and 64
3. **Number of Epochs**: Default is 15, with early stopping based on validation accuracy
4. **Optimizer**: Adam optimizer is currently used, with potential experiments using SGD

Initial experiments with a learning rate of 0.001 showed instability in the loss function after several epochs, suggesting that a lower learning rate might provide more stable convergence.

#### Comparison with Traditional Methods
To establish a comprehensive evaluation framework, the CNN approach is being compared with traditional machine learning methods:

1. **Feature Extraction**:
   - HOG (Histogram of Oriented Gradients)
   - LBP (Local Binary Patterns)
2. **Classifiers**:
   - SVM (Support Vector Machine)
   - Random Forest

Preliminary results indicate that while traditional methods achieve reasonable performance (approximately 76% accuracy with HOG+SVM), they struggle with distinguishing between visually similar defect categories. The computational requirements for feature extraction also present a significant bottleneck in the traditional pipeline.

### Current Progress and Challenges
The CNN implementation is complete, including the data loading pipeline, model architecture, training loop, and evaluation scripts. Initial training runs have been performed, and the model is being fine-tuned based on validation performance.

Several technical challenges have been addressed during implementation:
1. **Memory Management**: Resolving CUDA memory allocation issues required careful configuration of DataLoader parameters
2. **Class Imbalance**: The dataset exhibits imbalance across defect categories, requiring consideration in the evaluation metrics
3. **Computational Efficiency**: Feature extraction for traditional methods proved computationally intensive, highlighting an advantage of end-to-end deep learning approaches

The traditional machine learning comparison is also implemented and ready for comparative analysis, though the feature extraction process has proven significantly more time-intensive than the CNN forward pass.

### Next Steps
The remaining work for this project includes:

1. Completing hyperparameter optimization to maximize model performance
2. Conducting comprehensive evaluation on the test set, with particular focus on per-class performance metrics
3. Finalizing the comparison between CNN and traditional approaches, quantifying the performance gap
4. Analyzing failure cases to identify potential improvements to the model or preprocessing pipeline
5. Documenting findings and preparing the final report with visualizations of model performance
