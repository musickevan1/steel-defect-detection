# lightning_model.py
# Implements a PyTorch Lightning module for the CNN model

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import classification_report

from model import build_model

class SteelDefectClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for steel defect classification.
    
    This module wraps the ResNet model and handles training, validation, and testing
    with appropriate metrics tracking.
    """
    
    def __init__(self, num_classes=6, learning_rate=0.001, pretrained=True):
        """
        Initialize the Lightning module.
        
        Args:
            num_classes (int): Number of defect classes to classify
            learning_rate (float): Learning rate for the optimizer
            pretrained (bool): Whether to use pretrained weights for the model
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Build the model
        self.model = build_model(num_classes=num_classes, pretrained=pretrained)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average='macro')
        
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average='macro')
        
        self.test_confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        
        # For storing predictions and targets for later analysis
        self.test_predictions = []
        self.test_targets = []
        self.class_names = None  # Will be set during testing
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler."""
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3, 
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    def training_step(self, batch, batch_idx):
        """Training step for a single batch."""
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        
        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = self.train_acc(preds, targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for a single batch."""
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        
        # Calculate metrics
        preds = torch.argmax(outputs, dim=1)
        acc = self.val_acc(preds, targets)
        f1 = self.val_f1(preds, targets)
        precision = self.val_precision(preds, targets)
        recall = self.val_recall(preds, targets)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_epoch=True)
        self.log('val_precision', precision, on_epoch=True)
        self.log('val_recall', recall, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step for a single batch."""
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        
        # Calculate metrics
        preds = torch.argmax(outputs, dim=1)
        acc = self.test_acc(preds, targets)
        f1 = self.test_f1(preds, targets)
        precision = self.test_precision(preds, targets)
        recall = self.test_recall(preds, targets)
        
        # Update confusion matrix
        self.test_confmat(preds, targets)
        
        # Store predictions and targets for later analysis
        self.test_predictions.append(preds.cpu())
        self.test_targets.append(targets.cpu())
        
        # Log metrics
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', acc, on_epoch=True)
        self.log('test_f1', f1, on_epoch=True)
        self.log('test_precision', precision, on_epoch=True)
        self.log('test_recall', recall, on_epoch=True)
        
        return loss
    
    def on_test_epoch_end(self):
        """Called at the end of the test epoch to compute final metrics and visualizations."""
        # Compute confusion matrix
        confmat = self.test_confmat.compute().cpu().numpy()
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join('results', 'cnn')
        os.makedirs(results_dir, exist_ok=True)
        
        # Plot and save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confmat, 
            annot=True, 
            fmt='g', 
            cmap='Blues',
            xticklabels=self.class_names if self.class_names else range(self.hparams.num_classes),
            yticklabels=self.class_names if self.class_names else range(self.hparams.num_classes)
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.test_predictions).numpy()
        all_targets = torch.cat(self.test_targets).numpy()
        
        # Generate and save classification report
        if self.class_names:
            report = classification_report(
                all_targets, 
                all_preds, 
                target_names=self.class_names,
                output_dict=True
            )
            
            # Save report as text
            with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
                f.write(classification_report(all_targets, all_preds, target_names=self.class_names))
            
            # Plot and save per-class metrics
            plt.figure(figsize=(12, 6))
            classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
            precision = [report[cls]['precision'] for cls in classes]
            recall = [report[cls]['recall'] for cls in classes]
            f1 = [report[cls]['f1-score'] for cls in classes]
            
            x = np.arange(len(classes))
            width = 0.25
            
            plt.bar(x - width, precision, width, label='Precision')
            plt.bar(x, recall, width, label='Recall')
            plt.bar(x + width, f1, width, label='F1-score')
            
            plt.xlabel('Class')
            plt.ylabel('Score')
            plt.title('Per-class Performance Metrics')
            plt.xticks(x, classes, rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'per_class_metrics.png'))
            plt.close()
        
        # Reset stored predictions and targets
        self.test_predictions = []
        self.test_targets = []