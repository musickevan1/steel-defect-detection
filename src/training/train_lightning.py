# train_lightning.py
# Handles the training and evaluation of the CNN model using PyTorch Lightning.

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Import from our other scripts
from data_loader import load_data
from lightning_model import SteelDefectClassifier

# --- Parse Arguments --- 
parser = argparse.ArgumentParser(description='Train CNN Model with PyTorch Lightning')
parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory to save models')
parser.add_argument('--results_dir', type=str, default='results/cnn', help='Directory to save results')
parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
parser.add_argument('--pretrained', type=bool, default=True, help='Whether to use pretrained weights')
parser.add_argument('--precision', type=str, default='32-true', help='Precision for training (16-mixed, 32-true, etc.)')
parser.add_argument('--early_stopping', type=bool, default=True, help='Whether to use early stopping')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
args = parser.parse_args()

# --- Main Execution ---
def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, test_dataset, class_names = load_data(batch_size=args.batch_size)
    
    if not train_loader or not val_loader:
        print("Failed to load data. Exiting.")
        exit()
    
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Initialize model
    print("Initializing model...")
    model = SteelDefectClassifier(
        num_classes=num_classes,
        learning_rate=args.lr,
        pretrained=args.pretrained
    )
    
    # Set class names for visualization
    model.class_names = class_names
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename='cnn-{epoch:02d}-{val_acc:.4f}',
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    if args.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor='val_acc',
            patience=args.patience,
            verbose=True,
            mode='max'
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name='steel_defect_cnn',
        version=None  # Auto-increment version
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',  # Automatically choose GPU if available
        devices=1,
        callbacks=callbacks,
        logger=logger,
        precision=args.precision,
        log_every_n_steps=10
    )
    
    # Train model
    print("\n--- Starting CNN Training with PyTorch Lightning ---")
    trainer.fit(model, train_loader, val_loader)
    
    # Test model
    print("\n--- Testing Best Model ---")
    test_results = trainer.test(model, test_loader)
    test_acc = test_results[0]['test_acc']
    
    # Get the number of epochs trained
    epochs_trained = trainer.current_epoch + 1
    
    # Get the best epoch from the checkpoint path
    best_epoch = None
    if checkpoint_callback.best_model_path:
        try:
            best_epoch = int(checkpoint_callback.best_model_path.split('epoch=')[1].split('-')[0])
        except:
            best_epoch = trainer.current_epoch
    
    print(f"\nBest model path: {checkpoint_callback.best_model_path}")
    print(f"Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Epochs trained: {epochs_trained}")
    print(f"Best epoch: {best_epoch}")
    
    print("\n--- CNN Training and Testing Finished ---")
    print(f"Results saved to {args.results_dir}")
    print(f"Logs saved to {args.log_dir}")

if __name__ == '__main__':
    main()