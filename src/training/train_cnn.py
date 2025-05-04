# train_cnn.py
# Handles the training and validation of the CNN model.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import copy

# Import from our other scripts
from data_loader import load_data, IMG_SIZE # Assuming load_data returns loaders and class_names
from model import build_model

import argparse

# --- Parse --- 
parser = argparse.ArgumentParser(description='Train CNN Model')
parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory to save models')
parser.add_argument('--model_name', type=str, default='cnn_best_model.pth', help='Name of the best model file')
args = parser.parse_args()

# --- Configuration ---
MODEL_SAVE_DIR = args.save_dir
BEST_MODEL_NAME = args.model_name
NUM_EPOCHS = args.epochs
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size

# --- Helper Functions ---
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total

# --- Training and Validation Loop ---
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=NUM_EPOCHS):
    """Trains the model and validates it after each epoch."""
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Ensure the save directory exists
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_SAVE_DIR, BEST_MODEL_NAME)

    print("\n--- Starting CNN Training ---")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        # --- Training Phase ---
        model.train() # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

            if (i + 1) % 20 == 0: # Print progress every 20 batches
                 print(f"  Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")


        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # --- Validation Phase ---
        model.eval() # Set model to evaluate mode
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total_samples = 0

        with torch.no_grad(): # No need to track gradients during validation
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)
                val_total_samples += labels.size(0)

        val_epoch_loss = val_running_loss / val_total_samples
        val_epoch_acc = val_running_corrects.double() / val_total_samples
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        # Save the best model based on validation accuracy
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, model_save_path)
            print(f"*** Best model saved to {model_save_path} (Val Acc: {best_acc:.4f}) ***")

    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# --- Main Execution ---
if __name__ == '__main__':
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    # Use the batch size defined in this script for consistency
    train_loader, val_loader, test_loader, test_dataset, class_names = load_data(batch_size=BATCH_SIZE)

    if not train_loader or not val_loader:
        print("Failed to load data. Exiting.")
        exit()

    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")

    # Build model
    print("Building model...")
    model = build_model(num_classes=num_classes, pretrained=True)
    model = model.to(device) # Move model to device

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Example of optimizing only the final layer (fine-tuning)
    # optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=NUM_EPOCHS)

    print("\n--- CNN Training Finished ---")

    # TODO: Add evaluation on the test set using the best model
