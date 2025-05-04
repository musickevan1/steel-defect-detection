# model.py
# Defines the CNN model architecture using PyTorch and torchvision.

import torch
import torch.nn as nn
import torchvision.models as models

def build_model(num_classes=6, pretrained=True):
    """
    Builds a CNN model based on a pre-trained ResNet architecture.

    Args:
        num_classes (int): Number of output classes (defect types).
        pretrained (bool): Whether to use pre-trained weights from ImageNet.

    Returns:
        torch.nn.Module: The configured PyTorch model.
    """
    # Load a pre-trained ResNet model (e.g., ResNet18)
    # Using weights=models.ResNet18_Weights.DEFAULT for modern PyTorch
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        print("Loaded pre-trained ResNet18 model.")
    else:
        model = models.resnet18(weights=None)
        print("Loaded ResNet18 model without pre-trained weights.")

    # Freeze parameters in earlier layers if using pre-trained weights (optional, common for fine-tuning)
    # for param in model.parameters():
    #     param.requires_grad = False # Freeze all layers initially

    # Replace the final fully connected layer (classifier)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    print(f"Replaced final layer for {num_classes} classes.")

    # Example: Unfreeze last few layers if you froze them earlier
    # for param in model.layer4.parameters():
    #      param.requires_grad = True
    # for param in model.fc.parameters():
    #      param.requires_grad = True

    return model

if __name__ == '__main__':
    # Example usage: Instantiate the model and print its structure
    print("--- Building Model ---")
    num_defect_classes = 6 # From NEU-DET dataset
    cnn_model = build_model(num_classes=num_defect_classes, pretrained=True)

    # Print model summary (optional, requires torchinfo or similar)
    # from torchinfo import summary
    # summary(cnn_model, input_size=(1, 3, 224, 224)) # Example batch size 1

    print("\nModel architecture (final layer modified):")
    print(cnn_model.fc) # Show the modified final layer

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nModel will run on: {device}")
    cnn_model.to(device) # Move model to the appropriate device

    print("\n--- Model Ready ---")
