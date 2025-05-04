# run_final_evaluation.py
# Script to run the final test evaluation with a mock checkpoint for demonstration

import os
import argparse
import subprocess
import torch
import torch.nn as nn
from model import build_model
from lightning_model import SteelDefectClassifier

def create_mock_checkpoint():
    """Create a mock checkpoint if no real checkpoint exists."""
    print("Creating mock checkpoint for demonstration...")
    
    # Create directories
    os.makedirs('saved_models', exist_ok=True)
    
    # Check if a checkpoint already exists
    if any(f.endswith('.ckpt') for f in os.listdir('saved_models')):
        print("Checkpoint already exists in saved_models/. Using existing checkpoint.")
        return
    
    # Create a mock model
    model = build_model(num_classes=6, pretrained=False)
    
    # Create a Lightning module
    lightning_model = SteelDefectClassifier(num_classes=6, learning_rate=0.001, pretrained=False)
    
    # Set the model weights
    lightning_model.model = model
    
    # Save the checkpoint
    checkpoint_path = os.path.join('saved_models', 'mock_model-epoch=01-val_acc=0.8500.ckpt')
    torch.save(lightning_model.state_dict(), checkpoint_path)
    
    print(f"Mock checkpoint created at {checkpoint_path}")
    return checkpoint_path

def main():
    parser = argparse.ArgumentParser(description='Run Final Test Evaluation')
    parser.add_argument('--create_mock', action='store_true', help='Create a mock checkpoint for demonstration')
    parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--calc_inference_time', action='store_true', help='Calculate average inference time per image')
    args = parser.parse_args()
    
    checkpoint_path = args.checkpoint
    
    # Create a mock checkpoint if requested
    if args.create_mock:
        checkpoint_path = create_mock_checkpoint()
    
    # Run the final test evaluation
    cmd = ["python", "final_test_evaluation.py"]
    
    if checkpoint_path:
        cmd.extend(["--checkpoint", checkpoint_path])
    
    if args.calc_inference_time:
        cmd.append("--calc_inference_time")
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == '__main__':
    main()