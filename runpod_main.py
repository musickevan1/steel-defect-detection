#!/usr/bin/env python
"""
Main script for running experiments on RunPod.
This script:
1. Sets up the RunPod environment
2. Runs the experiments
3. Analyzes the results
"""

import os
import sys
import subprocess
import argparse
import time

def run_command(command, description=None):
    """Run a command and print its output."""
    if description:
        print(f"\n=== {description} ===\n")
    
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        return False
    
    return True

def setup_environment():
    """Set up the RunPod environment."""
    # Set the RUNPOD environment variable
    os.environ['RUNPOD'] = '1'
    
    # Run the setup script
    return run_command("python runpod/setup_runpod.py", "Setting up RunPod environment")

def run_experiments(args):
    """Run the experiments."""
    if args.debug:
        # Run a single experiment for debugging
        command = f"python debug_experiment.py --batch_size {args.batch_size} --lr {args.lr} --epochs {args.epochs}"
        return run_command(command, "Running debug experiment")
    else:
        # Run all experiments
        return run_command("python run_experiments.py", "Running experiments")

def analyze_results():
    """Analyze the results."""
    return run_command("python run_analysis.py", "Analyzing results")

def run_failure_analysis():
    """Run failure analysis."""
    return run_command("python run_failure_analysis.py", "Running failure analysis")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run experiments on RunPod")
    parser.add_argument("--debug", action="store_true", help="Run a single experiment for debugging")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for debug experiment")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for debug experiment")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for debug experiment")
    parser.add_argument("--skip_setup", action="store_true", help="Skip environment setup")
    parser.add_argument("--skip_analysis", action="store_true", help="Skip results analysis")
    parser.add_argument("--skip_failure_analysis", action="store_true", help="Skip failure analysis")
    args = parser.parse_args()
    
    print("=== RunPod Main Script ===")
    
    # Record start time
    start_time = time.time()
    
    # Setup environment
    if not args.skip_setup:
        if not setup_environment():
            print("Environment setup failed. Exiting.")
            return
    
    # Run experiments
    if not run_experiments(args):
        print("Experiments failed. Exiting.")
        return
    
    # Analyze results
    if not args.skip_analysis:
        if not analyze_results():
            print("Results analysis failed.")
    
    # Run failure analysis
    if not args.skip_failure_analysis:
        if not run_failure_analysis():
            print("Failure analysis failed.")
    
    # Record end time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Print summary
    print("\n=== RunPod Execution Summary ===")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("All tasks completed.")

if __name__ == "__main__":
    main()