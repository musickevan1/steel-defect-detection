# run_experiments.py
# Script to run experiments with different hyperparameters

import os
import subprocess
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config import BATCH_SIZES, LEARNING_RATES, RESULTS_DIR, LOG_DIR
from traditional_ml_enhanced import run_traditional_ml_pipeline

def run_cnn_experiments():
    """Run CNN experiments with different hyperparameters."""
    print("=== Running CNN Experiments ===")
    
    # Create results directory
    results_dir = os.path.join(RESULTS_DIR, 'cnn')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create experiments log
    experiments_log = os.path.join(results_dir, 'experiments_log.csv')
    if not os.path.exists(experiments_log):
        with open(experiments_log, 'w') as f:
            f.write('timestamp,batch_size,learning_rate,val_acc,test_acc,epochs_trained,best_epoch\n')
    
    # Run experiments for each combination of batch size and learning rate
    for batch_size, lr in itertools.product(BATCH_SIZES, LEARNING_RATES):
        print(f"\nRunning experiment with batch_size={batch_size}, lr={lr}")
        
        # Create experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"cnn_bs{batch_size}_lr{lr}_{timestamp}"
        
        # Run training
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to the train_lightning.py script
        train_script = os.path.join(script_dir, "train_lightning.py")
        
        cmd = [
            "python", train_script,
            "--batch_size", str(batch_size),
            "--lr", str(lr),
            "--save_dir", os.path.join(RESULTS_DIR, 'cnn', experiment_name, 'checkpoints'),
            "--results_dir", os.path.join(RESULTS_DIR, 'cnn', experiment_name, 'results'),
            "--log_dir", os.path.join(LOG_DIR, experiment_name)
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Use a different approach to capture output
        try:
            # Run the command and capture output in real-time
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Capture output in real-time
            stdout_lines = []
            stderr_lines = []
            
            # Read stdout
            for line in process.stdout:
                line = line.strip()
                print(f"STDOUT: {line}")
                stdout_lines.append(line)
            
            # Read stderr
            for line in process.stderr:
                line = line.strip()
                print(f"STDERR: {line}")
                stderr_lines.append(line)
            
            # Wait for process to complete
            process.wait()
            
            # Print process return code
            print(f"Process return code: {process.returncode}")
            
            # Combine output
            output = '\n'.join(stdout_lines)
            error_output = '\n'.join(stderr_lines)
            
        except Exception as e:
            print(f"Error running command: {e}")
            output = ""
            error_output = str(e)
        
        # Print stderr if there's any output
        if error_output:
            print("\nCommand stderr:")
            print(error_output)
        
        # Parse output to get validation accuracy, test accuracy, etc.
        val_acc = None
        test_acc = None
        epochs_trained = None
        best_epoch = None
        
        # Print the output for debugging
        print("\nCommand output:")
        print(output)
        
        for line in output.split('\n'):
            if "Best validation accuracy:" in line:
                try:
                    val_acc = float(line.split(": ")[1])
                except:
                    print(f"Warning: Could not parse validation accuracy from line: {line}")
            elif "Test accuracy:" in line:
                try:
                    test_acc = float(line.split(": ")[1])
                except:
                    print(f"Warning: Could not parse test accuracy from line: {line}")
            elif "Epochs trained:" in line:
                try:
                    epochs_trained = int(line.split(": ")[1])
                except:
                    print(f"Warning: Could not parse epochs trained from line: {line}")
            elif "Best epoch:" in line:
                try:
                    best_epoch = int(line.split(": ")[1])
                except:
                    print(f"Warning: Could not parse best epoch from line: {line}")
            # Fallback parsing for older format
            elif "Epoch" in line and "step=" in line:
                try:
                    epochs_trained = int(line.split("Epoch ")[1].split("/")[0])
                except:
                    pass
            elif "Best model path:" in line and "epoch=" in line:
                try:
                    best_epoch = int(line.split("epoch=")[1].split("-")[0])
                except:
                    pass
        
        # Log results
        with open(experiments_log, 'a') as f:
            f.write(f"{timestamp},{batch_size},{lr},{val_acc},{test_acc},{epochs_trained},{best_epoch}\n")
        
        print(f"Experiment completed. Results saved to {experiments_log}")

def run_traditional_ml_experiments():
    """Run traditional ML experiments with enhanced pipeline."""
    print("=== Running Traditional ML Experiments ===")
    
    # Create results directory
    results_dir = os.path.join(RESULTS_DIR, 'traditional_ml')
    os.makedirs(results_dir, exist_ok=True)
    
    # Run enhanced traditional ML pipeline
    print("Running enhanced traditional ML pipeline...")
    results = run_traditional_ml_pipeline(
        use_same_test_split=True,
        use_grid_search=True,
        use_cross_val=True
    )
    
    # Log results
    experiments_log = os.path.join(results_dir, 'experiments_log.csv')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not os.path.exists(experiments_log):
        with open(experiments_log, 'w') as f:
            f.write('timestamp,model,feature,accuracy,precision,recall,f1,inference_time_ms\n')
    
    # Log HOG+SVM results
    with open(experiments_log, 'a') as f:
        f.write(f"{timestamp},SVM,HOG,{results['hog_svm']['accuracy']:.4f},{results['hog_svm']['precision']:.4f},"
                f"{results['hog_svm']['recall']:.4f},{results['hog_svm']['f1']:.4f},"
                f"{results['hog_svm']['avg_inference_time']*1000:.2f}\n")
    
    # Log LBP+RF results
    with open(experiments_log, 'a') as f:
        f.write(f"{timestamp},RF,LBP,{results['lbp_rf']['accuracy']:.4f},{results['lbp_rf']['precision']:.4f},"
                f"{results['lbp_rf']['recall']:.4f},{results['lbp_rf']['f1']:.4f},"
                f"{results['lbp_rf']['avg_inference_time']*1000:.2f}\n")
    
    print(f"Traditional ML experiments completed. Results saved to {results_dir}")

def analyze_results():
    """Analyze and visualize experiment results."""
    print("=== Analyzing Results ===")
    
    # Load CNN experiment results
    cnn_log_path = os.path.join(RESULTS_DIR, 'cnn', 'experiments_log.csv')
    if os.path.exists(cnn_log_path):
        cnn_results = pd.read_csv(cnn_log_path)
        
        # Create heatmap of validation accuracy
        plt.figure(figsize=(10, 8))
        pivot_table = cnn_results.pivot_table(
            values='val_acc', 
            index='batch_size', 
            columns='learning_rate'
        )
        sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='viridis')
        plt.title('Validation Accuracy by Batch Size and Learning Rate')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'cnn', 'hyperparameter_heatmap.png'))
        
        # Find best hyperparameters
        best_idx = cnn_results['val_acc'].idxmax()
        best_config = cnn_results.iloc[best_idx]
        
        print(f"Best CNN configuration:")
        print(f"  Batch size: {best_config['batch_size']}")
        print(f"  Learning rate: {best_config['learning_rate']}")
        print(f"  Validation accuracy: {best_config['val_acc']:.4f}")
        print(f"  Test accuracy: {best_config['test_acc']:.4f}")
        
        # Save best configuration
        with open(os.path.join(RESULTS_DIR, 'cnn', 'best_config.txt'), 'w') as f:
            f.write(f"Best CNN configuration:\n")
            f.write(f"Batch size: {best_config['batch_size']}\n")
            f.write(f"Learning rate: {best_config['learning_rate']}\n")
            f.write(f"Validation accuracy: {best_config['val_acc']:.4f}\n")
            f.write(f"Test accuracy: {best_config['test_acc']:.4f}\n")
            f.write(f"Epochs trained: {best_config['epochs_trained']}\n")
            f.write(f"Best epoch: {best_config['best_epoch']}\n")
    else:
        print(f"No CNN experiment results found at {cnn_log_path}")

def main():
    """Main function to run experiments and analyze results."""
    print("=== Steel Defect Detection Experiments ===")
    
    # Ask user what to run
    print("\nWhat would you like to run?")
    print("1. CNN experiments")
    print("2. Traditional ML experiments")
    print("3. Both CNN and traditional ML experiments")
    print("4. Analyze results")
    print("5. Run final test evaluation")
    print("6. Run failure analysis")
    
    choice = input("Enter your choice (1-4): ")
    
    if choice == '1':
        run_cnn_experiments()
    elif choice == '2':
        run_traditional_ml_experiments()
    elif choice == '3':
        run_cnn_experiments()
        run_traditional_ml_experiments()
    elif choice == '4':
        analyze_results()
    elif choice == '5':
        # Run final test evaluation
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to the final_test_evaluation.py script
        eval_script = os.path.join(script_dir, "final_test_evaluation.py")
        
        cmd = ["python", eval_script, "--calc_inference_time"]
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd)
    elif choice == '6':
        # Run failure analysis
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to the failure_analysis.py script
        failure_script = os.path.join(script_dir, "failure_analysis.py")
        
        cmd = ["python", failure_script]
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd)
    else:
        print("Invalid choice. Exiting.")

if __name__ == '__main__':
    main()