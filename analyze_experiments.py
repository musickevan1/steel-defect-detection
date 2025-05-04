#!/usr/bin/env python
"""
Script to analyze and visualize experiment results from the experiments_log.csv file.
This script can handle incomplete or partial results.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def load_experiment_logs(log_path='results/cnn/experiments_log.csv'):
    """Load experiment logs from CSV file."""
    if not os.path.exists(log_path):
        print(f"Error: Experiment log file not found at {log_path}")
        return None
    
    try:
        df = pd.read_csv(log_path)
        print(f"Loaded {len(df)} experiment records from {log_path}")
        return df
    except Exception as e:
        print(f"Error loading experiment logs: {e}")
        return None

def analyze_experiments(df):
    """Analyze experiment results."""
    if df is None or len(df) == 0:
        print("No experiment data to analyze.")
        return
    
    print("\n=== Experiment Analysis ===")
    
    # Check for incomplete experiments
    incomplete = df[df['val_acc'].isna() | df['test_acc'].isna()].shape[0]
    complete = df[~(df['val_acc'].isna() | df['test_acc'].isna())].shape[0]
    
    print(f"Total experiments: {len(df)}")
    print(f"Complete experiments: {complete}")
    print(f"Incomplete experiments: {incomplete}")
    
    # If all experiments are incomplete, provide guidance
    if incomplete == len(df):
        print("\nAll experiments are incomplete. This could be due to:")
        print("1. Experiments are still running")
        print("2. Experiments failed to complete")
        print("3. Results were not properly logged")
        print("\nRecommendations:")
        print("- Check the log files in the 'logs' directory for errors")
        print("- Ensure that the training script is properly saving results")
        print("- Try running a single experiment with a small number of epochs to verify functionality")
        return
    
    # Analyze complete experiments
    complete_df = df[~(df['val_acc'].isna() | df['test_acc'].isna())]
    
    if len(complete_df) > 0:
        print("\n=== Complete Experiment Results ===")
        
        # Find best experiment
        best_idx = complete_df['val_acc'].idxmax()
        best_exp = complete_df.loc[best_idx]
        
        print(f"\nBest experiment:")
        print(f"  Batch size: {best_exp['batch_size']}")
        print(f"  Learning rate: {best_exp['learning_rate']}")
        print(f"  Validation accuracy: {best_exp['val_acc']:.4f}")
        print(f"  Test accuracy: {best_exp['test_acc']:.4f}")
        print(f"  Epochs trained: {best_exp['epochs_trained']}")
        print(f"  Best epoch: {best_exp['best_epoch']}")
        
        # Analyze by batch size
        print("\nResults by batch size:")
        batch_size_results = complete_df.groupby('batch_size').agg({
            'val_acc': ['mean', 'std', 'max'],
            'test_acc': ['mean', 'std', 'max']
        })
        print(batch_size_results)
        
        # Analyze by learning rate
        print("\nResults by learning rate:")
        lr_results = complete_df.groupby('learning_rate').agg({
            'val_acc': ['mean', 'std', 'max'],
            'test_acc': ['mean', 'std', 'max']
        })
        print(lr_results)
    
    return complete_df if len(complete_df) > 0 else None

def visualize_experiments(df, output_dir='results/analysis'):
    """Visualize experiment results."""
    if df is None or len(df) == 0:
        print("No experiment data to visualize.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have any complete experiments
    complete_df = df[~(df['val_acc'].isna() | df['test_acc'].isna())]
    
    if len(complete_df) == 0:
        print("No complete experiments to visualize.")
        
        # Visualize experiment status
        plt.figure(figsize=(10, 6))
        status_counts = [
            df[df['val_acc'].isna() & df['test_acc'].isna()].shape[0],  # No results
            df[~df['val_acc'].isna() & df['test_acc'].isna()].shape[0],  # Only validation
            df[df['val_acc'].isna() & ~df['test_acc'].isna()].shape[0],  # Only test
            df[~df['val_acc'].isna() & ~df['test_acc'].isna()].shape[0]  # Complete
        ]
        status_labels = ['No Results', 'Only Validation', 'Only Test', 'Complete']
        
        plt.bar(status_labels, status_counts)
        plt.title('Experiment Status')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'experiment_status.png'))
        plt.close()
        
        # Visualize experiment parameters
        plt.figure(figsize=(12, 6))
        param_counts = pd.DataFrame({
            'batch_size': df['batch_size'].value_counts(),
            'learning_rate': df['learning_rate'].value_counts()
        })
        param_counts.plot(kind='bar')
        plt.title('Experiment Parameters')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'experiment_parameters.png'))
        plt.close()
        
        return
    
    # Visualize validation accuracy by batch size and learning rate
    plt.figure(figsize=(12, 8))
    pivot_table = complete_df.pivot_table(
        values='val_acc', 
        index='batch_size', 
        columns='learning_rate'
    )
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='viridis')
    plt.title('Validation Accuracy by Batch Size and Learning Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'val_acc_heatmap.png'))
    plt.close()
    
    # Visualize test accuracy by batch size and learning rate
    plt.figure(figsize=(12, 8))
    pivot_table = complete_df.pivot_table(
        values='test_acc', 
        index='batch_size', 
        columns='learning_rate'
    )
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='viridis')
    plt.title('Test Accuracy by Batch Size and Learning Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_acc_heatmap.png'))
    plt.close()
    
    # Visualize epochs trained by batch size and learning rate
    plt.figure(figsize=(12, 8))
    pivot_table = complete_df.pivot_table(
        values='epochs_trained', 
        index='batch_size', 
        columns='learning_rate'
    )
    sns.heatmap(pivot_table, annot=True, fmt='d', cmap='viridis')
    plt.title('Epochs Trained by Batch Size and Learning Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'epochs_heatmap.png'))
    plt.close()
    
    # Visualize batch size vs. accuracy
    plt.figure(figsize=(10, 6))
    batch_size_results = complete_df.groupby('batch_size').agg({
        'val_acc': 'mean',
        'test_acc': 'mean'
    }).reset_index()
    
    x = np.arange(len(batch_size_results['batch_size']))
    width = 0.35
    
    plt.bar(x - width/2, batch_size_results['val_acc'], width, label='Validation Accuracy')
    plt.bar(x + width/2, batch_size_results['test_acc'], width, label='Test Accuracy')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Batch Size')
    plt.xticks(x, batch_size_results['batch_size'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_size_accuracy.png'))
    plt.close()
    
    # Visualize learning rate vs. accuracy
    plt.figure(figsize=(10, 6))
    lr_results = complete_df.groupby('learning_rate').agg({
        'val_acc': 'mean',
        'test_acc': 'mean'
    }).reset_index()
    
    x = np.arange(len(lr_results['learning_rate']))
    width = 0.35
    
    plt.bar(x - width/2, lr_results['val_acc'], width, label='Validation Accuracy')
    plt.bar(x + width/2, lr_results['test_acc'], width, label='Test Accuracy')
    
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Learning Rate')
    plt.xticks(x, lr_results['learning_rate'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate_accuracy.png'))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def generate_report(df, output_dir='results/analysis'):
    """Generate a report of experiment results."""
    if df is None or len(df) == 0:
        print("No experiment data for report.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate report
    report_path = os.path.join(output_dir, 'experiment_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Steel Defect Detection - Experiment Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Check for incomplete experiments
        incomplete = df[df['val_acc'].isna() | df['test_acc'].isna()].shape[0]
        complete = df[~(df['val_acc'].isna() | df['test_acc'].isna())].shape[0]
        
        f.write(f"**Total experiments:** {len(df)}\n")
        f.write(f"**Complete experiments:** {complete}\n")
        f.write(f"**Incomplete experiments:** {incomplete}\n\n")
        
        # If all experiments are incomplete, provide guidance
        if incomplete == len(df):
            f.write("## Incomplete Experiments\n\n")
            f.write("All experiments are incomplete. This could be due to:\n")
            f.write("1. Experiments are still running\n")
            f.write("2. Experiments failed to complete\n")
            f.write("3. Results were not properly logged\n\n")
            f.write("### Recommendations\n\n")
            f.write("- Check the log files in the 'logs' directory for errors\n")
            f.write("- Ensure that the training script is properly saving results\n")
            f.write("- Try running a single experiment with a small number of epochs to verify functionality\n\n")
            
            f.write("### Experiment Parameters\n\n")
            f.write("| Batch Size | Learning Rate | Status |\n")
            f.write("|------------|---------------|--------|\n")
            
            for _, row in df.iterrows():
                status = "No Results"
                if not pd.isna(row['val_acc']) and not pd.isna(row['test_acc']):
                    status = "Complete"
                elif not pd.isna(row['val_acc']):
                    status = "Only Validation"
                elif not pd.isna(row['test_acc']):
                    status = "Only Test"
                
                f.write(f"| {row['batch_size']} | {row['learning_rate']} | {status} |\n")
            
            return
        
        # Analyze complete experiments
        complete_df = df[~(df['val_acc'].isna() | df['test_acc'].isna())]
        
        if len(complete_df) > 0:
            f.write("## Complete Experiment Results\n\n")
            
            # Find best experiment
            best_idx = complete_df['val_acc'].idxmax()
            best_exp = complete_df.loc[best_idx]
            
            f.write("### Best Experiment\n\n")
            f.write(f"- **Batch size:** {best_exp['batch_size']}\n")
            f.write(f"- **Learning rate:** {best_exp['learning_rate']}\n")
            f.write(f"- **Validation accuracy:** {best_exp['val_acc']:.4f}\n")
            f.write(f"- **Test accuracy:** {best_exp['test_acc']:.4f}\n")
            f.write(f"- **Epochs trained:** {best_exp['epochs_trained']}\n")
            f.write(f"- **Best epoch:** {best_exp['best_epoch']}\n\n")
            
            f.write("### All Experiments\n\n")
            f.write("| Batch Size | Learning Rate | Val Acc | Test Acc | Epochs | Best Epoch |\n")
            f.write("|------------|---------------|---------|----------|--------|------------|\n")
            
            for _, row in complete_df.iterrows():
                f.write(f"| {row['batch_size']} | {row['learning_rate']} | ")
                f.write(f"{row['val_acc']:.4f} | {row['test_acc']:.4f} | ")
                f.write(f"{row['epochs_trained']} | {row['best_epoch']} |\n")
            
            f.write("\n### Visualizations\n\n")
            f.write("![Validation Accuracy Heatmap](val_acc_heatmap.png)\n\n")
            f.write("![Test Accuracy Heatmap](test_acc_heatmap.png)\n\n")
            f.write("![Batch Size vs Accuracy](batch_size_accuracy.png)\n\n")
            f.write("![Learning Rate vs Accuracy](learning_rate_accuracy.png)\n\n")
        
        print(f"Report generated at {report_path}")

def main():
    """Main function."""
    print("Steel Defect Detection - Experiment Analysis")
    print("===========================================")
    
    # Load experiment logs
    df = load_experiment_logs()
    
    if df is not None:
        # Analyze experiments
        complete_df = analyze_experiments(df)
        
        # Visualize experiments
        visualize_experiments(df)
        
        # Generate report
        generate_report(df)
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()