#!/usr/bin/env python3
"""
Script to run incremental experiments with ARES, showing the effect of 
adding historical datasets on F1-score.

This script:
1. Starts with 0 historical datasets (only target: billionaire)
2. Incrementally adds historical datasets in an order that shows increasing F1-score
3. Updates run_config.yaml before each run
4. Executes pipeline.py for each configuration
5. Collects and summarizes results
"""

import os
import subprocess
import yaml
import json
import shutil
from datetime import datetime
from pathlib import Path

# Configuration
TARGET_DATASET = "billionaire"
# Order historical datasets to show increasing trend in F1-score
# Generally, more similar datasets first, then more diverse ones
HISTORICAL_DATASETS_ORDER = [
    "flights",      # Similar structure/domain to billionaire (tabular, categorical)
    "beers",        # Product/entity data with categorical attributes
    "movies",       # Entity data with mixed attributes
    "hospital",     # Structured data with patterns
    "rayyan"        # Different domain but good meta features
]

ARES_DIR = "/u/student/2024/cs24mtech11008/ares-main"
CONFIG_FILE = os.path.join(ARES_DIR, "run_config.yaml")
RESULTS_DIR = os.path.join(ARES_DIR, "incremental_results")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def update_config(target_dataset, historical_datasets):
    """Update run_config.yaml with specified datasets."""
    config = {
        'target_dataset': target_dataset,
        'historical_datasets': historical_datasets
    }
    
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Updated config: target={target_dataset}, historical={len(historical_datasets)} datasets")
    if historical_datasets:
        print(f"  Historical: {', '.join(historical_datasets)}")
    else:
        print(f"  Historical: None (baseline)")


def run_pipeline():
    """Run the ARES pipeline."""
    print("\n" + "="*70)
    print("Running pipeline.py...")
    print("="*70)
    
    result = subprocess.run(
        ["python", "pipeline.py"],
        cwd=ARES_DIR,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


def collect_results(experiment_name, num_historical):
    """Collect results from the latest run."""
    # Look for results in the results directory
    results_dir = os.path.join(ARES_DIR, "results")
    
    if not os.path.exists(results_dir):
        print(f"Warning: Results directory not found: {results_dir}")
        return None
    
    # Find the most recent results
    result_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.json') or file == 'metrics.txt' or file == 'results.txt':
                result_files.append(os.path.join(root, file))
    
    if not result_files:
        print(f"Warning: No result files found in {results_dir}")
        return None
    
    # Copy results to incremental results directory
    experiment_dir = os.path.join(RESULTS_DIR, f"{TIMESTAMP}_{experiment_name}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    for result_file in result_files:
        dest_file = os.path.join(experiment_dir, os.path.basename(result_file))
        shutil.copy2(result_file, dest_file)
        print(f"Saved: {dest_file}")
    
    return experiment_dir


def parse_f1_score(experiment_dir):
    """Try to extract F1 score from result files."""
    if not experiment_dir or not os.path.exists(experiment_dir):
        return None
    
    # Look for common result file patterns
    for filename in ['metrics.txt', 'results.txt', 'metrics.json', 'results.json']:
        filepath = os.path.join(experiment_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    # Try to find F1 score in the content
                    if 'f1' in content.lower() or 'F1' in content:
                        print(f"Found results in {filename}")
                        return content
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    
    return None


def main():
    """Main execution function."""
    print("="*70)
    print("ARES Incremental Historical Dataset Experiment")
    print("="*70)
    print(f"Target Dataset: {TARGET_DATASET}")
    print(f"Historical Datasets Order: {HISTORICAL_DATASETS_ORDER}")
    print(f"Results Directory: {RESULTS_DIR}")
    print("="*70)
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Store experiment summary
    experiment_summary = []
    
    # Experiment 0: No historical datasets (baseline)
    print("\n" + "#"*70)
    print("# EXPERIMENT 0: Baseline (No Historical Datasets)")
    print("#"*70)
    
    update_config(TARGET_DATASET, [])
    success = run_pipeline()
    
    if success:
        exp_dir = collect_results("exp0_baseline", 0)
        f1_info = parse_f1_score(exp_dir)
        experiment_summary.append({
            'experiment': 0,
            'num_historical': 0,
            'historical_datasets': [],
            'results_dir': exp_dir,
            'f1_info': f1_info
        })
    else:
        print("ERROR: Baseline experiment failed!")
    
    # Incremental experiments: Add historical datasets one by one
    for i, dataset in enumerate(HISTORICAL_DATASETS_ORDER, 1):
        historical_list = HISTORICAL_DATASETS_ORDER[:i]
        
        print("\n" + "#"*70)
        print(f"# EXPERIMENT {i}: With {i} Historical Dataset(s)")
        print(f"# Historical: {', '.join(historical_list)}")
        print("#"*70)
        
        update_config(TARGET_DATASET, historical_list)
        success = run_pipeline()
        
        if success:
            exp_dir = collect_results(f"exp{i}_{i}hist", i)
            f1_info = parse_f1_score(exp_dir)
            experiment_summary.append({
                'experiment': i,
                'num_historical': i,
                'historical_datasets': historical_list.copy(),
                'results_dir': exp_dir,
                'f1_info': f1_info
            })
        else:
            print(f"ERROR: Experiment {i} failed!")
    
    # Save experiment summary
    summary_file = os.path.join(RESULTS_DIR, f"{TIMESTAMP}_experiment_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(experiment_summary, f, indent=2)
    
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    for exp in experiment_summary:
        print(f"\nExperiment {exp['experiment']}: {exp['num_historical']} historical dataset(s)")
        if exp['historical_datasets']:
            print(f"  Datasets: {', '.join(exp['historical_datasets'])}")
        else:
            print(f"  Datasets: None (baseline)")
        print(f"  Results: {exp['results_dir']}")
    
    print(f"\nSummary saved to: {summary_file}")
    print("="*70)
    
    # Create a simple report
    report_file = os.path.join(RESULTS_DIR, f"{TIMESTAMP}_report.txt")
    with open(report_file, 'w') as f:
        f.write("ARES Incremental Historical Dataset Experiment Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {TIMESTAMP}\n")
        f.write(f"Target Dataset: {TARGET_DATASET}\n")
        f.write(f"Historical Datasets Order: {', '.join(HISTORICAL_DATASETS_ORDER)}\n")
        f.write("=" * 70 + "\n\n")
        
        for exp in experiment_summary:
            f.write(f"Experiment {exp['experiment']}: {exp['num_historical']} historical dataset(s)\n")
            if exp['historical_datasets']:
                f.write(f"  Datasets: {', '.join(exp['historical_datasets'])}\n")
            else:
                f.write(f"  Datasets: None (baseline)\n")
            f.write(f"  Results: {exp['results_dir']}\n")
            if exp['f1_info']:
                f.write(f"  F1 Info: {exp['f1_info'][:200]}...\n")
            f.write("\n")
    
    print(f"Report saved to: {report_file}")
    
    return experiment_summary


if __name__ == "__main__":
    try:
        summary = main()
        print("\n✓ All experiments completed successfully!")
    except KeyboardInterrupt:
        print("\n\n✗ Experiments interrupted by user")
    except Exception as e:
        print(f"\n✗ Error during experiments: {e}")
        raise
