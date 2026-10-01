#!/bin/bash
# Wrapper script to run incremental historical dataset experiments with ARES

set -e  # Exit on error

echo "=================================="
echo "ARES Incremental Experiment Runner"
echo "=================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not available"
    exit 1
fi

# Activate conda environment
echo "Activating conda environment: ares"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ares

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'ares'"
    exit 1
fi

echo "Environment activated successfully"
echo ""

# Change to ares-main directory
cd /u/student/2024/cs24mtech11008/ares-main

# Run the incremental experiment script
echo "Starting incremental experiments..."
python run_incremental_historical.py

echo ""
echo "=================================="
echo "Experiments completed!"
echo "=================================="
