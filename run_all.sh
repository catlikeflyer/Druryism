#!/bin/bash
# Druryism pipeline automation script
# Author: Principal AI Software Engineer

set -e

# Resolve script directory path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================================================="
# Disclaimer & System Warning
echo "⚠️  System Compatibility Warning:"
echo "   This project works exclusively on Apple Silicon (M-series) Mac devices"
echo "   utilizing Apple's MLX machine learning framework."
echo "========================================================================="
echo ""

# 1. Dataset Generation Check
echo "=== Step 1: Checking Dataset Files ==="
if [ ! -d "data" ] || [ ! -f "data/train.jsonl" ] || [ ! -f "data/valid.jsonl" ]; then
    echo "Dataset not found. Running dataset generation..."
    python3 generate_dataset.py
else
    echo "Dataset directory and training files already exist. Skipping dataset generation."
fi
echo ""

# 2. QLoRA Adapters Training Check
echo "=== Step 2: Checking QLoRA Adapters ==="
if [ ! -d "drury_adapters" ] || [ ! -f "drury_adapters/adapters.safetensors" ]; then
    echo "Fine-tuned adapters not found. Starting style adapter fine-tuning..."
    # Ensure train.sh is executable and run it
    chmod +x train.sh
    ./train.sh
else
    echo "Fine-tuned Drury adapters already exist in './drury_adapters'. Skipping training."
fi
echo ""

# 3. Model Evaluation Suite
echo "=== Step 3: Running Model Evaluation Suite ==="
echo "Evaluating baseline SLM vs. fine-tuned SLM on 10 scenes..."
python3 eval.py
echo ""

# 4. End-to-End Pipeline Execution
echo "=== Step 4: Running Vision-to-Commentary Pipeline ==="
echo "Executing main pipeline on local image scene..."
python3 pipeline.py
echo ""

echo "========================================================================="
echo "🎉 Druryism Pipeline Orchestration Completed Successfully!"
echo "   - Interactive Showcase Page: docs/index.html"
echo "   - Comparative Evaluation Report: docs/eval_report.html"
echo "========================================================================="
