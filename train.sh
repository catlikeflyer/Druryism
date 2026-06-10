#!/bin/bash
# fine-tuning runner script
# Author: Principal AI Software Engineer

set -e

echo "=== Step 1: Installing/Updating MLX training dependencies ==="
pip install -U "mlx-lm[train]"

echo "=== Step 2: Generating fine-tuning dataset ==="
if [ ! -f "data/train.jsonl" ]; then
    python3 generate_dataset.py
else
    echo "Dataset files already exist in './data'."
fi

echo "=== Step 3: Running MLX QLoRA Trainer on Apple Silicon ==="
# We train for 200 iterations to adapt style while staying lightweight.
# Output directory is configured as './drury_adapters' to align with our pipeline script.
python3 -m mlx_lm lora \
  --model mlx-community/Phi-3-mini-4k-instruct-4bit \
  --train \
  --data ./data \
  --iters 200 \
  --batch-size 4 \
  --adapter-path ./drury_adapters

echo "=== Process Complete! ==="
echo "Your fine-tuned Drury adapters are saved in: ./drury_adapters"
echo "You can now execute your pipeline: python3 pipeline.py"
