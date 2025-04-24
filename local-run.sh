#!/bin/bash
set -e

# 加载 .env 文件中的环境变量（如 WANDB_API_KEY）
if [ -f .env ]; then
  echo "Loading environment variables from .env"
  source .env
fi

CONFIG_FILE="src/conf/toy.yaml"

if [ ! -z "$1" ]; then
  CONFIG_FILE="$1"
fi

echo "Using config file: ${CONFIG_FILE}"

if [ ! -z "$WANDB_API_KEY" ]; then
  echo "Logging into Weights & Biases..."
  wandb login "$WANDB_API_KEY"
else
  echo "Warning: WANDB_API_KEY not found. Skipping wandb login."
fi

echo "Starting training..."
python src/train.py --config "$CONFIG_FILE"

echo "Training complete."
