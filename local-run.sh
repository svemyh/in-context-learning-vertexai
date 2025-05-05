#!/bin/bash
set -e

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
# WANDB_MODE=disabled python src/train.py --config src/conf/toy.yaml

echo "Training complete."
