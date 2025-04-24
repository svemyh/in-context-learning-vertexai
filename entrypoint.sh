#!/bin/bash
set -e

# Default configuration file
CONFIG_FILE=${CONFIG_FILE:-src/conf/toy.yaml}

# Set up cloud storage integration if environment variables are provided
if [ ! -z "$GCS_BUCKET" ]; then
    # Create output directory structure for storing results
    RUN_ID=$(uuidgen)
    GCS_OUTPUT_DIR="gs://${GCS_BUCKET}/runs/${RUN_ID}"
    echo "Run ID: ${RUN_ID}"
    echo "Results will be stored in: ${GCS_OUTPUT_DIR}"
    
    # Update wandb configuration to use GCS path
    sed -i "s|your-entity|${WANDB_ENTITY:-'vertex-ai-run'}|g" src/conf/wandb.yaml
    
    # Create a modified config file with output directory pointing to GCS
    CONFIG_WITH_GCS="/tmp/config_with_gcs.yaml"
    cat $CONFIG_FILE > $CONFIG_WITH_GCS
    echo "out_dir: ${GCS_OUTPUT_DIR}" >> $CONFIG_WITH_GCS
    CONFIG_FILE=$CONFIG_WITH_GCS
fi

# Activate wandb if credentials are provided
if [ ! -z "$WANDB_API_KEY" ]; then
    wandb login $WANDB_API_KEY
fi

# Check for and convert --config-file argument to --config for train.py
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--config-file" ]]; then
        ARGS+=("--config")
    else
        ARGS+=("$arg")
    fi
done

# Run the training
echo "Starting training with config: $CONFIG_FILE"
python src/train.py --config $CONFIG_FILE "${ARGS[@]}"

# Upload results to GCS if configured
if [ ! -z "$GCS_BUCKET" ] && [ ! -z "$OUT_DIR" ] && [ -d "$OUT_DIR" ]; then
    echo "Uploading results to ${GCS_OUTPUT_DIR}"
    gsutil -m cp -r $OUT_DIR/* ${GCS_OUTPUT_DIR}/
fi