#!/bin/bash
set -e

# Load environment variables from .env file if it exists (Loads WANDB_API_KEY)
if [ -f .env ]; then
  echo "Loading environment variables from .env file"
  source .env
fi

# Configuration
PROJECT_ID="<gcp_project_id>"
LOCATION="us-central1"
REPOSITORY="<gcp_artifact_registry_name>"
BUCKET_NAME="<gcp_bucket_name>"
IMAGE_NAME="in-context-learning"
TAG="latest"
CONFIG_FILE="src/conf/toy.yaml"

# Compute configuration
MACHINE_TYPE="n1-standard-4"
ACCELERATOR_TYPE="NVIDIA_TESLA_T4"
ACCELERATOR_COUNT="1"
USE_PREEMPTIBLE="true"

# Set up artifacts registry path
ARTIFACT_REGISTRY="${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}"
CONTAINER_URI="${ARTIFACT_REGISTRY}/${IMAGE_NAME}:${TAG}"

echo "========================================================"
echo "Starting ML-OPS Pipeline for In-Context Learning Project"
echo "========================================================"
echo "Project ID: ${PROJECT_ID}"
echo "Container: ${CONTAINER_URI}"
echo "Config: ${CONFIG_FILE}"
echo "Bucket: ${BUCKET_NAME}"
echo "Machine Type: ${MACHINE_TYPE}"
echo "Accelerator: ${ACCELERATOR_TYPE} x ${ACCELERATOR_COUNT}"
echo "Preemptible VM: ${USE_PREEMPTIBLE}"
echo "========================================================"

# 1. Build Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${TAG} .

# 2. Tag for Google Artifact Registry
echo "Tagging Docker image for Artifact Registry..."
docker tag ${IMAGE_NAME}:${TAG} ${CONTAINER_URI}

# 3. Authenticate to artifact registry using service account key 
echo "Authenticating to Artifact Registry..."
cat service-account-key.json | base64 | docker login -u _json_key_base64 --password-stdin https://${ARTIFACT_REGISTRY}

# 4. Push to Google Artifact Registry
echo "Pushing Docker image to Artifact Registry..."
docker push ${CONTAINER_URI}

# 5. Launch training job on Vertex AI
echo "Launching training job on Vertex AI..."
python vertex_job.py \
  --project-id "${PROJECT_ID}" \
  --location "${LOCATION}" \
  --container-uri "${CONTAINER_URI}" \
  --config-file "${CONFIG_FILE}" \
  --bucket-name "${BUCKET_NAME}" \
  --machine-type "${MACHINE_TYPE}" \
  --accelerator-type "${ACCELERATOR_TYPE}" \
  --accelerator-count "${ACCELERATOR_COUNT}" \
  ${USE_PREEMPTIBLE:+--use-preemptible} \
  --service-account-path "service-account-key.json" \
  ${WANDB_API_KEY:+--wandb-api-key "${WANDB_API_KEY}"} \

echo "========================================================"
echo "ML-OPS Pipeline completed successfully!"
echo "Check Google Cloud Console for job status and results"
echo "Results will be stored in: gs://${BUCKET_NAME}/training_runs/"
echo "https://console.cloud.google.com/storage/browser/${BUCKET_NAME}/training_runs"
echo "========================================================"