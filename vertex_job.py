import argparse
import os
import json
from google.cloud import aiplatform
from datetime import datetime
from google.cloud.aiplatform.compat.types import custom_job as gca_custom_job_compat
from google.oauth2 import service_account

def initialize_vertex_ai(service_account_path, project_id, location):
    """Initialize the Vertex AI client with service account credentials.
    
    Args:
        service_account_path: Path to the service account key JSON file
        project_id: Your Google Cloud project ID
        location: Location for the job (e.g., 'us-central1')
    """
    # Load the service account credentials from the file
    credentials = service_account.Credentials.from_service_account_file(
        service_account_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    
    # Initialize Vertex AI with these credentials
    aiplatform.init(
        credentials=credentials,
        project=project_id,
        location=location
    )

def create_custom_training_job(
    project_id,
    location,
    container_uri,
    config_file="src/conf/toy.yaml",
    bucket_name="eecs282-project",
    machine_type="n1-standard-8",
    accelerator_type=None,
    accelerator_count=None,
    use_preemptible=False,
    wandb_api_key=None,
    wandb_entity=None,
):
    """Create and run a custom training job on Vertex AI.
    
    Args:
        project_id: Your Google Cloud project ID
        location: Location for the job (e.g., 'us-central1')
        container_uri: URI of the Docker container in Artifact Registry
        config_file: Path to the config YAML file to use
        bucket_name: GCS bucket to store results
        machine_type: Vertex AI machine type
        accelerator_type: GPU accelerator type
        accelerator_count: Number of GPUs
        use_preemptible: Use a preemptible VM instance
        wandb_api_key: Weights & Biases API key (optional)
        wandb_entity: Weights & Biases entity (optional)
    """
    # Generate a timestamp-based display name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    display_name = f"in-context-learning-training-{timestamp}"

    # Set up environment variables for the training container
    env_vars = {
        "CONFIG_FILE": config_file,
        "GCS_BUCKET": bucket_name,
    }
    
    # Add wandb credentials if provided
    if wandb_api_key:
        env_vars["WANDB_API_KEY"] = wandb_api_key
    if wandb_entity:
        env_vars["WANDB_ENTITY"] = wandb_entity

    # Prepare worker_pool_specs dictionary
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": machine_type,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": container_uri,
                "command": ["./entrypoint.sh"],
                "args": [
                    "--config-file", config_file,
                ],
                # Pass environment variables within container_spec for CustomJob
                "env": [{"name": k, "value": v} for k, v in env_vars.items()]
            },
        }
    ]

    # Conditionally add accelerator config to worker_pool_specs
    if accelerator_type and accelerator_count:
        worker_pool_specs[0]["machine_spec"]["accelerator_type"] = accelerator_type
        worker_pool_specs[0]["machine_spec"]["accelerator_count"] = int(accelerator_count)

    # Prepare the main job specification dictionary
    job_spec = {
        "worker_pool_specs": worker_pool_specs,
        # Define base output directory within job_spec
        "base_output_directory": {
             "output_uri_prefix": f"gs://{bucket_name}/aiplatform-custom-training-{display_name}"
        }
    }

    # Conditionally add scheduling config to job_spec (outside worker_pool_specs)
    if use_preemptible:
        # Specify a preemptible VM correctly at the job level
        job_spec["scheduling"] = {
            "preemptible": True
        }
        
        # Note: WorkerPoolSpec doesn't support a scheduling field, so we only set it at the job level

    # --- Use aiplatform.CustomJob instead --- 
    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=job_spec["worker_pool_specs"],  # Extract worker_pool_specs
        base_output_dir=job_spec["base_output_directory"]["output_uri_prefix"], # Extract base_output_dir
        project=project_id,
        location=location,
        staging_bucket=f"gs://{bucket_name}",
    )

    # Run the custom job
    # Note: Model registration is not automatic here
    job.run(
        service_account=None, # Use default service account
        sync=True # Block until the job completes
    )

    # --- Model registration would need separate logic if required ---
    # print(f"Job completed. Check output at: {job_spec['base_output_directory']['output_uri_prefix']}")
    # print(f"View Job:\n{job._dashboard_uri()}") # May not have dashboard URI directly
    print(f"Job Name: {job.resource_name}")
    print(f"Job State: {job.state}")

    # Return the job object itself, not a model object
    return job 

def main():
    parser = argparse.ArgumentParser(description="Launch a training job on Vertex AI")
    parser.add_argument("--project-id", required=True, help="Your Google Cloud project ID")
    parser.add_argument("--location", default="us-central1", help="Location for the job")
    parser.add_argument("--container-uri", required=True, help="URI of the Docker container in Artifact Registry")
    parser.add_argument("--config-file", default="src/conf/toy.yaml", help="Path to the config YAML file to use")
    parser.add_argument("--bucket-name", default="eecs282-project", help="GCS bucket to store results")
    parser.add_argument("--machine-type", default="n1-standard-8", help="Vertex AI machine type")
    parser.add_argument("--accelerator-type", help="GPU accelerator type")
    parser.add_argument("--accelerator-count", type=str, help="Number of accelerators (GPUs)")
    parser.add_argument("--use-preemptible", action='store_true', help="Use a preemptible VM instance")
    parser.add_argument("--wandb-api-key", help="Weights & Biases API key")
    parser.add_argument("--wandb-entity", help="Weights & Biases entity")
    parser.add_argument("--service-account-path", required=True, help="Path to the service account key JSON file")
    
    args = parser.parse_args()
    
    initialize_vertex_ai(args.service_account_path, args.project_id, args.location)
    
    create_custom_training_job(
        project_id=args.project_id,
        location=args.location,
        container_uri=args.container_uri,
        config_file=args.config_file,
        bucket_name=args.bucket_name,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        use_preemptible=args.use_preemptible,
        wandb_api_key=args.wandb_api_key,
        wandb_entity=args.wandb_entity,
    )

if __name__ == "__main__":
    main()