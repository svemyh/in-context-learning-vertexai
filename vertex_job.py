import argparse
import os
from datetime import datetime
from google.cloud import aiplatform
from google.oauth2 import service_account

def initialize_vertex_ai(service_account_path, project_id, location):
    credentials = service_account.Credentials.from_service_account_file(
        service_account_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    display_name = f"in-context-learning-training-{timestamp}"

    env_vars = {
        "CONFIG_FILE": config_file,
        "GCS_BUCKET": bucket_name,
    }
    if wandb_api_key:
        env_vars["WANDB_API_KEY"] = wandb_api_key
    if wandb_entity:
        env_vars["WANDB_ENTITY"] = wandb_entity

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": machine_type,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": container_uri,
                "command": ["/bin/bash", "/app/entrypoint.sh"],
                "args": [
                    "--config-file", config_file,
                ],
                "env": [{"name": k, "value": v} for k, v in env_vars.items()]
            },
        }
    ]

    if accelerator_type and accelerator_count:
        worker_pool_specs[0]["machine_spec"]["accelerator_type"] = accelerator_type
        worker_pool_specs[0]["machine_spec"]["accelerator_count"] = int(accelerator_count)

    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=f"gs://{bucket_name}/aiplatform-custom-training-{display_name}",
        project=project_id,
        location=location,
        staging_bucket=f"gs://{bucket_name}",
    )

    job.run(
        service_account=None,
        sync=True
    )

    print(f"Job Name: {job.resource_name}")
    print(f"Job State: {job.state}")

    return job

def main():
    parser = argparse.ArgumentParser(description="Launch a training job on Vertex AI")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--container-uri", required=True)
    parser.add_argument("--config-file", default="src/conf/toy.yaml")
    parser.add_argument("--bucket-name", default="eecs282-project")
    parser.add_argument("--machine-type", default="n1-standard-8")
    parser.add_argument("--accelerator-type")
    parser.add_argument("--accelerator-count", type=str)
    parser.add_argument("--use-preemptible", action='store_true')
    parser.add_argument("--wandb-api-key")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--service-account-path", required=True)

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
