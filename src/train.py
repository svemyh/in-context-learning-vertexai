import os
from random import randint
import uuid
import datetime
from pathlib import Path

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model

import wandb

# Add Google Cloud Storage imports
from google.cloud import storage

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def upload_to_gcs(local_path, bucket_name, gcs_path):
    """Upload a file to Google Cloud Storage bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        print(f"Uploaded {local_path} to gs://{bucket_name}/{gcs_path}")
        return f"gs://{bucket_name}/{gcs_path}"
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        return None


def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples
    
    # Dictionary to store metrics for later uploading
    metrics_log = {'loss': [], 'excess_loss': [], 'steps': []}

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)

        loss_func = task.get_training_metric()

        loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func)

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )
        
        # Store metrics for GCS upload
        if not args.test_run:
            metrics_log['loss'].append(loss)
            metrics_log['excess_loss'].append(loss / baseline_loss)
            metrics_log['steps'].append(i)

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            model_checkpoint_path = os.path.join(args.out_dir, f"model_{i}.pt")
            torch.save(model.state_dict(), model_checkpoint_path)
    
    # Save final metrics to a file
    if not args.test_run:
        import json
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Save metrics as JSON
        metrics_path = os.path.join(args.out_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics_log, f)
        
        # Create a loss curve plot
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_log['steps'], metrics_log['loss'], label='Training Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)
        loss_curve_path = os.path.join(args.out_dir, "loss_curve.png")
        plt.savefig(loss_curve_path)
        plt.close()


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    model.cuda()
    model.train()

    train(model, args)

    if not args.test_run:
        metrics = get_run_metrics(args.out_dir)  # precompute metrics for eval
        
        # Get the GCS bucket name from environment variable
        gcs_bucket = os.environ.get('GCS_BUCKET')
        
        # If GCS bucket is specified, upload results
        if gcs_bucket:
            print(f"Uploading training results to GCS bucket: {gcs_bucket}")
            
            # Create a timestamp-based folder name
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = args.wandb.name or f"model_{timestamp}"
            gcs_folder = f"training_runs/{model_name}_{timestamp}"
            
            # Upload model files
            for root, dirs, files in os.walk(args.out_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    rel_path = os.path.relpath(local_path, start=args.out_dir)
                    gcs_path = f"{gcs_folder}/{rel_path}"
                    upload_to_gcs(local_path, gcs_bucket, gcs_path)
            
            print(f"Uploaded training results to gs://{gcs_bucket}/{gcs_folder}/")
            
            # Save the GCS path for reference
            with open(os.path.join(args.out_dir, "gcs_path.txt"), "w") as f:
                f.write(f"gs://{gcs_bucket}/{gcs_folder}/")


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
