import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from eval import get_model_from_run, get_run_metrics
from torch.utils.data import DataLoader
from dataset import SeqToSeqDataset
from samplers import TimeFreqSampler

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Transformer model
model, conf = get_model_from_run("./models/training_ft_transformer")
model = model.to(device)
model.eval()

# Prepare new test data
sampler = TimeFreqSampler(n_dims=1)
n_points = 128
input_length = 64
pred_length = 64
num_samples = 500  # generate fresh 500 samples

dataset = SeqToSeqDataset(sampler, n_points, input_length, pred_length, num_samples)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Compute metrics function
def compute_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

# === Run Transformer on new data ===
all_preds = []
all_trues = []
with torch.no_grad():
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs, targets)
        all_preds.append(outputs.cpu().numpy())
        all_trues.append(targets.cpu().numpy())

y_pred_model = np.concatenate(all_preds, axis=0)
y_true = np.concatenate(all_trues, axis=0)

if y_true.ndim == 3 and y_true.shape[-1] == 1:
    y_true = np.squeeze(y_true, axis=-1)
if y_pred_model.ndim == 3 and y_pred_model.shape[-1] == 1:
    y_pred_model = np.squeeze(y_pred_model, axis=-1)

metrics_model = compute_metrics(y_true, y_pred_model)

# === Run naive baseline (copy last input value) ===
all_preds_baseline = []
for inputs, targets in data_loader:
    last_value = inputs[:, -1, :]  # shape [batch, 1]
    baseline_pred = np.repeat(last_value.cpu().numpy(), targets.shape[1], axis=1)
    all_preds_baseline.append(baseline_pred)

y_pred_baseline = np.concatenate(all_preds_baseline, axis=0)
metrics_baseline = compute_metrics(y_true, y_pred_baseline)

# === Print results ===
print("\n=== New Test Data Results ===")
print("Transformer:")
for k, v in metrics_model.items():
    print(f"  {k.upper()}: {v:.4f}")
print("Naive Baseline:")
for k, v in metrics_baseline.items():
    print(f"  {k.upper()}: {v:.4f}")

# === Plot comparison ===
os.makedirs("./eval_results", exist_ok=True)

def plot_metric_comparison(name, model_value, baseline_value, save_path):
    plt.figure(figsize=(6, 5))
    plt.bar(["Transformer", "Naive"], [model_value, baseline_value])
    plt.ylabel(name.upper())
    plt.title(f"{name.upper()} Comparison (New Test Data)")
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

for metric in metrics_model.keys():
    plot_metric_comparison(metric, metrics_model[metric], metrics_baseline[metric],
                           f"./eval_results/{metric}_comparison_test.png")
