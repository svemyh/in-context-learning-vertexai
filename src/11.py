import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from eval import get_model_from_run
from torch.utils.data import DataLoader
from dataset import SeqToSeqDataset
from samplers import TimeFreqSampler

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare test data
sampler = TimeFreqSampler(n_dims=1)
n_points = 128
input_length = 64
pred_length = 64
num_samples = 200

dataset = SeqToSeqDataset(sampler, n_points, input_length, pred_length, num_samples)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Collect ground truth
all_trues = []
for _, targets in data_loader:
    all_trues.append(targets.numpy())
y_true = np.concatenate(all_trues, axis=0)
if y_true.ndim == 3 and y_true.shape[-1] == 1:
    y_true = np.squeeze(y_true, axis=-1)

# Metrics function
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

summary = {}

# === Run Transformer ===
transformer, conf = get_model_from_run("./models/training_ft_transformer")
transformer = transformer.to(device)
transformer.eval()

all_preds = []
with torch.no_grad():
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = transformer(inputs, targets)
        all_preds.append(outputs.cpu().numpy())

y_pred_transformer = np.concatenate(all_preds, axis=0)
if y_pred_transformer.ndim == 3 and y_pred_transformer.shape[-1] == 1:
    y_pred_transformer = np.squeeze(y_pred_transformer, axis=-1)

summary["Transformer"] = compute_metrics(y_true, y_pred_transformer)

# === Define RNN and LSTM ===
class RNNBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNBaseline, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # predict full pred_length
        return out

class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMBaseline, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # predict full pred_length
        return out

# Initialize baselines
rnn_baseline = RNNBaseline(input_dim=1, hidden_dim=64, output_dim=pred_length).to(device)
lstm_baseline = LSTMBaseline(input_dim=1, hidden_dim=64, output_dim=pred_length).to(device)

# Run RNN
inputs_full, _ = next(iter(DataLoader(dataset, batch_size=len(dataset))))
inputs_full = inputs_full.to(device)

rnn_baseline.eval()
with torch.no_grad():
    rnn_pred = rnn_baseline(inputs_full).cpu().numpy()
summary["RNN"] = compute_metrics(y_true, rnn_pred)

# Run LSTM
lstm_baseline.eval()
with torch.no_grad():
    lstm_pred = lstm_baseline(inputs_full).cpu().numpy()
summary["LSTM"] = compute_metrics(y_true, lstm_pred)

# === Print and plot ===
print("\n=== Recomputed Metrics on Test Data ===")
for name, vals in summary.items():
    print(f"{name}:")
    for k, v in vals.items():
        print(f"  {k.upper()}: {v:.4f}")

os.makedirs("./eval_results", exist_ok=True)

def plot_metric_across_models(metric_name, summary, save_path):
    names = list(summary.keys())
    values = [summary[name][metric_name] for name in names]

    plt.figure(figsize=(10, 5))
    plt.bar(names, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric_name.upper())
    plt.title(f"{metric_name.upper()} Across Models (Test Data)")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

for metric in ["mse", "mae", "rmse", "r2"]:
    plot_metric_across_models(metric, summary, f"./eval_results/{metric}_across_models_test.png")
