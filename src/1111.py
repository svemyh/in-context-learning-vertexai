from eval import get_run_metrics, read_run_dir, get_model_from_run
import seaborn as sns
import matplotlib.pyplot as plt
import torch

model, conf = get_model_from_run("./models/training_ft_transformer")
metrics = get_run_metrics("./models/training_ft_transformer")

setting = "standard"

models_metrics = metrics[setting]

# metrics = get_run_metrics("./your_run_path", skip_baselines=True)
print("ALL setting:", metrics.keys())
setting = "standard"
# print(f"{setting}:", metrics[setting].keys())
model_name = list(metrics[setting].keys())[0]
print("Length of the model", len(metrics[setting][model_name]["mean"]))
print("mean:", metrics[setting][model_name]["mean"][:5])



model_names = list(models_metrics.keys())
model_names = ['gpt2_embd=256_layer=12_head=8', 'decision_tree_max_depth=4', 'mlp_seq=128_hid=256']

means = [models_metrics[m]["mean"][0] for m in model_names]

lows  = [models_metrics[m]["bootstrap_low"][0]  for m in model_names]
highs = [models_metrics[m]["bootstrap_high"][0] for m in model_names]
# model_names = ['Transformer', 'RNN', 'LSTM']


plt.figure(figsize=(8,5))
bars = plt.bar(model_names, means)
plt.xticks(rotation=45, ha="right")
plt.ylabel("MSE")
plt.title("Transformer vs. Baselines (standard)")
plt.tight_layout()
# plt.show()
plt.savefig("./eval.png")