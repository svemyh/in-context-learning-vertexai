import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from eval import get_model_from_run, get_run_metrics
from torch.utils.data import DataLoader
from dataset import SeqToSeqDataset
from samplers import TimeFreqSampler


metrics = get_run_metrics("./models/training_ft_transformer")
print(metrics.keys())  
print(metrics["standard"].keys()) 

state = torch.load("./models/training_ft_transformer/state.pt")
print(state["model_state_dict"].keys())


# print("metrics['loss']:", metrics['loss'])
# print("metrics['excess_loss']:", metrics['excess_loss'])
# print("metrics['steps']:", metrics['steps'])

# plt.plot(metrics['steps'], metrics['loss'])
# # plt.plot(metrics['steps'], metrics['excess_loss'])
# plt.show()