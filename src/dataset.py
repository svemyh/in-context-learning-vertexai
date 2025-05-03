import torch
from torch.utils.data import Dataset, DataLoader
from samplers import TimeFreqSampler

class SeqToSeqDataset(Dataset):
    def __init__(self, sampler, n_points, input_length, pred_length, num_samples):
        """
        sampler: your TimeFreqSampler instance
        n_points: total length of each sequence (input + prediction)
        input_length: length of input sequence
        pred_length: length of predicted sequence
        num_samples: total number of samples in dataset
        """
        self.sampler = sampler
        self.n_points = n_points
        self.input_length = input_length
        self.pred_length = pred_length
        self.num_samples = num_samples
        
        assert input_length + pred_length <= n_points, "input + pred exceeds total points"

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a single sample (shape: [n_points, 1])
        x = self.sampler.sample_xs(self.n_points, b_size=1)[0]  # shape: [n_points, 1]
        
        # Split into input and target
        input_seq = x[:self.input_length, :]  # shape: [input_length, 1]
        target_seq = x[self.input_length:self.input_length + self.pred_length, :]  # shape: [pred_length, 1]
        
        return input_seq, target_seq


