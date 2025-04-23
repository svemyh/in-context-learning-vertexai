import numpy as np
import torch


import numpy as np
import torch


def generate_mixture_batch(
    batch_size: int,
    seq_len: int,
    min_freq=0.5,
    max_freq=5.0,
    min_amp=0.5,
    max_amp=1.5,
    noise_std=0.05,
    max_components=5,
):
    """
    Generate a batch of time-series each composed of a random mixture of sines & cosines,
    plus noise, and return both the time-domain signals and their DFT magnitudes.

    Args:
        batch_size:      number of samples in the batch
        seq_len:         number of time points per sample
        min_freq:        minimum frequency (Hz)
        max_freq:        maximum frequency (Hz)
        min_amp:         minimum amplitude for each component
        max_amp:         maximum amplitude for each component
        noise_std:       std dev of additive Gaussian noise
        max_components:  maximum number of sin/cos components per signal

    Returns:
        xs: Tensor of shape [batch_size, seq_len, 1] (time domain)
        ys: Tensor of shape [batch_size, seq_len]   (DFT magnitudes)
    """
    t = np.linspace(0, 1, seq_len, endpoint=False)  # assume 1 s duration
    xs_list = []
    ys_list = []

    for _ in range(batch_size):
        # decide how many components (1 to max_components)
        n_comp = np.random.randint(1, max_components + 1)
        signal = np.zeros(seq_len, dtype=np.float32)

        for _c in range(n_comp):
            f = np.random.uniform(min_freq, max_freq)
            A = np.random.uniform(min_amp, max_amp)
            phase = np.random.uniform(0, 2 * np.pi)
            # randomly choose sine or cosine
            if np.random.rand() < 0.5:
                signal += A * np.sin(2 * np.pi * f * t + phase)
            else:
                signal += A * np.cos(2 * np.pi * f * t + phase)

        # add noise
        signal += np.random.normal(0, noise_std, size=seq_len).astype(np.float32)

        # compute DFT magnitude
        fft_vals = np.fft.fft(signal)
        mag = np.abs(fft_vals).astype(np.float32)

        xs_list.append(signal)
        ys_list.append(mag)

    xs = torch.from_numpy(np.stack(xs_list))       # [B, seq_len]
    xs = xs.unsqueeze(-1)                          # [B, seq_len, 1]
    ys = torch.from_numpy(np.stack(ys_list))       # [B, seq_len]
    return xs, ys


if __name__ == "__main__":
    BATCH_SIZE = 64
    SEQ_LEN = 128

    xs, ys = generate_mixture_batch(BATCH_SIZE, SEQ_LEN)
    print(f"xs shape: {xs.shape}, ys shape: {ys.shape}")

    torch.save({"xs": xs, "ys": ys}, "time_freq_mixture.pt")
    print("Saved mixture data to time_freq_mixture.pt")
