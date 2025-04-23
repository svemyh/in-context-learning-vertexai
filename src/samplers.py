import math
import torch

class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "timefreq": TimeFreqSampler, 
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b

class TimeFreqSampler(DataSampler):
    """
    Generates each sample as a random mixture of sinusoids (sine OR cosine)
    + noise, of length `n_points`.  n_dims should be 1.
    """

    def __init__(
        self,
        n_dims,
        min_freq=0.5,
        max_freq=5.0,
        min_amp=0.5,
        max_amp=1.5,
        noise_std=0.05,
        max_components=5,
    ):
        super().__init__(n_dims)
        assert n_dims == 1, "Time‐series sampler only supports n_dims=1"
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_amp = min_amp
        self.max_amp = max_amp
        self.noise_std = noise_std
        self.max_components = max_components

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        # t runs from 0→1 second
        t = torch.linspace(0, 1, n_points, device="cpu")  # [n_points]
        xs_b = torch.zeros(b_size, n_points, self.n_dims)

        for i in range(b_size):
            # decide how many sin/cos components
            n_comp = torch.randint(1, self.max_components + 1, (1,)).item()
            signal = torch.zeros(n_points)

            for _ in range(n_comp):
                f = torch.empty(1).uniform_(self.min_freq, self.max_freq).item()
                A = torch.empty(1).uniform_(self.min_amp, self.max_amp).item()
                phase = torch.empty(1).uniform_(0, 2 * math.pi).item()
                if torch.rand(1).item() < 0.5:
                    signal += A * torch.sin(2 * math.pi * f * t + phase)
                else:
                    signal += A * torch.cos(2 * math.pi * f * t + phase)

            # add noise
            signal += torch.randn(n_points) * self.noise_std
            xs_b[i, :, 0] = signal

        # ignore n_dims_truncated and seeds for now
        return xs_b