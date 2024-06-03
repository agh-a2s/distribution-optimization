import numpy as np
import torch

from .problem import DEFAULT_NR_OF_KERNELS
from .utils import optimal_no_bins


def cdf_mixtures_torch(
    kernel: torch.Tensor, means: torch.Tensor, sds: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    mixtures = torch.stack([torch.distributions.Normal(mean_, sd_).cdf(kernel) for mean_, sd_ in zip(means, sds)])
    return torch.matmul(mixtures.T, weights)


def bin_prob_for_mixtures_torch(
    means: torch.Tensor, sds: torch.Tensor, weights: torch.Tensor, breaks: torch.Tensor
) -> torch.Tensor:
    cdfs = cdf_mixtures_torch(breaks, means, sds, weights)
    return cdfs[1:] - cdfs[:-1]


class ChiSquareLossWithGradientModel(torch.nn.Module):
    def __init__(
        self,
        data: np.ndarray,
        nr_of_modes: int,
        nr_of_kernels: int | None = DEFAULT_NR_OF_KERNELS,
        initialize=None,
    ):
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.nr_of_modes = nr_of_modes
        self.N = len(data)
        self.nr_of_bins = optimal_no_bins(data)
        self.breaks = torch.linspace(self.data.min(), self.data.max(), self.nr_of_bins + 1)
        self.observed_bins = torch.histc(self.data, bins=self.nr_of_bins, min=self.data.min(), max=self.data.max())
        self.nr_of_kernels = nr_of_kernels if nr_of_kernels is not None else DEFAULT_NR_OF_KERNELS

        x = initialize()
        weights, stds, means = (
            x[: self.nr_of_modes],
            x[self.nr_of_modes : 2 * self.nr_of_modes],
            x[2 * self.nr_of_modes :],
        )
        self.means = torch.nn.Parameter(torch.tensor(means, dtype=torch.float32))
        self.sds = torch.nn.Parameter(torch.tensor(stds, dtype=torch.float32))
        self.weights = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float32))

    def similarity_error(self):
        normalized_weights = self.weights / torch.sum(self.weights)
        estimated_bins = bin_prob_for_mixtures_torch(self.means, self.sds, normalized_weights, self.breaks) * self.N
        norm = estimated_bins.clone()
        norm[norm < 1] = 1
        diffssq = torch.pow((self.observed_bins - estimated_bins), 2)
        diffssq[diffssq < 4] = 0
        return torch.sum(diffssq / norm) / self.N


class AdamSolver:
    def __init__(self, lr: float = 3e-4):
        self.lr = lr

    def __call__(
        self,
        model: ChiSquareLossWithGradientModel,
        max_n_evals: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        self.losses = []
        for _ in range(max_n_evals):
            optimizer.zero_grad()
            loss = model.similarity_error()
            loss.backward()
            optimizer.step()
            self.losses.append(loss.item())
        weights = model.weights.detach().numpy()
        weights = weights / np.sum(weights)
        sds = model.sds.detach().numpy()
        means = model.means.detach().numpy()

        return np.concatenate(
            [
                weights,
                sds,
                means,
            ]
        )
