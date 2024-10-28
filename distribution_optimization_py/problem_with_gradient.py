import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .problem import DEFAULT_NR_OF_KERNELS, DEFAULT_OVERLAP_TOLERANCE, INFINITY
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
        overlap_tolerance: float | None = DEFAULT_OVERLAP_TOLERANCE,
        initialize=None,
    ):
        assert initialize is not None, "You have to pass a method to initialize the solution."
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.nr_of_modes = nr_of_modes
        self.N = len(data)
        self.nr_of_bins = optimal_no_bins(data)
        self.breaks = torch.linspace(self.data.min(), self.data.max(), self.nr_of_bins + 1)
        self.observed_bins, _ = np.histogram(data, self.breaks.numpy())
        self.observed_bins = torch.tensor(self.observed_bins, dtype=torch.float32)
        self.nr_of_kernels = nr_of_kernels if nr_of_kernels is not None else DEFAULT_NR_OF_KERNELS
        self.overlap_tolerance = overlap_tolerance

        x = initialize()
        weights, stds, means = (
            x[: self.nr_of_modes],
            x[self.nr_of_modes : 2 * self.nr_of_modes],
            x[2 * self.nr_of_modes :],
        )
        self.means = torch.nn.Parameter(torch.tensor(means, dtype=torch.float32))
        self.sds = torch.nn.Parameter(torch.tensor(stds, dtype=torch.float32))
        self.weights = torch.tensor(
            weights, dtype=torch.float32
        )  # torch.nn.Parameter(torch.tensor(weights, dtype=torch.float32))

    def loss(self):
        # overlap_error = self.overlap_error_by_density()
        similarity_error = self.similarity_error()
        return similarity_error
        # return torch.where(
        #     overlap_error > self.overlap_tolerance,
        #     torch.tensor(INFINITY, dtype=overlap_error.dtype, device=overlap_error.device),
        #     similarity_error,
        # )

    def similarity_error(self):
        normalized_weights = self.weights / torch.sum(self.weights)
        estimated_bins = bin_prob_for_mixtures_torch(self.means, self.sds, normalized_weights, self.breaks) * self.N
        norm = estimated_bins.clone()
        norm[norm < 1] = 1
        diffssq = torch.pow((self.observed_bins - estimated_bins), 2)
        diffssq[diffssq < 4] = 0
        return torch.sum(diffssq / norm) / self.N

    def overlap_error_by_density(self) -> torch.Tensor:
        normalized_weights = self.weights / torch.sum(self.weights)
        kernels = torch.linspace(torch.min(self.data), torch.max(self.data), self.nr_of_kernels)
        densities = torch.stack(
            [
                torch.distributions.Normal(m, sd).log_prob(kernels).exp() * w
                for m, sd, w in zip(self.means, self.sds, normalized_weights)
            ]
        )

        overlap_in_component = torch.zeros_like(densities)

        for i in range(len(self.means)):
            max_other_modes = torch.max(torch.cat([densities[:i], densities[i + 1 :]], dim=0), dim=0).values
            overlap_in_component[i] = torch.minimum(densities[i], max_other_modes)

        area_in_component = torch.sum(densities, dim=1)
        overlap_in_components = torch.sum(overlap_in_component, dim=1)
        ov_ratio_in_component = overlap_in_components / area_in_component

        error_overlap_component = torch.max(ov_ratio_in_component)
        return error_overlap_component


class AdamSolver:
    def __init__(self, lr: float = 1e-1, patience: int = 10, factor: float = 0.1):
        self.lr = lr
        self.patience = patience
        self.factor = factor

    def __call__(
        self,
        model: ChiSquareLossWithGradientModel,
        max_n_evals: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            verbose=True,
        )
        self.losses = []
        for _ in range(max_n_evals):
            optimizer.zero_grad()
            loss = model.loss()
            if loss == 0 or loss == INFINITY:
                break
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
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


class LBFGSSolver:
    def __init__(self, lr: float = 1e-2, max_iter: int = 20):
        self.lr = lr
        self.max_iter = max_iter

    def __call__(
        self,
        model: ChiSquareLossWithGradientModel,
        max_n_evals: int,
        random_state: int | None = 42,
    ) -> np.ndarray:
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=self.lr,
            max_iter=self.max_iter,
            line_search_fn="strong_wolfe",  # Common choice for better accuracy
        )
        self.losses = []

        def closure():
            optimizer.zero_grad()
            loss = model.loss()
            loss.backward()
            return loss

        for _ in range(max_n_evals):
            loss = optimizer.step(closure)
            if loss.item() == 0 or loss.item() == INFINITY:
                break
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
