import numpy as np
from pyhms.core.problem import Problem, get_function_problem
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters, _estimate_log_gaussian_prob

from ..problem import GaussianMixtureProblem
from .protocol import Solution, Solver


def get_genome_from_gaussian_mixture(gmm: GaussianMixture) -> np.ndarray:
    gmm_solution = np.concatenate([gmm.weights_, np.sqrt(gmm.covariances_.flatten()), gmm.means_.flatten()])
    nr_of_modes = gmm.n_components
    means_order = np.argsort(gmm_solution[2 * nr_of_modes :])
    gmm_solution[:nr_of_modes] = gmm_solution[:nr_of_modes][means_order]
    gmm_solution[nr_of_modes : 2 * nr_of_modes] = gmm_solution[nr_of_modes : 2 * nr_of_modes][means_order]
    gmm_solution[2 * nr_of_modes :] = gmm_solution[2 * nr_of_modes :][means_order]
    return gmm_solution


REG_COVAR = 1e-6
COVARIANCE_TYPE = "diag"


def run_em_step(
    problem: Problem,
    genome: np.ndarray,
) -> np.ndarray:
    np.random.seed(42)
    gaussian_problem = get_function_problem(problem).fitness_function
    X = gaussian_problem.data.reshape(-1, 1)
    weights = genome[: gaussian_problem.nr_of_modes]
    precisions = (1 / genome[gaussian_problem.nr_of_modes : 2 * gaussian_problem.nr_of_modes] ** 2).reshape(-1, 1)
    precisions_cholesky = np.sqrt(precisions)
    means = genome[2 * gaussian_problem.nr_of_modes :].reshape(-1, 1)
    weighted_log_prob = _estimate_log_gaussian_prob(X, means, precisions_cholesky, COVARIANCE_TYPE) + np.log(weights)
    log_prob_norm = logsumexp(weighted_log_prob, axis=1)
    with np.errstate(under="ignore"):
        # ignore underflow
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
    weights, means, covariances = _estimate_gaussian_parameters(X, np.exp(log_resp), REG_COVAR, COVARIANCE_TYPE)
    weights /= weights.sum()
    std_devs = np.sqrt(covariances.flatten())
    new_genome = np.concatenate([weights, std_devs, means.flatten()])
    return np.clip(new_genome, gaussian_problem.data_lower, gaussian_problem.data_upper)


def run_em(problem: Problem, genome: np.ndarray, n_steps: int) -> np.ndarray:
    for _ in range(n_steps):
        genome = run_em_step(problem, genome)
    return genome


class EMSolver(Solver):
    def __call__(
        self,
        problem: GaussianMixtureProblem,
        max_n_evals: int,
        random_state: int | None = None,
    ) -> Solution:
        gmm = GaussianMixture(n_components=problem.nr_of_modes, n_init=random_state, max_iter=max_n_evals)
        gmm.fit(problem.data.reshape(-1, 1))
        genome = get_genome_from_gaussian_mixture(gmm)
        return Solution(
            fitness=problem(genome),
            genome=genome,
            scaled_genome=genome,
            log_likelihood=problem.log_likelihood(genome),
            fitness_values=None,
        )
