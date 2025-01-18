import random
from math import gamma

import numpy as np
from pyhms.core.individual import Individual
from pyhms.core.population import Population
from pyhms.core.problem import FunctionProblem
from pyhms.demes.single_pop_eas.common import VariationalOperator, apply_bounds

from ..evaluator.problem_wrapper import ProblemMonitor
from ..problem import GaussianMixtureProblem, ScaledGaussianMixtureProblem
from .protocol import CONVERGENCE_PLOT_STEP_SIZE, Solution, Solver

CS_CONFIG = {"pop_size": 50, "p": 0.25, "alpha": 1.0, "beta": 1.5}
CS_CONFIG_DE = {"pop_size": 50, "p": 0.25, "alpha": 1.0, "beta": 1.5}


def select_parents(population: Population) -> np.ndarray:
    choices = np.indices((population.size, population.size))[1]
    # For each individual, select 2 random indices
    indices = np.array([np.random.choice(row, size=2, replace=False) for row in choices])
    return population.genomes[indices]


class LevyFlight:
    def __init__(self, beta: float = 1.5):
        self.beta = beta

    def __call__(self, size: int) -> np.ndarray:
        sigma_u = (
            gamma(1 + self.beta)
            * np.sin(np.pi * self.beta / 2)
            / (gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))
        ) ** (1 / self.beta)
        sigma_v = 1
        u = np.random.normal(0, sigma_u, size)
        v = np.random.normal(0, sigma_v, size)
        step = u / np.abs(v) ** (1 / self.beta)
        return step


class CuckooSearchBase(VariationalOperator):
    def __init__(self, p: float = 0.25, alpha: float = 1.0, beta: float = 1.5):
        assert alpha >= 0, "alpha must be non-negative"
        assert 0 <= p <= 1, "pa must be in [0, 1]"
        assert beta > 0 and beta <= 2, "beta must be in (0, 2]"
        self.pa = p
        self.alpha = alpha
        self.levy_flight = LevyFlight(beta)

    def get_best_solution(self, population: Population) -> np.ndarray:
        best_index = np.argmax(population.fitnesses) if population.problem.maximize else np.argmin(population.fitnesses)
        return population.genomes[best_index]

    def abandon_nests(self, population: Population) -> Population:
        n_abandon = int(self.pa * population.size)
        sorted_pop_indices = np.argsort(population.fitnesses)
        abandon_indices = sorted_pop_indices[-n_abandon:]
        new_nests = np.random.uniform(
            population.problem.bounds[:, 0],
            population.problem.bounds[:, 1],
            size=(n_abandon, population.genomes.shape[1]),
        )
        population.genomes[abandon_indices] = new_nests
        population.fitnesses[abandon_indices] = np.nan
        return population

    def __call__(self, population: Population) -> Population:
        best_solution = self.get_best_solution(population)

        sample_count = population.size * population.genomes.shape[1]

        g = np.random.normal(0, 1, sample_count).reshape(population.genomes.shape)

        # Generate new solutions (but keep the current best)
        step_sizes = self.levy_flight(sample_count).reshape(population.genomes.shape)
        step = g * self.alpha * step_sizes
        new_genomes = population.genomes + step * (population.genomes - best_solution)
        new_genomes = apply_bounds(new_genomes, population.problem.bounds, "reflect")

        # Evaluate new solutions
        new_population = Population(new_genomes, np.full(population.size, np.nan), population.problem)
        new_population.evaluate()

        # Replace some nests by constructing new solutions
        new_population = self.abandon_nests(new_population)
        new_population.evaluate()

        # Keep the best solutions
        combined_population = population.merge(new_population)
        sorted_indices = np.argsort(combined_population.fitnesses)
        if population.problem.maximize:
            sorted_indices = sorted_indices[::-1]
        best_indices = sorted_indices[: population.size]

        return combined_population[best_indices]


class CuckooSearchDE(CuckooSearchBase):
    def get_new_nests(self, population: Population) -> Population:
        best_solution = self.get_best_solution(population)

        sample_count = population.size * population.genomes.shape[1]
        g = np.random.normal(0, 1, sample_count).reshape(population.genomes.shape)

        # Generate new solutions (but keep the current best)
        step_sizes = self.levy_flight(sample_count).reshape(population.genomes.shape)
        step = g * self.alpha * step_sizes

        new_genomes = population.genomes + step * (population.genomes - best_solution)
        new_genomes = apply_bounds(new_genomes, population.problem.bounds, "reflect")

        # Evaluate new solutions
        new_population = Population(new_genomes, np.full(population.size, np.nan), population.problem)
        new_population.evaluate()
        return new_population

    def abandon_nests(self, population: Population) -> Population:
        sample_count = population.size * population.genomes.shape[1]

        sorted_pop_indices = np.argsort(population.fitnesses)
        n_abandon = int(self.pa * population.size)
        indices_to_abandon = sorted_pop_indices[-n_abandon:]
        nests_to_abandon_indices = np.zeros(population.size, dtype=int)
        nests_to_abandon_indices[indices_to_abandon] = 1
        r = np.random.uniform(low=0, high=1, size=sample_count).reshape(population.genomes.shape)
        parents = select_parents(population)
        new_genomes = population.genomes + nests_to_abandon_indices[:, np.newaxis] * r * (parents[:, 0] - parents[:, 1])
        new_genomes = apply_bounds(new_genomes, population.problem.bounds, "reflect")
        population.update_genome(new_genomes)
        return population

    def __call__(self, population: Population) -> Population:
        new_population = self.get_new_nests(population)
        # Keep the best solutions
        new_population_indices = (
            (new_population.fitnesses >= population.fitnesses)
            if population.problem.maximize
            else (new_population.fitnesses <= population.fitnesses)
        )
        new_population = new_population[new_population_indices].merge(population[~new_population_indices])
        # Replace some nests by constructing new solutions
        new_population = self.abandon_nests(new_population)
        new_population.evaluate()

        return new_population


class CuckooSearchOptimizer:
    def __init__(
            self,
            p: float = 0.25,
            alpha: float = 1.0,
            beta: float = 1.5,
            use_de: bool = True,
    ):
        self.cuckoo_search = CuckooSearchBase(p, alpha, beta) if not use_de else CuckooSearchDE(p, alpha, beta)

    def run(self, parents: list[Individual]) -> list[Individual]:
        parent_population = Population.from_individuals(parents)
        new_population = self.cuckoo_search(parent_population)
        return new_population.to_individuals()


class CSSolver(Solver):
    def __call__(
            self,
            problem: GaussianMixtureProblem,
            max_n_evals: int,
            random_state: int | None = None,
            **kwargs,
    ):
        if random_state:
            np.random.seed(random_state)
            random.seed(random_state)
        bounds = np.column_stack((problem.lower, problem.upper))
        function_problem = FunctionProblem(problem, bounds=bounds, maximize=False)
        eval_cutoff_problem = ProblemMonitor(function_problem, max_n_evals, CONVERGENCE_PLOT_STEP_SIZE)
        cs = CuckooSearchOptimizer(
            p=kwargs.get("p", CS_CONFIG["p"]),
            alpha=kwargs.get("alpha", CS_CONFIG["alpha"]),
            beta=kwargs.get("beta", CS_CONFIG["beta"]),
            use_de=False,
        )
        all_populations = []
        population = Individual.create_population(
            pop_size=CS_CONFIG["pop_size"],
            problem=eval_cutoff_problem,
            initialize=problem.initialize,
        )
        iterations_count = int(max_n_evals / CS_CONFIG["pop_size"])
        for _ in range(iterations_count):
            population = cs.run(population)
            all_populations.extend(population)
        best_solution = max(all_populations)
        scaled_genome = (
            problem.reals_to_internal(best_solution.genome)
            if isinstance(problem, ScaledGaussianMixtureProblem)
            else None
        )
        return Solution(
            fitness=best_solution.fitness,
            genome=best_solution.genome,
            scaled_genome=scaled_genome,
            log_likelihood=problem.log_likelihood(best_solution.genome),
            fitness_values=np.array(eval_cutoff_problem._problem_values),
        )


class CSDESolver(Solver):
    def __call__(
        self,
        problem: GaussianMixtureProblem,
        max_n_evals: int,
        random_state: int | None = None,
        **kwargs,
    ):
        if random_state:
            np.random.seed(random_state)
            random.seed(random_state)
        bounds = np.column_stack((problem.lower, problem.upper))
        function_problem = FunctionProblem(problem, bounds=bounds, maximize=False)
        eval_cutoff_problem = ProblemMonitor(function_problem, max_n_evals, CONVERGENCE_PLOT_STEP_SIZE)
        cs = CuckooSearchOptimizer(
            p=kwargs.get("p", CS_CONFIG_DE["p"]),
            alpha=kwargs.get("alpha", CS_CONFIG_DE["alpha"]),
            beta=kwargs.get("beta", CS_CONFIG_DE["beta"]),
            use_de=True,
        )
        all_populations = []
        population = Individual.create_population(
            pop_size=CS_CONFIG["pop_size"],
            problem=eval_cutoff_problem,
            initialize=problem.initialize,
        )
        iterations_count = int(max_n_evals / CS_CONFIG["pop_size"])
        for _ in range(iterations_count):
            population = cs.run(population)
            all_populations.extend(population)
        best_solution = max(all_populations)
        scaled_genome = (
            problem.reals_to_internal(best_solution.genome)
            if isinstance(problem, ScaledGaussianMixtureProblem)
            else None
        )
        return Solution(
            fitness=best_solution.fitness,
            genome=best_solution.genome,
            scaled_genome=scaled_genome,
            log_likelihood=problem.log_likelihood(best_solution.genome),
            fitness_values=np.array(eval_cutoff_problem._problem_values),
        )
