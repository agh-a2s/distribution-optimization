from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator

import leap_ec.ops as lops
import numpy as np
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.representation import Representation
from leap_ec.util import wrap_curry
from pyhms.core.individual import Individual
from pyhms.core.problem import Problem, get_function_problem
from pyhms.initializers import sample_uniform
from toolz import pipe

from .solver.em import run_em

DEFAULT_K_ELITES = 1
DEFAULT_GENERATIONS = 1
DEFAULT_MUTATION_STD = 1.0


class AbstractEA(ABC):
    def __init__(
        self,
        problem: Problem,
        pop_size: int,
    ) -> None:
        super().__init__()
        self.problem = problem
        self.bounds = problem.bounds
        self.pop_size = pop_size

    @abstractmethod
    def run(self, parents: list[Individual] | None = None):
        raise NotImplementedError()

    @classmethod
    def create(cls, problem: Problem, pop_size: int, **kwargs):
        return cls(problem=problem, pop_size=pop_size, **kwargs)


class SimpleEA(AbstractEA):
    """
    A simple single population EA (SEA skeleton).
    """

    def __init__(
        self,
        problem: Problem,
        pop_size: int,
        pipeline: list[Any],
        generations: int | None = DEFAULT_GENERATIONS,
        k_elites: int | None = DEFAULT_K_ELITES,
        representation: Representation | None = None,
    ) -> None:
        super().__init__(problem, pop_size)
        self.generations = generations
        self.pipeline = pipeline
        self.k_elites = k_elites
        if representation is not None:
            self.representation = representation
        else:
            self.representation = Representation(initialize=sample_uniform(bounds=problem.bounds))

    def run(self, parents: list[Individual] | None = None) -> list[Individual]:
        if parents is None:
            parent_genomes = [self.representation.initialize() for _ in range(self.pop_size)]
            parents = [Individual(genome=genome, problem=self.problem) for genome in parent_genomes]
            parents = Individual.evaluate_population(parents)
        else:
            assert self.pop_size == len(parents)
        return pipe(parents, *self.pipeline, lops.elitist_survival(parents=parents, k=self.k_elites))


@wrap_curry
@lops.iteriter_op
def mutate_uniform(next_individual: Iterator, p_mutate: float = 0.1, bounds=(-np.inf, np.inf), fix=None) -> Iterator:
    """
    Mutate an individual by applying a uniform mutation to a single random gene.

    :param next_individual: Iterator of individuals to be mutated
    :param magnitude_range: A tuple specifying the range (min, max) for the uniform mutation
    :param expected_num_mutations: Not used in this function, kept for interface compatibility
    :param bounds: Tuple specifying the lower and upper bounds for the mutation
    :return: Iterator of mutated individuals
    """

    while True:
        individual = next(next_individual)
        if np.random.rand() <= p_mutate:
            gene_index = np.random.randint(0, len(individual.genome))
            mutated_gene = np.random.uniform(bounds[gene_index][0], bounds[gene_index][1])
            individual.genome[gene_index] = mutated_gene
            if fix is not None:
                individual.genome = fix(individual.genome)
            individual.fitness = None
        yield individual


@wrap_curry
@lops.iteriter_op
def mutate_aggresive_uniform(
    next_individual: Iterator, p_mutate: float = 0.1, bounds=(-np.inf, np.inf), fix=None
) -> Iterator:
    """
    Mutate an individual by applying a uniform mutation to a single random gene.

    :param next_individual: Iterator of individuals to be mutated
    :param magnitude_range: A tuple specifying the range (min, max) for the uniform mutation
    :param expected_num_mutations: Not used in this function, kept for interface compatibility
    :param bounds: Tuple specifying the lower and upper bounds for the mutation
    :return: Iterator of mutated individuals
    """

    while True:
        individual = next(next_individual)
        for gene_index in range(len(individual.genome)):
            if np.random.rand() <= p_mutate:
                mutated_gene = np.random.uniform(bounds[gene_index][0], bounds[gene_index][1])
                individual.genome[gene_index] = mutated_gene
                if fix is not None:
                    individual.genome = fix(individual.genome)
                individual.fitness = None
        yield individual


@wrap_curry
@lops.iteriter_op
def em(next_individual: Iterator, fix: Callable | None = None, n_steps: int = 10) -> Iterator:
    """
    Run Expectation Maximization on the individual's genome.
    """
    while True:
        individual = next(next_individual)
        new_genome = run_em(problem=individual.problem, genome=individual.genome, n_steps=n_steps)
        individual.genome = fix(new_genome)
        individual.fitness = None
        yield individual


class ArithmeticCrossover(lops.Crossover):
    def __init__(self, p_xover: float = 1.0, persist_children=False, fix=None):
        """
        Initialize the arithmetic crossover without a fixed alpha.
        Alpha will be sampled from a uniform distribution for each crossover event.
        :param p_xover: The probability of crossover.
        :param persist_children: Whether to persist children in the population.
        """
        super().__init__(p_xover=p_xover, persist_children=persist_children)
        self.fix = fix

    def recombine(self, parent_a, parent_b):
        """
        Perform arithmetic recombination between two parents to produce two new individuals.
        For each recombination, alpha is sampled from a uniform distribution [0, 1].
        """
        assert isinstance(parent_a.genome, np.ndarray) and isinstance(parent_b.genome, np.ndarray)

        if np.random.rand() <= self.p_xover:
            # Ensure both genomes are of the same length
            min_length = min(parent_a.genome.shape[0], parent_b.genome.shape[0])

            # Sample alpha from a uniform distribution for each crossover
            alpha = np.random.uniform(0, 1)

            # Create offspring by linear combination of parents' genomes
            offspring_a_genome = alpha * parent_a.genome[:min_length] + (1 - alpha) * parent_b.genome[:min_length]
            offspring_b_genome = (1 - alpha) * parent_a.genome[:min_length] + alpha * parent_b.genome[:min_length]

            # Update genomes of offspring
            parent_a.genome[:min_length] = offspring_a_genome
            parent_b.genome[:min_length] = offspring_b_genome
        if self.fix is not None:
            parent_a.genome = self.fix(parent_a.genome)
            parent_b.genome = self.fix(parent_b.genome)
        return parent_a, parent_b


class GAStyleEA(SimpleEA):
    """
    An implementation of SEA using LEAP.
    """

    def __init__(
        self,
        generations,
        problem,
        bounds,
        pop_size,
        k_elites=1,
        representation=None,
        p_mutation=1,
        p_crossover=1,
        use_warm_start=False,
        use_fix=True,
    ) -> None:
        self.gaussian_problem = get_function_problem(problem).fitness_function
        super().__init__(
            problem,
            pop_size,
            pipeline=[
                lops.tournament_selection,
                lops.clone,
                ArithmeticCrossover(
                    p_xover=p_crossover,
                    fix=self.gaussian_problem.fix if use_fix else None,
                ),
                mutate_uniform(
                    bounds=bounds,
                    p_mutate=p_mutation,
                    fix=self.gaussian_problem.fix if use_fix else None,
                ),
                lops.evaluate,
                lops.pool(size=pop_size),
            ],
            generations=generations,
            k_elites=k_elites,
            representation=representation,
        )
        self._use_fix = use_fix
        self._use_warm_start = use_warm_start

    def run(self, parents: list[Individual] | None = None, **kwargs: Any) -> list[Individual]:
        if parents is None:
            pop_size = self.pop_size if not self._use_warm_start else self.pop_size - 1
            parents = self.representation.create_population(pop_size=pop_size, problem=self.problem)
            if self._use_warm_start:
                x0 = self.gaussian_problem.initialize_warm_start()
                parents.append(Individual(genome=x0, problem=self.problem))
            parents = Individual.evaluate_population(parents)
        else:
            assert self.pop_size == len(parents)

        return pipe(parents, *self.pipeline, lops.elitist_survival(parents=parents, k=self.k_elites))

    @classmethod
    def create(cls, generations, problem, pop_size, **kwargs):
        k_elites = kwargs.get("k_elites") or 1
        p_mutation = kwargs.get("p_mutation") or 0.9
        p_crossover = kwargs.get("p_crossover") or 0.9
        return cls(
            generations=generations,
            problem=problem,
            bounds=problem.bounds,
            pop_size=pop_size,
            k_elites=k_elites,
            p_mutation=p_mutation,
            p_crossover=p_crossover,
        )


class GAStyleWithEM(SimpleEA):
    def __init__(
        self,
        generations: int,
        problem: Problem,
        bounds: np.ndarray,
        pop_size: int,
        k_elites: int = 1,
        representation: Representation | None = None,
        p_mutation: float = 1,
        p_crossover: float = 1,
        use_warm_start: bool = True,
        use_fix: bool = True,
        em_steps: int = 1,
    ) -> None:
        self.gaussian_problem = get_function_problem(problem).fitness_function
        super().__init__(
            problem,
            pop_size,
            pipeline=[
                lops.tournament_selection,
                lops.clone,
                ArithmeticCrossover(
                    p_xover=p_crossover,
                    fix=self.gaussian_problem.fix if use_fix else None,
                ),
                mutate_uniform(
                    bounds=bounds,
                    p_mutate=p_mutation,
                    fix=self.gaussian_problem.fix if use_fix else None,
                ),
                # em(
                #     n_steps=em_steps, fix=self.gaussian_problem.fix if use_fix else None
                # ),
                lops.evaluate,
                lops.pool(size=pop_size),
            ],
            generations=generations,
            k_elites=k_elites,
            representation=representation,
        )
        self._em_steps = em_steps
        self._use_warm_start = use_warm_start

    def run(self, parents: list[Individual] | None = None, **kwargs: Any) -> list[Individual]:
        if parents is None:
            initial_points = self.representation.create_population(pop_size=self.pop_size, problem=self.problem)
            parents = []
            for parent in initial_points:
                genome = run_em(self.problem, parent.genome, self._em_steps)
                genome = self.gaussian_problem.fix(genome)
                parents.append(
                    Individual(
                        genome=genome,
                        fitness=self.problem.evaluate(genome),
                        problem=self.problem,
                    )
                )
            Individual.evaluate_population(parents)

        else:
            assert self.pop_size == len(parents)

        return pipe(parents, *self.pipeline, lops.elitist_survival(parents=parents, k=self.k_elites))

    @classmethod
    def create(cls, generations, problem, pop_size, **kwargs):
        k_elites = kwargs.get("k_elites") or 1
        p_mutation = kwargs.get("p_mutation") or 0.9
        p_crossover = kwargs.get("p_crossover") or 0.9
        return cls(
            generations=generations,
            problem=problem,
            bounds=problem.bounds,
            pop_size=pop_size,
            k_elites=k_elites,
            p_mutation=p_mutation,
            p_crossover=p_crossover,
        )


class CustomSEA(SimpleEA):
    """
    An implementation of SEA using LEAP.
    """

    def __init__(
        self,
        generations,
        problem,
        bounds,
        pop_size,
        mutation_std=1.0,
        k_elites=1,
        p_mutation=1,
        p_crossover=0.5,
        representation=None,
    ) -> None:
        super().__init__(
            problem,
            pop_size,
            pipeline=[
                lops.tournament_selection,
                lops.clone,
                lops.UniformCrossover(p_xover=p_crossover),
                mutate_gaussian(std=mutation_std, bounds=bounds, expected_num_mutations=p_mutation),
                lops.evaluate,
                lops.pool(size=pop_size),
            ],
            generations=generations,
            k_elites=k_elites,
            representation=representation,
        )

    @classmethod
    def create(cls, generations, problem, pop_size, **kwargs):
        mutation_std = kwargs.get("mutation_std") or 1.0
        k_elites = kwargs.get("k_elites") or 1
        p_mutation = kwargs.get("p_mutation") or 1
        p_crossover = kwargs.get("p_crossover") or 0.5
        return cls(
            generations=generations,
            problem=problem,
            bounds=problem.bounds,
            pop_size=pop_size,
            mutation_std=mutation_std,
            k_elites=k_elites,
            p_mutation=p_mutation,
            p_crossover=p_crossover,
        )
