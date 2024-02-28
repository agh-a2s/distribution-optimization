from typing import Iterator

import leap_ec.ops as lops
import numpy as np
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.util import wrap_curry
from pyhms.demes.single_pop_eas.sea import SimpleEA


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
    ) -> None:
        super().__init__(
            generations,
            problem,
            bounds,
            pop_size,
            pipeline=[
                lops.tournament_selection,
                lops.clone,
                ArithmeticCrossover(
                    p_xover=p_crossover,
                ),
                mutate_uniform(
                    bounds=bounds,
                    p_mutate=p_mutation,
                ),
                lops.evaluate,
                lops.pool(size=pop_size),
            ],
            k_elites=k_elites,
            representation=representation,
        )

    @classmethod
    def create(cls, generations, problem, bounds, pop_size, **kwargs):
        k_elites = kwargs.get("k_elites") or 1
        p_mutation = kwargs.get("p_mutation") or 0.9
        p_crossover = kwargs.get("p_crossover") or 0.9
        return cls(
            generations=generations,
            problem=problem,
            bounds=bounds,
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
            generations,
            problem,
            bounds,
            pop_size,
            pipeline=[
                lops.tournament_selection,
                lops.clone,
                lops.UniformCrossover(p_xover=p_crossover),
                mutate_gaussian(std=mutation_std, bounds=bounds, expected_num_mutations=p_mutation),
                lops.evaluate,
                lops.pool(size=pop_size),
            ],
            k_elites=k_elites,
            representation=representation,
        )

    @classmethod
    def create(cls, generations, problem, bounds, pop_size, **kwargs):
        mutation_std = kwargs.get("mutation_std") or 1.0
        k_elites = kwargs.get("k_elites") or 1
        p_mutation = kwargs.get("p_mutation") or 1
        p_crossover = kwargs.get("p_crossover") or 0.5
        return cls(
            generations=generations,
            problem=problem,
            bounds=bounds,
            pop_size=pop_size,
            mutation_std=mutation_std,
            k_elites=k_elites,
            p_mutation=p_mutation,
            p_crossover=p_crossover,
        )
