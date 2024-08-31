from time import time

import numpy as np
import pandas as pd
import plotly.express as px
from multiprocess import Pool

from ..problem import GaussianMixtureProblem
from ..solver import Solver
from ..solver.protocol import CONVERGENCE_PLOT_STEP_SIZE, Solution


class ProblemEvaluator:
    def __init__(
        self,
        solvers: list[Solver],
        problems: list[GaussianMixtureProblem],
        seed_count: int = 30,
        max_n_evals: int = 10000,
        max_pool: int = 10,
    ):
        self.seed_count = seed_count
        self.solvers = solvers
        self.max_n_evals = max_n_evals
        self.max_pool = max_pool
        self.problems = problems

    def evaluate_solver(
        self,
        solver: Solver,
        problem: GaussianMixtureProblem,
        max_n_evals: int,
    ) -> list[Solution]:
        solutions = []
        for random_state in range(1, self.seed_count + 1):
            try:
                solution = solver(problem, max_n_evals, random_state)
                solutions.append(solution)
            except Exception as exc:
                print(f"{solver.__class__.__name__} failed, {exc}")
        return solutions

    def evaluate_problem(self, problem: GaussianMixtureProblem, precision: float) -> pd.DataFrame:
        start = time()
        rows = []
        solver_to_fitness_values = {}
        for solver in self.solvers:
            solutions = self.evaluate_solver(solver, problem, self.max_n_evals)
            fitness_values = np.array([solution.fitness for solution in solutions])
            rows.append(
                {
                    "problem_id": problem.id,
                    "solver": solver.__class__.__name__,
                    "fitness_mean": np.mean(fitness_values),
                    "fitness_std": np.std(fitness_values),
                    "fitness_values": fitness_values,
                    "success_rate": np.mean(fitness_values < precision),
                }
            )
            solver_to_fitness_values[solver.__class__.__name__] = [solution.fitness_values for solution in solutions]
        end = time()
        print(f"Problem {problem.id} evaluated in {(end - start):.2f} seconds")
        return pd.DataFrame(rows), solver_to_fitness_values

    def __call__(self, precision: float | None = 1e-8) -> pd.DataFrame:
        with Pool(self.max_pool) as p:
            pool_outputs = p.map(lambda problem: self.evaluate_problem(problem, precision), self.problems)
        dfs = [df for df, _ in pool_outputs]
        problem_id_to_solver_to_fitness_values = {
            problem.id: solver_to_fitness_values
            for problem, (_, solver_to_fitness_values) in zip(self.problems, pool_outputs)
        }
        return pd.concat(dfs), problem_id_to_solver_to_fitness_values


def convergence_plot(
    problem_id_to_solver_to_values: dict[str, dict[str, np.ndarray]],
    problem_id: str,
) -> None:
    solver_to_values = problem_id_to_solver_to_values[problem_id]
    solver_dfs = []
    for solver_name, values in solver_to_values.items():
        solver_df = pd.DataFrame(values).T
        solver_df["model"] = solver_name
        solver_df.index = (solver_df.index + 1) * CONVERGENCE_PLOT_STEP_SIZE
        solver_dfs.append(solver_df)

    df = pd.concat(solver_dfs)
    fig = px.box(df, x=df.index, y=df.columns, color="model", title=problem_id)
    fig.show()
