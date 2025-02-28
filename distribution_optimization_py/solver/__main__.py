from distribution_optimization_py.evaluator.evaluator import ProblemEvaluator
import os
from distribution_optimization_py.solver import (
    iLSHADESolver,
    SHADESolver,
    GASolver,
    LSHADESolver,
    DESolver,
)
from distribution_optimization_py.problem import (
    GaussianMixtureProblem,
)
import numpy as np

if __name__ == "__main__":
    solvers = [
        iLSHADESolver(),
        DESolver(),
        SHADESolver(),
        GASolver(),
        LSHADESolver(),
    ]

    synthetic_problems = []
    synthetic_problems_solutions = []
    synthetic_problem_id_to_fitness = {}
    results_dir_name = "./results"
    for nr_of_components in os.listdir(results_dir_name):
        for dataset_name in os.listdir(f"{results_dir_name}/{nr_of_components}"):
            if not dataset_name.startswith("dataset_"):
                continue
            dataset_idx = dataset_name.split("_")[1].replace(".npy", "")
            dataset = np.load(
                f"{results_dir_name}/{nr_of_components}/dataset_{dataset_idx}.npy"
            )
            parameters = np.load(
                f"{results_dir_name}/{nr_of_components}/parameters_{dataset_idx}.npy"
            )
            id = f"synthetic_{nr_of_components}_{dataset_idx}"
            problem = GaussianMixtureProblem(
                data=dataset,
                nr_of_modes=int(nr_of_components),
                id=id,
                bin_type="equal_probability",
                bin_number_method="mann_wald",
            )
            synthetic_problems_solutions.append(parameters)
            synthetic_problems.append(problem)
            synthetic_problem_id_to_fitness[id] = problem(parameters)

    evaluator = ProblemEvaluator(
        solvers=solvers, problems=synthetic_problems, max_n_evals=10000, seed_count=10
    )
    results = evaluator()

    results[0].to_csv(f"{results_dir_name}/optimization_results.csv")
