from .evaluator import ProblemEvaluator
from ..solver.protocol import DATASETS

if __name__ == "__main__":
    from ..solver.hms import HMSSolver
    from ..problem import ScaledGaussianMixtureProblem

    solvers = [HMSSolver()]
    problems = [
        ScaledGaussianMixtureProblem(dataset.data, dataset.nr_of_modes)
        for dataset in DATASETS
    ]
    evaluator = ProblemEvaluator(solvers, problems)
    results = evaluator()
    results.to_csv("results.csv", index=False)
