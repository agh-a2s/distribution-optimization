from ..solver.protocol import DATASETS
from .evaluator import ProblemEvaluator

if __name__ == "__main__":
    from ..problem import ScaledGaussianMixtureProblem
    from ..solver.hms import HMSSolver

    solvers = [HMSSolver()]
    problems = [ScaledGaussianMixtureProblem(dataset.data, dataset.nr_of_modes) for dataset in DATASETS]
    evaluator = ProblemEvaluator(solvers, problems)
    results = evaluator()
    results.to_csv("results.csv", index=False)
