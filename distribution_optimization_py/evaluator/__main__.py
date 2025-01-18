from ..datasets import DATASETS
from ..problem import GaussianMixtureProblem
from ..solver.cs import CSDESolver, CSSolver
from ..solver.de import iLSHADESolver
from ..solver.ga import GASolver
from .evaluator import ProblemEvaluator

if __name__ == "__main__":
    solvers = [
        iLSHADESolver(),
        GASolver(),
        CSSolver(),
        CSDESolver(),
    ]
    problems = [
        # DistributionOptimizationProblem(
        #     dataset.data,
        #     dataset.nr_of_modes,
        #     method=ConstraintHandlingTechnique.DEATH_PENALTY,
        #     id=dataset.name,
        #     use_repair=True,
        # )
        GaussianMixtureProblem(
            dataset.data,
            dataset.nr_of_modes,
            id=dataset.name,
        )
        for dataset in DATASETS
    ]

    solver_configs = {
        CSSolver.__name__: {
            # truck_driving_data
            DATASETS[0].name: {
                "alpha": 1,
            },
            # mixture3
            DATASETS[1].name: {
                "alpha": 0.5,
            },
            # textbook_data
            DATASETS[2].name: {
                "alpha": 0.5,
            },
            # iris_ica
            DATASETS[3].name: {
                "alpha": 0.51,
            },
            # chromatogram_time
            DATASETS[4].name: {
                "alpha": 0.49,
            },
            # atmosphere_data
            DATASETS[5].name: {
                "alpha": 0.45,
            },

            ### Reevaluate upper
        },
        CSDESolver.__name__: {
            # truck_driving_data
            DATASETS[0].name: {
                "alpha": 0.75,
            },
            # mixture3
            DATASETS[1].name: {
                "alpha": 0.65,
            },
            # textbook_data
            DATASETS[2].name: {
                "alpha": 0.7,
            },
            # iris_ica
            DATASETS[3].name: {
                "alpha": 0.255,
            },
            # chromatogram_time
            DATASETS[4].name: {
                "alpha": 0.26,
            },
            # atmosphere_data
            DATASETS[5].name: {
                "alpha": 0.408,
            },
        },
    }

    evaluator = ProblemEvaluator(solvers, problems, solver_configs=solver_configs)
    results = evaluator()
    results[0].to_csv("cs_results.csv", index=False)
