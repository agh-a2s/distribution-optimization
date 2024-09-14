import numpy as np
from pyhms.core.problem import EvalCutoffProblem, FunctionProblem, Problem


class ProblemMonitor(EvalCutoffProblem):
    def __init__(self, decorated_problem: Problem, eval_cutoff: int, n_steps: int = 100):
        super().__init__(decorated_problem, eval_cutoff)
        self._n_steps = n_steps
        self._current_best = None
        self._problem_values = []
        self._index = []

    def evaluate(self, phenome, *args, **kwargs):
        if self._n_evals >= self._eval_cutoff:
            return -np.inf if self._inner.maximize else np.inf
        fitness_value = super().evaluate(phenome, *args, **kwargs)
        if self._current_best is None or fitness_value < self._current_best:
            self._current_best = fitness_value
        if self.n_evaluations % self._n_steps == 0:
            self._problem_values.append(self._current_best)
            self._index.append(self.n_evaluations)
        return fitness_value
