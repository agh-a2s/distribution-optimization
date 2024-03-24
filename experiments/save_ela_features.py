import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time

import numpy as np
import pandas as pd
from ioh import ProblemClass, get_problem
from pflacco.classical_ela_features import (
    calculate_dispersion,
    calculate_ela_distribution,
    calculate_ela_level,
    calculate_ela_meta,
    calculate_information_content,
    calculate_nbc,
    calculate_pca,
)
from pflacco.sampling import create_initial_sample

SAMPLE_COEFFICIENT = 250


def evaluate_problem(fid):
    feature_dfs = []
    for dim in [8, 14]:  # 3 * n - 1
        for iid in range(1, 21):
            for seed in range(2):
                random.seed(seed)
                np.random.seed(seed)
                start = time()
                problem = get_problem(fid, iid, dim, problem_class=ProblemClass.BBOB)
                X = create_initial_sample(
                    dim,
                    lower_bound=-5,
                    upper_bound=5,
                    sample_coefficient=SAMPLE_COEFFICIENT,
                    seed=seed,
                )
                y = X.apply(lambda x: problem(x), axis=1)
                ela_meta = calculate_ela_meta(X, y)
                ela_distr = calculate_ela_distribution(X, y)
                ela_level = calculate_ela_level(X, y)
                nbc = calculate_nbc(X, y)
                disp = calculate_dispersion(X, y)
                ic = calculate_information_content(X, y, seed=seed)
                ela_pca = calculate_pca(X, y)
                end = time()
                print(f"fid: {fid}, dim: {dim}, iid: {iid}, time: {end - start}")

                data = pd.DataFrame(
                    {
                        **ic,
                        **ela_meta,
                        **ela_distr,
                        **ela_level,
                        **ela_pca,
                        **nbc,
                        **disp,
                        **{"fid": fid},
                        **{"dim": dim},
                        **{"iid": iid},
                        **{"seed": seed},
                    },
                    index=[0],
                )
                feature_dfs.append(data)

    return pd.concat(feature_dfs, ignore_index=True)


if __name__ == "__main__":
    feature_dfs = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for fid in range(1, 25):
            futures.append(executor.submit(evaluate_problem, fid))

        for future in as_completed(futures):
            feature_dfs.append(future.result())

    features_df = pd.concat(feature_dfs, ignore_index=True)
    features_df.to_csv("./experiments/ela_features_improved.csv", index=False)
