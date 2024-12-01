from multiprocessing import Pool
from .method import run_experiment_for_nr_of_components

# TODO:
# Make sure that new run doesn't overwrite old results (results.csv)

if __name__ == "__main__":
    components_to_test = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    components_to_nr_of_datasets = {
        2: 30,
        3: 30,
        4: 30,
        5: 30,
        6: 30,
        7: 30,
        8: 30,
        9: 30,
        10: 30,
        11: 10,
    }
    overlap_min_value = 0.4
    overlap_max_value = 0.5
    results_dir_name = "results_difficult"
    arg_list = [
        (
            nr_of_components,
            components_to_nr_of_datasets[nr_of_components],
            overlap_min_value,
            overlap_max_value,
            results_dir_name,
        )
        for nr_of_components in components_to_test
    ]

    with Pool(10) as p:
        p.starmap(
            run_experiment_for_nr_of_components,
            arg_list,
        )
