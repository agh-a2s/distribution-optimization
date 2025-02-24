from multiprocessing import Pool

from .method import (
    run_experiment_for_difficult_examples,
    run_experiment_for_nr_of_components,
)


# if __name__ == "__main__":
#     components_to_test = [2, 3, 4, 5, 6, 7, 8, 9, 10]
#     components_to_nr_of_datasets = {
#         2: 30,
#         3: 30,
#         4: 30,
#         5: 30,
#         6: 30,
#         7: 30,
#         8: 30,
#         9: 30,
#         10: 30,
#     }
#     min_overlap_value = 0.0
#     max_overlap_value = 0.2
#     results_dir_name = "results_difficult"
#     arg_list = [
#         (
#             nr_of_components,
#             components_to_nr_of_datasets[nr_of_components],
#             min_overlap_value,
#             max_overlap_value,
#             results_dir_name,
#         )
#         for nr_of_components in components_to_test
#     ]

#     with Pool(10) as p:
#         p.starmap(
#             run_experiment_for_nr_of_components,
#             arg_list,
#         )

# if __name__ == "__main__":
#     run_experiment_for_difficult_examples(
#         nr_of_datasets=10, results_dir_name="results_difficult_2"
#     )

if __name__ == "__main__":
    components_to_test = list(range(2, 11))
    components_to_nr_of_datasets = {i: 30 for i in components_to_test}
    min_overlap_value = 0.0
    max_overlap_value = 0.49
    results_dir_name = "results_23_02"
    arg_list = [
        (
            nr_of_components,
            components_to_nr_of_datasets[nr_of_components],
            min_overlap_value,
            max_overlap_value,
            results_dir_name,
        )
        for nr_of_components in components_to_test
    ]

    with Pool(10) as p:
        p.starmap(
            run_experiment_for_nr_of_components,
            arg_list,
        )
