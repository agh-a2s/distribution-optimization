import numpy as np
import pandas as pd
from distribution_optimization_py.problem import GaussianMixtureProblem
from distribution_optimization_py.solver import iLSHADESolver, GASolver
import random
from distribution_optimization_py.gaussian_mixture import GaussianMixture
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from distribution_optimization_py.datasets import DATASETS

pio.kaleido.scope.mathjax = None

np.random.seed(42)
random.seed(42)

DATASET_NAME_TO_TITLE = {
    "truck_driving_data": "Truck Driving",
    "mixture3": "Mixture 3",
    "textbook_data": "Textbook Data",
    "iris_ica": "Iris ICA",
    "chromatogram_time": "Chromatogram Time",
    "atmosphere_data": "Atmosphere Data",
}

if __name__ == "__main__":
    datasets = [DATASETS[0], DATASETS[3], DATASETS[4]]
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[DATASET_NAME_TO_TITLE.get(d.name) for d in datasets],
        vertical_spacing=0.08,
    )

    for subplot_idx, dataset in enumerate(datasets, 1):
        data = dataset.data
        nr_of_modes = dataset.nr_of_modes
        old_problem = GaussianMixtureProblem(
            data=data,
            nr_of_modes=nr_of_modes,
            bin_type="equal_width",
            bin_number_method="keating",
        )
        new_problem = GaussianMixtureProblem(
            data=data,
            nr_of_modes=nr_of_modes,
            bin_type="equal_probability",
            bin_number_method="mann_wald",
        )
        old_solver = GASolver()
        new_solver = iLSHADESolver()
        old_solutions = []
        new_solutions = []

        for i in range(10):
            random_state = 42 + i
            old_solution = old_solver(
                old_problem, max_n_evals=10000, random_state=random_state
            )
            new_solution = new_solver(
                new_problem, max_n_evals=10000, random_state=random_state
            )
            old_solutions.append(old_solution.genome)
            new_solutions.append(new_solution.genome)

        all_solutions = old_solutions + new_solutions
        all_labels = [f"Keating-{i+1}" for i in range(10)] + [
            f"Mann-Wald-{i+1}" for i in range(10)
        ]
        bins = 75

        gmms = [
            GaussianMixture(n_components=nr_of_modes, random_state=1).set_params(
                data, solution
            )
            for solution in all_solutions
        ]

        x = np.linspace(data.min(), data.max(), 1000)
        pdfs = [gmm.score_samples(x) for gmm in gmms]
        hist_data = np.histogram(data, bins=bins, density=True)

        data_color = "rgba(31, 119, 180, 0.3)"
        keating_color = "#ff7f0e"
        mann_wald_color = "#2ca02c"

        fig.add_trace(
            go.Bar(
                x=hist_data[1][:-1],
                y=hist_data[0],
                opacity=0.7,
                name="Empirical Data",
                marker_color=data_color,
                marker_line_width=0.2,
                legendgroup="data",
                showlegend=(subplot_idx == 1),
            ),
            row=subplot_idx,
            col=1,
        )

        for i, (pdf, label) in enumerate(zip(pdfs, all_labels)):
            dash_pattern = "dashdot"
            color = keating_color if "Keating" in label else mann_wald_color

            show_legend = subplot_idx == 1 and (
                (i == 0 and "Keating" in label) or (i == 10 and "Mann-Wald" in label)
            )
            legend_group = (
                "GA + Keating" if "Keating" in label else "iLSHADE + Mann-Wald"
            )
            legend_name = (
                "GA + Keating" if "Keating" in label else "iLSHADE + Mann-Wald"
            )

            fig.add_trace(
                go.Scatter(
                    x=x.flatten(),
                    y=pdf,
                    mode="lines",
                    name=legend_name if show_legend else None,
                    legendgroup=legend_group,
                    showlegend=show_legend,
                    line=dict(color=color, dash=dash_pattern, width=1.5),
                ),
                row=subplot_idx,
                col=1,
            )

    fig.update_layout(
        height=1200,
        width=1000,
        showlegend=True,
        template="plotly_white",
        font=dict(size=12),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
        ),
        margin=dict(l=60, r=100, t=60, b=60),
    )

    for i in range(1, 4):
        fig.update_xaxes(title_text="X" if i == 3 else None, row=i, col=1)
        fig.update_yaxes(title_text="Density" if i == 2 else None, row=i, col=1)

    fig.write_image(f"./images/comparison_plot.eps", scale=2)
