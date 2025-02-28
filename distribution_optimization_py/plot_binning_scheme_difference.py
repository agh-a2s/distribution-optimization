import numpy as np
from distribution_optimization_py.problem import GaussianMixtureProblem
from distribution_optimization_py.solver import iLSHADESolver
from distribution_optimization_py.synthetic_datasets import (
    generate_gaussian_mixture_data,
)
import random
from distribution_optimization_py.gaussian_mixture import GaussianMixture
import plotly.graph_objects as go
import plotly.io as pio

pio.kaleido.scope.mathjax = None

np.random.seed(42)
random.seed(42)

# nr_of_modes = 4
# min = 0.0
# max = 1.0
# sigma = (max - min) * 0.01 * 1.5
# expected_bin_width = (max - min) / 10
# step_size = expected_bin_width / 4
# best_solution = np.array([0.5, 0.5, sigma, sigma, min, max])
# data = generate_gaussian_mixture_data(
#     means=np.array([min, (min + max) / 3, 2 * (min + max) / 3, max]),
#     stds=np.array([sigma, sigma, sigma, sigma]),
#     weights=np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4]),
#     n_samples=1000,
# )

nr_of_modes = 3
min = 0.0
max = 1.0
sigma = (max - min) * 0.01 * 1.5
expected_bin_width = (max - min) / 10
step_size = expected_bin_width / 4
data = generate_gaussian_mixture_data(
    means=np.array([min, (min + max) / 2, max]),
    stds=np.array([sigma, sigma, sigma]),
    weights=np.array([1 / nr_of_modes] * nr_of_modes),
    n_samples=1000,
)

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

solver = iLSHADESolver()
old_solutions = []
new_solutions = []

for i in range(10):
    random_state = i + 42
    np.random.seed(random_state)
    random.seed(random_state)
    old_solution = solver(old_problem, max_n_evals=10000, random_state=random_state)
    new_solution = solver(new_problem, max_n_evals=10000, random_state=random_state)
    old_solutions.append(old_solution.genome)
    new_solutions.append(new_solution.genome)

all_solutions = old_solutions + new_solutions
all_labels = [f"Keating-{i+1}" for i in range(10)] + [
    f"Mann-Wald-{i+1}" for i in range(10)
]
bins = 100

gmms = [
    GaussianMixture(n_components=nr_of_modes, random_state=1).set_params(data, solution)
    for solution in all_solutions
]

x = np.linspace(data.min(), data.max(), 1000)
pdfs = [gmm.score_samples(x) for gmm in gmms]
hist_data = np.histogram(data, bins=bins, density=True)

data_color = "rgba(31, 119, 180, 1.0)"  # Blue color with full opacity
keating_color = "#ff7f0e"  # Orange
mann_wald_color = "#2ca02c"  # Green

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=hist_data[1][:-1],
        y=hist_data[0],
        opacity=0.8,
        name="Empirical Data",
        marker_color=data_color,
        marker_line_width=0.5,
        legendgroup="data",
    )
)

for i, (pdf, label) in enumerate(zip(pdfs, all_labels)):
    dash_pattern = "dashdot"
    color = keating_color if "Keating" in label else mann_wald_color

    show_legend = (i == 0 and "Keating" in label) or (i == 10 and "Mann-Wald" in label)
    legend_group = "Keating" if "Keating" in label else "Mann-Wald"
    legend_name = "Keating" if "Keating" in label else "Mann-Wald"

    fig.add_trace(
        go.Scatter(
            x=x.flatten(),
            y=pdf,
            mode="lines",
            name=legend_name if show_legend else None,
            legendgroup=legend_group,
            showlegend=show_legend,
            line=dict(color=color, dash=dash_pattern, width=2),
        )
    )

fig.update_layout(
    height=600,
    width=1200,
    showlegend=True,
    template="plotly_white",
    font=dict(size=18),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.0),
    margin=dict(l=60, r=120, t=80, b=60),
    barmode="overlay",
)

fig.update_yaxes(title_text="Density")

# fig.show()
fig.write_image("./images/bin_comparison.eps", scale=2)
fig.write_image("./images/bin_comparison.png", scale=2)
