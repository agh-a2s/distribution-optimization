import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.io as pio
import numpy as np

pio.kaleido.scope.mathjax = None

if __name__ == "__main__":
    pd.options.plotting.backend = "plotly"

    results_dir_name = "./results"
    results_df = pd.read_csv(f"{results_dir_name}/optimization_results.csv")
    df = results_df.pivot(
        index="problem_id",
        columns="solver",
        values=["fitness_mean", "fitness_std", "success_rate"],
    )
    df["number_of_components"] = df.index.str.split("_").str[1].astype(int)
    df["dataset_index"] = df.index.str.split("_").str[2]
    raw_fitness = df["fitness_mean"].rename(
        columns={
            "SHADESolver": "SHADE",
            "GASolver": "GA",
            "LSHADESolver": "LSHADE",
            "DESolver": "DE",
            "iLSHADESolver": "iLSHADE",
            "JADESolver": "JADE",
        }
    )

    problem_ranks = raw_fitness.rank(axis=1)

    grouped_ranks = problem_ranks.groupby(df["number_of_components"]).mean()
    grouped_fitness = raw_fitness.groupby(df["number_of_components"]).mean()

    colors = {
        "DE": "#1f77b4",
        "GA": "#ff7f0e",
        "SHADE": "#2ca02c",
        "LSHADE": "#d62728",
        "iLSHADE": "#9467bd",
        "JADE": "#8c564b",
    }

    fig = go.Figure()
    fig = make_subplots(rows=1, cols=2)

    for method in grouped_ranks.columns:
        fig.add_trace(
            go.Scatter(
                x=grouped_ranks.index,
                y=grouped_ranks[method],
                name=method,
                line=dict(color=colors[method]),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=grouped_fitness.index,
                y=grouped_fitness[method],
                name=method,
                line=dict(color=colors[method]),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        height=400,
        width=1200,
        showlegend=True,
        template="plotly_white",
        font=dict(size=18),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.0),
        margin=dict(l=60, r=120, t=40, b=60),
    )

    fig.update_xaxes(title_text="Number of Components", row=1, col=1)
    fig.update_xaxes(title_text="Number of Components", row=1, col=2)
    fig.update_yaxes(title_text="Average Rank", row=1, col=1)
    fig.update_yaxes(title_text="Average Function Value", row=1, col=2)

    fig.write_image("./images/optimization_methods_analysis.eps")

    # methods = raw_fitness.columns.tolist()
    # n_methods = len(methods)
    # win_matrix = np.zeros((n_methods, n_methods))
    # total_problems = len(raw_fitness)

    # for idx, row in raw_fitness.iterrows():
    #     for i, method1 in enumerate(methods):
    #         for j, method2 in enumerate(methods):
    #             if i == j:
    #                 win_matrix[i, j] = total_problems
    #             if i != j and row[method1] <= row[method2]:
    #                 win_matrix[i, j] += 1

    # win_percentage = (win_matrix / total_problems) * 100

    # fig_heatmap = go.Figure(
    #     data=go.Heatmap(
    #         z=win_percentage,
    #         x=methods,
    #         y=methods,
    #         colorscale="Blues",
    #         text=np.round(win_percentage, 1).astype(str),
    #         texttemplate="%{text}%",  # Add percentage symbol
    #         textfont={"size": 14, "color": "black"},
    #         hoverongaps=False,
    #         colorbar=dict(
    #             title="Win Percentage (%)",
    #             titlefont=dict(size=14, family="Arial"),
    #             tickfont=dict(size=12, family="Arial"),
    #             tickformat=".1f",
    #         ),
    #     )
    # )

    # # Configure layout for publication standards
    # fig_heatmap.update_layout(
    #     height=600,
    #     width=700,
    #     template="plotly_white",
    #     font=dict(size=16),
    #     xaxis=dict(
    #         title=dict(
    #             text="Algorithm (Column outperforms Row)",
    #             font=dict(size=16, family="Arial"),
    #         ),
    #         tickfont=dict(size=14, family="Arial"),
    #         tickangle=-30,  # Improve readability of labels
    #     ),
    #     yaxis=dict(
    #         title=dict(text="Algorithm", font=dict(size=16, family="Arial")),
    #         tickfont=dict(size=14, family="Arial"),
    #     ),
    #     margin=dict(l=80, r=80, t=100, b=100),
    #     paper_bgcolor="white",
    #     plot_bgcolor="white",
    # )
    # fig_heatmap.write_image(
    #     "./images/algorithm_win_matrix.eps",
    #     scale=3,
    #     width=700,
    #     height=600,
    # )
