import pandas as pd
import plotly.io as pio
import plotly.subplots as sp

pio.kaleido.scope.mathjax = None

if __name__ == "__main__":
    results_dir_name = "results_24_02"

    rows_rank = []
    rows_value = []
    for nr_of_components in range(2, 11):
        results = pd.read_csv(
            f"./{results_dir_name}/{nr_of_components}/results.csv", index_col=0
        )
        dict_result_rank = (
            results[["method", "nr_of_components", "dataset_idx", "JS"]]
            .groupby(["method", "nr_of_components", "dataset_idx"])
            .mean()
            .groupby(["dataset_idx"])
            .rank(method="min")
            .groupby("method")
            .mean()
            .sort_values("JS")
            .to_dict()["JS"]
        )
        rows_rank.append(dict_result_rank | {"nr_of_components": nr_of_components})

        dict_result_value = (
            results[["method", "nr_of_components", "dataset_idx", "JS"]]
            .groupby(["method", "nr_of_components", "dataset_idx"])
            .mean()
            .groupby("method")
            .mean()
            .sort_values("JS")
            .to_dict()["JS"]
        )
        rows_value.append(dict_result_value | {"nr_of_components": nr_of_components})

    fig = sp.make_subplots(
        rows=1, cols=2, subplot_titles=("Average Rank", "Average Value")
    )

    colors = {
        "DO": "#1f77b4",
        "GA Mann-Wald": "#ff7f0e",
        "iLSHADE Mann-Wald": "#2ca02c",
        "EM": "#d62728",
    }

    df_rank = pd.DataFrame(rows_rank).set_index("nr_of_components")[
        ["DO", "GA Mann-Wald", "iLSHADE Mann-Wald", "EM"]
    ]
    df_value = pd.DataFrame(rows_value).set_index("nr_of_components")[
        ["DO", "GA Mann-Wald", "iLSHADE Mann-Wald", "EM"]
    ]
    column_to_name = {
        "DO": "GA + Keating",
        "GA Mann-Wald": "GA + Mann-Wald",
        "iLSHADE Mann-Wald": "iLSHADE + Mann-Wald",
        "EM": "EM",
    }

    for method in df_rank.columns:
        fig.add_trace(
            dict(
                type="scatter",
                x=df_rank.index,
                y=df_rank[method],
                name=column_to_name[method],
                line=dict(color=colors[method]),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            dict(
                type="scatter",
                x=df_value.index,
                y=df_value[method],
                name=column_to_name[method],
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
        font=dict(size=14),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.0,
        ),
        margin=dict(l=60, r=120, t=40, b=60),
    )

    fig.update_xaxes(title_text="Number of Components", row=1, col=1)
    fig.update_xaxes(title_text="Number of Components", row=1, col=2)
    fig.update_yaxes(title_text="Average Rank", row=1, col=1)
    fig.update_yaxes(title_text="Average JS Value", row=1, col=2)

    fig.write_image("./images/js_analysis.png", scale=2)
