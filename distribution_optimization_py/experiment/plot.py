import pandas as pd
import plotly.io as pio
import plotly.subplots as sp

pio.kaleido.scope.mathjax = None

if __name__ == "__main__":
    results_dir_name = "results"

    rows_rank = []
    rows_value = []
    all_js_values = {"DO": [], "GA Mann-Wald": [], "iLSHADE Mann-Wald": [], "EM": []}
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

        for method in ["DO", "GA Mann-Wald", "iLSHADE Mann-Wald", "EM"]:
            method_js_values = (
                results[results["method"] == method]
                .groupby("dataset_idx")["JS"]
                .mean()
                .tolist()
            )
            all_js_values[method].append(
                {"values": method_js_values, "nr_of_components": nr_of_components}
            )

    fig = sp.make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("", "", ""),
        row_heights=[
            0.35,
            0.65,
        ],  # Adjusted row heights to reduce space between subplots
        specs=[[{}, {}], [{"colspan": 2}, None]],
        vertical_spacing=0.1,  # Added vertical spacing parameter to reduce gap
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

        x_coords = []
        y_coords = []
        for data in all_js_values[method]:
            method_offset = list(df_rank.columns).index(method) * 0.2 - 0.3
            x_coords.extend(
                [data["nr_of_components"] + method_offset] * len(data["values"])
            )
            y_coords.extend(data["values"])

        fig.add_trace(
            dict(
                type="box",
                x=x_coords,
                y=y_coords,
                name=column_to_name[method],
                marker=dict(color=colors[method]),
                boxpoints="all",  # Show all points
                jitter=0.3,  # Add random jitter to points for better visibility
                pointpos=0,  # Center the points
                line=dict(color=colors[method]),
                fillcolor="white",
                opacity=0.6,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        height=800,
        width=1200,
        showlegend=True,
        template="plotly_white",
        font=dict(size=18),
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
    fig.update_xaxes(title_text="Number of Components", row=2, col=1)
    fig.update_yaxes(title_text="Average Rank", row=1, col=1)
    fig.update_yaxes(title_text="Average JSD Value", row=1, col=2)
    fig.update_yaxes(title_text="JSD Value", row=2, col=1)

    fig.update_xaxes(
        tickmode="array",
        ticktext=list(range(2, 11)),
        tickvals=list(range(2, 11)),
        row=2,
        col=1,
    )

    fig.write_image("./images/js_analysis.eps", scale=2)
