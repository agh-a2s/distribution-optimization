import pandas as pd

raw_df = pd.read_csv("cs_results_original.csv")
raw_df["fitness"] = raw_df["fitness_mean"].round(5).astype(str) + " ± " + raw_df["fitness_std"].round(5).astype(str)

df = raw_df[["problem_id", "solver", "fitness"]].pivot(
    index="problem_id",
    columns="solver",
    values=["fitness"],
)

print(df.head())
