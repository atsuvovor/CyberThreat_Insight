import pandas as pd


def compute_threat_distribution(df: pd.DataFrame) -> dict:
    if "Threat Level" not in df.columns:
        return {}

    return df["Threat Level"].value_counts().to_dict()


def compute_issues_by_department(df: pd.DataFrame, top_n: int = 5) -> dict:
    if "Department Affected" not in df.columns:
        return {}

    return (
        df.groupby("Department Affected")
          .size()
          .sort_values(ascending=False)
          .head(top_n)
          .to_dict()
    )


def compute_cost_statistics(df: pd.DataFrame) -> dict:
    if "Cost" not in df.columns:
        return {}

    return {
        "total_cost": float(df["Cost"].sum()),
        "average_cost": float(df["Cost"].mean()),
        "max_cost": float(df["Cost"].max())
    }
