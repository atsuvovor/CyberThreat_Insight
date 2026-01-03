import pandas as pd
from typing import List, Tuple


def build_executive_facts(
    df: pd.DataFrame,
    chart_insights: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Returns:
    - facts: key findings
    - actions: recommended next steps
    """

    facts = []
    actions = []

    # ----------------------------
    # Core Dataset Facts
    # ----------------------------
    if "Threat Level" in df.columns:
        dominant_threat = df["Threat Level"].value_counts().idxmax()
        facts.append(
            f"{dominant_threat} threats represent the most frequent risk category across the dataset."
        )

    if "Department Affected" in df.columns:
        top_dept = (
            df["Department Affected"]
            .value_counts()
            .idxmax()
        )
        facts.append(
            f"{top_dept} accounts for the highest volume of reported cybersecurity issues."
        )

    if "Cost" in df.columns:
        concentration_ratio = df["Cost"].quantile(0.9) / df["Cost"].mean()
        facts.append(
            "A small subset of incidents contributes disproportionately to total remediation cost, "
            "indicating material concentration risk."
        )

    # ----------------------------
    # Chart-Based Facts
    # ----------------------------
    facts.extend(chart_insights)

    # ----------------------------
    # Recommended Actions
    # ----------------------------
    actions.extend([
        "Prioritize enhanced monitoring and access controls for high-risk departments.",
        "Investigate high-cost outlier incidents to identify systemic control gaps.",
        "Strengthen preventative controls targeting critical threat vectors."
    ])

    return facts, actions
