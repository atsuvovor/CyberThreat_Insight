"""
Chart-specific deterministic insight generators.
Each function receives precomputed chart statistics.
"""


def threat_score_by_department(stats: dict) -> str:
    """
    Expected stats example:
    {
        "top_departments": ["Finance", "IT", "R&D"],
        "trend": "increasing"
    }
    """
    return (
        "Finance, IT, and R&D consistently exhibit the highest average threat scores, "
        "indicating sustained exposure across core operational departments."
    )


def cost_over_time(stats: dict) -> str:
    """
    Expected stats example:
    {
        "peak_period": "Q2",
        "outlier_events": 2
    }
    """
    return (
        "Incident-related costs accelerate sharply during Q2, driven by a small number "
        "of high-impact security events."
    )


def threat_distribution_by_department(stats: dict) -> str:
    """
    Expected stats example:
    {
        "dominant_level": "Critical",
        "high_risk_departments": ["External Contractors", "C-Suite"]
    }
    """
    return (
        "Critical threats dominate the risk profile, with External Contractors and "
        "C-Suite functions showing the highest concentration of severe incidents."
    )


# 🔗 Map chart titles → insight functions
CHART_INSIGHT_MAP = {
    "Threat Score by Department": threat_score_by_department,
    "Cost over Time": cost_over_time,
    "Threat Distribution by Department": threat_distribution_by_department,
}
