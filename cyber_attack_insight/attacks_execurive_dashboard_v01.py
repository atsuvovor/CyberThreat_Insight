"""
summary_executive_dashboard.py
--------------------------------
Executive-level cybersecurity simulation dashboard.

This module:
- Aggregates attack and incident metrics
- Generates executive KPIs
- Produces visual summaries (bar & donut charts)
- Generates a PDF executive report
"""

# ============================
# Required Libraries (Only)
# ============================

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from fpdf import FPDF
from IPython.display import display


# ==========================================================
# Executive Report Aggregation
# ==========================================================

def generate_executive_report(df: pd.DataFrame) -> dict:
    """
    Generate executive KPI summaries from the simulated attacks dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Simulated cybersecurity incidents data

    Returns
    -------
    dict
        Dictionary of aggregated metrics for visualization
    """

    # Aggregate threat and severity distributions
    total_theats = df.groupby("Threat Level").size()
    severity_stats = df.groupby("Severity").size()

    # Calculate financial impact (in millions)
    impact_cost_stats = round(df.groupby("Severity")["Cost"].sum() / 1_000_000)

    # Incident resolution status
    resolved_stats = df[df["Status"].isin(["Resolved", "Closed"])].groupby("Threat Level").size()
    out_standing_issues = df[df["Status"].isin(["Open", "In Progress"])].groupby("Threat Level").size()

    # Average response times by status
    outstanding_issues_avg_resp_time = round(
        df[df["Status"].isin(["Open", "In Progress"])]
        .groupby("Threat Level")["Issue Response Time Days"].mean()
    )

    solved_issues_avg_resp_time = round(
        df[df["Status"].isin(["Resolved", "Closed"])]
        .groupby("Threat Level")["Issue Response Time Days"].mean()
    )

    # Identify top 5 highest-risk incidents
    top_issues = df.nlargest(5, "Threat Score")

    # Overall average response time
    overall_avg_response_time = df["Issue Response Time Days"].mean()

    # Consolidated executive metrics dictionary
    report_summary_data_dic = {
        "Total Attack": total_theats,
        "Attack Volume Severity": severity_stats,
        "Impact in Cost(M$)": impact_cost_stats,
        "Resolved Issues": resolved_stats,
        "Outstanding Issues": out_standing_issues,
        "Outstanding Issues Avg Response Time": outstanding_issues_avg_resp_time,
        "Solved Issues Avg Response Time": solved_issues_avg_resp_time,
        "Top 5 Issues": top_issues.to_dict(),
        "Overall Average Response Time(days)": overall_avg_response_time
    }

    # Prepare Top 5 issues table
    top_five_issues_df = pd.DataFrame(report_summary_data_dic.pop("Top 5 Issues"))
    top_five_issues_df["cost"] = top_five_issues_df["Cost"].apply(lambda x: round(x / 1_000_000))

    # Convert overall response time into multiple units
    avg_days = round(report_summary_data_dic.pop("Overall Average Response Time(days)"))
    average_response_time = {
        "Average Response Time in days": avg_days,
        "Average Response Time in hours": avg_days * 24,
        "Average Response Time in minutes": avg_days * 1440
    }

    # Ensure numeric consistency before dataframe creation
    for col in [
        "Impact in Cost(M$)",
        "Outstanding Issues Avg Response Time",
        "Solved Issues Avg Response Time"
    ]:
        report_summary_data_dic[col] = pd.to_numeric(
            report_summary_data_dic[col], errors="coerce"
        )

    # Create executive summary dataframe
    report_summary_df = pd.DataFrame(report_summary_data_dic)
    report_summary_df = report_summary_df.apply(
        lambda x: round(x) if x.dtype.kind in "biufc" else x
    )

    # Display executive summaries
    print("\nExecutive Summary Metrics\n")
    display(report_summary_df)

    print("\nAverage Response Time\n")
    display(pd.DataFrame(average_response_time, index=[0]))

    print("\nTop 5 Issues Impact with Adaptive Defense Mechanism\n")
    display(
        top_five_issues_df[
            [
                "Issue ID", "Threat Level", "Severity",
                "Issue Response Time Days", "Department Affected",
                "Cost", "Defense Action"
            ]
        ]
    )

    return report_summary_data_dic


# ============================================
