"""
Executive Cybersecurity Reporting Pipeline
------------------------------------------
This module generates executive-level cybersecurity reports including:
- KPI aggregation and summaries
- Visual analytics (bar charts, donut charts)
- Incident and attack scenario analysis
- Automated PDF executive reporting

Author: Atsu Vovor
"""

# ============================
# Required Libraries
# ============================

import os # Import the os module to create directories
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from fpdf import FPDF

from IPython.display import display
from CyberThreat_Insight.cyber_attack_insight.attack_simulation_v02 import main_attacks_simulation_pipeline

NEW_DATA_URL = "https://drive.google.com/file/d/1Nr9PymyvLfDh3qTfaeKNVbvLwt7lNX6l/view?usp=sharing"
DATA_FOLDER_PATH =  "CyberThreat_Insight/cybersecurity_data"
executive_cybersecurity_attack_report_on_drive = os.path.join(DATA_FOLDER_PATH, "Executive_Cybersecurity_Attack_Report.pdf")



def normalize_threat_level(df, THREAT_LEVEL = "Threat Level"):
    # Normalize Threat Level
    THREAT_LEVEL_MAP = {
        0: "Low",
        1: "Medium",
        2: "High",
        3: "Critical"
        }
    df[THREAT_LEVEL] = (
        df[THREAT_LEVEL]
        .map(THREAT_LEVEL_MAP)
        .fillna(df[THREAT_LEVEL])
    )
    return df
    
# ============================
# Executive Report Aggregation
# ============================

def generate_executive_report(df: pd.DataFrame) -> dict:
    """
    Generate high-level executive metrics from cybersecurity incidents.

    Parameters
    ----------
    df : pd.DataFrame
        Cybersecurity incidents dataset

    Returns
    -------
    dict
        Dictionary containing aggregated metrics used for plotting
    """

    # ---- Threat and Severity Statistics ----
    total_theats = df.groupby("Threat Level").size()
    severity_stats = df.groupby("Severity").size()

    # Total impact cost in millions
    impact_cost_stats = round(df.groupby("Severity")["Cost"].sum() / 1_000_000)

    # Resolved vs outstanding issues
    resolved_stats = df[df["Status"].isin(["Resolved", "Closed"])].groupby("Threat Level").size()
    out_standing_issues = df[df["Status"].isin(["Open", "In Progress"])].groupby("Threat Level").size()

    # Average response times
    outstanding_issues_avg_resp_time = round(
        df[df["Status"].isin(["Open", "In Progress"])]
        .groupby("Threat Level")["Issue Response Time Days"].mean()
    )

    solved_issues_avg_resp_time = round(
        df[df["Status"].isin(["Resolved", "Closed"])]
        .groupby("Threat Level")["Issue Response Time Days"].mean()
    )

    # ---- Top 5 Highest-Risk Issues ----
    top_issues = df.nlargest(5, "Threat Score")

    # ---- Overall KPI ----
    overall_avg_response_time = df["Issue Response Time Days"].mean()

    # ---- Consolidated Executive Dictionary ----
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

    # ---- Top 5 Issues Table ----
    top_five_issues_df = pd.DataFrame(report_summary_data_dic.pop("Top 5 Issues"))
    top_five_issues_df["cost"] = top_five_issues_df["Cost"].apply(lambda x: round(x / 1_000_000))

    # ---- Average Response Time Conversions ----
    average_response_time_days = round(
        report_summary_data_dic.pop("Overall Average Response Time(days)")
    )

    average_response_time = {
        "Average Response Time in days": average_response_time_days,
        "Average Response Time in hours": average_response_time_days * 24,
        "Average Response Time in minutes": average_response_time_days * 1440
    }

    # ---- Create Executive Summary DataFrame ----
    for col in [
        "Impact in Cost(M$)",
        "Outstanding Issues Avg Response Time",
        "Solved Issues Avg Response Time"
    ]:
        report_summary_data_dic[col] = pd.to_numeric(
            report_summary_data_dic[col], errors="coerce"
        )

    report_summary_df = pd.DataFrame(report_summary_data_dic)
    report_summary_df = report_summary_df.apply(
        lambda x: round(x) if x.dtype.kind in "biufc" else x
    )

    # ---- Executive Displays ----
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


# ==================================
# Executive Report Visualization
# ==================================

def plot_executive_report_bars(data_dic: dict) -> None:
    """
    Plot horizontal bar charts for executive KPIs.

    Parameters
    ----------
    data_dic : dict
        Aggregated executive metrics
    """

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    axes = axes.flatten()

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c",
        "#d62728", "#9467bd", "#8c564b", "#e377c2"
    ]

    for i, (title, data) in enumerate(data_dic.items()):
        if i >= len(axes):
            break

        sorted_data = data.sort_values()
        ax = axes[i]
        ax.barh(sorted_data.index, sorted_data.values, color=colors[i % len(colors)])

        ax.set_title(title, fontsize=14)
        ax.set_facecolor("#f5f5f5")
        ax.xaxis.set_visible(False)

        for spine in ax.spines.values():
            spine.set_visible(False)

        for j, v in enumerate(sorted_data.values):
            ax.text(v, j, str(v), va="center", fontsize=10)

    # Remove extra subplots if fewer data points
    for i in range(len(data_dic), len(axes)):
        fig.delaxes(axes[i])

    # Display the plots
    plt.tight_layout()
    
    plt.show()


def plot_executive_report_donut_charts(data_dic: dict) -> None:
    """
    Plot donut charts showing threat distribution by severity.

    Parameters
    ----------
    data_dic : dict
        Aggregated executive metrics
    """

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    axes = axes.flatten()

    color_map = {
        "Critical": "darkred",
        "High": "red",
        "Medium": "orange",
        "Low": "green"
    }

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color, label=level)
        for level, color in color_map.items()
    ]

    fig.legend(handles, color_map.keys(), loc="upper right", title="Threat Level")

    for i, (title, data) in enumerate(data_dic.items()):
        if i >= len(axes):
            break

        labels = data.index
        values = data.values
        colors = [color_map.get(label, "gray") for label in labels]
        total = values.sum()

        axes[i].pie(
            values,
            labels=[f"{l}\n{v} ({v/total:.0%})" for l, v in zip(labels, values)],
            startangle=90,
            colors=colors,
            wedgeprops=dict(width=0.4)
        )

        axes[i].text(0, 0, str(total), ha="center", va="center", fontsize=14, fontweight="bold")
        axes[i].set_title(title, fontsize=14)

    # Remove extra subplots if fewer data points
    for i in range(len(data_dic), len(axes)):
        fig.delaxes(axes[i])

    # Display the plots
    plt.tight_layout()
    
    plt.show()


# ============================
# PDF Executive Report Class
# ============================

class ExecutiveReport(FPDF):
    """
    Custom PDF class for executive cybersecurity reporting.
    """

    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Executive Report: Cybersecurity Incident Analysis", align="C", ln=True)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title: str):
        """Add a section title."""
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=True)
        self.ln(5)

    def section_body(self, body: str):
        """Add a paragraph body."""
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_table(self, headers, data, col_widths):
        """Add a formatted table."""
        self.set_font("Arial", "B", 10)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 10, header, border=1, align="C")
        self.ln()

        self.set_font("Arial", "", 10)
        for row in data:
            for i, item in enumerate(row):
                self.cell(col_widths[i], 10, str(item), border=1, align="C")
            self.ln()


# =====================================================
# Main Execution Pipelines (Dashboard & PDF)
# =====================================================

def main_executive_report_pipeline(df: pd.DataFrame) -> None:
    """Run executive KPI aggregation and visualization."""
    report_summary_data_dic = generate_executive_report(df)
    plot_executive_report_bars(report_summary_data_dic)
    plot_executive_report_donut_charts(report_summary_data_dic)


def main_dashboard(NEW_DATA_URL = None,
                   simulated_attacks_file_path = None) -> None:
    #simulated_attacks_file_path: str =
    #"CyberThreat_Insight/cybersecurity_data/combined_normal_and_simulated_attacks_class_df.csv"

    """
    Main dashboard execution entry point.

    Parameters
    ----------
    NEW_DATA_URL: sty #  simulated= attacks file URL 
    simulated_attacks_file_path: sty #  simulated= attacks file path
    """

    #attack_simulation_df = pd.read_csv(simulated_attacks_file_path)
    attack_simulation_df = main_attacks_simulation_pipeline(NEW_DATA_URL)
    
     #load attacks data 
    if simulated_attacks_file_path is not None:
        df = pd.read_csv(simulated_attacks_file_path)
        attack_simulation_df = normalize_threat_level(df)
    else:
        df =  main_attacks_simulation_pipeline(NEW_DATA_URL)
        attack_simulation_df = normalize_threat_level(df)
                       
    print("\nDashboar main_attacks_executive_reporting_pipeline\n")
    main_executive_report_pipeline(attack_simulation_df)
        
    #print("\nRunning Executive KPI Dashboard\n")
    #print("\nDashboar attacks_executive_summary_reporting_pipeline\n")
   # main_attacks_executive_summary_reporting_pipeline(attack_simulation_df)
   

if __name__ == "__main__":
    main_dashboard(NEW_DATA_URL)
