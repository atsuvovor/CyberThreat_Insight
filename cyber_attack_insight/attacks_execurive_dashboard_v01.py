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
from CyberThreat_Insight.cyber_attack_insight.attack_simulation_secenario_v01 import get_attacks_data


NEW_DATA_URL = "https://drive.google.com/file/d/1Nr9PymyvLfDh3qTfaeKNVbvLwt7lNX6l/view?usp=sharing"


def generate_executive_report(df):
    # Threat statistics
    total_theats = df.groupby("Threat Level").size()
    severity_stats = df.groupby("Severity").size()
    impact_cost_stats = round(df.groupby("Severity")["Cost"].sum()/ 1_000_000)
    resolved_stats = df[df["Status"].isin(["Resolved", "Closed"])].groupby("Threat Level").size()
    out_standing_issues = df[df["Status"].isin(["Open", "In Progress"])].groupby("Threat Level").size()
    outstanding_issues_avg_resp_time = round(df[df["Status"].isin(["Open", "In Progress"])].groupby("Threat Level")["Issue Response Time Days"].mean())
    solved_issues_avg_resp_time = round(df[df["Status"].isin(["Resolved", "Closed"])].groupby("Threat Level")["Issue Response Time Days"].mean())


    # Top 5 issues
    top_issues = df.nlargest(5, "Threat Score")

    # Average response time
    overall_avg_response_time = df["Issue Response Time Days"].mean()

    # Save to CSV
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

    top_five_issues_df = pd.DataFrame(report_summary_data_dic.pop("Top 5 Issues"))
    top_five_issues_df["cost"] =  top_five_issues_df["Cost"].apply(lambda x: round(x/1_000_000))
    average_response_time = round(report_summary_data_dic.pop("Overall Average Response Time(days)"))

    # Convert numeric columns to numeric type before creating the DataFrame
    for col in ["Impact in Cost(M$)", "Outstanding Issues Avg Response Time", "Solved Issues Avg Response Time"]:
        report_summary_data_dic[col] = pd.to_numeric(report_summary_data_dic[col], errors='coerce')


    # Create report_summary_df from report_summary_data_dic
    report_summary_df = pd.DataFrame(report_summary_data_dic)

    # Apply round to numeric columns only after creating the DataFrame
    report_summary_df = report_summary_df.apply(lambda x: round(x) if x.dtype.kind in 'biufc' else x)

    top_five_incidents_defense_df = top_five_issues_df[["Issue ID", "Threat Level", "Severity",
                                                        "Issue Response Time Days", "Department Affected", "Cost", "Defense Action"]]
    days = 184
    hours = days * 24
    minutes = days * 1440
    average_response_time ={
        "Average Response Time in days" : average_response_time,
        "Average Response Time in hours" : hours,
        "Average Response Time in minutes" : minutes
        }

    average_response_time_df = pd.DataFrame(average_response_time, index=[0])

    print("\nreport_summary_df\n")
    display(report_summary_df)
    print("\naverage_response_time\n")
    display(average_response_time_df)
    print("\nTop 5 issues impact with  Addaptative Defense Mechanism\n")
    display(top_five_incidents_defense_df)

    return report_summary_data_dic

#------------------------- Plot Executive Report metrics--------------------------------------------

#Bar chart--

def plot_executive_report_bars(data_dic):
    # Define the number of plots
    num_plots = len(data_dic)

    # Create a figure with 2 rows and 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    axes = axes.flatten()  # Flatten the axes for easier indexing

    # Define the colors for each plot
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    # Iterate over the data dictionary and create each subplot
    for i, (title, data) in enumerate(data_dic.items()):
        if i >= len(axes):  # Break if more plots than subplots
            break
        ax = axes[i]

        # Sort data for ascending bars
        sorted_data = data.sort_values()

        # Plot the horizontal bar chart
        ax.barh(sorted_data.index, sorted_data.values, color=colors[i % len(colors)])

        # Customize the subplot
        ax.set_title(title, fontsize=14)
        ax.set_facecolor("#f5f5f5")  # Light gray background
        ax.spines['top'].set_visible(False)  # Remove top border
        ax.spines['right'].set_visible(False)  # Remove right border
        ax.spines['left'].set_visible(False)  # Remove left border
        ax.spines['bottom'].set_visible(False)  # Remove bottom border
        ax.xaxis.set_visible(False)  # Hide the x-axis
        for j, v in enumerate(sorted_data.values):
            ax.text(v, j, str(v), va='center', fontsize=10)  # Add labels

    # Remove extra subplots if fewer data points
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    # Display the plots
    plt.show()

#donut chart---------------------

def plot_executive_report_donut_charts(data_dic):
    # Define the number of plots
    num_plots = len(data_dic)

    # Create a figure with 2 rows and 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    axes = axes.flatten()  # Flatten the axes for easier indexing

    # Define the color mapping
    color_map = {
        "Critical": "darkred",
        "High": "red",
        "Medium": "orange",
        "Low": "green"
    }

    # Create a single legend for the entire figure
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=level,
                          markersize=10, markerfacecolor=color) for level, color in color_map.items()]
    fig.legend(handles, color_map.keys(), loc='upper right', fontsize=12, title="Threat Level")

    # Iterate over the data dictionary and create each subplot
    for i, (title, data) in enumerate(data_dic.items()):
        if i >= len(axes):  # Break if more plots than subplots
            break
        ax = axes[i]

        # Prepare data for the pie chart
        labels = data.index
        values = data.values
        colors = [color_map[label] for label in labels]
        total = values.sum()  # Total sum of values

        # Create a donut plot
        wedges, texts, autotexts = ax.pie(
            values,
            labels=[f"{label}\n{value} ({value/total:.0%})" for label, value in zip(labels, values)],
            autopct='',
            startangle=90,
            colors=colors,
            wedgeprops=dict(width=0.4)
        )

        # Add the total sum at the center of the donut
        ax.text(0, 0, str(total), ha='center', va='center', fontsize=14, fontweight='bold')

        # Set title
        ax.set_title(title, fontsize=14)

    # Remove extra subplots if fewer data points
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    # Display the plots
    plt.show()

#---------------------------------------------Generate Executive Summary------------------------------------------------
# Generate executive Summary
class ExecutiveReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Executive Report: Cybersecurity Incident Analysis', align='C', ln=True)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=True)
        self.ln(5)

    def section_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_table(self, headers, data, col_widths):
        self.set_font('Arial', 'B', 10)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 10, header, border=1, align='C')
        self.ln()
        self.set_font('Arial', '', 10)
        for row in data:
            for i, item in enumerate(row):
                self.cell(col_widths[i], 10, str(item), border=1, align='C')
            self.ln()

# Extract attacks key metrics for the report
def extract_attacks_key_metrics(df):

    critical_issues_df = df[df["Severity"] == "Critical"]
    resolved_issues_df = df[df["Status"].isin(["Resolved", "Closed"])]
    attack_types = ["Phishing", "Malware", "DDOS", "Data Leak"]

    phishing_attack_departement_affected = df[df["Login Attempts"] > 10]
    malware_attack_departement_affected = df[df["Num Files Accessed"] > 50]
    ddos_attack_departement_affected = df[df["Session Duration in Second"] > 3600]
    ddos_attack_departement_affected = df[df["Data Transfer MB"] > 500]
    data_leak_attack_departement_affected = df[df["Data Transfer MB"] > 500]

    attack_type_departement_affected_dic = {
        "Phishing": phishing_attack_departement_affected,
        "Malware": malware_attack_departement_affected,
        "DDOS": ddos_attack_departement_affected,
        "Data Leak": data_leak_attack_departement_affected
    }

    metrics_dic = {
        "Total Issues": len(df),
        "Critical Issues": len(critical_issues_df),
        "Resolved Issues": len(resolved_issues_df),
        "Unresolved Issues": len(df) - len(resolved_issues_df),
        "Phishing Attacks": len(df[df["Login Attempts"] > 10]),
        "Malware Attacks": len(df[df["Num Files Accessed"] > 50]),
        "DDOS Attacks": len(df[df["Session Duration in Second"] > 3600]),
        "Data Leak Attacks": len(df[df["Data Transfer MB"] > 500]),
    }

    attack_metrics_df = pd.DataFrame(metrics_dic, index=["Value"]).T

    Incident_summary_dic = {
        "Total Issues": metrics_dic["Total Issues"],
        "Critical Issues": metrics_dic["Critical Issues"],
        "Resolved Issues": metrics_dic["Resolved Issues"],
        "Unresolved Issues": metrics_dic["Unresolved Issues"]}

    Insident_summary_df = pd.DataFrame(Incident_summary_dic, index=["Value"]).T

    attack_scenarios_dic = {
        "Phishing Attacks": metrics_dic['Phishing Attacks'],
        "Malware Attacks": metrics_dic['Malware Attacks'],
        "DDOS Attacks": metrics_dic['DDOS Attacks'],
        "Data Leak Attacks": metrics_dic['Data Leak Attacks']}

    attack_scenarios_df = pd.DataFrame(attack_scenarios_dic, index=["Value"]).T

    critical_issues_sample_df = critical_issues_df.head(10)[["Issue ID", "Category",
                                                             "Threat Level", "Severity",
                                                             "Status", "Risk Level", "Impact Score",
                                                             "Issue Response Time Days", "Department Affected",
                                                             "Cost", "Defense Action"]]


    return metrics_dic, Incident_summary_dic, attack_scenarios_dic, attack_type_departement_affected_dic, critical_issues_df, critical_issues_sample_df

#-------------------------------plot incident_summary  and  attack_scenario----------------------------------

def millions_formatter(x, pos):
    return f"{x / 1e6:.1f}"

def plot_attacks_metrics(incident_summary_dic, attack_scenarios_dic, attack_type_departement_affected_dic):
    # Convert dictionaries to dataframes
    incident_summary_df = pd.DataFrame(incident_summary_dic, index=["Value"]).T
    attack_scenarios_df = pd.DataFrame(attack_scenarios_dic, index=["Value"]).T

    # Extract the attack dataframes
    phishing_df = attack_type_departement_affected_dic["Phishing"]
    malware_df = attack_type_departement_affected_dic["Malware"]
    ddos_df = attack_type_departement_affected_dic["DDOS"]
    data_leak_df = attack_type_departement_affected_dic["Data Leak"]

    # List of all data to plot
    plot_data = [
        (incident_summary_df, "Incident Summary", "index", "Value"),
        (attack_scenarios_df, "Attack Scenarios", "index", "Value"),
        (phishing_df, "Phishing Attack - Dept vs Cost", "Department Affected", "Cost"),
        (malware_df, "Malware Attack - Dept vs Cost", "Department Affected", "Cost"),
        (ddos_df, "DDOS Attack - Dept vs Cost", "Department Affected", "Cost"),
        (data_leak_df, "Data Leak Attack - Dept vs Cost", "Department Affected", "Cost")
    ]

    # Define a color palette for the subplots
    colors = ['steelblue', 'darkorange', 'seagreen', 'crimson', 'gold', 'purple']

    # Create subplots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, (df, title, x_col, y_col) in enumerate(plot_data):
        ax = axes[i]

        # Assign a unique color to each plot
        color = colors[i]

        if not df.empty:  # Ensure dataframe is not empty
            if x_col == "index":  # Handle incident_summary_df and attack_scenarios_df
                df_sorted = df.sort_values(by=y_col, ascending=False)
                ax.barh(df_sorted.index, df_sorted[y_col], color=color, edgecolor='none')

                ax.set_title(title, fontsize=12)
                ax.set_xlabel(y_col)
                ax.set_ylabel(x_col)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            else:  # Handle attack-type dataframes
                df_sorted = df.sort_values(by=y_col, ascending=False)
                ax.barh(df_sorted[x_col], df_sorted[y_col], color=color, edgecolor='none')

                # Format x-axis values as "M $"
                ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))

                ax.set_title(title, fontsize=12)
                ax.set_xlabel(y_col if y_col != "Cost" else "Cost (in M $)")
                ax.set_ylabel(x_col)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        else:
            # Handle empty dataframes
            ax.text(0.5, 0.5, "No Data Available", horizontalalignment='center', verticalalignment='center', fontsize=12)
            ax.set_title(title, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # Hide any unused axes if fewer than 6 plots
    for j in range(len(plot_data), len(axes)):
        axes[j].axis("off")

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


# Generate the PDF report
def generate_attacks_pdf_report(metrics, Insident_summary, attack_scenarios, critical_issues_df):
    report = ExecutiveReport()
    report.add_page()

    report.section_title("Incident Summary")
    summary_body = (
        f"Total Issues: {metrics['Total Issues']}\n"
        f"Critical Issues: {metrics['Critical Issues']}\n"
        f"Resolved Issues: {metrics['Resolved Issues']}\n"
        f"Unresolved Issues: {metrics['Unresolved Issues']}\n"
        )
    report.section_body(summary_body)

    report.section_title("Attack Scenarios")
    attack_body = (
        f"Phishing Attacks: {metrics['Phishing Attacks']}\n"
        f"Malware Attacks: {metrics['Malware Attacks']}\n"
        f"DDOS Attacks: {metrics['DDOS Attacks']}\n"
        f"Data Leak Attacks: {metrics['Data Leak Attacks']}\n"
        )
    report.section_body(attack_body)

    report.section_title("Critical Issues Overview")
    critical_issues_sample_df = critical_issues_df.head(10)[["Issue ID", "Category", "Threat Level", "Severity", "Status", "Risk Level",
         "Impact Score", "Issue Response Time Days", "Department Affected", "Cost", "Defense Action"]]

    headers = critical_issues_sample_df.columns.tolist()
    data = critical_issues_sample_df.values.tolist()
    col_widths = [30, 40, 30, 30, 30, 30, 30, 30, 100, 30, 100]
    report.add_table(headers, data, col_widths)


    # Save the report
    report.output(Executive_Cybersecurity_Attack_Report_on_google_drive)
    print(f"Executive Report saved to {Executive_Cybersecurity_Attack_Report_on_google_drive}")

#------------Metric extraction pipiline------------
def attacks_key_metrics_pipeline(df):

    metrics_dic, incident_summary_dic, attack_scenarios_dic, attack_type_departement_affected_dic, \
                            critical_issues_df, critical_issues_sample_df = extract_attacks_key_metrics(df)

    print("\n")

    plot_attacks_metrics(incident_summary_dic, attack_scenarios_dic, attack_type_departement_affected_dic)
    print("\n")
    print("\nCritical Issues Sample\n")
    display(critical_issues_sample_df)

    return  metrics_dic, incident_summary_dic, attack_scenarios_dic, critical_issues_df

def plot_executive_report_metrics(data_dic):
    plot_executive_report_bars(data_dic)
    print("\n")
    print("\n")
    plot_executive_report_donut_charts(data_dic)

#-------------------------------------------Main Pipeline----------------------------------------------------------------------------
def main_executive_report_pipeline(df):

    report_summary_data_dic = generate_executive_report(df)
    plot_executive_report_metrics(report_summary_data_dic)

def main_attacks_executive_summary_reporting_pipeline(df):
    metrics, incident_summary, attack_scenarios, critical_issues_df = attacks_key_metrics_pipeline(df)
    generate_attacks_pdf_report(metrics, incident_summary, attack_scenarios, critical_issues_df)
#-----------------------------------------Main Dashboard-----------------------------------------------------------------------------

def main_dashboard(NEW_DATA_URL = None,
                   simulated_attacks_file_path = None) -> None:
                       
    """
    Entry point for the simulation executive dashboard.
    Loads data and runs all executive summaries.
    """
    #load attacks data 
    if simulated_attacks_file_path is not None:
        attack_simulation_df = pd.read_csv(simulated_attacks_file_path)
    else:
        attack_simulation_df = get_attacks_data(NEW_DATA_URL)

        print("\nDashboar main_attacks_executive_summary_reporting_pipeline\n")
        main_executive_report_pipeline(attack_simulation_df)

    print("\nDashboar attacks_executive_summary_reporting_pipeline\n")
    main_attacks_executive_summary_reporting_pipeline(attack_simulation_df)

if __name__ == "__main__":
    main_dashboard(NEW_DATA_URL)
