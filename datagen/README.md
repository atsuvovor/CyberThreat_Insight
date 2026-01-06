
<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/cybersecurity_data_generator2.png" 
       alt="Centered Image" 
       style="width: 100%; height: auto;">
</p>
🔗 Live Dashboard:
<a 
  href="https://colab.research.google.com/github/atsuvovor/Cybersecurity-Data-Generator/blob/main/CyberInsightDataGenerator.ipynb" 
  target="_blank"
>
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<details>

<summary>Click to view the code</summary>

```python

# -*- coding: utf-8 -*-
"""
datagen/cyberdatagen.py
CyberDataGen
Anomalous Behavior Detection in Cybersecurity Analytics using Generative AI
-------------------------------------------------------------------------------
Portable version:
- Works in Colab, JupyterLab, or local Python
- Saves to CyberThreat_Insight/cybersecurity_data/
- Displays datasets (optional)
- Saves 5 CSVs + prints summary
- Prompts user for optional local ZIP download
Author: Atsu Vovor
"""

import os
import shutil
import zipfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from IPython.display import display
import argparse
import random
import warnings
warnings.filterwarnings("ignore")
# Try to import Colab-specific files.download (ignored if not available)
try:
    from google.colab import files
    COLAB = True
except ImportError:
    COLAB = False

from  synthetic_data_plot import explaratory_data_analysis_pipeline
# =====================================================================
# Configuration
# =====================================================================
class DataConfig:
    def __init__(self):
        # ------------------ Parameters ------------------
        self.num_normal_issues = 800
        self.num_anomalous_issues = 200
        self.total_issues = self.num_normal_issues + self.num_anomalous_issues
        self.num_users = 100
        self.num_reporters = 10
        self.num_assignees = 20
        self.num_departments = 5
        self.current_date = datetime.now()
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(self.current_date.year, self.current_date.month, self.current_date.day)

        # ------------------ Paths ------------------
        self.github_repo_folder = "/content/CyberThreat_Insight/cybersecurity_data"
        os.makedirs(self.github_repo_folder, exist_ok=True)

        self.normal_data_file = os.path.join(self.github_repo_folder, "cybersecurity_dataset_normal.csv")
        self.anomalous_data_file = os.path.join(self.github_repo_folder, "cybersecurity_dataset_anomalous.csv")
        self.combined_data_file = os.path.join(self.github_repo_folder, "cybersecurity_dataset_combined.csv")
        self.key_threat_indicators_file = os.path.join(self.github_repo_folder, "key_threat_indicators.csv")
        self.scenarios_with_colors_file = os.path.join(self.github_repo_folder, "scenarios_with_colors.csv")
        self.zip_file = os.path.join(self.github_repo_folder, "cybersecurity_data.zip")

        # ------------------ Metadata ------------------
        self.issue_ids = [f"ISSUE-{i:04d}" for i in range(1, self.num_normal_issues + 1)]
        self.issue_keys = [f"KEY-{i:04d}" for i in range(1, self.num_normal_issues + 1)]

        self.KPI_list = [
            "Network Security","Access Control","System Vulnerability",
            "Penetration Testing Effectiveness","Management Oversight",
            "Procurement Security", "Control Effectiveness",
            "Asset Inventory Accuracy", "Vulnerability Remediation",
            "Risk Management Maturity", "Risk Assessment Coverage"
        ]
        self.KRI_list = [
            "Data Breach", "Phishing Attack","Malware","Data Leak",
            "Legal Compliance","Risk Exposure", "Cloud Security Posture",
            "Unauthorized Access", "DDOS"
        ]
        self.categories = self.KPI_list + self.KRI_list
        self.severities = ["Low", "Medium", "High", "Critical"]
        self.statuses = ["Open", "In Progress", "Resolved","Closed"]
        self.reporters = [f"Reporter {i}" for i in range(1, self.num_reporters + 1)]
        self.assignees = [f"Assignee {i}" for i in range(1, self.num_assignees + 1)]
        self.users = [f"User_{i}" for i in range(1, self.num_users + 1)]
        self.departments = ["IT", "Finance", "Operations", "HR", "Legal",
                            "Sales", "C-Suite Executives", "External Contractors"]
        self.locations = ["CANADA", "USA", "Unknown", "EU", "DE", "FR", "JP", "CN", "AU", "IN", "UK"]

        self.columns = [
            "Issue ID", "Issue Key", "Issue Name", "Issue Volume", "Category",
            "Severity", "Status", "Reporters", "Assignees", "Date Reported",
            "Date Resolved", "Issue Response Time Days", "Impact Score", "Risk Level",
            "Department Affected", "Remediation Steps", "Cost", "KPI/KRI", "User ID",
            "Timestamps", "Activity Type","User Location", "IP Location",
            "Session Duration in Second", "Num Files Accessed", "Login Attempts",
            "Data Transfer MB", "CPU Usage %", "Memory Usage MB", "Threat Score",
            "Threat Level", "Defense Action"
        ]

        # ------------------ Extra DataFrames ------------------
        self.ktis_data = {
            "KIT": [
                "Severity", "Impact Score", "Risk Level", "Response Time", "Category",
                "Activity Type", "Login Attempts", "Num Files Accessed", "Data Transfer MB",
                "CPU Usage %", "Memory Usage MB"
            ],
            "Condition": [
                "Critical = 10, High = 8, Medium = 5, Low = 2",
                "1 to 10 (already a score)",
                "High = 8, Medium = 5, Low = 2",
                ">7 days = 5, 3-7 days = 3, <3 days = 1",
                "Unauthorized Access = 8, Phishing = 6, etc.",
                "High-risk types (e.g., login, data_transfer)",
                ">5 = 5, 3-5 = 3, <3 = 1",
                ">10 = 5, 5-10 = 3, <5 = 1",
                ">100 MB = 5, 50-100 MB = 3, <50 MB = 1",
                ">80% = 5, 60-80% = 3, <60% = 1",
                ">8000 MB = 5, 4000-8000 MB = 3, <4000 MB = 1"
            ],
            "Score": [
                "2 - 10", "1 - 10", "2 - 8", "1 - 5", "1 - 8", "1 - 5", "1 - 5", "1 - 5", "1 - 5", "1 - 5", "1 - 5"
            ]
        }
        self.ktis_key_threat_indicators_df = pd.DataFrame(self.ktis_data)

        self.scenario_data = {
            "Scenario": list(range(1, 17)),
            "Threat Level": [
                "Critical", "Critical", "Critical", "Critical",
                "High", "High", "High", "High",
                "Medium", "Medium", "Medium", "Medium",
                "Low", "Low", "Low", "Low"
            ],
            "Severity": [
                "Critical", "High", "Medium", "Low",
                "Critical", "High", "Medium", "Low",
                "Critical", "High", "Medium", "Low",
                "Critical", "High", "Medium", "Low"
            ],
            "Suggested Color": [
                "Dark Red", "Red", "Orange-Red", "Orange",
                "Red", "Orange-Red", "Orange", "Yellow-Orange",
                "Orange", "Yellow-Orange", "Yellow", "Light Yellow",
                "Yellow", "Light Yellow", "Green-Yellow", "Green"
            ]
        }
        self.scenarios_with_colors_df = pd.DataFrame(self.scenario_data)

        #---------------------------------------------Define columns---------------------------------------------------
        self.numerical_columns = [
            "Timestamps", "Issue Response Time Days", "Impact Score", "Cost",
            "Session Duration in Second", "Num Files Accessed", "Login Attempts",
            "Data Transfer MB", "CPU Usage %", "Memory Usage MB", "Threat Score"
            ]


        self.explanatory_data_analysis_columns = [
            "Date Reported", "Issue Response Time Days", "Impact Score", "Cost",
            "Session Duration in Second", "Num Files Accessed", "Login Attempts",
            "Data Transfer MB", "CPU Usage %", "Memory Usage MB", "Threat Score"
            ]

        self.user_activity_features = [
            "Risk Level", "Issue Response Time Days", "Impact Score", "Cost",
            "Session Duration in Second", "Num Files Accessed", "Login Attempts",
            "Data Transfer MB", "CPU Usage %", "Memory Usage MB", "Threat Score"
            ]


        self.initial_dates_columns = ["Date Reported", "Date Resolved", "Timestamps"]

        self.categorical_columns = ["Issue ID", "Issue Key", "Issue Name", "Category", "Severity", "Status", "Reporters",
                               "Assignees", "Risk Level", "Department Affected", "Remediation Steps", "KPI/KRI",
                               "User ID", "Activity Type", "User Location", "IP Location", "Threat Level",      "Defense Action", "Color"
                               ]
        self.features_engineering_columns = [
            "Issue Response Time Days", "Impact Score", "Cost",
            "Session Duration in Second", "Num Files Accessed", "Login Attempts",
            "Data Transfer MB", "CPU Usage %", "Memory Usage MB", "Threat Score", "Threat Level"
            ]
        self.numerical_behavioral_features = [
            "Login Attempts", "Data Transfer MB", "CPU Usage %", "Memory Usage MB",
            "Session Duration in Second", "Num Files Accessed", "Threat Score"
            ]

        #IP addresses, port numbers, packet sizes, and time intervals
        # ---------------------Generate user activity metadata------------------------
        self.activity_types = ["login", "file_access", "data_modification"]

    def get_column_dic(self):
        """
        Returns a dictionary containing lists of column names categorized by type.
        """
        columns_dic = {
            "numerical_columns": self.numerical_columns,
            "explanatory_data_analysis_columns": self.explanatory_data_analysis_columns,
            "user_activity_features": self.user_activity_features,
            "initial_dates_columns": self.initial_dates_columns,
            "categorical_columns": self.categorical_columns,
            "features_engineering_columns": self.features_engineering_columns
        }
        return columns_dic

# =====================================================================
# Data Generator (keep your original generation logic here)
# =====================================================================
class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.user_profiles = {user: {'baseline_activity': random.uniform(0.5, 1.5),
                                     'risk_tolerance': random.uniform(0.8, 1.2)} for user in self.config.users}
        self.department_profiles = {dept: {'baseline_risk': random.uniform(0.5, 1.5),
                                         'activity_multiplier': random.uniform(0.8, 1.2)} for dept in self.config.departments}
        self.anomalous_issue_ids = [f"ISSUE-{i:04d}" for i in range(self.config.num_normal_issues +1, self.config.total_issues + 1)]
        self.anomalous_issue_keys = [f"KEY-{i:04d}" for i in range(self.config.num_normal_issues +1, self.config.total_issues + 1)]
    

    #                   -----------------------------------------------------------------------
    #                      Generate normal issue names for each KPI and KRI by Mapping
    #                      normal issue name to issue category using a dictionary
    #                   ---------------------------------------------------------------------
    def generate_normal_issues_name(self, category):# Mapping issue name to issue category using a dictionary
        """
        Maps a category to a synthetic normal issue name.

        Args:
            category (str): The category of the issue.

        Returns:
            str: The corresponding normal issue name, or "Unknown Issue" if not found.
        """
        issue_mapping = {
            "Network Security": "Inadequate Firewall Configurations",
            "Access Control": "Weak Authentication Protocols",
            "System Vulnerability": "Outdated Operating System Components",
            "Penetration Testing Effectiveness": "Unresolved Vulnerabilities from Latest Penetration Test",
            "Management Oversight": "Inconsistent Review of Security Policies",
            "Procurement Security": "Supplier Security Compliance Gaps",
            "Control Effectiveness": "Insufficient Access Control Measures",
            "Asset Inventory Accuracy": "Missing or Inaccurate Asset Records",
            "Vulnerability Remediation": "Delayed Patching of Known Vulnerabilities",
            "Risk Management Maturity": "Incomplete Risk Management Framework",
            "Risk Assessment Coverage": "Insufficient Coverage in Annual Risk Assessment",
            "Data Breach": "Unauthorized Access Leading to Data Exposure",
            "Phishing Attack": "Successful Phishing Attempt Targeting Executives",
            "Malware": "Detected Malware Infiltration in Internal Systems",
            "Data Leak": "Sensitive Data Leak via Misconfigured Cloud Storage",
            "Legal Compliance": "Non-Compliance with Data Protection Regulations",
            "Risk Exposure": "Increased Exposure due to Insufficient Data Encryption",
            "Cloud Security Posture": "Weak Cloud Storage Access Controls",
            "Unauthorized Access": "Access by Unauthorized Personnel Detected",
            "DDOS": "High-Volume Distributed Denial-of-Service Attack"
        }

        return issue_mapping.get(category, "Unknown Issue")


    #                                 -------------------------------------------------------------------
    #                                   Generate anomalous issue names for each KPI and KRI by Mapping
    #                                   anomalous issue name to issue category using a dictionary
    #                                  ------------------------------------------------------------------
    def generate_anomalous_issue_name(self, category):
        """
        Maps a category to a synthetic anomalous issue name.

        Args:
            category (str): The category of the issue.

        Returns:
            str: The corresponding anomalous issue name, or "Unknown Issue" if not found.
        """

        anomalous_issue_mapping = {
            "Network Security": "Sudden Increase in Unfiltered Traffic",
            "Access Control": "Multiple Unauthorized Access Attempts Detected",
            "System Vulnerability": "Newly Discovered Vulnerabilities in Core Systems",
            "Penetration Testing Effectiveness": "Critical Issues Not Detected in Last Penetration Test",
            "Management Oversight": "High Frequency of Policy Violations",
            "Procurement Security": "Supplier Network Breach Exposure",
            "Control Effectiveness": "Ineffective Access Controls in High-Sensitivity Areas",
            "Asset Inventory Accuracy": "Significant Number of Untracked Devices",
            "Vulnerability Remediation": "Delayed Patching of Critical Vulnerabilities",
            "Risk Management Maturity": "Lack of Updated Risk Management Procedures",
            "Risk Assessment Coverage": "Unassessed High-Risk Areas",
            "Data Breach": "Unusual Data Transfer Volumes Detected",
            "Phishing Attack": "Targeted Phishing Campaign Against Executives",
            "Malware": "Malware Detected in Core System Components",
            "Data Leak": "Unusual Data Access from External Locations",
            "Legal Compliance": "Potential Non-Compliance Detected in Sensitive Data Handling",
            "Risk Exposure": "Unanticipated Increase in Risk Exposure",
            "Cloud Security Posture": "Weak Access Controls on Critical Cloud Resources",
            "Unauthorized Access": "Spike in Unauthorized Access Attempts",
            "DDOS": "High-Volume Distributed Denial-of-Service Attack from Multiple Sources"
        }

        return anomalous_issue_mapping.get(category, "Unknown Issue")


    #-------------------------Implementation-----------------------------------
    # filter KPI Vs KRI
    def filter_kpi_and_kri(self, category):
        """
        Categorizes an issue as either a Key Performance Indicator (KPI) or a Key Risk Indicator (KRI).

        Args:
            category (str): The category of the issue.

        Returns:
            str: 'KPI' if the category is in the KPI list, otherwise 'KRI'.
        """
        if category in self.config.KPI_list:
            return 'KPI'
        else:
            return 'KRI'
#
    # Function to generate a random start date within a specific date range--
    def random_date(self, start, end):
        """
        Generates a random datetime object within a specified date range.

        Args:
            start (datetime): The start date of the range.
            end (datetime): The end date of the range.

        Returns:
            datetime: A random datetime object within the specified range.
        """
        # Calculate the difference in days and add a random number of days to the start date
        return start + timedelta(days=np.random.randint(0, (end - start).days))

    # ----------------------------------Define threat level calculation-----------------------------------------------------
    def calculate_threat_level(self, severity, impact_score, risk_level, response_time_days,
                               login_attempts, num_files_accessed, data_transfer_MB,
                               cpu_usage_percent, memory_usage_MB):
        """
        Calculates a threat score and assigns a threat level based on various issue attributes.

        Args:
            severity (str): The severity of the issue ('Low', 'Medium', 'High', 'Critical').
            impact_score (int or float): The impact score of the issue (1-10).
            risk_level (str): The risk level of the issue ('Low', 'Medium', 'High', 'Critical').
            response_time_days (int): The response time in days for the issue.
            login_attempts (int): The number of login attempts.
            num_files_accessed (int): The number of files accessed.
            data_transfer_MB (int or float): The amount of data transferred in MB.
            cpu_usage_percent (int or float): The CPU usage percentage.
            memory_usage_MB (int or float): The memory usage in MB.

        Returns:
            tuple: A tuple containing the calculated threat level (str) and threat score (float).
        """
        # Define scores based on input criteria
        severity_score = {"Critical": 10, "High": 8, "Medium": 5, "Low": 2}.get(severity, 1)
        risk_score = {"Critical": 10, "High": 8, "Medium": 5, "Low": 2}.get(risk_level, 1)

        response_time_score = 5 if response_time_days > 7 else 3 if response_time_days > 3 else 1
        login_attempts_score = 5 if login_attempts > 5 else 3 if login_attempts > 3 else 1
        files_accessed_score = 5 if num_files_accessed > 10 else 3 if num_files_accessed > 5 else 1
        data_transfer_score = 5 if data_transfer_MB > 100 else 3 if data_transfer_MB > 50 else 1

        # New metrics: CPU usage and memory usage
        cpu_usage_score = 5 if cpu_usage_percent > 85 else 3 if cpu_usage_percent > 60 else 1
        memory_usage_score = 5 if memory_usage_MB > 10000 else 3 if memory_usage_MB > 6000 else 1

        # Aggregate the scores
        threat_score = (
            0.25 * severity_score +
            0.2 * impact_score +
            0.15 * risk_score +
            0.1 * response_time_score +
            0.05 * login_attempts_score +
            0.05 * files_accessed_score +
            0.05 * data_transfer_score +
            0.075 * cpu_usage_score +
            0.075 * memory_usage_score
        )

        # Determine threat level based on the calculated score
        if threat_score >= 9:
            return "Critical", threat_score
        elif threat_score >= 7:
            return "High", threat_score
        elif threat_score >= 4:
            return "Medium", threat_score
        else:
            return "Low", threat_score
    
        #--------------------- Adaptive defense mechanism based on threat level and conditions----------------------------------

    def adaptive_defense_mechanism(self, row):
        """
        Determines the adaptive response based on threat level, severity, and activity context.

        Args:
            row (pd.Series): A row from the DataFrame representing a single issue.

        Returns:
            str: The suggested defense action(s).
        """
        action = "Monitor"

        # Map the threat level and severity to actions based on scenarios
        threat_severity_actions = {
            ("Critical", "Critical"): "Immediate System-wide Shutdown & Investigation",
            ("Critical", "High"): "Escalate to Security Operations Center (SOC) & Block User",
            ("Critical", "Medium"): "Isolate Affected System & Restrict User Access",
            ("Critical", "Low"): "Increase Monitoring & Schedule Review",
            ("High", "Critical"): "Escalate to SOC & Restrict Critical System Access",
            ("High", "High"): "Restrict User Activity & Monitor Logs",
            ("High", "Medium"): "Alert Security Team & Review Logs",
            ("High", "Low"): "Flag for Review",
            ("Medium", "Critical"): "Increase Monitoring & Investigate",
            ("Medium", "High"): "Schedule Investigation",
            ("Medium", "Medium"): "Routine Monitoring",
            ("Medium", "Low"): "Log Activity for Reference",
            ("Low", "Critical"): "Log and Notify",
            ("Low", "High"): "Routine Monitoring",
            ("Low", "Medium"): "Log for Reference",
            ("Low", "Low"): "No Action Needed"
        }

        # Assign action based on scenario
        action = threat_severity_actions.get((row["Threat Level"], row["Severity"]), action)

        # Additional responses based on user behavior and thresholds
        if row["Threat Level"] in ["Critical", "High"] and row["Login Attempts"] > 5:
            action += " | Lock Account & Alert"
        if row["Activity Type"] == "File Access" and row["Num Files Accessed"] > 15:
            action += " | Restrict File Access"
        if row["Activity Type"] == "Login" and row["Login Attempts"] > 10:
            action += " | Require Multi-Factor Authentication (MFA)"
        if row["Data Transfer MB"] > 100:
            action += " | Limit Data Transfer"

        return action


    def generate_normal_issues_df(self, p_issue_ids, p_issue_keys):
        """Generates a DataFrame of synthetic normal cybersecurity issue data with enhanced logic."""
        normal_issues_data = []
        time_difference_days = (self.config.end_date - self.config.start_date).days
        # Handle the case where the time difference is zero or negative
        days_increment = max(1, time_difference_days)

        # Ensure the divisor for date calculation is at least 1
        date_divisor = max(1, self.config.num_normal_issues // days_increment)


        for i, (issue_id, issue_key) in enumerate(zip(p_issue_ids, p_issue_keys)):
            issue_volume = 1
            category = random.choice(self.config.categories)
            issue_name = self.generate_normal_issues_name(category)
            severity = random.choice(self.config.severities)
            status = random.choice(self.config.statuses)
            reporter = random.choice(self.config.reporters)
            assignee = random.choice(self.config.assignees)

            # Temporal Pattern: Daily/Weekly spikes and overall trend
            # Adjusted calculation to avoid division by zero
            date_reported = self.config.start_date + timedelta(days=i // date_divisor,
                                                   hours=random.randint(0, 23), minutes=random.randint(0, 59))
            # Add weekly peak
            if date_reported.weekday() in [4, 5]: # Friday, Saturday
                 date_reported += timedelta(hours=random.randint(2, 6))
            # Add daily peak (e.g., morning)
            if date_reported.hour in [9, 10, 11]:
                 date_reported += timedelta(minutes=random.randint(15, 45))


            # Remediation Effectiveness: Depends on severity, status, and a simulated assignee workload
            assignee_workload = random.uniform(0.5, 1.5) # Simulate workload
            severity_factor = {"Low": 0.8, "Medium": 1.0, "High": 1.5, "Critical": 2.0}.get(severity, 1.0)
            status_factor = 1.0 if status in ["Resolved", "Closed"] else 2.0 # Open/In Progress might take longer
            avg_resolution_days = 7 * severity_factor * assignee_workload * status_factor # Base resolution time
            issue_response_time_days = max(1, int(np.random.normal(loc=avg_resolution_days, scale=avg_resolution_days/3)))
            date_resolved = date_reported + timedelta(days=issue_response_time_days) if status in ["Resolved", "Closed"] else self.config.current_date + timedelta(days=random.randint(30,180)) # Simulate future resolution for open issues

            # Feature Dependencies and Realistic Distributions
            user_id = random.choice(self.config.users)
            department_affected = random.choice(self.config.departments)
            user_profile = self.user_profiles[user_id]
            department_profile = self.department_profiles[department_affected]

            timestamp = date_reported + timedelta(hours=np.random.randint(0, 24), minutes=np.random.randint(0, 60))

            activity_type = random.choice(self.config.activity_types)
            user_location = random.choice(self.config.locations) # User location should be generated per issue
            ip_location = user_location if np.random.rand() > 0.8 else random.choice([loc for loc in self.config.locations if loc != user_location])

            # Session Duration: Exponential distribution, adjusted by activity type and profiles
            base_session_duration = 600 # seconds
            activity_multiplier = {"login": 0.5, "file_access": 1.5, "data_modification": 2.0}.get(activity_type, 1.0)
            session_duration = max(10, int(np.random.exponential(scale=base_session_duration * activity_multiplier * user_profile['baseline_activity'])))

            # Num Files Accessed: Poisson or Negative Binomial, adjusted by activity type and profiles
            base_files_accessed = 5
            activity_multiplier_files = {"login": 0.1, "file_access": 2.0, "data_modification": 1.5}.get(activity_type, 1.0)
            num_files_accessed = int(max(1, np.random.poisson(lam=base_files_accessed * activity_multiplier_files * user_profile['baseline_activity'] * department_profile['activity_multiplier'])))


            # Login Attempts: Negative Binomial (for bursty attempts), adjusted by profiles
            base_login_attempts = 3
            login_attempts = max(1, np.random.negative_binomial(n=3, p=0.5) + base_login_attempts * user_profile['baseline_activity'] * department_profile['baseline_risk'])


            # Data Transfer MB: Pareto distribution (few large, many small), adjusted by activity type and profiles
            base_data_transfer = 10 # MB
            activity_multiplier_transfer = {"login": 0.1, "file_access": 1.5, "data_modification": 2.5}.get(activity_type, 1.0)
            data_transfer_MB = max(0.1, np.random.pareto(a=2.0) * base_data_transfer * activity_multiplier_transfer * user_profile['baseline_activity'] * department_profile['activity_multiplier'])

            # CPU/Memory Usage: Dependent on activity type and session duration
            base_cpu = 30
            base_mem = 4000
            cpu_usage_percent = max(1, min(100, np.random.normal(loc=base_cpu + session_duration/300, scale=10)))
            memory_usage_MB = max(512, np.random.normal(loc=base_mem + session_duration*5, scale=1000))


            # Impact Score and Cost: Dependent on Severity and Category
            base_impact = 5
            base_cost = 5000
            severity_impact_multiplier = {"Low": 1.0, "Medium": 1.5, "High": 2.5, "Critical": 4.0}.get(severity, 1.0)
            category_impact_multiplier = {"Data Breach": 3.0, "Malware": 2.0, "Unauthorized Access": 2.5}.get(category, 1.0) # Example category impact
            impact_score = max(1, min(10, int(np.random.normal(loc=base_impact * severity_impact_multiplier * category_impact_multiplier * user_profile['risk_tolerance'], scale=3))))
            cost = max(100, np.random.normal(loc=base_cost * severity_impact_multiplier * category_impact_multiplier * department_profile['baseline_risk'], scale=2000))

            # Risk Level Calculation (was missing)
            risk_level = 'Critical' if impact_score > 8 else 'High' if impact_score > 5 else 'Medium' if impact_score > 3 else 'Low'

            # KPI/KRI Calculation (was missing)
            kpi_kri = self.filter_kpi_and_kri(category)

            remediation_steps = f"Steps to resolve {issue_name}" # Remediation Steps (was missing)


            threat_level, threat_score = self.calculate_threat_level(
                severity, impact_score, risk_level, issue_response_time_days,
                login_attempts, num_files_accessed, data_transfer_MB,
                cpu_usage_percent, memory_usage_MB
            )

            row_data = {
                "Severity": severity, "Impact Score": impact_score, "Risk Level": risk_level,
                "Issue Response Time Days": issue_response_time_days, "Login Attempts": login_attempts,
                "Num Files Accessed": num_files_accessed, "Data Transfer MB": data_transfer_MB,
                "CPU Usage %": cpu_usage_percent, "Memory Usage MB": memory_usage_MB,
                "Threat Level": threat_level, "Activity Type": activity_type
            }
            defense_action = self.adaptive_defense_mechanism(row_data)


            normal_issues_data.append([
                issue_id, issue_key, issue_name, issue_volume, category, severity, status, reporter, assignee,
                date_reported, date_resolved, issue_response_time_days, impact_score, risk_level, department_affected,
                remediation_steps, cost, kpi_kri, user_id, timestamp, activity_type, user_location, ip_location,
                session_duration, num_files_accessed, login_attempts, data_transfer_MB,
                cpu_usage_percent, memory_usage_MB, threat_score, threat_level, defense_action
            ])

        df = pd.DataFrame(normal_issues_data, columns=self.config.columns)
        return df


    def generate_anomalous_issues_df(self, p_anomalous_issue_ids, p_anomalous_issue_keys):
        """Generates a DataFrame of synthetic anomalous cybersecurity issue data with enhanced logic."""
        anomalous_issues_data = []

        for i, (issue_id, issue_key) in enumerate(zip(p_anomalous_issue_ids, p_anomalous_issue_keys)):
            issue_volume = 1
            category = random.choice(self.config.categories)
            issue_name = self.generate_anomalous_issue_name(category)
            severity = np.random.choice(self.config.severities, p=[0.05, 0.15, 0.4, 0.4]) # Higher probability for High/Critical
            status = random.choice(self.config.statuses)
            reporter = random.choice(self.config.reporters)
            assignee = random.choice(self.config.assignees)

            # Temporal Pattern: Deviations from normal patterns
            date_reported = self.random_date(self.config.start_date, self.config.end_date) + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
            # Introduce activity outside typical hours for some anomalies
            if np.random.rand() < 0.4: # 40% chance of off-hours activity
                 date_reported = date_reported.replace(hour=random.choice([0, 1, 2, 3, 4, 5, 6, 22, 23]))

            # Remediation Effectiveness: Can be slower for anomalies
            assignee_workload = random.uniform(1.0, 2.0) # Simulate higher workload for anomalies
            severity_factor = {"Low": 1.5, "Medium": 2.0, "High": 3.0, "Critical": 4.0}.get(severity, 2.0)
            status_factor = 1.5 if status in ["Resolved", "Closed"] else 2.5 # Can take even longer
            avg_resolution_days = 14 * severity_factor * assignee_workload * status_factor # Higher base
            issue_response_time_days = max(1, int(np.random.normal(loc=avg_resolution_days, scale=avg_resolution_days/2)))
            date_resolved = date_reported + timedelta(days=issue_response_time_days) if status in ["Resolved", "Closed"] else self.config.current_date + timedelta(days=random.randint(60,240))


            # Feature Dependencies and Realistic Distributions (shifted for anomalies)
            user_id = random.choice(self.config.users)
            department_affected = random.choice(self.config.departments)
            user_profile = self.user_profiles[user_id]
            department_profile = self.department_profiles[department_affected]


            timestamp = date_reported + timedelta(hours=np.random.randint(0, 24), minutes=np.random.randint(0, 60))

            activity_type = np.random.choice(self.config.activity_types, p=[0.2, 0.4, 0.4]) # Higher chance of file_access, data_modification
            user_location = random.choice(self.config.locations) # User location should be generated per issue
            ip_location = random.choice([loc for loc in self.config.locations if loc != user_location]) # More likely to be from unusual location


            # Session Duration: Exponential distribution, adjusted and potentially shorter/longer for anomalies
            base_session_duration = 900 # seconds
            activity_multiplier = {"login": 0.8, "file_access": 2.0, "data_modification": 3.0}.get(activity_type, 1.5)
            session_duration = max(5, int(np.random.exponential(scale=base_session_duration * activity_multiplier * user_profile['baseline_activity'] * 1.5))) # Higher scale for anomalies

            # Num Files Accessed: Negative Binomial, adjusted and higher for anomalies
            base_files_accessed = 20
            activity_multiplier_files = {"login": 0.5, "file_access": 3.0, "data_modification": 2.5}.get(activity_type, 2.0)
            num_files_accessed = int(max(5, np.random.negative_binomial(n=5, p=0.3) + base_files_accessed * activity_multiplier_files * user_profile['baseline_activity'] * department_profile['activity_multiplier'] * 2.0)) # Higher mean/variance


            # Login Attempts: Negative Binomial (for bursty attempts), adjusted and much higher for anomalies
            base_login_attempts = 10
            login_attempts = max(5, np.random.negative_binomial(n=10, p=0.3) + base_login_attempts * user_profile['baseline_activity'] * department_profile['baseline_risk'] * 3.0) # Much higher mean/variance


            # Data Transfer MB: Pareto distribution, adjusted and much higher for anomalies
            base_data_transfer = 100 # MB
            activity_multiplier_transfer = {"login": 0.5, "file_access": 2.0, "data_modification": 4.0}.get(activity_type, 2.5)
            data_transfer_MB = max(1, np.random.pareto(a=1.5) * base_data_transfer * activity_multiplier_transfer * user_profile['baseline_activity'] * department_profile['activity_multiplier'] * 3.0) # Higher scale, lower 'a' for heavier tail


            # CPU/Memory Usage: Dependent on activity type and session duration, often higher for anomalies
            base_cpu = 60
            base_mem = 8000
            cpu_usage_percent = max(1, min(100, np.random.normal(loc=base_cpu + session_duration/200, scale=15)))
            memory_usage_MB = max(1000, np.random.normal(loc=base_mem + session_duration*10, scale=2000))


            # Nuanced Anomalous Patterns: Unusual combinations
            # Example: High data transfer with very low session duration, or file access from unusual location
            if np.random.rand() < 0.3: # 30% chance of injecting this pattern
                 if data_transfer_MB > 1000 and session_duration < 300:
                      session_duration = max(10, int(session_duration * np.random.uniform(0.2, 0.5))) # Make session duration even shorter
                 if activity_type == 'file_access' and ip_location == user_location and np.random.rand() < 0.5:
                      ip_location = random.choice([loc for loc in self.config.locations if loc != user_location]) # Force unusual location


            # Impact Score and Cost: Dependent on Severity and Category, often higher for anomalies
            base_impact = 7
            base_cost = 10000
            severity_impact_multiplier = {"Low": 1.5, "Medium": 2.0, "High": 3.0, "Critical": 5.0}.get(severity, 2.0)
            category_impact_multiplier = {"Data Breach": 4.0, "Malware": 3.0, "Unauthorized Access": 3.5}.get(category, 1.5) # Example category impact
            impact_score = max(3, min(10, int(np.random.normal(loc=base_impact * severity_impact_multiplier * category_impact_multiplier * user_profile['risk_tolerance'] * 1.2, scale=4))))
            cost = max(500, np.random.normal(loc=base_cost * severity_impact_multiplier * category_impact_multiplier * department_profile['baseline_risk'] * 1.5, scale=5000))


            # Risk Level Calculation (was missing)
            risk_level = 'Critical' if impact_score > 8 else 'High' if impact_score > 5 else 'Medium' if impact_score > 3 else 'Low'


            # KPI/KRI Calculation (was missing)
            kpi_kri = self.filter_kpi_and_kri(category)


            remediation_steps = f"Steps to resolve {issue_name}" # Remediation Steps (was missing)

            threat_level, threat_score = self.calculate_threat_level(
                severity, impact_score, risk_level, issue_response_time_days,
                login_attempts, num_files_accessed, data_transfer_MB,
                cpu_usage_percent, memory_usage_MB
            )

            row_data = {
                "Severity": severity, "Impact Score": impact_score, "Risk Level": risk_level,
                "Issue Response Time Days": issue_response_time_days, "Login Attempts": login_attempts,
                "Num Files Accessed": num_files_accessed, "Data Transfer MB": data_transfer_MB,
                "CPU Usage %": cpu_usage_percent, "Memory Usage MB": memory_usage_MB,
                "Threat Level": threat_level, "Activity Type": activity_type
            }
            defense_action = self.adaptive_defense_mechanism(row_data)

            anomalous_issues_data.append([
                issue_id, issue_key, issue_name, issue_volume, category, severity, status, reporter, assignee,
                date_reported, date_resolved, issue_response_time_days, impact_score, risk_level, department_affected,
                remediation_steps, cost, kpi_kri, user_id, timestamp, activity_type, user_location, ip_location,
                session_duration, num_files_accessed, login_attempts, data_transfer_MB,
                cpu_usage_percent, memory_usage_MB, threat_score, threat_level, defense_action
            ])

        df = pd.DataFrame(anomalous_issues_data, columns=self.config.columns)
        return df


    def data_generation_pipeline(self):
        normal_df = self.generate_normal_issues_df(self.config.issue_ids, self.config.issue_keys)
        normal_df["Is Anomaly"] = 0
        anomaly_df = self.generate_anomalous_issues_df(self.anomalous_issue_ids, self.anomalous_issue_keys)
        anomaly_df["Is Anomaly"] = 1
        combined_df = pd.concat([normal_df, anomaly_df], ignore_index=True)
        return normal_df, anomaly_df, combined_df


# =====================================================================
# Processor
# =====================================================================
class DataProcessor:
    def map_threat_severity_to_color(self, df):
        def assign_color(threat, severity):
            if threat == "Critical":
                if severity == "Critical": return "Dark Red"
                elif severity == "High": return "Red"
                elif severity == "Medium": return "Orange-Red"
                else: return "Orange"
            elif threat == "High":
                if severity == "Critical": return "Red"
                elif severity == "High": return "Orange-Red"
                elif severity == "Medium": return "Orange"
                else: return "Yellow-Orange"
            elif threat == "Medium":
                if severity == "Critical": return "Orange"
                elif severity == "High": return "Yellow-Orange"
                elif severity == "Medium": return "Yellow"
                else: return "Light Yellow"
            else:
                if severity == "Critical": return "Yellow"
                elif severity == "High": return "Light Yellow"
                elif severity == "Medium": return "Green-Yellow"
                else: return "Green"
        df["Color"] = df.apply(lambda row: assign_color(row["Threat Level"], row["Severity"]), axis=1)
        return df


# =====================================================================
# Data Display
# =====================================================================
class DataDisplay:
    """Handles displaying dataframes."""
    def display_the_data_frames(self, p_normal_issues_df, p_anomalous_issues_df, p_normal_and_anomalous_df,
                                p_ktis_key_threat_indicators_df, p_scenarios_with_colors_df):
        """Displays info, description, and head for multiple DataFrames."""

        print('Normal_issues_df Data structure\n')
        display(p_normal_issues_df.info())
        print('\nData statistics summary\n')
        display(p_normal_issues_df.describe().transpose())
        print('\nNormal_issues_df\n')
        display(p_normal_issues_df.head())

        print('\nAnomalous_issues_df Data structure\n')
        display(p_anomalous_issues_df.info())
        print('\nAnomalous_issues_df statistics summary\n')
        display(p_anomalous_issues_df.describe().transpose())
        print('\nAnomalous_issues_df\n')
        display(p_anomalous_issues_df.head())

        print('\nNormal & anomalous combined Data structure\n')
        display(p_normal_and_anomalous_df.info())
        print('\nCombined statistics summary\n')
        display(p_normal_and_anomalous_df.describe().transpose())
        print('\nNormal & anomalous combined Data\n')
        display(p_normal_and_anomalous_df.head())

        print('\nKey Threat Indicators\n')
        display(p_ktis_key_threat_indicators_df)

        print('\nScenarios with Colors\n')
        display(p_scenarios_with_colors_df)
        print('\n')

        #pcharts plotting
        explaratory_data_analysis_pipeline(p_normal_and_anomalous_df)

# =====================================================================
# Saver
# =====================================================================

class DataSaver:
    def save_dataframe(self, df, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✅ Saved to {save_path}")

    def print_summary(self, file_paths):
        summary = []
        for path in file_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                size_kb = os.path.getsize(path) / 1024
                summary.append([os.path.basename(path), df.shape[0], df.shape[1], f"{size_kb:.1f} KB"])
        print("\n📊 Dataset Summary")
        print(pd.DataFrame(summary, columns=["File", "Rows", "Columns", "Size"]).to_string(index=False))

    def save_data_option(self, config, no_prompt=False, auto_download=False):
        """Optionally download a ZIP of the datasets."""
        temp_dir = "/tmp/cybersecurity_data"
        zip_path = "/tmp/cybersecurity_data.zip"

        def make_zip():
            os.makedirs(temp_dir, exist_ok=True)
            shutil.copy(config.normal_data_file, temp_dir)
            shutil.copy(config.anomalous_data_file, temp_dir)
            shutil.copy(config.combined_data_file, temp_dir)
            shutil.copy(config.key_threat_indicators_file, temp_dir)
            shutil.copy(config.scenarios_with_colors_file, temp_dir)
            shutil.make_archive("/tmp/cybersecurity_data", 'zip', temp_dir)

        # Auto mode (no user interaction)
        if no_prompt:
            print("Skipping local download (no-prompt mode).")
            return
        if auto_download:
            print("Preparing files for automatic download...")
            make_zip()
            if COLAB:
                files.download(zip_path)
                print("Files downloaded locally as cybersecurity_data.zip")
            else:
                print(f"ZIP created at {zip_path} (please download manually).")
            return

        # Interactive prompt
        while True:
            choice = input("Would you like to download the data files locally as well? (yes/no): ").strip().lower()
            if choice == 'yes':
                print("Preparing files for download...")
                make_zip()
                if COLAB:
                    files.download(zip_path)
                    print("Files downloaded locally as cybersecurity_data.zip")
                else:
                    print(f"ZIP created at {zip_path} (please download manually).")
                break
            elif choice == 'no':
                print("Files saved to repository only. No local download.")
                break
            else:
                print("Invalid choice. Please enter 'yes' or 'no'.")


# =====================================================================
# Main pipeline
# =====================================================================
def cybersecurity_data_pipeline(show_data=True, no_prompt=False, auto_download=False):
    config = DataConfig()
    generator = DataGenerator(config)
    processor = DataProcessor()
    saver = DataSaver()
    display_handler = DataDisplay()

    normal_df, anomaly_df, combined_df = generator.data_generation_pipeline()
    combined_df = processor.map_threat_severity_to_color(combined_df)

    # Display if requested
    if show_data:
        display_handler.display_the_data_frames(
            normal_df, anomaly_df, combined_df,
            config.ktis_key_threat_indicators_df,
            config.scenarios_with_colors_df
        )

    # Save all datasets
    saver.save_dataframe(normal_df, config.normal_data_file)
    saver.save_dataframe(anomaly_df, config.anomalous_data_file)
    saver.save_dataframe(combined_df, config.combined_data_file)
    saver.save_dataframe(config.ktis_key_threat_indicators_df, config.key_threat_indicators_file)
    saver.save_dataframe(config.scenarios_with_colors_df, config.scenarios_with_colors_file)

    # Print summary
    saver.print_summary([
        config.normal_data_file,
        config.anomalous_data_file,
        config.combined_data_file,
        config.key_threat_indicators_file,
        config.scenarios_with_colors_file
    ])

    # Prompt or auto ZIP download
    saver.save_data_option(config, no_prompt=no_prompt, auto_download=auto_download)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cybersecurity Data Generator")
    parser.add_argument("--no-prompt", action="store_true", help="Skip download prompt")
    parser.add_argument("--auto-download", action="store_true", help="Auto-download ZIP without prompt")
    parser.add_argument("--no-display", action="store_true", help="Skip DataFrame display")

    args = parser.parse_args()

    cybersecurity_data_pipeline(
        show_data=not args.no_display,
        no_prompt=args.no_prompt,
        auto_download=args.auto_download
    )



```

</details>

## Overview

This project provides a Python script designed to generate synthetic cybersecurity issue data. The goal is to create a realistic dataset that simulates both normal and anomalous cybersecurity events, which can be used for various purposes such as:

*   **Training anomaly detection models:** The generated data includes labeled normal and anomalous samples.
*   **Testing security information and event management (SIEM) systems:** The dataset mimics real-world security logs and incidents.
*   **Developing and validating security analytics:** The diverse set of features allows for exploring various security metrics and indicators.
*   **Demonstrating cybersecurity concepts:** The dataset can be used for educational purposes to illustrate different types of attacks and their characteristics.

The script generates data with a variety of attributes, including issue details, user activity, system metrics, and threat indicators. It also incorporates a simple adaptive defense mechanism based on the calculated threat level and severity.  



**Core Data Schema**:
   Each column will be structured to simulate real-world attributes.  

  
   - `Issue ID`, `Issue Key`: Unique identifiers.
   - `Issue Name`, `Category`, `Severity`: Descriptive issue metadata with categorical values.
   - `Status`, `Reporters`, `Assignees`: Status categories and personnel involved.
   - `Date Reported`, `Date Resolved`: Randomized dates across a timeline.
   - `Impact Score`, `Risk Level`: Randomized scores to reflect varying severity.
   - `Cost`: Randomized to reflect the volatility in month-over-month impact.  

**User Activity Columns**:
   Columns like `user_id`, `timestamp`, `activity_type`, `location`, `session_duration`, and `data_transfer_MB` will be generated to simulate behavioral patterns.

**Monthly Volatility**:  
  - **Impact Score**, **Cost**, and **data_transfer_MB** We use synthetic techniques to create spikes or drops in activity between months, simulating the volatility in issues or user activity.
  - For example, we use random walks to vary values in a non-linear fashion to capture realistic volatility.  

**Data Augmentation**:

* **Scaling Data Points**: During the early stages of feature engineering, SMOTE and controlled random sampling are applied particularly to categorical features to increase data diversity and address class imbalance.
* **Label Perturbation for `Assignees` and `Departments`**: Categorical labels are periodically reassigned to simulate organizational role changes and dynamic team structures commonly observed in real-world environments.
* **Time-Series Variability Injection**: Synthetic timestamps are generated both within and across sessions to model realistic behavioral dynamics, including bursts in login attempts, spikes in data transfer activity, and variations in session duration.


**User activity features:**  

  - user_id: Identifier for each user.  
  - timestamp: Time of the activity.  
  - activity_type: Type of activity (e.g., "login," "file_access," "data_modification").  
  - location: User's location (e.g., IP region).  
  - session_duration: Length of session in seconds.  
  - num_files_accessed: Number of files accessed in a session.  
  - ogin_attempts: Number of login attempts in a session.  
  - data_transfer_MB: Amount of data transferred (MB).  

**Anomalies:**  

  - We include some rows with anomalous patterns like high login attempts, unusual session duration and high data transfer volumes from unexpected locations

**Explanation of Key Parts:**
- **Volatile Data Generation**: The `generate_volatile_data` function adds random fluctuations to values, simulating high month-over-month volatility.
- **User Activity Features**: Columns like `activity_type`, `session_duration`, `num_files_accessed`, `login_attempts`, and `data_transfer_MB` are varied to reflect real user behaviors.
- **Random Timestamps**: Activity timestamps are spread across the timeline from `start_date` to `end_date`.

- **Generate normal issues dataset**: First, we a normal issue dataset with almost no data anomaly
- **Generate anomalous issues dataset**: The we introduce anomaly to the detaset
- **Combine normal and anomalous data**: We combine both normal and anomalous datasets
- * **Addressing Class Imbalance**: During the feature analysis stage, the **Synthetic Minority Over-sampling Technique (SMOTE)** is applied to mitigate class imbalance and ensure that minority threat classes are adequately represented in the dataset.  

  
**User Activities Generation Metrics Formula**  

The expression:  base_value + base_value * volatility * (np.random.randn()) * (1.2 if severity in ['High', 'Critical'] else 1)

means that we’re generating a value based on a starting point (base_value) and adjusting it for both randomness and severity level. Here's a breakdown:

- **base_value**: This is the initial value that the output is based on.
- **volatility * (np.random.randn())**: This part adds a random fluctuation around the base_value. `np.random.randn()` generates a value from a standard normal distribution (centered around 0), so it could be positive or negative, creating variation. Multiplying by volatility scales the randomness, making the fluctuation stronger or weaker.
- **(1.2 if severity in ['High', 'Critical'] else 1)**: This adds an additional factor to increase the outcome by 20% if the severity is "High" or "Critical." If severity isn’t in these categories, the factor is simply 1, meaning no extra adjustment.

So, if severity is "High" or "Critical," the result is a base value adjusted for both volatility and severity; otherwise, it’s just the base value with volatility adjustment.  


**Treat level Identification and Adaptive Defense Systems Setting**

Threat levels are derived from the generated cybersecurity dataset using a composite threat scoring model. This model integrates multiple relevant features to quantitatively assess risk and dynamically support adaptive defense and response mechanisms.

**Key Threat Indicators (KTIs) Definition**  

The following columns are used as key threat indicators (KTIs):  

   - **Severity**: Indicates the criticality of the issue.
   - **Impact Score**: Represents the potential damage if the threat is realized.
   - **Risk Level**: A general indicator of risk associated with each issue.
   - **Issue Response Time Days**: The longer it takes to respond, the higher the threat level could be.
   - **Category**: Certain categories (e.g., unauthorized access) carry a higher base threat level.
   - **Activity Type**: Suspicious activity types (e.g., high login attempts, data modification) indicate a greater threat.
   - **Login Attempts**: Unusually high login attempts signal a brute force attack.
   - **Num Files Accessed** and **Data Transfer MB**: Large data transfers or access to many files in a session could indicate data exfiltration or suspicious activity.

**KTIs based Scoring**
  
For each KTI we define the acriteria to be used to assigne a score  

| KTI               | Condition                                      | Score   |
|-------------------|------------------------------------------------|---------|
| Severity          | Critical = 10, High = 8, Medium = 5, Low = 2   | 2 - 10  |
| Impact Score      | 1 to 10 (already a score)                      | 1 - 10  |
| Risk Level        | High = 8, Medium = 5, Low = 2                  | 2 - 8   |
| Response Time     | >7 days = 5, 3-7 days = 3, <3 days = 1         | 1 - 5   |
| Category          | Unauthorized Access = 8, Phishing = 6, etc.    | 1 - 8   |
| Activity Type     | High-risk types (e.g., login, data_transfer)   | 1 - 5   |
| Login Attempts    | >5 = 5, 3-5 = 3, <3 = 1                        | 1 - 5   |
| Num Files Accessed| >10 = 5, 5-10 = 3, <5 = 1                      | 1 - 5   |
| Data Transfer MB  | >100 MB = 5, 50-100 MB = 3, <50 MB = 1         | 1 - 5   |  

  


**Threat Score Calculation**
The threat level is calculated as a weighted sum of these scores. For example:

*Threat Score = 0.3 × Severity + 0.2 × Impact Score + 0.2 × Risk Level + 0.1 × Response Time + 0.1 × Login Attempts + 0.05 × Num Files Accessed + 0.05 × Data Transfer MB*

Note: The weights could be adjusted based on the importance of each factor in your specific cybersecurity context.

**Threat Level Thresholds Definition**  

We use the final threat score to categorize the threat level:
   - **Low Threat**: 0–3
   - **Medium Threat**: 4–6
   - **High Threat**: 7–9
   - **Critical Threat**: 10+

**Real-Time Calculation and Monitoring Implementation**
To implement this dynamically we :
   - Calculate and log the threat score whenever new data is added.
   - Set up alerts for high and critical threat scores.
   - Integrate this scoring model into a real-time dashboard or cybersecurity scorecard.

This method provides a structured and quantifiable approach to assessing the threat level based on multiple relevant indicators from the initial dataset.  

**Rule-based Adaptive Defense Mechanism**  

The system incorporates a rule-based defense layer that evaluates threat conditions against defined control thresholds and response policies. High-risk scenarios trigger automated flags, enhanced audit logging, and prescribed mitigation actions, ensuring consistent and traceable responses. This approach supports governance objectives by promoting transparency, repeatability, and alignment with risk management and compliance requirements.


**Rules Definition**
The following features are used to define rule-based conditions for identifying potential threats and recommending appropriate defensive actions: `Threat Level`, `Severity`, `Impact Score`, `Login Attempts`, `Risk Level`, `Issue Response Time (Days)`, `Num Files Accessed`, and `Data Transfer (MB)`.

**Adaptive Defense Mechanism**
The system responds dynamically by applying flags and assigning customized defensive actions based on rule evaluations and scenario classifications. Each issue is assigned an adaptive `Defense Action` aligned with its assessed threat conditions, providing an additional layer of automated response across varying threat levels and behavioral patterns.

To enhance interpretability and prioritization, threat conditions are visualized using color-coded cybersecurity scenarios. This approach enables rapid risk assessment and effective communication by mapping threat severity and urgency to intuitive color intensities, **red, orange, yellow, and green**, with increasing severity represented by deeper color intensity.  

**Color Scheme**
- **Critical Threat & Severity**: **Dark Red** – Highest urgency.
- **High Threat or Severity**: **Orange** – Serious, but not the highest urgency.
- **Medium Threat or Severity**: **Yellow** – Moderate concern.
- **Low Threat & Severity**: **Green** – Low concern, monitor as needed.

Here is the table **with colored emoji icons added** beside each color name in **Suggested Color**.
These icons render correctly in GitHub, Markdown, Slack, Teams, Notion, and Streamlit.  






## **Scenarios with Colors**

| **Scenario** | **Threat Level** | **Severity** | **Suggested Color**  | **Rationale**                                                                         |
| ------------ | ---------------- | ------------ | -------------------- | ------------------------------------------------------------------------------------- |
| **1**        | Critical         | Critical     | 🔴 **Dark Red**      | Maximum urgency, both threat and impact are critical. Immediate action required.      |
| **2**        | Critical         | High         | 🟥 **Red**           | Very high risk, threat is critical and impact is significant. Prioritize response.    |
| **3**        | Critical         | Medium       | 🟧 **Orange-Red**    | Significant threat but moderate impact. Act promptly to prevent escalation.           |
| **4**        | Critical         | Low          | 🟧 **Orange**        | High potential risk, current impact is minimal. Monitor closely and mitigate quickly. |
| **5**        | High             | Critical     | 🟥 **Red**           | High threat combined with critical impact. Needs immediate action.                    |
| **6**        | High             | High         | 🟧 **Orange-Red**    | High threat and significant impact. Prioritize response.                              |
| **7**        | High             | Medium       | 🟧 **Orange**        | Elevated threat and moderate impact. Requires attention.                              |
| **8**        | High             | Low          | 🟨 **Yellow-Orange** | High threat with low impact. Proactive monitoring recommended.                        |
| **9**        | Medium           | Critical     | 🟧 **Orange**        | Moderate threat with critical impact. Prioritize addressing the severity.             |
| **10**       | Medium           | High         | 🟨 **Yellow-Orange** | Medium threat with high impact. Needs resolution soon.                                |
| **11**       | Medium           | Medium       | 🟨 **Yellow**        | Medium threat and impact. Plan to address it.                                         |
| **12**       | Medium           | Low          | 🟨 **Light Yellow**  | Moderate threat, minimal impact. Monitor as needed.                                   |
| **13**       | Low              | Critical     | 🟨 **Yellow**        | Low threat but high impact. Address severity first.                                   |
| **14**       | Low              | High         | 🟨 **Light Yellow**  | Low threat with significant impact. Plan mitigation.                                  |
| **15**       | Low              | Medium       | 💛 **Green-Yellow**  | Low threat, moderate impact. Routine monitoring.                                      |
| **16**       | Low              | Low          | 🟩 **Green**         | Minimal risk. No immediate action required.                                           |  





This color based scenarios approach aligns urgency with the dual factors of **threat level** and **severity**, ensuring quick comprehension and appropriate prioritization.  



#### **2. Explanatory Data Analysis(EDA)**

The following steps were implemented in the exploratory data analysis (EDA) pipeline to analyze the dataset's key features and distribution patterns:

**Data Normalization:**
   - Implemented a function to normalize numerical features using Min-Max Scaling for consistent feature scaling.

**Time-Series Visualization:**
   - Plotted daily distribution of numerical features pre- and post-normalization using line plots for visualizing trends over time.

**Statistical Feature Analysis:**
   - Developed histograms and boxplots for all features, including overlays of statistical metrics (mean, standard deviation, skewness, kurtosis) for numerical features.
   - Integrated risk levels with customized color palettes for categorical data.

**Scatter Plot and Correlation Analysis:**
   - Created scatter plots to analyze relationships between key features such as session duration, login attempts, data transfer, and user location.
   - Generated a correlation heatmap to visualize interdependencies among numerical features.

**Distribution Analysis Pipeline:**
   - Built a modular pipeline to evaluate and compare the distribution of activity features across daily and aggregated reporting frequencies (e.g., monthly, quarterly).

**Comprehensive Feature Analysis:**
   - Combined scatter plots, heatmaps, and distribution visualizations into a unified framework for insights into user behavior and feature relationships.

**Dynamic Layouts and Annotations:**
   - Optimized subplot layouts to handle a variable number of features and annotated plots with key statistics for enhanced interpretability.

This pipeline provides a detailed understanding of numerical and categorical feature behaviors while highlighting correlations and potential anomalies in the dataset.

## Script Structure and Functionality

The script is organized into several classes, each responsible for a specific part of the data generation and processing pipeline:

*   **`DataConfig`**: This class holds all the configuration parameters and constants used throughout the script. This includes the number of normal and anomalous issues, user and department counts, date ranges, file paths for saving data to Google Drive, lists of possible categories, severities, statuses, reporters, assignees, users, departments, locations, and column names. It also stores pre-defined DataFrames for Key Threat Indicators (KTIs) and Scenarios with Colors.

*   **`DataGenerator`**: This class is responsible for generating the synthetic data for both normal and anomalous issues.
    *   It includes methods to map issue categories to synthetic normal and anomalous issue names (`generate_normal_issues_name`, `generate_anomalous_issue_name`).
    *   It filters categories into KPIs and KRIs (`filter_kpi_and_kri`).
    *   It generates random dates within a specified range (`random_date`).
    *   A key function is `calculate_threat_level`, which computes a threat score and assigns a threat level based on several input features.
    *   The `adaptive_defense_mechanism` function determines suggested defense actions based on threat level, severity, and activity context.
    *   The core data generation logic resides in `generate_normal_issues_df` and `generate_anomalous_issues_df`, which create pandas DataFrames for each type of issue. These functions include enhanced logic to simulate more realistic data distributions and dependencies between features, as well as temporal patterns and nuanced anomalous behaviors.
    *   The `data_generation_pipeline` orchestrates the generation of both normal and anomalous data and combines them into a single DataFrame, adding an "Is Anomaly" label.

*   **`DataProcessor`**: This class handles data processing tasks.
    *   The `map_threat_severity_to_color` method adds a "Color" column to the DataFrame based on the threat level and severity, providing a visual indicator of the issue's risk.

*   **`DataSaver`**: This class is responsible for saving the generated DataFrames.
    *   The `save_dataframe_to_google_drive` method saves a single DataFrame to a specified path in Google Drive as a CSV file. It also includes error handling to ensure the directory exists.
    *   The `save_the_data_to_CSV_to_google_drive` method calls the single-save method for all the generated DataFrames.

*   **`DataDisplay`**: This class provides functionality to display information about the generated DataFrames.
    *   The `display_the_data_frames` method uses `display()` to show the head, info, and describe outputs for each generated DataFrame, allowing for a quick overview of the data structure and statistics.  
    *   Display Plottings  
        1. Descriptive Statistics
           Computes summary statistics (.describe()).
           Detects categorical vs numerical features.
           Highlights distributions, outliers, and correlations.  
        📌 The core of classical EDA, enabling users to spot anomalies and trends.

        2. Visualization
            Plots histograms, correlation heatmaps, and categorical distributions.
            Supports matplotlib and seaborn for visual clarity.  
        📌 Visual EDA complements numerical summaries by revealing hidden structures.
       
## How to Use the Script

1.  **Run the Code Cell:** Execute the Python code cell containing the script.
2.  **Data Generation and Saving:** The script will automatically:
    *   Clone the projet repository from GitHub to Colab environment /content/
    *  Run the exec file cyberdatagen.py
    *   Initialize the configuration.
    *   Generate the normal and anomalous datasets.
    *   Combine the datasets.
    *   Apply the color mapping based on threat level and severity.
    *   Display the head, info, and describe outputs for the generated dataframes in the Colab output.  
    *   Prompt the user to download the generated dataframes (normal, anomalous, combined, KTIs, and scenarios with colors) as CSV files   

3.  **Access the Data:** The generated CSV files will be available locally in the user download folder   

## Customization

We can customize the data generation process by modifying the parameters in the `DataConfig` class. This includes:

*   Changing the number of normal and anomalous issues.
*   Adjusting the number of unique users, reporters, and assignees.
*   Modifying the date ranges for data generation.
*   Updating the lists of categories, severities, statuses, etc.
*   Adjusting the parameters within the `DataGenerator` class methods to fine-tune the distributions and relationships between features for both normal and anomalous data.

## Future Enhancements

Potential future enhancements for this project include:

*   Implementing more sophisticated anomaly generation techniques to simulate a wider variety of attack types and patterns.
*   Adding more complex dependencies and interactions between features to increase the realism of the dataset.
*   Incorporating network traffic data, log file entries, and other types of security-related information.
*   Developing a user interface or command-line tool for easier configuration and execution of the data generator.
*   Exploring different data formats for saving the generated data (e.g., Parquet, JSON).

```python
!git clone https://github.com/atsuvovor/Cybersecurity-Data-Generator.git
%run /content/Cybersecurity-Data-Generator/cyberdatagen.py
```


## Next Step: Feature Engineering
<a 
  href="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/feature_engineering/README.md"
  target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---

## 🤝 Connect With Me
I am always open to collaboration and discussion about new projects or technical roles.

Atsu Vovor  
Consultant, Data & Analytics    
Ph: 416-795-8246 | ✉️ atsu.vovor@bell.net    
🔗 <a href="https://www.linkedin.com/in/atsu-vovor-mmai-9188326/" target="_blank">LinkedIn</a> | <a href="https://atsuvovor.github.io/projects_portfolio.github.io/" target="_blank">GitHub</a> | <a href="https://public.tableau.com/app/profile/atsu.vovor8645/vizzes" target="_blank">Tableau Portfolio</a>    
📍 Mississauga ON      

### Thank you for visiting!🙏
