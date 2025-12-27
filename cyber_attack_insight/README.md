
<div align="center">
<h2>  CyberThreat-Insight - Attack Simulation & Stacked Anomaly Detection Platform</h2>
 <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/cyber_attack_simulation_engine2.png" 
       alt="Cyber Attack symulation Engine" 
       style="width: 600px; height: 40%;">

**Anomalous Behavior Detection in Cybersecurity Analytics using Generative AI & Stacked Generalization**

</div>

  

**Author:** Atsu Vovor  
**Title:** Data & Analytics Consultant | Cybersecurity | AI Reporting  
**Location:** Toronto, Canada  



## Overview

**CyberThreat-Insight** is an **enterprise-grade cyber attack simulation, anomaly detection, and AI-assisted reporting platform** designed to support:

* Cyber risk scenario modeling
* Threat detection validation
* Executive and board-level reporting
* Model Risk Management (MRM)
* Regulatory and audit review readiness

The platform augments **real operational cybersecurity data** with **statistically governed attack simulations**, applies a **stacked ensemble anomaly detection model**, and produces **audit-ready outputs** for dashboards, reports, and AI-generated executive summaries.

This solution is intentionally designed to align with how **financial institutions, insurers, and regulated enterprises** evaluate analytics platforms during **Model Risk Committee, Internal Audit, and regulator reviews**.




## Platform Capabilities at a Glance & High-Level Architecture  

<div align="center">
<img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/high_level_cyber_threat_architecture_file_path.png" 
       alt="Cyber Threat Detection Engine" 
       style="width: 40%; height: Aoto;">

</div>

* Multi-vector cyber attack simulation
* MITRE ATT&CK–aligned threat modeling
* Production-grade stacked anomaly detection
* ML-safe data engineering & inference
* Full data lineage & governance controls
* AI-assisted executive reporting
* Audit- and regulator-ready documentation

### Outputs

* Simulated datasets with attack annotations
* Anomaly scores & threat levels
* Dashboard-ready tables
* AI-generated executive summaries


##   Attack Simulation Model – Class & Function Reference

<div align="center">
<img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/core multiple attack simulation engine.png" 
       alt="Cyber Threat Detection Engine" 
       style="width: 50%; height: Aoto;">

</div>  



This module implements a **controlled cyber-attack simulation framework** designed to augment operational cybersecurity datasets with realistic attack behaviors prior to **ML anomaly detection and risk scoring**.

It supports **MITRE ATT&CK–aligned attack types**, ML-safe data handling, and production-grade inference pipelines.



### Base Classes

#### `BaseAttack`

**Purpose:**
Abstract parent class for all simulated attack types.
Provides **shared utilities** for numeric casting, safe noise generation, and metric bounding to ensure **data realism and ML safety**.

**Key Responsibilities:**

* Enforces numeric type consistency
* Applies metric clipping based on domain limits
* Provides reusable noise generators
* Prevents unrealistic or invalid values during simulation

**Core Attributes:**

* `NUMERIC_COLS`: List of metrics eligible for numeric manipulation
* `LIMITS`: Upper and lower bounds for critical system and security metrics

**Key Methods:**

* `_cast_numeric()` – Coerces numeric columns to `float64`
* `_bounded_lognormal()` – Generates multiplicative noise without runaway values
* `_clip_metrics()` – Enforces operational bounds on metrics
* `apply()` – Abstract method implemented by all attack subclasses



### Attack Simulation Classes

Each attack class simulates **statistically distinct behavior patterns** consistent with real-world cyber threats.



#### `PhishingAttack`

**Threat Modeled:** Credential abuse / Initial access
**MITRE ATT&CK:** T1566

**Behavior Simulated:**

* Elevated login attempts
* Moderate increases in impact and threat scores
* Targets access-control related records

**Key Metrics Affected:**

* Login Attempts
* Impact Score
* Threat Score



#### `MalwareAttack`

**Threat Modeled:** Malicious execution & enumeration
**MITRE ATT&CK:** T1204

**Behavior Simulated:**

* Increased file access activity
* Elevated impact and threat scores
* Targets system vulnerability categories

**Key Metrics Affected:**

* Num Files Accessed
* Impact Score
* Threat Score



#### `DDoSAttack`

**Threat Modeled:** Resource exhaustion
**MITRE ATT&CK:** T1499

**Behavior Simulated:**

* Abnormally long sessions
* Elevated login activity
* Exponential growth in impact and threat

**Key Metrics Affected:**

* Session Duration
* Login Attempts
* Impact Score
* Threat Score



#### `DataLeakAttack`

**Threat Modeled:** Data exfiltration
**MITRE ATT&CK:** T1041

**Behavior Simulated:**

* Large outbound data transfers
* High impact and threat escalation
* Log-normal transfer amplification

**Key Metrics Affected:**

* Data Transfer MB
* Impact Score
* Threat Score



### `InsiderThreatAttack`

**Threat Modeled:** Privileged misuse outside business hours
**MITRE ATT&CK:** T1078

**Behavior Simulated:**

* Suspicious after-hours activity
* Restricted file access
* Elevated data transfer volumes

**Key Metrics Affected:**

* Data Transfer MB
* Impact Score
* Threat Score
* Access Restricted Files (flag)



#### `RansomwareAttack`

**Threat Modeled:** Mass encryption & system impact
**MITRE ATT&CK:** T1486

**Behavior Simulated:**

* Memory and CPU spikes
* Excessive file access
* Severe threat and impact escalation

**Key Metrics Affected:**

* Memory Usage
* CPU Usage
* Num Files Accessed
* Impact Score
* Threat Score



###  Utility & Support Classes

### `IPAddressGenerator`

**Purpose:**
Generates realistic IPv4 source/destination pairs for simulated network activity.

**Methods:**

* `generate_random_ip()` – Produces a random IPv4 address
* `generate_ip_pair()` – Produces a source–destination IP pair



###  Data Preparation & ML Safety

#### `sanitize_for_ml(df)`

**Purpose:**
Ensures simulated data is **safe, clean, and schema-consistent** before ML inference.

**Controls Applied:**

* Replaces infinite values
* Imputes missing values using medians
* Casts numeric features to `float32`
* Preserves non-numeric metadata

**Governance Role:**
Acts as an **ML safety gate** prior to model scoring.



###  Orchestration Functions

#### `run_selected_attacks(df, selected_attacks, verbose=True)`

**Purpose:**
Sequentially applies selected attack simulations to an operational dataset.

**Features:**

* Validates DataFrame integrity
* Ensures each attack returns a valid dataset
* Provides verbose execution logging



### `main_attacks_simulation_pipeline(URL=None)`

**Purpose:**
End-to-end **production attack simulation and ML inference pipeline**.

**Pipeline Stages:**

1. Load operational dataset (CSV / Google Drive)
2. Apply selected cyber-attack simulations
3. Sanitize data for ML inference
4. Validate required schema fields
5. Persist augmented dataset
6. Run stacked anomaly detection model
7. Save predictions and risk scores

**Outputs:**

* Augmented datasets with simulated attacks
* ML-generated anomaly predictions
* Persisted CSV outputs for dashboards and reporting



### Governance & Model Risk Notes

* All simulations are **bounded and controlled**
* Statistical assumptions are transparent and reproducible
* ML inference occurs only after explicit sanitation and validation
* Designed to support **SR 11-7 / OSFI model governance expectations**


## Mathematical Foundations of Attack Simulation

Each attack is governed by **explicit probabilistic models** to ensure realism, explainability, and repeatability.  
  
> **Audience:** Executives, Audit, Risk Committees, non-technical stakeholders  

> **Purpose:**
> This appendix provides a formal mathematical description of the cyber-attack simulation logic, suitable for **academic review, regulator submission, or independent model validation**.
> No narrative interpretation is included in this section.  

### Phishing — Credential Abuse

$$
X_{\text{login}} \sim \text{Poisson}(\lambda)
$$

$$
\text{Impact} \sim \mathcal{N}(5, 3^2)
$$

$$
\text{Threat} \sim \mathcal{N}(6, 3^2)
$$

*Login attempts follow a Poisson distribution* because phishing attacks generate **many small, repeated login attempts**.
Impact and threat scores use a **normal distribution** to reflect moderate but consistent operational ris  


### Malware — System Enumeration

$$
X_{\text{files}} \sim \text{Poisson}(\lambda)
$$

$$
\text{Impact},; \text{Threat} \sim \mathcal{N}(7, 4^2)
$$

Malware tends to **scan files repeatedly**, which is well modeled by a Poisson process.
Severity scores are centered higher than phishing, reflecting **greater system compromise risk**.  


### DDoS — Resource Saturation

$$
X_{\text{session}} \sim \text{Exponential}(\beta)
$$

$$
\text{Impact},; \text{Threat} \sim \text{Exponential}(8)
$$

Session durations follow an **exponential distribution**, capturing the fact that most attacks are short, but a few last a very long time.
Severity escalates rapidly as resources are exhausted.   


### Data Leak — Exfiltration

$$
X = \mu \cdot e^{\sigma Z}, \quad Z \sim \mathcal{N}(0,1)
$$

$$
\text{Impact},; \text{Threat} \sim \mathcal{N}(12, 5^2)
$$


Data exfiltration follows a **lognormal distribution**, reflecting that:  

* Most leaks are small  
* A few rare events cause massive losses  

This aligns with real-world breach patterns.  

### Insider Threat — Time-Based Abuse  

$$
\text{hour} < 6 \quad \text{or} \quad \text{hour} > 23
$$

$$
X_{\text{transfer}} \sim \text{LogNormal}(\sigma = 0.3)
$$

Insider activity is flagged **outside normal business hours**.  
Data transfers follow a lognormal pattern, modeling **stealthy but potentially severe misuse**.  

### Ransomware — Encryption Storms

$$
X_{\text{CPU}} \sim \mathcal{N}(20, 10^2)
$$

$$
X_{\text{memory}} \sim \text{LogNormal}(\sigma = 0.5)
$$

$$
\text{Impact},; \text{Threat} \sim \mathcal{N}(15, 5^2)
$$


CPU and memory usage spike sharply during encryption.
Severity scores are the highest, reflecting **business-critical impact and recovery cost**.  

## Notes for Reviewers & Regulators

* All distributions are **explicitly defined** for transparency and reproducibility
* Parameters are **bounded in code** to enforce operational realism
* Scores represent **system risk indicators**, not attacker intent or attribution



## ML-Safe Data Engineering

### Sanitization Controls

* NaN / ±Inf handling
* Median imputation
* Numeric casting → `float32`
* Metadata preservation
* Schema enforcement

This prevents **inference-time failures** and supports **model reproducibility**.


## Stacked Anomaly Detection Model

The anomaly detection engine is a **stacked ensemble** consisting of:

* Isolation Forest
* One-Class SVM
* Local Outlier Factor (LOF)
* DBSCAN
* KMeans (distance-based features)
* Dense Autoencoder
* LSTM Autoencoder

A **Gradient Boosting meta-model** produces the final anomaly classification and threat score.



## Project Structure

```
CyberThreat_Insight/
│
├── cyber_attack_insight/
│   └── attack_simulation.py
│
├── production/
│   └── stacked_ad_classifier_prod.py
│
├── stacked_models_deployment/
│   ├── scaler.joblib
│   ├── gb_meta.joblib
│   ├── kmeans.joblib
│   ├── dense_autoencoder.keras
│   └── lstm_autoencoder.keras
│
├── cybersecurity_data/
│   ├── operational_data.csv
│   └── simulated_with_predictions_YYYYMMDD.csv
```



## Data Lineage & Model Governance

### End-to-End Lineage

| Stage          | Control                   |
| -------------- | ------------------------- |
| Data ingestion | Schema validation         |
| Simulation     | Bounded stochastic models |
| Feature prep   | Type enforcement          |
| Inference      | Version-locked models     |
| Output         | Timestamped CSV + hash    |
| Reporting      | Prompt versioning         |



### Governance Objectives

* OSFI Model Risk expectations
* SR 11-7 principles
* Explainable AI (XAI) standards



## Risk & Bias Analysis

### Identified Risks

| Risk                  | Mitigation          |
| --------------------- | ------------------- |
| Over-simulation       | Controlled sampling |
| False positives       | Ensemble modeling   |
| Severity inflation    | Hard clipping       |
| Distribution mismatch | Lognormal realism   |

### Bias Considerations

* No PII
* No demographic inference
* Scores represent **system risk**, not user intent

Residual risk is **intentional** to support stress-testing.



## Regulatory & Compliance Alignment

| Framework | Coverage                    |
| --------- | --------------------------- |
| OSFI E-23 | Data governance & lineage   |
| SR 11-7   | Model validation & controls |
| ISO 27001 | Security monitoring         |
| NIST CSF  | Detect / Respond            |


## Regulatory Compliance Mapping

| Pipeline Stage                   | Controls / Governance                                       | Applicable Standard / Framework                   | Notes / Implementation                                                                |
| -------------------------------- | ----------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Data Sources**                 | Source validation, hash checks, access controls             | NIST SP 800-53 AC-1, ISO 27001 A.8.1              | Verify CSV integrity, restrict access to Google Drive files                           |
| **Data Loader**                  | Schema enforcement, missing value handling, type validation | SOC 2 CC6.1, ISO 27001 A.12.5                     | Ensure operational dataset matches expected structure, prevent corrupt data ingestion |
| **Attack Simulation Engine**     | Parameter bounds, random seed, reproducibility              | NIST CSF PR.DS-1, ISO 27001 A.12.6                | Log attack parameters; ensure reproducible simulations for audit                      |
| **MITRE ATT&CK Mapping**         | Mapping audit, logging of attacks                           | MITRE ATT&CK Enterprise, NIST CSF DE.CM-1         | Each simulated attack mapped to MITRE techniques for threat modeling                  |
| **Sanitization / ML Safety**     | NaN / Inf handling, float32 casting, numeric scaling        | ISO 27001 A.12.4, SOC 2 CC7.1                     | Prevent ML errors and maintain safe numerical operations                              |
| **Stacked Anomaly Detection**    | Model versioning, validation, threshold enforcement         | NIST AI RMF, ISO/IEC 42001, EU AI Act (high-risk) | Track base and meta model versions; log model inputs and outputs                      |
| **Predictions & Risk Scores**    | Output verification, scoring consistency                    | SOC 2 CC7.1, NIST CSF DE.CM-7                     | Validate prediction outputs for completeness and numeric stability                    |
| **Dashboards / Reports**         | Access controls, versioning, audit trail                    | ISO 27001 A.9, SOC 2 CC6.1, GDPR Art. 32          | Executive dashboards only accessible to approved users; track exported reports        |
| **Audit / Logging Layer**        | Full pipeline traceability                                  | SOC 2 CC3.1, NIST SP 800-53 AU-2                  | Store timestamps, attack types, model versions, validation logs                       |
| **Model Governance & Oversight** | Committee approvals, documentation, risk assessment         | NIST AI RMF, EU AI Act, ISO/IEC 42001             | Periodic review by Model Risk Committee; governance reports for regulators            |

## MITRE ATT&CK–Mapped Architecture View

Each simulated attack is explicitly mapped to **MITRE ATT&CK tactics and techniques**, enabling:

* Threat coverage validation
* Blue-team alignment
* Regulatory traceability

| Attack Type    | MITRE Technique ID | Technique Name               |
| -------------- | ------------------ | ---------------------------- |
| Phishing       | T1566              | Phishing                     |
| Malware        | T1204              | User Execution               |
| Malware        | T1059              | Command & Scripting          |
| DDoS           | T1499              | Network Denial of Service    |
| Data Leak      | T1041              | Exfiltration Over C2 Channel |
| Insider Threat | T1078              | Valid Accounts               |
| Insider Threat | T1087              | Account Discovery            |
| Ransomware     | T1486              | Data Encrypted for Impact    |
| Ransomware     | T1490              | Inhibit System Recovery      |



## Key Features

### Multi-Attack Simulation Framework

Supported attack vectors:

* Phishing
* Malware
* Distributed Denial-of-Service (DDoS)
* Data Leakage
* Insider Threat
* Ransomware

Each attack:

* Targets realistic subsets of operational data
* Applies statistically bounded perturbations
* Preserves schema integrity and numeric limits


## Attacks  Architecture Diagram with Governance Overlay

<div align="center">
<img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/attacks_architecture_diagram_governance.png" 
       alt="Attacks Architecture Diagram Governance" 
       style="width: 100%; height: Aoto;">

</div>





## Intended Use & Limitations

**Appropriate for:**

* Threat modeling
* AI validation
* Executive risk dashboards

**Not intended for:**

* Attribution of real attackers
* User performance monitoring
* Law-enforcement decisioning



## Design Philosophy

CyberThreat-Insight prioritizes:

* Statistical realism
* Explainability
* Governance readiness
* Executive trust

It bridges **cybersecurity, data science, and AI governance** into a single, auditable framework.


## Model Validation Checklist (MRM / Audit)

> **Purpose:**
> To support **independent model validation, audit review, and regulatory challenge**

---

### Conceptual Soundness

| Check                     | Description                                       | Status |
| ------------------------- | ------------------------------------------------- | ------ |
| Statistical justification | Each attack mapped to an appropriate distribution | ✅      |
| Domain alignment          | Distributions align with real cyber behavior      | ✅      |
| Severity calibration      | Impact & Threat scores bounded and interpretable  | ✅      |



### Implementation Verification

| Check             | Reference                                | Status |
| ----------------- | ---------------------------------------- | ------ |
| Poisson logic     | Phishing, Malware (A.1, A.2)             | ✅      |
| Exponential logic | DDoS (A.3)                               | ✅      |
| Lognormal logic   | Data Leak, Insider, Ransomware (A.4–A.6) | ✅      |
| Numeric bounds    | Clipping applied post-simulation         | ✅      |



### Data Integrity Controls

| Control            | Description                      | Status |
| ------------------ | -------------------------------- | ------ |
| NaN / Inf handling | Sanitized pre-inference          | ✅      |
| Precision control  | Numeric features cast to float32 | ✅      |
| Schema enforcement | Required columns validated       | ✅      |



### Output Reasonableness

| Check               | Description                                    |
| ------------------- | ---------------------------------------------- |
| Distribution review | Simulated values reviewed vs historical ranges |
| Anomaly rate        | Monitored for inflation or collapse            |
| Stress behavior     | Rare events intentionally amplified            |



### Model Limitations  

* Simulations are **synthetic**, not attributional
* Severity scores represent **system risk**, not user intent
* Insider logic is heuristic, not behavioral profiling



##  Final Note for Regulators & Committees  

This framework was designed to:  

✔ Support **stress testing and scenario analysis**  
✔ Enable **transparent validation**  
✔ Align with **financial-services model governance standards**  
✔ Provide **explainable, auditable outputs**  

<img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/cyber_threat_detection_engine4.png" 
       alt="Cyber Threat Detection Engine" 
       style="width: 600px; height: 40%;">


## 👤 Author

**Atsu Vovor**
Data & Analytics Consultant | Cybersecurity | AI Reporting
📧 [atsu.vovor@bell.net](mailto:atsu.vovor@bell.net)
🔗 LinkedIn: [https://www.linkedin.com/in/atsu-vovor-mmai-9188326/](https://www.linkedin.com/in/atsu-vovor-mmai-9188326/)
💻 GitHub: [https://github.com/atsuvovor](https://github.com/atsuvovor)


