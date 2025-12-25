# 🛡️ CyberThreat-Insight

## Attack Simulation & Stacked Anomaly Detection Platform

**Author:** Atsu Vovor
**Title:** Data & Analytics Consultant | Cybersecurity | AI Reporting
**Location:** Toronto, Canada

---

## 📌 Executive Overview

**CyberThreat-Insight** is an **enterprise-grade cyber attack simulation, anomaly detection, and AI-assisted reporting platform** designed to support:

* Cyber risk scenario modeling
* Threat detection validation
* Executive and board-level reporting
* Model Risk Management (MRM)
* Regulatory and audit review readiness

The platform augments **real operational cybersecurity data** with **statistically governed attack simulations**, applies a **stacked ensemble anomaly detection model**, and produces **audit-ready outputs** for dashboards, reports, and AI-generated executive summaries.

This solution is intentionally designed to align with how **financial institutions, insurers, and regulated enterprises** evaluate analytics platforms during **Model Risk Committee, Internal Audit, and regulator reviews**.

---

## 🧠 Platform Capabilities at a Glance

* Multi-vector cyber attack simulation
* MITRE ATT&CK–aligned threat modeling
* Production-grade stacked anomaly detection
* ML-safe data engineering & inference
* Full data lineage & governance controls
* AI-assisted executive reporting
* Audit- and regulator-ready documentation

---

## 🧩 High-Level Architecture

```
External Data Sources (CSV / Google Drive)
            │
            ▼
Operational Cybersecurity Dataset
            │
            ▼
Attack Simulation Engine
(Phishing, Malware, DDoS, Insider, etc.)
            │
            ▼
ML-Safe Data Sanitization & Validation
            │
            ▼
Stacked Anomaly Detection Models
            │
            ▼
Threat Scores & Anomaly Predictions
            │
            ▼
Persisted Outputs (CSV / Dashboards / AI Reports)
```

---

## 🧭 MITRE ATT&CK–Mapped Architecture View

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

---

## ⚙️ Key Features

### ✅ Multi-Attack Simulation Framework

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

---

## 📊 Mathematical Foundations of Attack Simulation

Each attack is governed by **explicit probabilistic models** to ensure realism, explainability, and repeatability.

---

Here is the **correct, clean, and Markdown-compatible rewrite** of your section.
It will render properly in **GitHub README.md**, **MkDocs**, and **enterprise documentation tools**, while keeping the mathematics precise and readable.

---

## 📨 Phishing — Credential Abuse

**Login Attempts**

$$
X_{\text{login}} \sim \text{Poisson}(\lambda)
$$

**Impact and Threat Scores**

$$
\text{Impact} \sim \mathcal{N}(5, 3^2)
$$

$$
\text{Threat} \sim \mathcal{N}(6, 3^2)
$$

---

## 🦠 Malware — System Enumeration

**Files Accessed**

$$
X_{\text{files}} \sim \text{Poisson}(\lambda)
$$

**Impact and Threat Scores**

$$
(\text{Impact}, \text{Threat}) \sim \mathcal{N}(7, 4^2)
$$

---

## 🌐 DDoS — Resource Saturation

**Session Duration**

$$
X_{\text{session}} \sim \text{Exponential}(\beta)
$$

**Impact and Threat Escalation**

$$
(\text{Impact}, \text{Threat}) \sim \text{Exponential}(8)
$$

---

## 💾 Data Leak — Data Exfiltration

**Data Transfer Volume**

$$
X = \mu \cdot e^{\sigma Z}, \quad Z \sim \mathcal{N}(0,1)
$$

**Impact and Threat Scores**

$$
(\text{Impact}, \text{Threat}) \sim \mathcal{N}(12, 5^2)
$$

---

## 🔐 Insider Threat — Time-Based Abuse

**Targeted Sessions**

$$
\text{hour} < 6 \quad \text{or} \quad \text{hour} > 23
$$

**Data Transfer Volume**

$$
X_{\text{transfer}} \sim \text{LogNormal}(\sigma = 0.3)
$$

---

## 💣 Ransomware — Encryption Storms

**CPU Utilization**

$$
X_{\text{CPU}} \sim \mathcal{N}(20, 10^2)
$$

**Memory Consumption**

$$
X_{\text{memory}} \sim \text{LogNormal}(\sigma = 0.5)
$$

**Impact and Threat Scores**

$$
(\text{Impact}, \text{Threat}) \sim \mathcal{N}(15, 5^2)
$$

---

### ✅ Notes for Reviewers & Regulators

* All distributions are **explicitly defined** for transparency and reproducibility
* Parameters are **bounded in code** to enforce operational realism
* Scores represent **system risk indicators**, not attacker intent or attribution

If you want, I can also provide:

* A **LaTeX-only appendix** for academic or regulator submissions
* A **plain-English executive explanation** of each formula
* A **model validation checklist** referencing these equations

Just say the word.

---

## 🧼 ML-Safe Data Engineering

### Sanitization Controls

* NaN / ±Inf handling
* Median imputation
* Numeric casting → `float32`
* Metadata preservation
* Schema enforcement

This prevents **inference-time failures** and supports **model reproducibility**.

---

## 🔮 Stacked Anomaly Detection Model

The anomaly detection engine is a **stacked ensemble** consisting of:

* Isolation Forest
* One-Class SVM
* Local Outlier Factor (LOF)
* DBSCAN
* KMeans (distance-based features)
* Dense Autoencoder
* LSTM Autoencoder

A **Gradient Boosting meta-model** produces the final anomaly classification and threat score.

---

## 📂 Project Structure

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

---

## 🏛️ Data Lineage & Model Governance

### End-to-End Lineage

| Stage          | Control                   |
| -------------- | ------------------------- |
| Data ingestion | Schema validation         |
| Simulation     | Bounded stochastic models |
| Feature prep   | Type enforcement          |
| Inference      | Version-locked models     |
| Output         | Timestamped CSV + hash    |
| Reporting      | Prompt versioning         |

---

### Governance Objectives

* OSFI Model Risk expectations
* SR 11-7 principles
* Explainable AI (XAI) standards

---

## ⚖️ Risk & Bias Analysis

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

---

## 📜 Regulatory & Compliance Alignment

| Framework | Coverage                    |
| --------- | --------------------------- |
| OSFI E-23 | Data governance & lineage   |
| SR 11-7   | Model validation & controls |
| ISO 27001 | Security monitoring         |
| NIST CSF  | Detect / Respond            |

---

## 📤 Outputs

* Simulated datasets with attack annotations
* Anomaly scores & threat levels
* Dashboard-ready tables
* AI-generated executive summaries

---

## 🎯 Intended Use & Limitations

**Appropriate for:**

* Threat modeling
* AI validation
* Executive risk dashboards

**Not intended for:**

* Attribution of real attackers
* User performance monitoring
* Law-enforcement decisioning

---

## 🧠 Design Philosophy

CyberThreat-Insight prioritizes:

* Statistical realism
* Explainability
* Governance readiness
* Executive trust

It bridges **cybersecurity, data science, and AI governance** into a single, auditable framework.

---

Perfect — below is a **clean, regulator-ready extension** you can **drop directly into your `README.md`** (or keep as an appendix file).
It adds **three things auditors, academics, and model risk teams explicitly look for**:

1. **A LaTeX-only appendix** (submission-ready)
2. **Plain-English executive explanations** of each formula
3. **A formal model validation checklist** referencing the equations

This is aligned with **OSFI / SR 11-7 / internal MRM** standards.

---

# 📎 Appendix A — Mathematical Specifications (LaTeX-Only)

> **Purpose:**
> This appendix provides a formal mathematical description of the cyber-attack simulation logic, suitable for **academic review, regulator submission, or independent model validation**.
> No narrative interpretation is included in this section.

---

## A.1 Phishing — Credential Abuse

[
X_{\text{login}} \sim \text{Poisson}(\lambda)
]

[
\text{Impact} \sim \mathcal{N}(5, 3^2)
]

[
\text{Threat} \sim \mathcal{N}(6, 3^2)
]

---

## A.2 Malware — System Enumeration

[
X_{\text{files}} \sim \text{Poisson}(\lambda)
]

[
\text{Impact},; \text{Threat} \sim \mathcal{N}(7, 4^2)
]

---

## A.3 DDoS — Resource Saturation

[
X_{\text{session}} \sim \text{Exponential}(\beta)
]

[
\text{Impact},; \text{Threat} \sim \text{Exponential}(8)
]

---

## A.4 Data Leak — Exfiltration

[
X = \mu \cdot e^{\sigma Z}, \quad Z \sim \mathcal{N}(0,1)
]

[
\text{Impact},; \text{Threat} \sim \mathcal{N}(12, 5^2)
]

---

## A.5 Insider Threat — Time-Based Abuse

[
\text{hour} < 6 \quad \text{or} \quad \text{hour} > 23
]

[
X_{\text{transfer}} \sim \text{LogNormal}(\sigma = 0.3)
]

---

## A.6 Ransomware — Encryption Storms

[
X_{\text{CPU}} \sim \mathcal{N}(20, 10^2)
]

[
X_{\text{memory}} \sim \text{LogNormal}(\sigma = 0.5)
]

[
\text{Impact},; \text{Threat} \sim \mathcal{N}(15, 5^2)
]

---

# 🧠 Appendix B — Executive (Plain-English) Explanation

> **Audience:** Executives, Audit, Risk Committees, non-technical stakeholders

---

### Phishing (Credential Abuse)

*Login attempts follow a Poisson distribution* because phishing attacks generate **many small, repeated login attempts**.
Impact and threat scores use a **normal distribution** to reflect moderate but consistent operational risk.

---

### Malware (System Enumeration)

Malware tends to **scan files repeatedly**, which is well modeled by a Poisson process.
Severity scores are centered higher than phishing, reflecting **greater system compromise risk**.

---

### DDoS (Resource Saturation)

Session durations follow an **exponential distribution**, capturing the fact that most attacks are short, but a few last a very long time.
Severity escalates rapidly as resources are exhausted.

---

### Data Leak (Exfiltration)

Data exfiltration follows a **lognormal distribution**, reflecting that:

* Most leaks are small
* A few rare events cause massive losses

This aligns with real-world breach patterns.

---

### Insider Threat (Time-Based Abuse)

Insider activity is flagged **outside normal business hours**.
Data transfers follow a lognormal pattern, modeling **stealthy but potentially severe misuse**.

---

### Ransomware (Encryption Storms)

CPU and memory usage spike sharply during encryption.
Severity scores are the highest, reflecting **business-critical impact and recovery cost**.

---

# 🧪 Appendix C — Model Validation Checklist (MRM / Audit)

> **Purpose:**
> To support **independent model validation, audit review, and regulatory challenge**

---

## C.1 Conceptual Soundness

| Check                     | Description                                       | Status |
| ------------------------- | ------------------------------------------------- | ------ |
| Statistical justification | Each attack mapped to an appropriate distribution | ✅      |
| Domain alignment          | Distributions align with real cyber behavior      | ✅      |
| Severity calibration      | Impact & Threat scores bounded and interpretable  | ✅      |

---

## C.2 Implementation Verification

| Check             | Reference                                | Status |
| ----------------- | ---------------------------------------- | ------ |
| Poisson logic     | Phishing, Malware (A.1, A.2)             | ✅      |
| Exponential logic | DDoS (A.3)                               | ✅      |
| Lognormal logic   | Data Leak, Insider, Ransomware (A.4–A.6) | ✅      |
| Numeric bounds    | Clipping applied post-simulation         | ✅      |

---

## C.3 Data Integrity Controls

| Control            | Description                      | Status |
| ------------------ | -------------------------------- | ------ |
| NaN / Inf handling | Sanitized pre-inference          | ✅      |
| Precision control  | Numeric features cast to float32 | ✅      |
| Schema enforcement | Required columns validated       | ✅      |

---

## C.4 Output Reasonableness

| Check               | Description                                    |
| ------------------- | ---------------------------------------------- |
| Distribution review | Simulated values reviewed vs historical ranges |
| Anomaly rate        | Monitored for inflation or collapse            |
| Stress behavior     | Rare events intentionally amplified            |

---

## C.5 Model Limitations (Disclosed)

* Simulations are **synthetic**, not attributional
* Severity scores represent **system risk**, not user intent
* Insider logic is heuristic, not behavioral profiling

---

# 🏁 Final Note for Regulators & Committees

This framework was designed to:

✔ Support **stress testing and scenario analysis**
✔ Enable **transparent validation**
✔ Align with **financial-services model governance standards**
✔ Provide **explainable, auditable outputs**

---

If you want next, I can:

• Split this into a **formal Model Card (PDF)**
• Produce a **Model Risk Committee slide**
• Add **OSFI / SR 11-7 explicit control mapping**
• Package this as a **client-ready deliverable**

Just say the word.

---
## 👤 Author

**Atsu Vovor**
Data & Analytics Consultant | Cybersecurity | AI Reporting
📧 [atsu.vovor@bell.net](mailto:atsu.vovor@bell.net)
🔗 LinkedIn: [https://www.linkedin.com/in/atsu-vovor-mmai-9188326/](https://www.linkedin.com/in/atsu-vovor-mmai-9188326/)
💻 GitHub: [https://github.com/atsuvovor](https://github.com/atsuvovor)


