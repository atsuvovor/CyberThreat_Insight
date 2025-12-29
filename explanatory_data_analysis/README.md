<h3 align="center">Cyber Threat Unsight -  Explanatory Data Analysis (EDA) with AI-Powered Summarization  </h3>
<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/lagacy_model_dev_github.png" 
       alt="Centered Image" 
       style="width: 800px; height: 40%;">
</p>  


**Toronto, November 01 2024**  
**Autor : Atsu Vovor**
>Master of Management in Artificial Intelligence    
>Consultant Data Analytics Specialist | Machine Learning |  
Data science | Quantitative Analysis | AI Reporting| French & English Bilingual  


<a 
  href="https://github.com/atsuvovor/CyberThreat_Insight/tree/main/explanatory_data_analysis#readme"
  target="_parent">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Overview

The `eda.py` script in the **CyberThreat_Insight** repository automates **Explanatory Data Analysis (EDA)** by combining traditional **pandas/numpy-based data inspection** with **AI-powered summarization**. It is designed for portability across **Google Colab**, **Jupyter Notebook**, and **local Python environments**, enabling cybersecurity analysts and data scientists to quickly understand datasets and generate insights.



## Introduction

Data analysis is an essential step in cybersecurity threat modeling and anomaly detection. Analysts often need to explore datasets quickly to identify trends, distributions, and anomalies. The `eda.py` script bridges **classical EDA techniques** (descriptive statistics, missing value checks, etc.) with **Large Language Models (LLMs)** that summarize findings into **human-readable insights**.

This approach reduces manual effort and accelerates reporting, making it valuable for executive-level presentations and real-time simulation environments.


## Code Structure & Commentary

### 1. Imports and Environment Setup

* Uses standard Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`.
* Integrates Hugging Face pipelines (local CPU fallback).
* Includes `safe_import` for robust package handling (installs missing dependencies automatically).

📌 Ensures portability and resilience—users can run the script without worrying about missing libraries.



### 2. Data Loading

* Accepts CSV input (default dataset or user-provided).
* Prints dataset info: shape, column names, types, and missing values.

📌 Provides analysts with an immediate overview of dataset structure and data quality.



### 3. Descriptive Statistics

* Computes summary statistics (`.describe()`).
* Detects categorical vs numerical features.
* Highlights distributions, outliers, and correlations.

📌 The core of classical EDA, enabling users to spot anomalies and trends.



### 4. Visualization

* Plots histograms, correlation heatmaps, and categorical distributions.
* Supports `matplotlib` and `seaborn` for visual clarity.

📌 Visual EDA complements numerical summaries by revealing hidden structures.


### 5. AI-Powered Summarization

* Uses Hugging Face models (`distilgpt2`, or local alternatives if unavailable).
* Summarizes dataset insights: e.g., trends, anomalies, correlations.
* Outputs **natural language explanations** for decision-makers.

📌 The innovative part—LLMs transform raw statistics into **narrative insights**.



### 6. Execution Flow

When run, the script:

1. Clones the repository (first-time setup).
2. Loads the dataset.
3. Generates descriptive statistics & plots.
4. Produces an AI-powered executive summary.



## Usage & Portability

### 🔹 In Google Colab

```bash
!git clone https://github.com/atsuvovor/CyberThreat_Insight.git
%cd CyberThreat_Insight/explanatory_data_analysis
%run eda.py
# Cyber Threat Unsight -  Explanatory Data Analysis (EDA) with AI-Powered Summarization  

**Toronto, November 01 2024**  
**Autor : Atsu Vovor**
>Master of Management in Artificial Intelligence    
>Consultant Data Analytics Specialist | Machine Learning |  
Data science | Quantitative Analysis | AI Reporting| French & English Bilingual  


## Overview

The `eda.py` script in the **CyberThreat_Insight** repository automates **Explanatory Data Analysis (EDA)** by combining traditional **pandas/numpy-based data inspection** with **AI-powered summarization**. It is designed for portability across **Google Colab**, **Jupyter Notebook**, and **local Python environments**, enabling cybersecurity analysts and data scientists to quickly understand datasets and generate insights.



## Introduction

Data analysis is an essential step in cybersecurity threat modeling and anomaly detection. Analysts often need to explore datasets quickly to identify trends, distributions, and anomalies. The `eda.py` script bridges **classical EDA techniques** (descriptive statistics, missing value checks, etc.) with **Large Language Models (LLMs)** that summarize findings into **human-readable insights**.

This approach reduces manual effort and accelerates reporting, making it valuable for executive-level presentations and real-time simulation environments.


## Code Structure & Commentary

### 1. Imports and Environment Setup

* Uses standard Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`.
* Integrates Hugging Face pipelines (local CPU fallback).
* Includes `safe_import` for robust package handling (installs missing dependencies automatically).

📌 Ensures portability and resilience—users can run the script without worrying about missing libraries.



### 2. Data Loading

* Accepts CSV input (default dataset or user-provided).
* Prints dataset info: shape, column names, types, and missing values.

📌 Provides analysts with an immediate overview of dataset structure and data quality.



### 3. Descriptive Statistics

* Computes summary statistics (`.describe()`).
* Detects categorical vs numerical features.
* Highlights distributions, outliers, and correlations.

📌 The core of classical EDA, enabling users to spot anomalies and trends.



### 4. Visualization

* Plots histograms, correlation heatmaps, and categorical distributions.
* Supports `matplotlib` and `seaborn` for visual clarity.

📌 Visual EDA complements numerical summaries by revealing hidden structures.


### 5. AI-Powered Summarization

* Uses Hugging Face models (`distilgpt2`, or local alternatives if unavailable).
* Summarizes dataset insights: e.g., trends, anomalies, correlations.
* Outputs **natural language explanations** for decision-makers.

📌 The innovative part—LLMs transform raw statistics into **narrative insights**.



### 6. Execution Flow

When run, the script:

1. Clones the repository (first-time setup).
2. Loads the dataset.
3. Generates descriptive statistics & plots.
4. Produces an AI-powered executive summary.



## Usage & Portability

### 🔹 In Google Colab

```bash
!git clone https://github.com/atsuvovor/CyberThreat_Insight.git
%cd CyberThreat_Insight/explanatory_data_analysis
%run eda.py
```

---

### 🔹 In Jupyter Notebook

```bash
!git clone https://github.com/atsuvovor/CyberThreat_Insight.git
%cd CyberThreat_Insight/explanatory_data_analysis
%run eda.py
```


### 🔹 Locally

```bash
git clone https://github.com/atsuvovor/CyberThreat_Insight.git
cd CyberThreat_Insight/explanatory_data_analysis
python eda.py
```


## System Specifications

* **Python version:** 3.8+
* **Libraries:** pandas, numpy, matplotlib, seaborn, transformers (Hugging Face)
* **Hardware:** Runs on CPU (GPU optional for faster inference).
* **Memory:** ≥4 GB RAM recommended for medium datasets.


## Weaknesses of LLMs in EDA

* **Context length limits** → Summarization may miss details in large datasets.
* **Hallucinations** → AI may infer non-existent relationships.
* **Performance overhead** → Hugging Face models can be slow on CPU.
* **Data sensitivity** → LLM summaries may not be suitable for confidential datasets without governance.


## Next Steps to Improve

1. **Model Upgrades:** Replace `distilgpt2` with **Mistral 7B**, **Mixtral 8x7B**, or **LLaMA 3-8B-Instruct** for better summarization.
2. **Chunked Summarization:** Break large datasets into sections to fit within context limits.
3. **Interactive Dashboards:** Integrate with **Streamlit** or **Plotly Dash** for real-time EDA.
4. **Governance Layer:** Add data masking, compliance checks, and audit trails for regulatory use.
5. **Explainability:** Pair LLM summaries with **statistical justifications** (e.g., "95% of variance explained by top 3 features").


## Conclusion

The `eda.py` script provides a **scalable, portable, and AI-augmented** approach to exploratory data analysis in cybersecurity. By merging **traditional statistical techniques** with **AI-powered summarization**, it accelerates insight generation and supports both technical and executive stakeholders. While LLMs introduce challenges like hallucination and performance overhead, future improvements (stronger models, chunking, governance) will make this tool an indispensable part of the data analysis pipeline.

---

### 🔹 In Jupyter Notebook

```bash
!git clone https://github.com/atsuvovor/CyberThreat_Insight.git
%cd CyberThreat_Insight/explanatory_data_analysis
%run eda.py
```


### 🔹 Locally

```bash
git clone https://github.com/atsuvovor/CyberThreat_Insight.git
cd CyberThreat_Insight/explanatory_data_analysis
python eda.py
```


## System Specifications

* **Python version:** 3.8+
* **Libraries:** pandas, numpy, matplotlib, seaborn, transformers (Hugging Face)
* **Hardware:** Runs on CPU (GPU optional for faster inference).
* **Memory:** ≥4 GB RAM recommended for medium datasets.


## Weaknesses of LLMs in EDA

* **Context length limits** → Summarization may miss details in large datasets.
* **Hallucinations** → AI may infer non-existent relationships.
* **Performance overhead** → Hugging Face models can be slow on CPU.
* **Data sensitivity** → LLM summaries may not be suitable for confidential datasets without governance.


## Next Steps to Improve

1. **Model Upgrades:** Replace `distilgpt2` with **Mistral 7B**, **Mixtral 8x7B**, or **LLaMA 3-8B-Instruct** for better summarization.
2. **Chunked Summarization:** Break large datasets into sections to fit within context limits.
3. **Interactive Dashboards:** Integrate with **Streamlit** or **Plotly Dash** for real-time EDA.
4. **Governance Layer:** Add data masking, compliance checks, and audit trails for regulatory use.
5. **Explainability:** Pair LLM summaries with **statistical justifications** (e.g., "95% of variance explained by top 3 features").


## Conclusion

The `eda.py` script provides a **scalable, portable, and AI-augmented** approach to exploratory data analysis in cybersecurity. By merging **traditional statistical techniques** with **AI-powered summarization**, it accelerates insight generation and supports both technical and executive stakeholders. While LLMs introduce challenges like hallucination and performance overhead, future improvements (stronger models, chunking, governance) will make this tool an indispensable part of the data analysis pipeline.
