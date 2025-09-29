# CyberDataGen

`cyberdatagen.py` is a **portable data generation script** for simulating cybersecurity incidents, anomalies, and KPIs/KRIs for analytics and reporting.  
It generates **synthetic datasets** that can be used for dashboards, anomaly detection models, and executive reporting.

---

## 🚀 Features

- Generates **5 CSV datasets**:
  1. `cybersecurity_dataset_normal.csv` – normal security issues  
  2. `cybersecurity_dataset_anomalous.csv` – anomalous issues (attacks, critical incidents)  
  3. `cybersecurity_dataset_combined.csv` – merged dataset of normal + anomalous issues  
  4. `key_threat_indicators.csv` – key threat indicators (KRIs)  
  5. `scenarios_with_colors.csv` – scenarios mapped to severity levels & colors  

- Always saves to:
  ```
  CyberThreat_Insight/cybersecurity_data/
  ```

- Prints a **summary table** (rows, columns, size).  
- Optionally displays `.info()`, `.describe()`, and `.head()` for each dataset.  
- Prompts the user to **download a ZIP of all datasets** (automatic in Colab).  
- Portable: runs in **Google Colab, JupyterLab, or local Python**.  

---

## 📂 Output Structure

After running, the repo will contain:

```
CyberThreat_Insight/
  datagen/
    cyberdatagen.py
  cybersecurity_data/
    cybersecurity_dataset_normal.csv
    cybersecurity_dataset_anomalous.csv
    cybersecurity_dataset_combined.csv
    key_threat_indicators.csv
    scenarios_with_colors.csv
```

Optionally (if downloaded/created):
```
/tmp/cybersecurity_data.zip
```

---

## 🔧 Installation

Clone the repository (if running locally):

```bash
git clone https://github.com/atsuvovor/CyberThreat_Insight.git
cd CyberThreat_Insight/datagen
```

Dependencies (install via pip):

```bash
pip install pandas numpy matplotlib
```

Optional (for Colab ZIP download):
```bash
pip install google-colab
```

---

## ▶️ Usage

### 1. Google Colab

Run directly from GitHub without cloning:

```python
%run https://raw.githubusercontent.com/atsuvovor/CyberThreat_Insight/main/datagen/cyberdatagen.py
```

👉 At the end, you’ll be prompted:

```
Would you like to download the data files locally as well? (yes/no):
```

- **yes** → creates & downloads `cybersecurity_data.zip`  
- **no** → keeps datasets in the repo folder only  

---

### 2. Local / JupyterLab

Run with Python:

```bash
python datagen/cyberdatagen.py
```

CLI options:

- `--no-prompt` → skip ZIP prompt (useful for automation)  
- `--auto-download` → auto-create ZIP without asking (downloads in Colab, creates `/tmp/cybersecurity_data.zip` locally)  
- `--no-display` → skip verbose `.info()` and `.describe()` outputs  

Examples:

```bash
python datagen/cyberdatagen.py --no-prompt
python datagen/cyberdatagen.py --auto-download
python datagen/cyberdatagen.py --no-display
```

---

## 📊 Example Output (Summary Table)

```
📊 Dataset Summary
                         File   Rows  Columns    Size
 cybersecurity_dataset_normal.csv   800      8  120.3 KB
cybersecurity_dataset_anomalous.csv 200      8   30.4 KB
   cybersecurity_dataset_combined.csv 1000     9  150.7 KB
          key_threat_indicators.csv    3       2    2.1 KB
         scenarios_with_colors.csv    3       3    2.2 KB
```

---

## 📈 Example Plots

After generating the data, you can quickly visualize trends.  
Example in Python (Jupyter/Colab):

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load combined dataset
df = pd.read_csv("cybersecurity_data/cybersecurity_dataset_combined.csv")

# Threat severity distribution
df['severity'].value_counts().plot(kind='bar', color='orange', title="Threat Severity Distribution")
plt.show()

# Normal vs Anomalous counts
df['severity_color'].value_counts().plot(kind='pie', autopct='%1.1f%%', title="Normal vs Anomalous Issues")
plt.show()
```

These plots help quickly understand the dataset composition.

---

## 📌 Notes

- The generated data is **synthetic** and designed for **simulation/testing only**.  
- Useful for:
  - Cyber attack simulators  
  - Dashboard prototyping  
  - AI/ML anomaly detection experiments  
  - Executive reporting demos  

---

## 🧑‍💻 Author

**Atsu Vovor**  
Data & Analytics Consultant | Cybersecurity | AI Reporting  
[LinkedIn](https://www.linkedin.com/in/atsu-vovor-mmai-9188326/) | [GitHub](https://github.com/atsuvovor)
