 ![Project Header Image](images/cyberThreat-Insight.png)  
 <div align="center">
  
**Detecting Anomalous Behavior in Cybersecurity Analytics with Generative AI**
</div>

**Toronto, November 01 2024**  
**Autor : Atsu Vovor**
>Master of Management in Artificial Intelligence    
>Consultant Data Analytics Specialist | Machine Learning |  
Data science | Quantitative Analysis | AI Reporting| French & English Bilingual  

---

## **Abstract**
The CyberThreat Insight project leverages data analytics and machine learning to detect and analyze anomalous behavior in user accounts and network systems. Using synthetic data generated through advanced augmentation techniques, the project investigates patterns in cybersecurity issues, enabling proactive threat detection and response. This research-driven approach provides actionable intelligence, that can help organizations reduce risk from internal and external threats.

This project is a research-focused initiative aimed at exploring the potential of generative algorithms in cybersecurity analytics. The methods implemented are designed to simulate data that emulate real-world cyberattack scenarios. It is important to note that the data used in this project is entirely synthetic, with no initial dataset sourced externally for baseline reference.

---

## **Introduction**
In today’s evolving cybersecurity landscape, identifying subtle and  anomalous behaviors is essential for combating sophisticated cyber threats. The **CyberThreat Insight** project aims to harness machine learning to understand and address complex cybersecurity challenges. By analyzing synthetic data that mirrors real-world cybersecurity issues, this project will identify unusual behaviors such as high login attempts, extended session durations, or significant data transfers. The findings will support organizations in developing proactive detection capabilities, improving their ability to respond swiftly to internal and external threats.

---

## **Project Description**
The **CyberThreat Insight** project will focus on the following key areas to build an anomaly detection framework for cybersecurity analytics:

1. **Research and Analysis Objectives**: This project is designed for research and analysis purposes, investigating how machine learning techniques can enhance understanding and detection of complex cybersecurity issues. By identifying patterns that signify potential threats, the project is intended to improve decision-making and support risk mitigation.

2. **Synthetic Data Generation**: Using data augmentation techniques—such as SMOTE, GANs, label shuffling, time-series variability, and noise addition—the project will create a synthetic dataset with realistic, month-over-month volatility. This data will include anomalies that reflect potential security concerns, such as unusually high login attempts, extended session durations amd large data transfer volumes.

3. **Anomaly Detection with Machine Learning**: Machine learning models will be applied to identify and classify unusual patterns within the dataset. Techniques like Isolation Forests, Autoencoders, and DBSCAN will help in detecting anomalies, enabling the system to pinpoint behaviors that deviate from established baselines.

4. **Proactive Threat Detection and Response**: The project will integrate these models with alerting mechanisms, providing security teams with actionable insights for early threat response. By identifying suspicious activity patterns in real-time, the system will offer timely intelligence for mitigating internal and external threats.

5. **Continuous Model Improvement**: Feedback from detection results and analysts’ input will be incorporated to refine models, ensuring that they adapt to emerging threat patterns and reduce false positives.

6. **Project Outcome and Impact**: The final deliverable will be an anomaly detection framework capable of analyzing user behaviors and system interactions, alerting security teams to potentially malicious activities. By proactively identifying threats, the CyberThreat Insight project will help organizations enhance their
 cybersecurity resilience, gaining valuable insights for future threat prevention.


---
## **Scope of the Project**  

#### **1.Data Preparation( Data synthetization & Preprocessing)**    
In this section, we will use data augmentation techniques (SMOTE, GAN, label shuffling or permutation, time series variability, and noise addition) to generate a synthetic cybersecurity issues dataset   that will include month-to-month volatility and significant anomalies (such as high login attempts, unusual session durations, or high data transfer volumes). The goal here is to reduce the umbalance of data classes.  

shouldn't it be better to generate  # Create anomalous issues dataset
def generate_anomalous_issues_df(p_anomalous_issue_ids, p_anomalous_issue_keys) module based on the generate_normal_issues_df output( generate_normal_issues_df output to be used as imput to generate anomalous issues dataset)? I think this will help to customize or ajuste the anomalous or update the columns included in the anomalous when there is not enougt anomalous.

absolutely. That’s a great idea, and it’ll give you more flexibility and realism in your dataset. By basing the anomalous issues generation on the normal issues dataset, you:

Benefits
Ensure Column Consistency
Any schema changes to the normal dataset automatically propagate.

Simplify Maintenance
No need to manage column lists or logic redundantly in two places.

Customizable Anomalies
You can select rows from the normal set and tweak them (e.g., elevate threat metrics, distort values).

Guarantee Coverage
Especially useful when you don’t have enough anomalies — you can "morph" some normal rows into anomalies.

Support Semi-Synthetic Modeling
Which is closer to real-world threats — abnormal behavior typically starts from a normal baseline.

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
   - **Scaling Up Data Points**: We will use SMOTE or random sampling for categorical columns to add diversity.
   - **Label Swapping for `Assignees`, `Departments`**: Here, we randomly reassign categories periodically to simulate changing roles.
   - **Time-Series Variability**: We use simulated timestamps within and across sessions to show login attempts, data transfer spikes, and session durations.

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
- **Adressing class imbalance in datasets**:Using SMOTE (Synthetic Minority Over-sampling Technique) we make sure that class imbalance in the dataset is resolved.  
all the data files are saves on google drive  

  
**User Activities Generation Metrics Formula**  

The expression:  base_value + base_value * volatility * (np.random.randn()) * (1.2 if severity in ['High', 'Critical'] else 1)

means that we’re generating a value based on a starting point (base_value) and adjusting it for both randomness and severity level. Here's a breakdown:

- **base_value**: This is the initial value that the output is based on.
- **volatility * (np.random.randn())**: This part adds a random fluctuation around the base_value. `np.random.randn()` generates a value from a standard normal distribution (centered around 0), so it could be positive or negative, creating variation. Multiplying by volatility scales the randomness, making the fluctuation stronger or weaker.
- **(1.2 if severity in ['High', 'Critical'] else 1)**: This adds an additional factor to increase the outcome by 20% if the severity is "High" or "Critical." If severity isn’t in these categories, the factor is simply 1, meaning no extra adjustment.

So, if severity is "High" or "Critical," the result is a base value adjusted for both volatility and severity; otherwise, it’s just the base value with volatility adjustment.  


**Treat level Identification and Adaptive Defense Systems Setting**

We will set up a threat level based our cybersecurity dataset generated. We will create a threat scoring model that combines multiple relevant features.  

**Key Threat Indicators (KTIs) Definition**  

The following columns will be uses as key threat indicators (KTIs):  

   - **Severity**: Indicates the criticality of the issue.
   - **Impact Score**: Represents the potential damage if the threat is realized.
   - **Risk Level**: A general indicator of risk associated with each issue.
   - **Issue Response Time Days**: The longer it takes to respond, the higher the threat level could be.
   - **Category**: Certain categories (e.g., unauthorized access) carry a higher base threat level.
   - **Activity Type**: Suspicious activity types (e.g., high login attempts, data modification) indicate a greater threat.
   - **Login Attempts**: Unusually high login attempts signal a brute force attack.
   - **Num Files Accessed** and **Data Transfer MB**: Large data transfers or access to many files in a session could indicate data exfiltration or suspicious activity.

**KTIs based Scoring**
  
For each KTI we will define the acriteria to be used to assigne a score  

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

Here we will add logic that monitors specific threat conditions in real-time and adapt responses based on defined rules. This will include automatic flagging of high-threat issues, increasing logging frequency for suspicious activities, and assigning specific mitigation actions based on the threat level and activity context.

**Rules Definition**  
 We will use the following features to define rules that will be applied to identify potential threats and recommend defensive actions: `Threat Level`, `Severity`, `Impact Score`, `Login Attempts`, `Risk Level`, `Issue Response Time Days`, `Num Files Accessed`,`Data Transfer MB`.   

**Defense Mechanism**: The system will respond adaptively by adding flags and assigning custom actions based on the rule evaluations and scenarios colors

   
 The defense mechanism assigns an adaptive `Defense Action` to each issue based on threat conditions, adding an extra layer of automated response for varying threat levels and behaviors.
 The treat conditions are implemented by Color-coding cybersecurity scenarios, we bealieve, is a helpful way to quickly communicate risk levels and prioritize response actions. Here's a suggested approach to buld the scenarios, where we use **intensity of red, orange, yellow, and green** to represent risk:

**Color Scheme**
- **Critical Threat & Severity**: **Dark Red** – Highest urgency.
- **High Threat or Severity**: **Orange** – Serious, but not the highest urgency.
- **Medium Threat or Severity**: **Yellow** – Moderate concern.
- **Low Threat & Severity**: **Green** – Low concern, monitor as needed.

**Scenarios with Colors**

| **Scenario** | **Threat Level** | **Severity** | **Suggested Color** | **Rationale** |
|---------------|------------------|--------------|----------------------|----------------|
| **1** | Critical | Critical | **Dark Red** | Maximum urgency, both threat and impact are critical. Immediate action required. |
| **2** | Critical | High | **Red** | Very high risk, threat is critical and impact is significant. Prioritize response. |
| **3** | Critical | Medium | **Orange-Red** | Significant threat but moderate impact. Act promptly to prevent escalation. |
| **4** | Critical | Low | **Orange** | High potential risk, current impact is minimal. Monitor closely and mitigate quickly. |
| **5** | High | Critical | **Red** | High threat combined with critical impact. Needs immediate action. |
| **6** | High | High | **Orange-Red** | High threat and significant impact. Prioritize response. |
| **7** | High | Medium | **Orange** | Elevated threat and moderate impact. Requires attention. |
| **8** | High | Low | **Yellow-Orange** | High threat with low impact. Proactive monitoring recommended. |
| **9** | Medium | Critical | **Orange** | Moderate threat with critical impact. Prioritize addressing the severity. |
| **10** | Medium | High | **Yellow-Orange** | Medium threat with high impact. Needs resolution soon. |
| **11** | Medium | Medium | **Yellow** | Medium threat and impact. Plan to address it. |
| **12** | Medium | Low | **Light Yellow** | Moderate threat, minimal impact. Monitor as needed. |
| **13** | Low | Critical | **Yellow** | Low threat but high impact. Address severity first. |
| **14** | Low | High | **Light Yellow** | Low threat with significant impact. Plan mitigation. |
| **15** | Low | Medium | **Green-Yellow** | Low threat, moderate impact. Routine monitoring. |
| **16** | Low | Low | **Green** | Minimal risk. No immediate action required. |

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


  
#### **3. Features Engineering Pipeline**

The feature engineering pipeline was designed to simulate realistic cybersecurity scenarios, enhance anomaly detection, and prepare the dataset for effective model training. It involved the following key steps:

* **Synthetic Data Load**: Real-time behavioral data was simulated to represent normal system activity.
* **Anomaly Injection (Cholesky Perturbation)**: Statistically realistic anomalies were introduced to compensate for the natural scarcity of threat events.
* **Feature Normalization**: All features were scaled using Min-Max and Z-score methods to ensure consistent input ranges.
* **Correlation Analysis**: Pearson and Spearman heatmaps helped identify and mitigate multicollinearity among variables.
* **Feature Importance (Random Forest)**: The most influential threat indicators were identified for model optimization.
* **Model Explainability (SHAP)**: SHAP values provided interpretability for each prediction, essential for SOC analysts.
* **Dimensionality Reduction (PCA)**: Principal Component Analysis reduced noise while preserving important behavioral patterns.
* **Data Augmentation (SMOTE + GANs)**: Oversampling techniques balanced the dataset by generating synthetic threat instances.

This workflow produced a clean, balanced, and interpretable feature set optimized for machine learning–based cyber threat classification.



#### **4. Train-Test Split**
A function is defined to split the augmented feature matrix and target vector
Assign the results to variables representing the training and testing data```
Define a function for splitting the dataset into training and testing subsets
Use train_test_split from sklearn to randomly split the data
`test_size=0.2` specifies that 20% of the data will be allocated to the testing set.    


#### **5. Anomaly Detection Models Developement**
We implemented two supervised machine learning algorithms(Random Forest
Gradient Boosting), six unsupervised machine learning algorithms(Isolation Forest
One-Class SVM, DBSCAN, Autoencoder, K-means Clustering, Local Outlier Factor (LOF)) and one mixed superviced and unsupervised machine learning algorithm(LSTM (Long Short-Term Memory))  
  

#### **6. Best model Selection**  
We chosed the most performing algorithm based on each model 'Overall Model Accuracy'.  


#### **7. Best Model Deployment**  
We deployed the winning medel to myGoogle drive.

Through this systematic approach, CyberThreat Insight will contribute to a deeper understanding of behavioral anomalies, equipping organizations with the tools needed to anticipate and mitigate cybersecurity risks effectively.

#### **8. Best Model Testing**
As testing strategy, we will uploard the best model and run it with the inital synthetic data that serve as real time production data. The main reason of using the initial synthetic data is that the model was developed using augmented data. The purpuse of our testing strategy is to capture the model performance on the real data.   
We will run the model performance visualization charts like:  
- Squater Plot on y = Data Transfer, X = Session Duration  
- ROC curve(Y=True Positive Rate, X= false positive rate)  
- Precision recall curve(y= precition, X= recall)


As part of our testing strategy, we will evaluate the best-performing model using the **initial synthetic dataset**. This dataset simulates real-time production environments and is independent of the augmented data used during training. This approach allows us to evaluate **how well the model generalizes to operational-like conditions** and to identify any overfitting to the augmented training data.

##### **Evaluation Metrics**

We will generate the following performance outputs and charts to interpret model behavior across all threat levels:

#####  **1. Confusion Matrix**

* **Purpose**: Visualize how well the model classifies each threat category (`Threat Level`).
* **Interpretation**: Shows the counts of true vs. predicted labels.

  * Diagonal values = correct classifications
  * Off-diagonal values = misclassifications
* **Helps Identify**:

  * Whether the model is confusing *High* with *Medium* threats, etc.
  * If there's any class imbalance affecting performance


#####  **2. Classification Report**

* **Includes**:

  * **Precision**: How many predicted labels were actually correct?
  * **Recall**: How many true labels were correctly predicted?
  * **F1-score**: Harmonic mean of precision and recall
  * **Support**: Number of actual instances per class
* **Purpose**: Detailed per-class evaluation — crucial for **cybersecurity**, where missing a *high threat* is more costly than misclassifying a *low threat*.


#####  **3. ROC Curve**

* **X-axis**: False Positive Rate
* **Y-axis**: True Positive Rate (Recall)
* **Multiclass**: Will be plotted using a One-vs-Rest strategy
* **Purpose**: Shows how well the model distinguishes between each threat level at different thresholds


#####  **4. Precision-Recall Curve**

* **X-axis**: Recall
* **Y-axis**: Precision
* **Multiclass**: One-vs-Rest approach
* **Purpose**: Ideal for **imbalanced classes** (e.g., rare high-risk attacks)
* **Key Insight**: Focus on how well the model maintains precision as it tries to improve recall



#####  **5. Scatter Plot**

* **X-axis**: `Session Duration in Second`
* **Y-axis**: `Data Transfer MB`
* **Color**: Model-predicted `Threat Level`
* **Purpose**: Visual exploratory view to see how predicted threat levels distribute across session metrics


  
#####  **Feature Set Used**

| Feature                      | Description                                       |
| ---------------------------- | ------------------------------------------------- |
| `Issue Response Time Days`   | How long it took to respond to the issue          |
| `Impact Score`               | Estimated impact of the session                   |
| `Cost`                       | Operational or financial impact                   |
| `Session Duration in Second` | Length of the session                             |
| `Num Files Accessed`         | Number of files accessed during session           |
| `Login Attempts`             | Count of login attempts                           |
| `Data Transfer MB`           | Volume of data moved                              |
| `CPU Usage %`                | Average CPU usage during session                  |
| `Memory Usage MB`            | RAM usage in megabytes                            |
| `Threat Score`               | Model-assigned risk score based on prior analysis |  

  
Here’s an expanded and more detailed rewrite of your section on **Cyber Attack Simulation**:

  

#### **9. Cyber Attack Simulation**

As part of the next phase of the project, we will extend the platform to simulate a range of high-impact cyber attacks. These simulations will provide a dynamic testing environment to evaluate detection capabilities, assess organizational vulnerabilities, and enhance the system’s AI-powered threat response mechanisms. The simulated attack types will include:

* **Phishing Attacks:** Simulate social engineering campaigns to test user susceptibility to deceptive emails, credential harvesting, and fraudulent access attempts.
* **Malware Attacks:** Model the behavior and spread of malicious software such as keyloggers, spyware, trojans, and worms to assess endpoint defenses and containment strategies.
* **Distributed Denial-of-Service (DDoS) Attacks:** Emulate volumetric and application-layer attacks aimed at overwhelming network resources, disrupting services, and testing resilience under stress.
* **Data Leak Attacks:** Mimic unauthorized data exfiltration scenarios, both accidental and malicious, to evaluate monitoring, detection, and containment protocols.
* **Insider Threats:** Simulate misuse of access privileges by employees or contractors, focusing on the detection of anomalous behaviors within internal systems.
* **Ransomware Attacks:** Recreate file encryption and ransom demand scenarios to test system backups, alerting systems, and recovery processes.

Each simulation will be integrated into the platform’s AI analytics engine and risk dashboards, providing real-time threat scoring, response playbooks, and post-event analysis to support training, governance, and resilience planning.
---

## **Project Files**
Here’s a summary of important files in the repository:

- **[`.gitignore`](./.gitignore)** - This file lists all the files and directories to be ignored by Git.
- **[`requirements.txt`](./requirements.txt)** - Contains the dependencies required to run the project.
- **[`setup.py`](./setup.py)** - Setup script for packaging and distributing the project.
- **[`src/`](./src)** - Contains the main Python source code files for the project.
  - **[`main.py`](./src/main.py)** - Entry point for the project.
- **[`tests/`](./tests)** - Includes the unit tests for the project.
  - **[`test_main.py`](./tests/test_main.py)** - Unit tests for the `main.py` script.
- **[`notebooks/`](./notebooks)** - Contains Jupyter Notebooks for data analysis and modeling.
  - **[`example_notebook.ipynb`](./notebooks/example_notebook.ipynb)** - Example of how the data is analyzed.

---

## **Installation**
To install the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/my-python-project.git
   cd my-python-project

