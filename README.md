---

# Predictive Maintenance for Industrial Pumps -- Accenture AI Studio

---

### 👥 **Team Members**

| Name          | GitHub Handle | Contribution                                                      |
| ------------- | ------------- | ----------------------------------------------------------------- |
| Ynalois Pangilinan   | *@ynaloisp* | Project coordination, EDA, Data Preprocessing |
| Lanaya Carbonell | *@LanayaC*    | Data cleaning, results interpretation, documentation              |
| Fatima Rasheed | *@fzrasheed*    | Feature engineering, model development, outlier analysis           |
| Osewuike Igue | *@autowiki4*    | Model training, hyperparameter tuning, evaluation                 |
| Samiksha Gaherwar | *@Samiksha-Gah*    | Model training/testing, evaluation             |

---

## 🎯 **Project Highlights**

* Developed a **machine learning–based predictive maintenance system** to forecast industrial pump failures within a **24-hour horizon** using multivariate time-series sensor data.
* Engineered a rich set of **temporal, statistical, and anomaly-based features**, including rolling statistics, slopes, robust z-scores, and outlier rates.
* Achieved a **ROC-AUC of 0.86** and **PR-AUC of 0.40** using an **XGBoost classifier**, with **high recall (>0.80)** for failure events—critical for maintenance planning.
* Addressed real-world constraints such as **class imbalance**, **pump-specific behavior**, and **time-series data leakage** using careful validation strategies.
* Generated actionable insights to support **proactive maintenance scheduling**, reducing downtime and operational risk.

---

## 👩🏽‍💻 **Setup and Installation**

Follow the steps below to reproduce the analysis and results.

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Accenture2A_Project.git
cd Accenture2A_Project
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost numba
```

### 3. Dataset access

* The dataset is accessed via a **Google Sheet**, but is also a CSV file.
* Ensure you have permission to access the sheet.
* The `sheet_id` is specified directly in the notebook.

### 4. Run the notebook

* Open the notebook in **Google Colab** (recommended) or Jupyter Notebook.
* Run all cells sequentially to reproduce data cleaning, EDA, feature engineering, and model training.

---

## 🏗️ **Project Overview**

This project was completed as part of the **Break Through Tech AI Program – AI Studio**, where teams collaborate on real-world, industry-inspired machine learning challenges.

**Project Objective:**
To build a predictive maintenance model capable of identifying **impending pump failures up to 24 hours in advance** using historical sensor data.

**Scope & Real-World Significance:**
Unplanned pump failures can lead to costly downtime, safety risks, and operational inefficiencies. By accurately predicting failures ahead of time, maintenance teams can:

* Schedule repairs proactively
* Reduce emergency shutdowns
* Optimize resource allocation

This project demonstrates how **data science and machine learning** can be applied to industrial systems to deliver tangible operational value.

---

## 📊 **Data Exploration**

### Dataset Description

* **Size:** ~720,000 records
* **Pumps:** 50 unique pumps
* **Frequency:** 10-minute intervals
* **Features:** Throughput, pressure, vibration, bearing temperature, operating status

### Key EDA Insights

* **Severe class imbalance:** ~93.3% `RUNNING`, ~6.7% `DOWN`
* During `DOWN` events:

  * Throughput, pressure, and vibration drop to ~0
  * Bearing temperature cools toward ambient (~25°C)
* **Temporal patterns**:

  * Failures peak between **2–5 AM**, especially around **4 AM**
  * Slightly fewer failures occur on Sundays
* **Failure duration**:

  * Most failures last **13–18 hours**, suggesting consistent repair times
* **Outlier behavior**:

  * Outlier rates increase steadily in the **24 hours leading up to failure**
  * Near-failure windows show outlier rates approaching 100%

### Challenges & Assumptions

* Sensor outliers during `DOWN` states were treated as **expected behavior**, not noise.
* Care was taken to avoid **data leakage** when computing rolling features.

---

## 🧠 **Model Development**

### Feature Engineering

* Rolling mean, standard deviation, and range (1h, 6h, 12h windows)
* Short- and medium-term slopes to capture trends
* Robust z-scores using **median and MAD** (per pump)
* Outlier flags and rolling outlier rates
* Temporal cyclical features (hour and day-of-week using sine/cosine)
* Interaction features combining multiple sensor signals

### Models Explored

* Logistic Regression (baseline)
* Random Forest
* Extra Trees
* Gradient Boosting
* Linear SVC
* XGBoost (best-performing model)

### Training Setup

* **Chronological split:** 70% train / 30% test
* **Group K-Fold cross-validation** by `pump_number`
* Evaluation focused on **PR-AUC and Recall**, prioritizing failure detection

---

## 📈 **Results & Key Findings**

### Best Model: XGBoost

* **ROC-AUC:** 0.86
* **PR-AUC:** 0.40
* **Recall:** ~0.82 (after threshold tuning)

### Key Takeaways

* Advanced feature engineering significantly improved predictive performance.
* Outlier-based and trend-based features were strong early indicators of failure.
* High recall is achievable without an extreme loss in precision, making the model suitable for real-world maintenance workflows.

---

## 🚀 **Next Steps**

* Incorporate **cost-sensitive learning** to explicitly balance false positives and false negatives.
* Explore **sequence-based models** (e.g., LSTM or Temporal CNNs).
* Evaluate model performance under **concept drift** as equipment ages.
* Integrate predictions into a **real-time monitoring dashboard**.
* Test generalization on data from **new pumps or facilities**.

---

## 📄 **References**

* Scikit-learn Documentation
* XGBoost Documentation
* Time-Series Feature Engineering for Predictive Maintenance (industry articles)

---

## 🙏 **Acknowledgements**

We thank our **Break Through Tech AI Studio advisors**, teaching assistants, and peers for their guidance and feedback throughout this project.

---
