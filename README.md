# Predictive Maintenance for Industrial Pumps

## Overview
This notebook demonstrates a comprehensive approach to building a predictive maintenance system for industrial pumps using time-series sensor data. The goal is to predict pump failures within a 24-hour horizon, enabling proactive maintenance and minimizing downtime. The project covers data loading, extensive cleaning, exploratory data analysis (EDA), advanced feature engineering, and machine learning model training and evaluation.

## Dataset
The dataset contains time-series sensor readings from multiple industrial pumps, including:
-   `timestamp`: Date and time of the reading.
-   `pump_throughput_m3ph`: Fluid throughput (cubic meters per hour).
-   `operating_pressure_bar`: Operating pressure (bar).
-   `vibration_mm_s`: Vibration level (millimeters per second).
-   `bearing_temp_C`: Bearing temperature (Celsius).
-   `status`: Operating status (`RUNNING` or `DOWN`).
-   `pump_number`: Unique identifier for each pump.

The dataset comprises over 720,000 records across 50 pumps, with a logging interval of 10 minutes.

## Data Cleaning and Preparation
**Key Steps:**
1.  **Timestamp Conversion**: Converted `timestamp` to datetime format and handled one missing entry.
2.  **Status Standardization**: Standardized the `status` column to `RUNNING` or `DOWN` to ensure consistency.
3.  **Missing Values & Duplicates**: Checked for and confirmed no significant missing values or duplicate rows (`(timestamp, pump_number)`).
4.  **Outlier Detection**: Identified outliers in sensor readings (throughput, pressure, vibration, temperature) using the IQR method per pump. Noted that a significant portion of outliers occur during `DOWN` events, as expected.
5.  **Time Series Consistency**: Verified consistent 10-minute logging intervals and sorted data chronologically per pump.

## Exploratory Data Analysis (EDA)
**Key Findings:**
-   **Class Imbalance**: The dataset is highly imbalanced, with approximately 93.3% `RUNNING` states and 6.7% `DOWN` states.
-   **Sensor Distributions**: Clear distinctions in sensor values between `RUNNING` and `DOWN` states:
    -   `RUNNING`: Throughput (99.88 m³/h), Pressure (9.79 bar), Vibration (1.33 mm/s), Temperature (66.68 °C).
    -   `DOWN`: All operational sensors (throughput, pressure, vibration) drop to ~0, while bearing temperature cools to ambient levels (~25 °C).
-   **Temporal Patterns**: Pump failures show recurring patterns, with spikes in `DOWN` events over time. Failure events tend to peak between 2 AM and 5 AM, particularly at 4 AM, and are fairly consistent across days of the week, with slightly fewer on Sundays.
-   **Failure Duration**: Most pump failures last between 13-18 hours, with a peak around 15-16 hours, indicating a consistent repair/recovery time.
-   **Outlier-Failure Correlation**: System-wide outlier rates closely align with `DOWN` events, with the outlier fraction steadily rising from 0% at 24 hours pre-failure to nearly 100% at failure, suggesting strong predictive signal.
-   **Sensor Correlations**: Strong correlation between `throughput` and `operating_pressure` (0.97). `Vibration` and `bearing_temp_C` also show moderate correlations with other sensors, providing complementary information.

## Feature Engineering
To enhance prediction capabilities, a rich set of features was engineered:
-   **Rolling Statistics**: Mean, standard deviation, and range for all sensor columns over 1-hour, 6-hour, and 12-hour windows (lagged by one time step to prevent data leakage).
-   **Slopes**: Short-term and medium-term trends (1-hour and 6-hour slopes) for sensor readings, indicating rates of change.
-   **Robust Z-Scores**: Normalized sensor values based on per-pump median and Median Absolute Deviation (MAD) from the training set, to identify deviations from normal operating conditions.
-   **Outlier Flags and Rates**: Flags for individual sensor outliers and a rolling 1-hour rate of `any_outlier` events, indicating increasing anomaly prevalence.
-   **Temporal Features**: Sine and cosine transformations of `hour` and `dayofweek` to capture cyclical patterns.
-   **Interaction Ratios**: Combined sensor signals (e.g., `temp_vib_ratio`, `throughput_vib_ratio`, `pressure_throughput_product`) to capture complex relationships.
-   **Target Variable**: `y_failure_24h`, a binary label indicating whether a pump will fail within the next 24 hours.

## Model Training and Evaluation
The data was split chronologically into training (70%) and testing (30%) sets to ensure no future data was used for training.

**Models Explored:**
-   **Logistic Regression (Baseline)**: Established a baseline performance with `ROC-AUC: 0.8192` and `PR-AUC: 0.3742`. Achieved a recall of ~0.72 for failures at a default threshold, and ~0.80 recall when tuning the threshold for higher sensitivity.
-   **Random Forest Classifier**: Showed improved performance over Logistic Regression with `ROC-AUC: 0.8459` and `PR-AUC: 0.3856` (from full data run).
-   **XGBoost Classifier**: Achieved the best performance among the single models with `ROC-AUC: 0.8622` and `PR-AUC: 0.4016` (using a tuned threshold, recall was 0.8192).

**Model Comparison (subsampled data for quick iteration):**
| Model                  | Fit Time (s) | PR-AUC   | ROC-AUC  | Accuracy | F1 Score | Precision | Recall   |
|------------------------|--------------|----------|----------|----------|----------|-----------|----------|
| Linear SVC (scaled)    | 2.08         | **0.3966** | 0.8604   | 0.7813   | 0.4124   | 0.2827    | 0.7623   |
| Extra Trees            | 4.88         | 0.3966   | **0.8631** | 0.7457   | 0.3981   | 0.2613    | **0.8355** |
| Logistic Regression    | 2.12         | 0.3964   | 0.8600   | 0.7861   | 0.4149   | 0.2863    | 0.7531   |
| Gradient Boosting      | 17.14        | 0.3897   | 0.8588   | 0.8347   | 0.4353   | 0.3317    | 0.6330   |
| Random Forest          | 17.14        | 0.3856   | 0.8594   | 0.7667   | 0.4067   | 0.2733    | 0.7945   |
| Hist Gradient Boosting | 3.54         | 0.3769   | 0.8582   | 0.8377   | 0.4418   | 0.3378    | 0.6381   |
| Gaussian NB            | 0.29         | 0.1313   | 0.6278   | 0.2192   | 0.2047   | 0.1140    | 0.9982   |

*Note: Metrics for the model comparison table were generated using a subsampled training set for faster iteration. Full data training (e.g., for XGBoost) yielded higher scores.* 

**Cross-Validation**: Group K-Fold cross-validation (by `pump_number`) was performed to ensure robustness against pump-specific biases.

## Visualization of Predictions
Visualizations of predicted failure probabilities against actual failures for individual pumps highlight the model's ability to identify impending failures, often showing a rising probability trend before the actual `DOWN` event.

## Conclusion
The project successfully demonstrates that advanced feature engineering combined with robust machine learning models like XGBoost can effectively predict pump failures within a 24-hour window. The high recall achieved is critical for predictive maintenance applications, where minimizing false negatives (missed failures) is paramount.

## Setup and Usage
This notebook is designed to run in a Google Colab environment. 

**Dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost numba
```

**How to Run:**
1.  Open the notebook in Google Colab.
2.  Ensure you have access to the Google Sheet dataset (the `sheet_id` is provided in the notebook).
3.  Run all cells sequentially to reproduce the analysis and model training.
