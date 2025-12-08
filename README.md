# GPU Cluster CPU Predictor

Machine learning project to predict CPU usage in a cloud GPU cluster environment using real-world trace data from Alibaba GPU-2020. The goal is to support capacity planning and autoscaling decisions by accurately forecasting CPU utilization from system metrics and machine specifications. [web:24]

---

## 📌 Project Overview

Modern cloud platforms must continuously balance performance, reliability, and cost. One key challenge is predicting CPU usage ahead of time to provision just enough resources without over- or under-provisioning.

In this project, we:

- Use the **Alibaba GPU-2020 cluster trace** as a real-world dataset of production workloads. [web:24]
- Formulate CPU usage prediction as a **supervised regression** task.
- Build and compare several models (from baselines to ensembles).
- Deliver a **LaTeX report** and **clean Python notebook** suitable for academic grading and portfolio review.

---

## 🧠 Problem Statement

**Objective**  
Predict the CPU usage (`cpu_usage`) of a machine at a given time based on:

- Sensor metrics (GPU usage, memory usage, I/O operations).
- Machine hardware specifications (CPU, RAM, GPU capacity).
- Encoded categorical identifiers (GPU type, machine ID).

**Business impact**

Accurate CPU prediction enables:

- Smarter **capacity planning**.
- Proactive **autoscaling**.
- Better **cost optimization** and **quality of service** in cloud environments.

---

## 📂 Dataset

- Source: Alibaba GPU-2020 cluster trace (via Kaggle). [web:24]  
- Link: https://www.kaggle.com/datasets/derrickmwiti/cluster-trace-gpu-v2020  
- Size (after cleaning and merge):  
  - ~908k rows  
  - 15 features used for modeling

Key feature groups:

- **Sensors**: `gpu_wrk_util`, `avg_mem`, `max_mem`, `avg_gpu_wrk_mem`, `max_gpu_wrk_mem`, `read`, `write`, `read_count`, `write_count`
- **Machine specs**: `cap_cpu`, `cap_mem`, `cap_gpu`
- **Categoricals (encoded)**: `gpu_name_encoded`, `gpu_type_encoded`, `machine_encoded`

Target variable: `cpu_usage` (continuous).

---

## 🛠️ Tech Stack

- **Language**: Python (Jupyter Notebook)
- **Core libraries**:
  - `pandas`, `numpy` for data handling
  - `matplotlib`, `seaborn` for visualization
  - `scikit-learn` for classical ML models and pipelines
  - `xgboost` for gradient boosting [web:30]

- **Report**: Full project report written in **LaTeX**.

---

## 🔍 Modeling Approach

1. **Data preparation**
   - Merge sensor and machine specification tables.
   - Handle missing values (e.g., GPU metrics set to 0 when not used).
   - Remove rows with missing target or invalid values (inf/NaN).
   - Split into train/test (80/20, `random_state=42`).
   - Build a preprocessing pipeline:
     - `StandardScaler` on numerical features.
     - Pass-through for encoded categorical features. [web:30]

2. **Baselines**
   - `LinearRegression`
   - `DecisionTreeRegressor` (depth-limited)

3. **Model improvements**
   - K-Fold cross-validation (k=5) on the decision tree to assess stability.
   - Advanced models:
     - **Random Forest**
     - **XGBoost**
     - Bagging / tree ensembles

4. **Metrics**
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)
   - R² (coefficient of determination)

---

## 📊 Key Results

On the held-out test set:

| Model                     | MAE    | RMSE   | R²    |
|--------------------------|--------|--------|-------|
| Random Forest (best)     | 45.59  | 193.52 | 0.882 |
| XGBoost                  | 65.21  | 207.47 | 0.864 |
| Bagging Tree             | 78.43  | 296.15 | 0.723 |
| Decision Tree (depth=12) | 96.32  | 317.88 | 0.681 |
| Linear Regression        | 222.85 | 537.79 | 0.086 |

Highlights:

- Tree-based and ensemble methods clearly outperform linear regression, confirming the **non-linear** nature of the problem.
- Random Forest achieves strong performance with good stability and reasonable training cost.
- K-Fold validation shows low variance across folds, indicating **robust generalization**.

---

## ⚠️ Challenges Faced

- **Dataset size**: ~900k rows made some models and exhaustive grid searches too slow to run end-to-end on typical hardware and Google Colab.
- **Computation time**:
  - Some models (e.g., SVR, large ensembles) were too costly to train on the full dataset.
  - Mitigation: random subsampling (e.g., 300k rows) for cross-validation and heavy tuning loops.
- **Infrastructure limits**:
  - Long-running jobs hitting session limits.
  - Need to balance performance vs. practicality when selecting models and hyperparameters.

These constraints guided us towards **scalable
