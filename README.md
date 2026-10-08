# 💼 Employee Performance Prediction & HR Analytics Platform

An enterprise-grade **Machine Learning & HR Analytics Web Application** built with **Python, Scikit-Learn, and Streamlit**. It enables HR leaders, talent managers, and executives to forecast employee performance tiers, benchmark ML models, detect productivity drivers via Explainable AI, and perform batch talent audits.

---

## 🚀 Live Demo & Quick Start

```bash
# 1. Clone repository
git clone <your-repo-url>
cd employee-performance-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit Application
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## 🌟 Key Features

### 1. 👤 Individual Employee Performance Forecasting
- Interactive inputs for 20+ organizational, demographic, and behavioral attributes (e.g., Department, Job Level, Attendance Rate, Training Hours, Previous Performance, Overtime).
- Real-time ML inference producing:
  - **Performance Tier Badge:** `HIGH`, `MEDIUM`, or `LOW`.
  - **Continuous Score:** `0.0 - 100.0`.
  - **Confidence Metric & Probability Breakdown:** (e.g., `92.4%`).
  - **Actionable HR Recommendations:** Automated talent guidance.

### 2. 🔍 Explainable AI (XAI) - "Why this prediction?"
- Transparent factor breakdown illustrating top positive and negative drivers influencing an individual employee's rating (e.g., `+ High Impact` on Attendance, `- Medium Impact` on Overtime Burnout).
- Global feature importance ranking calculated directly from the trained ML models.

### 3. 📁 Dataset Explorer & In-App Model Retraining
- Upload custom HR `.csv` datasets with automatic schema validation and statistical summaries.
- Train & compare three state-of-the-art machine learning algorithms on the fly:
  - **Random Forest Classifier**
  - **Gradient Boosting Classifier**
  - **Logistic Regression**
- Automated benchmarking comparing **Accuracy, Precision, Recall, F1-Score**, and **Confusion Matrices**.
- Automatically selects and saves the top-performing algorithm.

### 4. ⚡ Batch Prediction & CSV Export
- Process thousands of employee records simultaneously through the preprocessing & inference pipeline.
- Interactive results table with instant search and filtering.
- One-click export of predictions to `employee_predictions.csv`.

### 5. 📊 Executive HR Analytics Dashboard
- KPI summaries: Total Headcount, Average Score, Top Performers count, Attendance Rate, Training Volume.
- Interactive Plotly visualizations:
  - Performance score distribution & density curves.
  - Department-wise performance comparisons.
  - Experience vs. Performance correlation with OLS trendlines.
  - Attendance & Training Hours impact on productivity.
  - Job Satisfaction vs. Performance box plots.
  - Compensation (Monthly Income) distribution.

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend & UI** | [Streamlit](https://streamlit.io/) | Interactive web interface with responsive layouts, metric cards, and widgets |
| **Machine Learning** | [Scikit-Learn](https://scikit-learn.org/) | Preprocessing pipelines, Random Forest, Gradient Boosting, Logistic Regression |
| **Data Processing** | [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) | Vectorized data manipulation, schema validation, statistical modeling |
| **Data Visualization** | [Plotly](https://plotly.com/python/) | High-performance interactive charts, heatmaps, and probability graphs |
| **Model Persistence** | [Joblib](https://joblib.readthedocs.io/) | Serialization of trained ML pipelines and metadata |

---

## 🧠 Machine Learning Architecture & Workflow

```text
Raw Employee Data (CSV / Form Input)
             │
             ▼
┌────────────────────────────────────────┐
│      Preprocessing & Feature Pipeline   │
│  ├─ Median Imputation & StandardScaler │ (Numerical features)
│  └─ Frequent Imputer & OneHotEncoder  │ (Categorical features)
└────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│          Model Benchmarking            │
│  ├─ Random Forest Classifier           │
│  ├─ Gradient Boosting Classifier       │
│  └─ Logistic Regression                │
└────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│       Evaluation & Auto-Selection      │
│  ├─ Accuracy, Precision, Recall, F1    │
│  └─ Confusion Matrix Heatmap           │
└────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│     Inference & Explainability Engine  │
│  ├─ Categorical Tier (High/Med/Low)    │
│  ├─ Calibrated Score (0 - 100)         │
│  ├─ Confidence Probability             │
│  └─ Local & Global Feature Impact      │
└────────────────────────────────────────┘
```

### Centralized Performance Thresholds
- 🟢 **High Performer:** `80.0 – 100.0`
- 🟡 **Medium Performer:** `60.0 – 79.9`
- 🔴 **Low Performer:** `0.0 – 59.9`

---

## 📂 Project Directory Structure

```text
employee-performance-prediction/
│
├── app.py                     # Streamlit application entry point
├── requirements.txt           # Python dependencies for local & cloud deploy
├── README.md                  # Complete documentation and deployment guide
│
├── data/
│   └── sample_employee_data.csv # Synthetic demo dataset (1,200 records)
│
├── models/
│   └── employee_performance_model.pkl # Serialized best ML pipeline
│
├── src/
│   ├── config.py              # Centralized thresholds, schema, and constants
│   ├── preprocessing.py       # Scikit-learn ColumnTransformer & data validators
│   ├── train_model.py         # Model training, benchmarking, and dataset generation
│   ├── prediction.py          # Single/batch inference & explainability logic
│   └── visualization.py       # Interactive Plotly charts & custom themes
│
└── .streamlit/
    └── config.toml            # Streamlit theme and server configuration
```

---

## ☁️ Deployment Guide

### Deploying to Streamlit Community Cloud (Recommended)

1. **Push code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "feat: complete employee performance prediction streamlit application"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```

2. **Connect to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
   - Click **"New app"**.
   - Select your repository, branch (`main`), and set Main file path to:
     ```text
     app.py
     ```
   - Click **"Deploy!"**.
   - Your app will be live with a public HTTPS URL.

---

## 🎓 Viva / Project Defense Questions & Answers

<details>
<summary><b>1. Why was a ColumnTransformer pipeline used?</b></summary>
A <code>ColumnTransformer</code> encapsulates numerical scaling (StandardScaler) and categorical encoding (OneHotEncoder) into a single object. This guarantees zero data leakage during training and test splits, and guarantees that single-row user form inputs receive the exact same mathematical transformations as the training dataset.
</details>

<details>
<summary><b>2. How does the model compare different algorithms?</b></summary>
The system simultaneously trains Random Forest, Gradient Boosting, and Logistic Regression on identical stratified splits. It evaluates weighted F1-score and accuracy, generating interactive confusion matrices before deploying the superior model.
</details>

<details>
<summary><b>3. How are Explainable AI factors generated?</b></summary>
Explanations calculate the directional delta between an employee's feature values and benchmark distributions, weighted by the model's feature importance vectors.
</details>

---

## 📄 License
MIT License. Built for educational, demonstration, and HR analytics purposes.
