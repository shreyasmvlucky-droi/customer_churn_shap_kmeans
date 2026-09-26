# 🔮 Customer Churn Prediction with SHAP Explainability & K-Means Segmentation

An end-to-end Enterprise Machine Learning system combining **Supervised Learning (XGBoost)**, **Explainable AI (SHAP)**, and **Unsupervised Learning (K-Means)** with an interactive **Streamlit Web Dashboard**.

---

## 🌟 What Makes This Project Unique?

Most machine learning churn projects only perform binary classification (predicting whether a customer will churn or not). This project builds a **3-Tier Predictive & Prescriptive Pipeline**:

1. **Supervised Classification (XGBoost & Random Forest):**
   - Answers **WHO** will churn (94.5% Accuracy, 0.986 ROC-AUC).
2. **Explainable AI (SHAP - SHapley Additive exPlanations):**
   - Answers **WHY** a specific customer is churning using game-theoretic feature attributions at both global and individual levels.
3. **SHAP-Space K-Means Persona Clustering:**
   - Innovation: Instead of clustering on raw demographic features, this project clusters customers in **SHAP feature space** (clustering by *churn drivers*).
   - Groups churners into actionable business personas (e.g., *Price-Sensitive*, *Support-Frustrated*, *Onboarding At-Risk*).
4. **Prescriptive Action Engine:**
   - Translates model insights directly into targeted business retention actions (e.g., targeted discounts, VIP support routing, contract lock-ins).

---

## 📐 Project Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    Customer Data Generation                 │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Exploratory Data Analysis (EDA)             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│            Feature Preprocessing (ColumnTransformer)        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Classification (XGBoost & RandomForest)         │
│                     [Predicts WHO Churns]                   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               Explainable AI (SHAP TreeExplainer)           │
│                    [Explains WHY They Churn]                │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             K-Means Persona Clustering (SHAP-Space)         │
│                 [Defines HOW to Target Retentions]          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Streamlit Web Dashboard (App)               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset & Train Pipeline
```bash
python data/generate_dataset.py
python src/model_pipeline.py
```

### 3. Launch Streamlit Web Application
```bash
streamlit run app.py
```

---

## 📁 Repository Directory Structure

```
customer_churn_shap_kmeans/
├── data/
│   ├── generate_dataset.py       # Realistic synthetic churn data generator
│   └── customer_churn_data.csv   # Generated CSV dataset (5,000 samples)
├── src/
│   ├── eda.py                    # Summary statistics & Plotly chart generators
│   ├── model_pipeline.py         # Sklearn & XGBoost pipeline, metrics, ROC
│   ├── shap_explainability.py    # SHAP TreeExplainer, global & local force breakdown
│   └── kmeans_segmentation.py    # Elbow, Silhouette, K-Means & persona profiler
├── models/                       # Saved trained models & transformers
├── app.py                        # Multi-tab Streamlit dashboard application
├── requirements.txt              # Environment dependencies
└── README.md                     # Project documentation & interview guide
```

---

## 🎙️ Interview Speaking Points Guide

When presenting this project in a data science or machine learning interview, emphasize the following key talking points:

1. **Pandas / NumPy Data Processing:** Handled complex data types, scaling, encoding categorical variables using scikit-learn's `ColumnTransformer`.
2. **Exploratory Data Analysis (EDA):** Identified core correlations such as tenure, contract duration, support calls, and monthly charges impacting customer attrition.
3. **Classification Modeling:** Benchmark comparison between **XGBoost** and **Random Forest**. Handled class imbalance using stratifying, achieving 94.5% accuracy and 0.986 ROC-AUC.
4. **Model Evaluation:** Utilized Precision-Recall trade-offs, Confusion Matrices, and ROC curves rather than relying solely on raw accuracy.
5. **SHAP Explainability:** Utilized `shap.TreeExplainer` to overcome the "black-box" nature of ensemble models, allowing business stakeholders to audit model decisions.
6. **K-Means Clustering:** Applied Elbow method and Silhouette scores for $K$ selection. Clustered on SHAP feature space to group customers by churn root cause.
7. **Business Interpretation & Prescriptive Analytics:** Built an automated decision engine linking cluster personas to specific high-ROI retention offers.
8. **Streamlit Deployment:** Designed an intuitive dashboard for business users, features team members, and executive decision-makers.
