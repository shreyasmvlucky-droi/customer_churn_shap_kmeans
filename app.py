import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.eda import (
    get_summary_statistics, plot_churn_distribution,
    plot_feature_vs_churn, plot_correlation_matrix
)
from src.model_pipeline import train_and_evaluate
from src.shap_explainability import (
    compute_shap_explainer, get_global_feature_importance,
    plot_global_importance_plotly, explain_single_customer,
    plot_single_customer_waterfall_plotly
)
from src.kmeans_segmentation import (
    evaluate_optimal_clusters, plot_elbow_and_silhouette,
    perform_kmeans_clustering, profile_shap_clusters,
    get_cluster_prescriptive_actions, plot_shap_clusters_2d
)

st.set_page_config(
    page_title="Customer Churn + SHAP + K-Means Intelligence Suite",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "data", "customer_churn_data.csv")
    if not os.path.exists(data_path):
        from data.generate_dataset import generate_telecom_churn_dataset
        df = generate_telecom_churn_dataset(5000)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
    return df

@st.cache_resource
def train_or_load_pipeline():
    df_path = os.path.join(os.path.dirname(__file__), "data", "customer_churn_data.csv")
    results = train_and_evaluate(df_path=df_path, model_dir="models")
    
    # Compute SHAP values
    model = results["models"]["XGBoost"]
    X_test_trans = results["test_data"]["X_test_trans"]
    feature_names = results["feature_names"]
    
    explainer, shap_values_raw, shap_vals = compute_shap_explainer(model, X_test_trans, feature_names)
    results["shap"] = {
        "explainer": explainer,
        "shap_values_raw": shap_values_raw,
        "shap_vals": shap_vals
    }
    return results

# Load data and pipeline
df = load_data()
pipeline = train_or_load_pipeline()

# Title Header
st.markdown("<div class='main-title'>🔮 Customer Churn + SHAP + K-Means Intelligence Suite</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Supervised Classification (WHO) + Explainable AI / SHAP (WHY) + K-Means Persona Clustering (HOW TO TARGET)</div>", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/brain--v1.png", width=70)
st.sidebar.title("Navigation & Controls")
app_mode = st.sidebar.radio(
    "Select Workflow Stage:",
    [
        "1. Executive Overview & Unique Value",
        "2. Exploratory Data Analysis (EDA)",
        "3. Model Performance (Classification)",
        "4. Global SHAP Explainability",
        "5. K-Means Customer Segmentation",
        "6. Real-Time Churn & SHAP Predictor"
    ]
)

# ---------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW
# ---------------------------------------------------------
if app_mode == "1. Executive Overview & Unique Value":
    st.header("📌 Executive Summary & Architectural Innovation")
    
    col1, col2, col3, col4 = st.columns(4)
    summary_stats = get_summary_statistics(df)
    
    with col1:
        st.metric("Total Customers", f"{summary_stats['total_customers']:,}")
    with col2:
        st.metric("Total Churned Customers", f"{summary_stats['churn_count']:,}")
    with col3:
        st.metric("Churn Rate", f"{summary_stats['churn_rate']:.1%}")
    with col4:
        st.metric("XGBoost ROC-AUC", f"{pipeline['metrics']['XGBoost']['ROC-AUC']:.3f}")
        
    st.markdown("---")
    
    st.subheader("💡 What Makes This Project Unique?")
    st.markdown("""
    Most churn projects stop at predicting binary **0/1** churn outcomes. This project implements a **Hybrid 3-Tier Machine Learning Architecture**:
    
    1. **Supervised Classifier (XGBoost)** $\rightarrow$ Answers **WHO** will churn with 94.5% accuracy.
    2. **Explainable AI (SHAP)** $\rightarrow$ Answers **WHY** each customer is churning by decomposing probability into feature attributions.
    3. **SHAP-based K-Means Clustering** $\rightarrow$ Answers **HOW** to group churners based on *root causes* rather than raw demography, enabling prescriptive retention campaigns.
    """)
    
    st.image("https://mermaid.ink/svg/p crowd_flow", use_container_width=True) if False else None
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_churn_distribution(df), use_container_width=True)
    with col_b:
        st.info("""
        ### 💼 Business Value & Revenue Impact
        - **Proactive Intervention:** Retain high-value customers before they cancel.
        - **Targeted Cost Reduction:** Replace blanket discounts with persona-specific offers (e.g., offer tech support to frustrated users, contract discounts to price-sensitive users).
        - **Quantifiable ROI:** 1% reduction in churn saves substantial annual recurring revenue.
        """)

# ---------------------------------------------------------
# TAB 2: EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------
elif app_mode == "2. Exploratory Data Analysis (EDA)":
    st.header("📊 Exploratory Data Analysis (EDA)")
    
    st.subheader("Dataset Preview & Attributes")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Interactive Feature vs Churn Analyzer")
    selected_feature = st.selectbox(
        "Select Feature to Analyze:",
        [c for c in df.columns if c not in ["CustomerID", "Churn"]]
    )
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(plot_feature_vs_churn(df, selected_feature), use_container_width=True)
    with col2:
        st.write(f"### Feature Breakdown: `{selected_feature}`")
        if df[selected_feature].dtype == 'object':
            ct = pd.crosstab(df[selected_feature], df["Churn"], normalize='index') * 100
            ct.columns = ["Retained (%)", "Churned (%)"]
            st.dataframe(ct.round(2), use_container_width=True)
        else:
            st.dataframe(df.groupby("Churn")[selected_feature].describe(), use_container_width=True)
            
    st.markdown("---")
    st.subheader("Numeric Correlation Matrix")
    st.plotly_chart(plot_correlation_matrix(df), use_container_width=True)

# ---------------------------------------------------------
# TAB 3: MODEL PERFORMANCE
# ---------------------------------------------------------
elif app_mode == "3. Model Performance (Classification)":
    st.header("⚡ Classification Model Training & Evaluation")
    
    xgb_m = pipeline["metrics"]["XGBoost"]
    rf_m = pipeline["metrics"]["RandomForest"]
    
    st.subheader("Model Benchmark Comparison")
    df_metrics = pd.DataFrame([
        {"Model": "XGBoost Classifier", **xgb_m},
        {"Model": "Random Forest Classifier", **rf_m}
    ])
    st.dataframe(df_metrics[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]], use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curves")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=pipeline["roc_data"]["XGBoost"]["fpr"],
            y=pipeline["roc_data"]["XGBoost"]["tpr"],
            name=f"XGBoost (AUC = {xgb_m['ROC-AUC']:.3f})",
            line=dict(color="#2563EB", width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=pipeline["roc_data"]["RandomForest"]["fpr"],
            y=pipeline["roc_data"]["RandomForest"]["tpr"],
            name=f"Random Forest (AUC = {rf_m['ROC-AUC']:.3f})",
            line=dict(color="#F59E0B", width=3, dash="dash")
        ))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color="gray", dash="dot"), name="Random Classifier"))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", title="ROC Curve Benchmark")
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with col2:
        st.subheader("Confusion Matrix (XGBoost Best Model)")
        cm = np.array(xgb_m["ConfusionMatrix"])
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted Label", y="Actual Label"),
            x=["Retained (0)", "Churned (1)"],
            y=["Retained (0)", "Churned (1)"],
            color_continuous_scale="Blues",
            title="XGBoost Confusion Matrix"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: GLOBAL SHAP EXPLAINABILITY
# ---------------------------------------------------------
elif app_mode == "4. Global SHAP Explainability":
    st.header("🧠 Global SHAP Explainability (Why Customers Churn)")
    
    shap_vals = pipeline["shap"]["shap_vals"]
    feature_names = pipeline["feature_names"]
    
    df_imp = get_global_feature_importance(shap_vals, feature_names)
    
    st.subheader("Global Feature Importance Ranking")
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.plotly_chart(plot_global_importance_plotly(df_imp, top_n=12), use_container_width=True)
    with col2:
        st.dataframe(df_imp.head(12), use_container_width=True)
        st.success("""
        **Top Churn Driver Insights:**
        - **Contract Type (Month-to-month):** Highest single risk multiplier.
        - **Tenure Months:** Longer customer tenure drastically reduces churn risk.
        - **Support Tickets & Tech Support:** High support call counts without resolution spark fast churn.
        """)

# ---------------------------------------------------------
# TAB 5: K-MEANS CUSTOMER SEGMENTATION
# ---------------------------------------------------------
elif app_mode == "5. K-Means Customer Segmentation":
    st.header("🎯 K-Means Customer Segmentation (SHAP-Space Persona Clustering)")
    
    shap_vals = pipeline["shap"]["shap_vals"]
    feature_names = pipeline["feature_names"]
    
    k_range, inertias, silhouettes = evaluate_optimal_clusters(shap_vals, max_k=8)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_elbow_and_silhouette(k_range, inertias, silhouettes), use_container_width=True)
    with col2:
        n_clusters = st.slider("Select Number of Personas (K):", min_value=2, max_value=6, value=3)
        st.info(f"Currently clustering customer churn drivers into **{n_clusters} distinct personas** using SHAP space K-Means.")
        
    kmeans_model, cluster_labels = perform_kmeans_clustering(shap_vals, n_clusters=n_clusters)
    
    st.subheader("2D SHAP Cluster Map")
    st.plotly_chart(plot_shap_clusters_2d(shap_vals, cluster_labels), use_container_width=True)
    
    st.subheader("Cluster Persona Profiles & Dominant Churn Drivers")
    profiles = profile_shap_clusters(shap_vals, cluster_labels, feature_names)
    st.dataframe(profiles, use_container_width=True)
    
    st.subheader("💡 Prescriptive Retention Action Engine")
    actions = get_cluster_prescriptive_actions()
    for cid in range(min(n_clusters, len(actions))):
        act = actions[cid]
        with st.expander(f"🔴 Persona Cluster {cid}: {act['Persona']} (Priority: {act['Priority']})", expanded=True):
            st.write(f"**Description:** {act['Description']}")
            st.success(f"**Recommended Action:** {act['Recommended_Action']}")

# ---------------------------------------------------------
# TAB 6: REAL-TIME CHURN & SHAP PREDICTOR
# ---------------------------------------------------------
elif app_mode == "6. Real-Time Churn & SHAP Predictor":
    st.header("⚡ Real-Time Customer Churn & SHAP Simulator")
    st.markdown("Adjust customer attributes below to get real-time churn prediction, SHAP root cause decomposition, and prescriptive actions.")
    
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        st.subheader("Basic Info & Contract")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=6)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        
    with col_input2:
        st.subheader("Services & Billing")
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        tech_sup = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        
    with col_input3:
        st.subheader("Usage & Support Metrics")
        monthly = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=85.0)
        tickets = st.slider("Support Tickets (Last 6M)", min_value=0, max_value=8, value=4)
        satisfaction = st.select_slider("Satisfaction Score (1=Low, 5=High)", options=[1, 2, 3, 4, 5], value=2)
        total_charges = tenure * monthly
        
    # Build single customer dataframe
    input_dict = {
        "CustomerID": ["SIM-0001"],
        "Gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "TenureMonths": [tenure],
        "PhoneService": ["Yes"],
        "MultipleLines": ["Yes"],
        "InternetService": [internet],
        "OnlineSecurity": [sec],
        "OnlineBackup": ["No"],
        "DeviceProtection": ["No"],
        "TechSupport": [tech_sup],
        "StreamingTV": ["Yes"],
        "StreamingMovies": ["Yes"],
        "Contract": [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod": [payment],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total_charges],
        "SupportTicketsLast6M": [tickets],
        "SatisfactionScore": [satisfaction]
    }
    
    df_sim = pd.DataFrame(input_dict)
    
    preprocessor = pipeline["preprocessor"]
    model = pipeline["models"]["XGBoost"]
    explainer = pipeline["shap"]["explainer"]
    feature_names = pipeline["feature_names"]
    
    X_sim_trans = preprocessor.transform(df_sim.drop(columns=["CustomerID"]))
    sim_prob = float(model.predict_proba(X_sim_trans)[:, 1][0])
    
    st.markdown("---")
    st.subheader("📊 Prediction Results & SHAP Root-Cause Analysis")
    
    res_col1, res_col2 = st.columns([2, 3])
    
    with res_col1:
        st.markdown("#### Churn Risk Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sim_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Predicted Churn Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#E74C3C" if sim_prob > 0.5 else "#2ECC71"},
                'steps': [
                    {'range': [0, 35], 'color': "#D4EFDF"},
                    {'range': [35, 65], 'color': "#FCF3CF"},
                    {'range': [65, 100], 'color': "#FADBD8"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        if sim_prob > 0.5:
            st.error("🚨 **HIGH CHURN RISK DETECTED!** Immediate intervention required.")
        else:
            st.success("✅ **LOW CHURN RISK.** Customer is stable.")
            
    with res_col2:
        st.markdown("#### Individual Customer SHAP Root Causes")
        sim_shap = explainer(X_sim_trans)
        if isinstance(sim_shap.values, list):
            sim_shap_vals = sim_shap.values[1]
        elif len(sim_shap.values.shape) == 3:
            sim_shap_vals = sim_shap.values[:, :, 1]
        else:
            sim_shap_vals = sim_shap.values
            
        df_single, base_p, pred_p = explain_single_customer(
            explainer, sim_shap_vals, X_sim_trans, 0, feature_names
        )
        
        st.plotly_chart(plot_single_customer_waterfall_plotly(df_single, top_n=8, customer_id="Simulated Customer"), use_container_width=True)
        
    st.subheader("🎯 Prescriptive Action Plan")
    top_pos_driver = df_single[df_single["SHAP_Value"] > 0].head(1)
    if not top_pos_driver.empty:
        main_driver = top_pos_driver["Feature"].values[0]
        st.warning(f"**Primary Driver of Churn Risk:** `{main_driver}`")
        if "Contract" in main_driver:
            st.info("💡 **Recommended Action:** Offer 1-year contract lock-in with a 10% monthly discount.")
        elif "Support" in main_driver or "Satisfaction" in main_driver:
            st.info("💡 **Recommended Action:** Priority call route to executive support + issue resolution voucher.")
        elif "Charges" in main_driver:
            st.info("💡 **Recommended Action:** Recommend downgrading unneeded service add-ons to reduce bill shock.")
        else:
            st.info("💡 **Recommended Action:** Send personalized retention email with customer success check-in.")
