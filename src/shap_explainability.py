import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Tuple

def compute_shap_explainer(model, X_trans: np.ndarray, feature_names: list) -> Tuple[shap.TreeExplainer, shap.Explanation, np.ndarray]:
    """
    Computes SHAP TreeExplainer and SHAP values for tree-based models.
    """
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer(X_trans)
    
    # Handle multi-output or single array shap values
    if isinstance(shap_values_raw.values, list):
        shap_vals = shap_values_raw.values[1] # positive class
    elif len(shap_values_raw.values.shape) == 3:
        shap_vals = shap_values_raw.values[:, :, 1]
    else:
        shap_vals = shap_values_raw.values
        
    return explainer, shap_values_raw, shap_vals

def get_global_feature_importance(shap_vals: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Calculates mean absolute SHAP value for each feature."""
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    
    return df_imp

def plot_global_importance_plotly(df_imp: pd.DataFrame, top_n: int = 15):
    """Plotly horizontal bar plot for global SHAP feature importances."""
    df_top = df_imp.head(top_n).sort_values(by="Mean_Abs_SHAP", ascending=True)
    fig = px.bar(
        df_top,
        x="Mean_Abs_SHAP",
        y="Feature",
        orientation="h",
        title=f"Top {top_n} Global Features Driving Customer Churn (SHAP)",
        labels={"Mean_Abs_SHAP": "Impact on Model Output (|SHAP Value|)"},
        color="Mean_Abs_SHAP",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(height=500)
    return fig

def explain_single_customer(
    explainer: shap.TreeExplainer,
    shap_vals: np.ndarray,
    X_trans: np.ndarray,
    customer_idx: int,
    feature_names: list,
    raw_customer_row: pd.DataFrame = None
) -> Tuple[pd.DataFrame, float, float]:
    """
    Provides individual level feature contribution breakdown for a specific customer.
    Returns DataFrame of top positive and negative risk drivers, base probability, and final predicted probability.
    """
    sample_shap = shap_vals[customer_idx]
    sample_values = X_trans[customer_idx]
    
    base_val = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[1])
    
    df_single = pd.DataFrame({
        "Feature": feature_names,
        "FeatureValue": sample_values,
        "SHAP_Value": sample_shap
    }).sort_values(by="SHAP_Value", key=abs, ascending=False).reset_index(drop=True)
    
    # Calculate approx probability contribution
    total_shap_sum = float(np.sum(sample_shap))
    predicted_log_odds = base_val + total_shap_sum
    predicted_prob = 1 / (1 + np.exp(-predicted_log_odds))
    base_prob = 1 / (1 + np.exp(-base_val))
    
    return df_single, base_prob, predicted_prob

def plot_single_customer_waterfall_plotly(df_single: pd.DataFrame, top_n: int = 10, customer_id: str = "Target Customer"):
    """Plotly Waterfall chart representing individual customer SHAP drivers."""
    df_top = df_single.head(top_n).copy()
    
    # Sort by SHAP value
    df_top["Direction"] = df_top["SHAP_Value"].apply(lambda x: "Increases Churn Risk" if x > 0 else "Decreases Churn Risk")
    
    fig = px.bar(
        df_top.sort_values(by="SHAP_Value", ascending=True),
        x="SHAP_Value",
        y="Feature",
        orientation="h",
        color="Direction",
        color_discrete_map={
            "Increases Churn Risk": "#E74C3C",
            "Decreases Churn Risk": "#2ECC71"
        },
        title=f"Root-Cause Breakdown for {customer_id} (Top {top_n} Features)"
    )
    fig.update_layout(xaxis_title="SHAP Value (Contribution to Churn Probability)")
    return fig
