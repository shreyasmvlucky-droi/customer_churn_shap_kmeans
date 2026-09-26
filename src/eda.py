import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

def get_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates high-level statistical summaries of the dataset."""
    total_customers = len(df)
    churn_count = int(df["Churn"].sum())
    churn_rate = float(df["Churn"].mean())
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'CustomerID' in categorical_cols:
        categorical_cols.remove('CustomerID')
        
    return {
        "total_customers": total_customers,
        "churn_count": churn_count,
        "churn_rate": churn_rate,
        "numeric_summary": df[numeric_cols].describe().to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols
    }

def plot_churn_distribution(df: pd.DataFrame):
    """Returns a plotly Donut Chart of Churn Distribution."""
    churn_counts = df["Churn"].value_counts().reset_index()
    churn_counts.columns = ["Churn_Status", "Count"]
    churn_counts["Label"] = churn_counts["Churn_Status"].map({0: "Retained", 1: "Churned"})
    
    fig = px.pie(
        churn_counts,
        values="Count",
        names="Label",
        title="Overall Customer Churn Ratio",
        hole=0.4,
        color="Label",
        color_discrete_map={"Retained": "#2ECC71", "Churned": "#E74C3C"}
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_feature_vs_churn(df: pd.DataFrame, feature: str):
    """Plot feature vs churn depending on categorical or numeric data type."""
    if df[feature].dtype == 'object' or feature in ['SeniorCitizen', 'SatisfactionScore', 'SupportTicketsLast6M']:
        grouped = df.groupby([feature, 'Churn']).size().reset_index(name='Count')
        grouped['Churn_Label'] = grouped['Churn'].map({0: 'Retained', 1: 'Churned'})
        fig = px.bar(
            grouped,
            x=feature,
            y='Count',
            color='Churn_Label',
            barmode='group',
            title=f"Churn Distribution by {feature}",
            color_discrete_map={'Retained': '#2ECC71', 'Churned': '#E74C3C'}
        )
        return fig
    else:
        fig = px.box(
            df,
            x='Churn',
            y=feature,
            color='Churn',
            labels={'Churn': 'Churn Status (0 = Retained, 1 = Churned)'},
            title=f"Distribution of {feature} by Churn Status",
            color_discrete_map={0: '#2ECC71', 1: '#E74C3C'}
        )
        return fig

def plot_correlation_matrix(df: pd.DataFrame):
    """Generates correlation heatmap for numerical features."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Matrix"
    )
    return fig
