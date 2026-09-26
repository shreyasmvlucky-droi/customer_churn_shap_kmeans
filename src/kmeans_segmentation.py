import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Tuple

def evaluate_optimal_clusters(data_matrix: np.ndarray, max_k: int = 8) -> Tuple[list, list, list]:
    """Computes Inertia and Silhouette scores across K=2..max_k."""
    k_range = list(range(2, max_k + 1))
    inertias = []
    silhouettes = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_matrix)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(data_matrix, labels))
        
    return k_range, inertias, silhouettes

def plot_elbow_and_silhouette(k_range: list, inertias: list, silhouettes: list):
    """Plotly chart showing Elbow Curve and Silhouette Scores side-by-side."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=k_range, y=inertias, mode='lines+markers', name='Inertia (Elbow)',
        line=dict(color='#3498DB', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=k_range, y=silhouettes, mode='lines+markers', name='Silhouette Score',
        yaxis='y2', line=dict(color='#2ECC71', width=3, dash='dash')
    ))
    
    fig.update_layout(
        title="Optimal Cluster Evaluation (Elbow & Silhouette)",
        xaxis=dict(title="Number of Clusters (K)"),
        yaxis=dict(title="Inertia"),
        yaxis2=dict(
            title="Silhouette Score",
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.6, y=0.95)
    )
    return fig

def perform_kmeans_clustering(data_matrix: np.ndarray, n_clusters: int = 3) -> Tuple[KMeans, np.ndarray]:
    """Fits KMeans model and returns cluster labels."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_matrix)
    return kmeans, labels

def profile_shap_clusters(
    shap_vals: np.ndarray,
    cluster_labels: np.ndarray,
    feature_names: list,
    raw_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Profiles clusters based on mean SHAP values per feature per cluster.
    Identifies top positive churn drivers for each persona cluster.
    """
    df_shap = pd.DataFrame(shap_vals, columns=feature_names)
    df_shap["Cluster"] = cluster_labels
    
    cluster_means = df_shap.groupby("Cluster").mean()
    
    profiles = []
    for cluster_id in range(len(np.unique(cluster_labels))):
        c_row = cluster_means.loc[cluster_id]
        top_drivers = c_row.sort_values(ascending=False).head(3)
        top_driver_names = [f"{feat} ({val:+.2f})" for feat, val in top_drivers.items()]
        
        count = (cluster_labels == cluster_id).sum()
        pct = count / len(cluster_labels)
        
        profiles.append({
            "Cluster_ID": f"Persona Cluster {cluster_id}",
            "Customer_Count": count,
            "Percentage": f"{pct:.1%}",
            "Primary_Churn_Driver_1": top_driver_names[0],
            "Primary_Churn_Driver_2": top_driver_names[1],
            "Primary_Churn_Driver_3": top_driver_names[2]
        })
        
    return pd.DataFrame(profiles)

def get_cluster_prescriptive_actions() -> Dict[int, Dict[str, str]]:
    """Returns business strategy map for churn persona clusters."""
    return {
        0: {
            "Persona": "Price-Sensitive High-Tier Customers",
            "Description": "High monthly charges combined with month-to-month contracts and paperless billing.",
            "Recommended_Action": "Offer 15% discount on annual plan upgrade, targeted bundle deals, or loyalty rewards.",
            "Priority": "HIGH"
        },
        1: {
            "Persona": "Low-Support / Onboarding At-Risk Customers",
            "Description": "Short tenure, no tech support, no online security, high support call friction.",
            "Recommended_Action": "Deploy proactive customer success onboarding agent, offer 3 months free Premium Tech Support.",
            "Priority": "URGENT"
        },
        2: {
            "Persona": "Service Friction & High-Ticket Churners",
            "Description": "Multiple unresolved support tickets, low satisfaction score, fiber optic speed issues.",
            "Recommended_Action": "Immediate VIP tier escalation, dedicated account manager check-in, service credit refund.",
            "Priority": "CRITICAL"
        }
    }

def plot_shap_clusters_2d(shap_vals: np.ndarray, cluster_labels: np.ndarray):
    """PCA/2D projection plot of SHAP clusters for visualization."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(shap_vals)
    
    df_pca = pd.DataFrame({
        "PCA1": coords[:, 0],
        "PCA2": coords[:, 1],
        "Cluster": [f"Cluster {c}" for c in cluster_labels]
    })
    
    fig = px.scatter(
        df_pca,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        title="2D SHAP Cluster Map (Customer Segments by Churn Drivers)",
        labels={"PCA1": "SHAP Principal Component 1", "PCA2": "SHAP Principal Component 2"},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    return fig
