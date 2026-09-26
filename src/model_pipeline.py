import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

def prepare_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    """Identify numeric and categorical features and build ColumnTransformer."""
    X = df.drop(columns=["CustomerID", "Churn"])
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )
    
    return preprocessor, numeric_features, categorical_features

def get_feature_names(preprocessor: ColumnTransformer, numeric_features: list, categorical_features: list) -> list:
    """Extract encoded feature names after ColumnTransformer fit."""
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_cols = cat_encoder.get_feature_names_out(categorical_features).tolist()
    return numeric_features + encoded_cat_cols

def train_and_evaluate(df_path: str = "data/customer_churn_data.csv", model_dir: str = "models") -> Dict[str, Any]:
    """Train XGBoost & Random Forest models, evaluate, and save best model."""
    df = pd.read_csv(df_path)
    X = df.drop(columns=["CustomerID", "Churn"])
    y = df["Churn"]
    
    preprocessor, numeric_features, categorical_features = prepare_preprocessor(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fit preprocessor on training data
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)
    
    # Train XGBoost
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train_trans, y_train)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42
    )
    rf_model.fit(X_train_trans, y_train)
    
    # Predictions
    xgb_preds = xgb_model.predict(X_test_trans)
    xgb_probs = xgb_model.predict_proba(X_test_trans)[:, 1]
    
    rf_preds = rf_model.predict(X_test_trans)
    rf_probs = rf_model.predict_proba(X_test_trans)[:, 1]
    
    def calc_metrics(y_true, preds, probs):
        return {
            "Accuracy": accuracy_score(y_true, preds),
            "Precision": precision_score(y_true, preds),
            "Recall": recall_score(y_true, preds),
            "F1-Score": f1_score(y_true, preds),
            "ROC-AUC": roc_auc_score(y_true, probs),
            "ConfusionMatrix": confusion_matrix(y_true, preds).tolist()
        }
        
    xgb_metrics = calc_metrics(y_test, xgb_preds, xgb_probs)
    rf_metrics = calc_metrics(y_test, rf_preds, rf_probs)
    
    # Save artifacts
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(xgb_model, os.path.join(model_dir, "xgb_model.joblib"))
    joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.joblib"))
    joblib.dump(feature_names, os.path.join(model_dir, "feature_names.joblib"))
    
    # Calculate ROC Curves
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
    
    results = {
        "models": {
            "XGBoost": xgb_model,
            "RandomForest": rf_model
        },
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "metrics": {
            "XGBoost": xgb_metrics,
            "RandomForest": rf_metrics
        },
        "roc_data": {
            "XGBoost": {"fpr": xgb_fpr.tolist(), "tpr": xgb_tpr.tolist()},
            "RandomForest": {"fpr": rf_fpr.tolist(), "tpr": rf_tpr.tolist()}
        },
        "test_data": {
            "X_test": X_test,
            "X_test_trans": X_test_trans,
            "y_test": y_test,
            "y_probs": xgb_probs
        }
    }
    
    return results

if __name__ == "__main__":
    res = train_and_evaluate(
        df_path="data/customer_churn_data.csv",
        model_dir="models"
    )
    print("Model Training Completed Successfully!")
    print(f"XGBoost Metrics: {res['metrics']['XGBoost']}")
    print(f"RandomForest Metrics: {res['metrics']['RandomForest']}")
