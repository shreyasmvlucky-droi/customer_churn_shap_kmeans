import os
import numpy as np
import pandas as pd

def generate_telecom_churn_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic Telco Customer Churn dataset with correlated feature patterns.
    """
    np.random.seed(random_state)
    
    customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]
    gender = np.random.choice(["Male", "Female"], size=n_samples)
    senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], size=n_samples, p=[0.30, 0.70])
    
    # Tenure months (1 to 72)
    tenure = np.random.exponential(scale=25, size=n_samples).clip(1, 72).astype(int)
    
    phone_service = np.random.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
            
    internet_service = np.random.choice(["Fiber optic", "DSL", "No"], size=n_samples, p=[0.44, 0.34, 0.22])
    
    online_security = []
    online_backup = []
    device_protection = []
    tech_support = []
    streaming_tv = []
    streaming_movies = []
    
    for net in internet_service:
        if net == "No":
            online_security.append("No internet service")
            online_backup.append("No internet service")
            device_protection.append("No internet service")
            tech_support.append("No internet service")
            streaming_tv.append("No internet service")
            streaming_movies.append("No internet service")
        else:
            online_security.append(np.random.choice(["Yes", "No"], p=[0.35, 0.65]))
            online_backup.append(np.random.choice(["Yes", "No"], p=[0.40, 0.60]))
            device_protection.append(np.random.choice(["Yes", "No"], p=[0.38, 0.62]))
            tech_support.append(np.random.choice(["Yes", "No"], p=[0.32, 0.68]))
            streaming_tv.append(np.random.choice(["Yes", "No"], p=[0.49, 0.51]))
            streaming_movies.append(np.random.choice(["Yes", "No"], p=[0.50, 0.50]))
            
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.23, 0.22])
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])
    payment_method = np.random.choice([
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ], size=n_samples, p=[0.34, 0.22, 0.22, 0.22])
    
    # Monthly Charges based on internet service & extras
    base_charge = np.where(internet_service == "Fiber optic", 75.0, np.where(internet_service == "DSL", 45.0, 20.0))
    extras_count = (
        (np.array(online_security) == "Yes").astype(int) +
        (np.array(online_backup) == "Yes").astype(int) +
        (np.array(device_protection) == "Yes").astype(int) +
        (np.array(tech_support) == "Yes").astype(int) +
        (np.array(streaming_tv) == "Yes").astype(int) +
        (np.array(streaming_movies) == "Yes").astype(int)
    )
    monthly_charges = np.round(base_charge + extras_count * 7.5 + np.random.normal(0, 3, n_samples), 2)
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)
    
    total_charges = np.round(tenure * monthly_charges + np.random.normal(0, 20, n_samples), 2)
    total_charges = np.clip(total_charges, 18.25, 8600.0)
    
    # Customer support tickets last 6 months
    support_tickets = np.random.poisson(lam=1.5, size=n_samples)
    
    # Customer Satisfaction Score (1 to 5)
    satisfaction_score = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.15, 0.20, 0.30, 0.25, 0.10])
    
    # Calculate log-odds of churn based on realistic business factors
    log_odds = (
        -1.2
        + 1.5 * (contract == "Month-to-month")
        - 0.8 * (contract == "Two year")
        + 0.025 * (monthly_charges - 60)
        - 0.04 * tenure
        + 0.6 * (tech_support == "No")
        + 0.5 * (online_security == "No")
        + 0.45 * (payment_method == "Electronic check")
        + 0.4 * (support_tickets >= 3)
        - 0.5 * (satisfaction_score >= 4)
        + 0.6 * (satisfaction_score <= 2)
        + np.random.normal(0, 0.35, n_samples)
    )
    
    prob_churn = 1 / (1 + np.exp(-log_odds))
    churn = (prob_churn > 0.5).astype(int)
    
    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "TenureMonths": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SupportTicketsLast6M": support_tickets,
        "SatisfactionScore": satisfaction_score,
        "Churn": churn
    })
    
    return df

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "customer_churn_data.csv")
    
    df = generate_telecom_churn_dataset(5000)
    df.to_csv(out_path, index=False)
    print(f"Dataset generated successfully at: {os.path.abspath(out_path)}")
    print(f"Shape: {df.shape}")
    print(f"Churn rate: {df['Churn'].mean():.2%}")
