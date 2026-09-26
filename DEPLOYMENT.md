# 🚀 Deployment Guide: Customer Churn + SHAP + K-Means Dashboard

This document provides step-by-step instructions to deploy your Streamlit application to free and production cloud hosting platforms.

---

## 🌟 Option 1: Streamlit Community Cloud (Recommended - 100% Free)

Streamlit Community Cloud connects directly to your GitHub repository and deploys your web app in under 2 minutes.

### Step 1: Push Code to GitHub
Open terminal in your project directory:
```bash
git init
git add .
git commit -m "Initial commit: Customer Churn SHAP K-Means Dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/customer-churn-shap-kmeans.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `YOUR_USERNAME/customer-churn-shap-kmeans`.
4. Set **Branch**: `main`.
5. Set **Main file path**: `app.py`.
6. Click **"Deploy!"**.

> 🎉 Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

---

## 🤗 Option 2: Hugging Face Spaces (100% Free)

Hugging Face Spaces provides free hosting for machine learning apps.

### Step 1: Create Space
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Set Space name (e.g. `customer-churn-intelligence`).
3. Select **Space SDK**: **Streamlit**.
4. Set Space visibility to **Public**.

### Step 2: Upload Files or Push Git Repository
```bash
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/customer-churn-intelligence
git push hf main
```
Your app will build automatically and display your live URL.

---

## 🐳 Option 3: Docker Deployment (Render / Railway / GCP Cloud Run)

You can build and run your app as a Docker container anywhere.

### Build & Run Container Locally:
```bash
docker build -t customer-churn-app .
docker run -p 8501:8501 customer-churn-app
```

### Deploy to Render (Free Tier):
1. Sign up at [render.com](https://render.com/).
2. Click **"New +"** $\rightarrow$ **"Web Service"**.
3. Connect your GitHub repository.
4. Select **Environment**: **Docker**.
5. Click **"Create Web Service"**.

---

## 📋 Pre-Deployment Verification Checklist

- [x] `.streamlit/config.toml` configured with headless mode.
- [x] `requirements.txt` contains all dependencies (`shap`, `xgboost`, `streamlit`, `plotly`, `scikit-learn`).
- [x] `Dockerfile` and `.dockerignore` included for containerization.
- [x] Automatic dataset generation fallback integrated in `app.py`.
