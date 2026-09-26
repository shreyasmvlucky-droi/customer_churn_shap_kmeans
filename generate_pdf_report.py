import os
from fpdf import FPDF

class PDFReport(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 10)
        self.set_text_color(100, 116, 139)
        self.cell(0, 10, "CUSTOMER CHURN + SHAP + K-MEANS | COMPLETE PROJECT REPORT", border=False, ln=1, align="L")
        self.set_draw_color(226, 232, 240)
        self.line(10, 18, 200, 18)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 9)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()} | Built from Scratch & Deployed in Production", align="C")

    def section_title(self, title):
        self.set_font("helvetica", "B", 13)
        self.set_text_color(30, 58, 138)
        self.set_fill_color(239, 246, 255)
        self.cell(0, 9, f"  {title}", fill=True, ln=1)
        self.ln(3)

    def subsection_title(self, title):
        self.set_font("helvetica", "B", 11)
        self.set_text_color(15, 23, 42)
        self.cell(0, 7, title, ln=1)
        self.ln(1)

    def body_text(self, text):
        self.set_font("helvetica", "", 10)
        self.set_text_color(51, 65, 85)
        self.multi_cell(0, 5.5, text)
        self.ln(3)

    def bullet_point(self, bold_prefix, text):
        self.set_font("helvetica", "B", 10)
        self.set_text_color(30, 58, 138)
        self.cell(6, 5.5, "-", ln=0)
        self.cell(self.get_string_width(bold_prefix) + 2, 5.5, bold_prefix, ln=0)
        self.set_font("helvetica", "", 10)
        self.set_text_color(51, 65, 85)
        self.multi_cell(0, 5.5, text)
        self.ln(1.5)

def build_pdf():
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title Block
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 10, "Customer Churn + SHAP + K-Means", ln=1, align="C")
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 7, "End-to-End System Implementation & Architectural Documentation", ln=1, align="C")
    pdf.ln(4)

    # Key Metadata Box
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, pdf.get_y(), 190, 22)
    start_y = pdf.get_y()
    
    pdf.set_xy(14, start_y + 2)
    pdf.set_font("helvetica", "B", 9.5)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(45, 5, "Implementation Status:", ln=0)
    pdf.set_font("helvetica", "", 9.5)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(100, 5, "100% Completed From Scratch", ln=1)
    
    pdf.set_x(14)
    pdf.set_font("helvetica", "B", 9.5)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(45, 5, "Deployment URL:", ln=0)
    pdf.set_font("helvetica", "U", 9.5)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(100, 5, "https://customerchurnshapkmeans-nktdukaakqzpzaceypnvzd.streamlit.app", ln=1, link="https://customerchurnshapkmeans-nktdukaakqzpzaceypnvzd.streamlit.app")
    
    pdf.set_x(14)
    pdf.set_font("helvetica", "B", 9.5)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(45, 5, "GitHub Repository:", ln=0)
    pdf.set_font("helvetica", "U", 9.5)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(100, 5, "https://github.com/shreyasmvlucky-droi/customer_churn_shap_kmeans", ln=1, link="https://github.com/shreyasmvlucky-droi/customer_churn_shap_kmeans")
    pdf.set_y(start_y + 26)

    # 1. Executive Summary & What is Unique
    pdf.section_title("1. Executive Summary & Project Uniqueness")
    pdf.body_text(
        "Standard customer churn projects focus solely on binary classification (predicting 0 or 1). This project introduces a 3-tier hybrid machine learning paradigm combining Supervised Learning, Explainable AI (XAI), and Unsupervised Persona Clustering to deliver end-to-end predictive and prescriptive business value."
    )
    
    pdf.subsection_title("What is Unique & Innovative in This Project?")
    pdf.bullet_point("3-Tier ML Architecture: ", "Combines XGBoost (WHO will churn), SHAP TreeExplainer (WHY they churn), and K-Means (HOW to target retentions).")
    pdf.bullet_point("SHAP-Space Persona Clustering: ", "Rather than clustering on raw demographic features, K-Means is applied to the SHAP feature attribution matrix. This clusters customers by their root causes of churn.")
    pdf.bullet_point("Prescriptive Action Engine: ", "Automatically assigns actionable, tailored business interventions (e.g. 15% discount for price-sensitive churners, VIP support escalation for ticket-fatigued users).")
    pdf.bullet_point("Production Web Dashboard: ", "Built an interactive 6-stage Streamlit web app with real-time customer churn simulation, SHAP waterfall plot breakdown, and risk gauges.")
    pdf.ln(2)

    # 2. Tools & Technologies
    pdf.section_title("2. Tools & Technologies Used")
    
    pdf.set_font("helvetica", "B", 9)
    pdf.set_fill_color(30, 58, 138)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(45, 7, " Category", border=1, fill=True, ln=0)
    pdf.cell(60, 7, " Technologies / Libraries", border=1, fill=True, ln=0)
    pdf.cell(85, 7, " Purpose in Project", border=1, fill=True, ln=1)
    
    tools_data = [
        ("Programming Language", "Python 3.11+", "Core language for end-to-end data pipeline and web app"),
        ("Data Processing", "Pandas, NumPy", "Data manipulation, feature engineering, missing value handling"),
        ("Machine Learning", "Scikit-Learn, XGBoost", "ColumnTransformer, StandardScaler, RandomForest & XGBoost"),
        ("Explainable AI (XAI)", "SHAP (TreeExplainer)", "Decomposing model predictions into exact feature attributions"),
        ("Unsupervised Learning", "K-Means, Silhouette Score", "Persona segmentation based on SHAP churn driver matrices"),
        ("Data Visualization", "Plotly, Seaborn, Matplotlib", "Interactive donut charts, ROC curves, and SHAP waterfalls"),
        ("Web Framework", "Streamlit", "Building production web app dashboard with reactive state"),
        ("Version Control & Cloud", "Git, GitHub, Docker, Streamlit Cloud", "CI/CD version control, dockerization, and cloud deployment")
    ]
    
    pdf.set_font("helvetica", "", 8.5)
    pdf.set_text_color(30, 41, 59)
    fill = False
    for cat, tech, purp in tools_data:
        pdf.set_fill_color(241, 245, 249) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.cell(45, 6, f" {cat}", border=1, fill=fill, ln=0)
        pdf.cell(60, 6, f" {tech}", border=1, fill=fill, ln=0)
        pdf.cell(85, 6, f" {purp}", border=1, fill=fill, ln=1)
        fill = not fill
    pdf.ln(4)

    # 3. System Architecture & Workflow
    pdf.section_title("3. System Architecture & Workflow")
    pdf.body_text(
        "The system follows a modular data science architecture built from scratch. Data moves seamlessly from initial generation through transformation, modeling, explainability, segmentation, and interactive presentation."
    )
    
    pdf.bullet_point("Step 1: Data Generation: ", "Simulated 5,000 telco customer records with realistic non-linear churn signals across tenure, monthly charges, contract types, tech support, and support tickets.")
    pdf.bullet_point("Step 2: Preprocessing: ", "Used ColumnTransformer with StandardScaler for numerical attributes and OneHotEncoder for categorical features.")
    pdf.bullet_point("Step 3: Supervised Classification: ", "Trained XGBoost and Random Forest models using stratified train-test splitting (80/20). XGBoost achieved 94.5% Accuracy and 0.987 ROC-AUC.")
    pdf.bullet_point("Step 4: SHAP Attribution Engine: ", "Computed exact SHAP values using TreeExplainer, extracting both global feature importance rankings and local customer waterfall vectors.")
    pdf.bullet_point("Step 5: SHAP K-Means Clustering: ", "Evaluated optimal clusters via Elbow Method & Silhouette Scores, partitioning customers into 3 core churn personas.")
    pdf.bullet_point("Step 6: Streamlit Web Suite: ", "Developed a multi-tab web application for EDA, model benchmarking, SHAP explorer, cluster personas, and live customer simulator.")
    pdf.ln(2)

    # 4. Detailed Module Implementation Breakdown
    pdf.section_title("4. Implementation Breakdown (Built From Scratch)")
    
    pdf.subsection_title("A. Dataset Generator (data/generate_dataset.py)")
    pdf.body_text("Script generates 5,000 realistic customer records with mathematically defined log-odds churn distributions driven by contract length, monthly charges, tenure decay, and support ticket frequency.")

    pdf.subsection_title("B. Exploratory Data Analysis Module (src/eda.py)")
    pdf.body_text("Provides functions for summary statistics, interactive Plotly donut charts for churn ratios, dynamic box plots / bar charts for feature vs. churn breakdown, and numeric correlation matrices.")

    pdf.subsection_title("C. Classification Pipeline (src/model_pipeline.py)")
    pdf.body_text("Implements scikit-learn preprocessing pipelines, fits XGBoost and Random Forest classifiers, computes Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, and saves binary model artifacts (.joblib).")

    pdf.subsection_title("D. SHAP Explainability Engine (src/shap_explainability.py)")
    pdf.body_text("Leverages SHAP TreeExplainer to compute global feature rankings and individual customer waterfall charts showing positive (risk-increasing) and negative (risk-reducing) feature impacts.")

    pdf.subsection_title("E. K-Means Persona Clustering (src/kmeans_segmentation.py)")
    pdf.body_text("Applies K-Means clustering directly on SHAP value matrices. Profiles cluster centroids to automatically categorize churners into actionable business personas.")

    pdf.subsection_title("F. Web Dashboard & Entrypoint (app.py & streamlit_app.py)")
    pdf.body_text("Streamlit web app featuring 6 dedicated workflow stages, custom CSS styling, cached model loading, and real-time interactive churn risk simulation.")
    pdf.ln(2)

    # 5. Model Evaluation & Benchmark Results
    pdf.section_title("5. Model Benchmark Results & Persona Actions")
    
    pdf.set_font("helvetica", "B", 9)
    pdf.set_fill_color(30, 58, 138)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(50, 7, " Model Algorithm", border=1, fill=True, ln=0)
    pdf.cell(28, 7, " Accuracy", border=1, fill=True, ln=0)
    pdf.cell(28, 7, " Precision", border=1, fill=True, ln=0)
    pdf.cell(28, 7, " Recall", border=1, fill=True, ln=0)
    pdf.cell(28, 7, " F1-Score", border=1, fill=True, ln=0)
    pdf.cell(28, 7, " ROC-AUC", border=1, fill=True, ln=1)
    
    pdf.set_font("helvetica", "", 8.5)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(50, 6, " XGBoost Classifier", border=1, ln=0)
    pdf.cell(28, 6, " 94.5%", border=1, ln=0)
    pdf.cell(28, 6, " 91.4%", border=1, ln=0)
    pdf.cell(28, 6, " 88.4%", border=1, ln=0)
    pdf.cell(28, 6, " 89.9%", border=1, ln=0)
    pdf.cell(28, 6, " 0.987", border=1, ln=1)
    
    pdf.cell(50, 6, " Random Forest Classifier", border=1, ln=0)
    pdf.cell(28, 6, " 91.6%", border=1, ln=0)
    pdf.cell(28, 6, " 87.8%", border=1, ln=0)
    pdf.cell(28, 6, " 80.8%", border=1, ln=0)
    pdf.cell(28, 6, " 84.2%", border=1, ln=0)
    pdf.cell(28, 6, " 0.974", border=1, ln=1)
    pdf.ln(3)

    pdf.subsection_title("SHAP Cluster Personas & Prescriptive Action Plan")
    pdf.bullet_point("Persona 0 (Price-Sensitive High-Tier): ", "Month-to-month contracts + high monthly charges. Action: 15% discount on annual plan upgrade.")
    pdf.bullet_point("Persona 1 (Low Support / Onboarding At-Risk): ", "Short tenure + no tech support. Action: Free 3 months Premium Tech Support & onboarding check-in.")
    pdf.bullet_point("Persona 2 (Service Friction & High Tickets): ", "4+ support tickets + low satisfaction score. Action: Priority VIP support escalation + refund voucher.")
    pdf.ln(2)

    # 6. Interview Presentation Guide
    pdf.section_title("6. Interview Talking Points & Portfolio Guide")
    pdf.bullet_point("Data Science Rigor: ", "Explain how ColumnTransformer handles data leakage by fitting transformers strictly on training splits.")
    pdf.bullet_point("Explainable AI (XAI): ", "Demonstrate how SHAP TreeExplainer converts complex XGBoost trees into interpretable game-theoretic feature attributions.")
    pdf.bullet_point("Unsupervised Innovation: ", "Highlight the shift from standard attribute clustering to SHAP-based root cause clustering.")
    pdf.bullet_point("End-to-End Delivery: ", "Emphasize building full-stack code, unit compilation, containerization, Git version control, and live Streamlit Cloud deployment.")

    out_path = os.path.join("C:\\Users\\Shreyas M\\Desktop\\customer_churn_shap_kmeans", "Customer_Churn_SHAP_KMeans_Project_Report.pdf")
    pdf.output(out_path)
    print(f"PDF report generated successfully at: {out_path}")

if __name__ == "__main__":
    build_pdf()
