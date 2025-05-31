readme_content = """
# 🧠 Mlops-Final-Student-Depression

A machine learning pipeline for predicting student depression using academic, lifestyle, and psychological data. Includes a Flask web app with dashboard visualization and model evaluation.

---

## 📘 Project Overview

This project presents a complete machine learning pipeline designed to predict depression in students based on a rich set of features including academic pressure, lifestyle habits, and psychological indicators. The objective is to provide early warning signals for mental health intervention through a deployable web application. The system offers both prediction capabilities and data visualization through a dashboard, enabling actionable insights for educational institutions, researchers, and policy makers.

---

## 📂 Dataset

- **Source:** [Student Depression Dataset – Kaggle](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset/data)
- **Format:** CSV
- **Description:**
  - **Demographics:** Age, Gender, City  
  - **Academic Indicators:** CGPA, Academic Pressure, Study Satisfaction  
  - **Lifestyle & Wellbeing:** Sleep Duration, Work Pressure, Job Satisfaction, Work/Study Hours, Dietary Habits  
  - **Additional Factors:** Profession, Degree, Financial Stress, Family History of Mental Illness, Suicidal Thoughts  
  - **Target:** `Depression_Status` (Yes/No)

> The dataset is ethically collected and anonymized for responsible use in mental health research and applications.

---

## 🔁 Pipeline Steps

### 1. Data Preprocessing
- Handle missing values
- Convert duration (e.g., "5 hours") to numeric
- Scale GPA to 10
- Encode categorical variables

### 2. Feature Engineering
- Create `Total Pressure` = Academic + Work
- Normalize derived numerical features

### 3. Model Training
- Logistic Regression
- LightGBM Classifier
- Train-test split (80/20), models saved as `.pkl` in `models/`

### 4. Evaluation
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Evaluation results saved as `.json`

### 5. Deployment
- Flask app (`app.py`)
- Input via form, output saved to `data/predictions.csv`
- Dashboard with Plotly charts
- HTML pages: `index.html`, `results.html`, `dashboard.html`

---

## ⚙️ Setup Instructions

---

### 🛠 Environment Setup

in bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\\Scripts\\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

### 🔧 Folder Structure


![image](https://github.com/user-attachments/assets/c241a75b-e673-44be-aaa9-9892a07b4b87)

### Launvh Flask App


# python app.py


# Then open your browser and go to:


# http://127.0.0.1:5000 


# or click on the link through the terminal
 
