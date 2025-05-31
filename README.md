# Mlops-Final-Student-Depression-
A machine learning pipeline for predicting student depression using academic, lifestyle, and psychological data. Includes a Flask web app with dashboard visualization and model evaluation.


readme_content = """
# 🧠 Student Depression Prediction Pipeline

## 📘 Project Overview

This project presents a complete machine learning pipeline designed to predict depression in students based on a rich set of features including academic pressure, lifestyle habits, and psychological indicators. The objective is to provide early warning signals for mental health intervention through a deployable web application. The system offers both prediction capabilities and data visualization through a dashboard, enabling actionable insights for educational institutions, researchers, and policy makers.

## 📂 Dataset

- *Source:* [Student Depression Dataset – Kaggle](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset/data)
- *Format:* CSV
- *Description:*  
  This dataset provides anonymized records for individual students, incorporating the following:

  - *Demographics:* Age, Gender, City
  - *Academic Indicators:* CGPA, Academic Pressure, Study Satisfaction
  - *Lifestyle & Wellbeing:* Sleep Duration, Work Pressure, Job Satisfaction, Work/Study Hours, Dietary Habits
  - *Additional Factors:* Profession, Degree, Financial Stress, Family History of Mental Illness, Suicidal Thoughts
  - *Target Variable:* Depression_Status – A binary outcome indicating if the student is experiencing depression (Yes/No)

This dataset is ethically collected and anonymized for responsible use in mental health research and applications.

## 🔁 Pipeline Steps

### 1. Data Preprocessing
- Missing values are removed or replaced with appropriate defaults.
- Text values such as "5 hours" in sleep duration are converted to numerical values using regex.
- GPA is rescaled from a 4-point scale to a standard 10-point scale.
- Categorical features (e.g., gender, profession) are encoded for model compatibility.

### 2. Feature Engineering
- *Total Pressure* is created by summing academic and work pressure scores.
- Satisfaction and stress scales are standardized to a uniform scale.
- Derived numerical features are normalized to improve model performance.

### 3. Model Training
- Two classification models are trained using scikit-learn and lightgbm:
  - *Logistic Regression* – baseline linear model
  - *LightGBM Classifier* – gradient boosting tree-based model
- Models are trained using an 80/20 train-test split and saved in models/ as .pkl files.

### 4. Evaluation
- For each model, the following metrics are computed:
  - Accuracy, Precision, Recall, F1-Score, and ROC-AUC
- Evaluation results are exported to JSON files and later used for comparison.
- The better model (usually LightGBM) is chosen for deployment.

### 5. Deployment
- The project includes a Flask web application (app.py) that:
  - Accepts form inputs from users
  - Displays predictions with confidence levels
  - Stores inputs and outputs in data/predictions.csv
- A dashboard interface visualizes prediction statistics using a responsive Plotly pie chart.
- HTML templates (index.html, results.html, dashboard.html) manage page structure and flow.

## ⚙ Setup Instructions
Folder structure 
![image](https://github.com/user-attachments/assets/ca56fb88-a892-445b-9375-f641eb779808)

To launch the Flaks App in the terminal enter:
Python app.py


