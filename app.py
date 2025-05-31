from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
import os
import json

app = Flask(__name__)

# Load trained LightGBM model
model = joblib.load("models/lightgbm_model.pkl")

# Path for logging
csv_path = "data/predictions.csv"
os.makedirs("data", exist_ok=True)

# Input fields
FIELDS = [
    "Gender", "Age", "City", "Profession", "Academic Pressure", "Work Pressure",
    "CGPA", "Study Satisfaction", "Job Satisfaction", "Sleep Duration",
    "Dietary Habits", "Degree", "Have you ever had suicidal thoughts ?",
    "Work/Study Hours", "Financial Stress", "Family History of Mental Illness"
]

# Create CSV if missing
if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
    pd.DataFrame(columns=["Model", "Prediction", "Confidence"] + FIELDS + ["Total Pressure"]).to_csv(csv_path, index=False)

# Preprocess function
def preprocess_input(form):
    """Convert raw form data to encoded DataFrame with engineered features."""
    data = []
    for field in FIELDS:
        val = form[field]
        try:
            val = float(val) if '.' in val or val.isdigit() else val
        except:
            pass
        data.append(val)
    

    df = pd.DataFrame([data], columns=FIELDS)
    
    # Ensure capped values for 1–5 scale features
    for col in ["Academic Pressure", "Work Pressure", "Study Satisfaction", "Job Satisfaction", "Financial Stress"]:
        df[col] = df[col].clip(upper=5)

    # Binary encodings
    binary_enc = {
        "Gender": {"Male": 0, "Female": 1},
        "Have you ever had suicidal thoughts ?": {"No": 0, "Yes": 1},
        "Family History of Mental Illness": {"No": 0, "Yes": 1}
    }
    for col, mapping in binary_enc.items():
        df[col] = df[col].map(mapping).fillna(0).astype(int)

    # Multi-category encodings
    city_enc = {"Cairo": 0, "Alexandria": 1, "Giza": 2, "Other": 3}
    profession_enc = {"Student": 0, "Freelancer": 1, "Employee": 2, "Unemployed": 3, "Other": 4}
    degree_enc = {"BSc": 0, "MSc": 1, "PhD": 2, "Other": 3}

    df["City"] = df["City"].map(city_enc).fillna(city_enc["Other"]).astype(int)
    df["Profession"] = df["Profession"].map(profession_enc).fillna(profession_enc["Other"]).astype(int)
    df["Degree"] = df["Degree"].map(degree_enc).fillna(degree_enc["Other"]).astype(int)

    # Feature engineering
    df["GPA_10"] = df["CGPA"] * 2.5
    df["Total Pressure"] = df["Academic Pressure"] + df["Work Pressure"]
    df["Satisfaction_Diff"] = df["Study Satisfaction"] - df["Job Satisfaction"]
    df["Stress_Index"] = df["Total Pressure"] + df["Financial Stress"]
    df["Sleep_to_Work"] = df["Sleep Duration"] / (df["Work/Study Hours"] + 1)
    df["Wellness_Score"] = df["Study Satisfaction"] + df["Sleep Duration"]
    df["Academic_to_Work"] = df["Academic Pressure"] / (df["Work Pressure"] + 1)
    df["Sleep_Quality"] = df["Sleep Duration"] * df["Dietary Habits"]

    return df

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        df = preprocess_input(request.form)
        prediction = model.predict(df)[0]
        confidence = float(np.max(model.predict_proba(df)) * 100)

        # Save full input + prediction
        row = {
            "Model": "LightGBM",
            "Prediction": "Depressed" if prediction else "Not Depressed",
            "Confidence": f"{confidence:.2f}"
        }
        for field in FIELDS:
            row[field] = request.form[field]
        row["Total Pressure"] = float(request.form["Academic Pressure"]) + float(request.form["Work Pressure"])

        # Append to CSV
        existing = pd.read_csv(csv_path)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        updated.to_csv(csv_path, index=False)

        return render_template("results.html", prediction=row["Prediction"], confidence=f"{confidence:.2f}")

    return render_template("index.html", fields=FIELDS)

# Dashboard route
@app.route("/dashboard")
def dashboard():
    import json

    # Initialize safe defaults
    df = pd.DataFrame(columns=["Model", "Prediction", "Confidence"] + FIELDS + ["Total Pressure"])
    pie_counts = [0, 0]
    metrics_lr = {}
    metrics_lgbm = {}

    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            # Pie chart counts
            depressed = int((df["Prediction"] == "Depressed").sum())
            not_depressed = int((df["Prediction"] == "Not Depressed").sum())
            pie_counts = [not_depressed, depressed]
    except Exception as e:
        print("CSV loading error:", e)

    try:
        with open("/Users/home/Main/Work/MLOPs/4Am/models/logistic_regression_metrics.json") as f:
            metrics_lr = json.load(f)
    except Exception as e:
        print("LR metrics error:", e)
        metrics_lr = {"Error": "Logistic Regression metrics not available."}

    try:
        with open("/Users/home/Main/Work/MLOPs/4Am/models/lightgbm_metrics.json") as f:
            metrics_lgbm = json.load(f)
    except Exception as e:
        print("LightGBM metrics error:", e)
        metrics_lgbm = {"Error": "LightGBM metrics not available."}

    return render_template("dashboard.html",
                           columns=df.columns,
                           rows=df.to_dict("records"),
                           pie_counts=pie_counts,
                           metrics_lr=metrics_lr,
                           metrics_lgbm=metrics_lgbm)

if __name__ == "__main__":
    app.run(debug=True)