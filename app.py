from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load("model/churn_model.pkl")
columns = joblib.load("model/feature_columns.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    tenure = float(request.form["tenure"])
    monthly = float(request.form["MonthlyCharges"])
    total = float(request.form["TotalCharges"])
    contract = request.form["Contract"]
    internet = request.form["InternetService"]

    # Create empty dataframe with correct columns
    input_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    # Fill numeric features
    input_df["tenure"] = tenure
    input_df["MonthlyCharges"] = monthly
    input_df["TotalCharges"] = total

    # Correct One-Hot Encoding (based on training)
    # Contract
    if contract == "One year":
        input_df["Contract_One year"] = 1
    elif contract == "Two year":
        input_df["Contract_Two year"] = 1
    # Month-to-month = baseline → all 0

    # Internet Service
    if internet == "Fiber optic":
        input_df["InternetService_Fiber optic"] = 1
    elif internet == "No":
        input_df["InternetService_No"] = 1
    # DSL = baseline → all 0

    # Prediction
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        result = "⚠ High Churn Risk"
    else:
        result = "✅ Customer is Safe"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)