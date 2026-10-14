from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# --------------------------------------------------
# Flask App
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"
DATA_CANDIDATES = [
    BASE_DIR / "data" / "ai4i2020.csv",
    BASE_DIR / "data" / "raw" / "machine_sensor_data.csv",
]
DATA_PATH = next((path for path in DATA_CANDIDATES if path.exists()), DATA_CANDIDATES[0])


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Features used during training
# --------------------------------------------------

FEATURE_COLUMNS = [
    "Air_temperature_K",
    "Process_temperature_K",
    "Rotational_speed_rpm",
    "Torque_Nm",
    "Tool_wear_min",
    "Type_L",
    "Type_M"
]


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

data = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# Prepare Dataset
# --------------------------------------------------

def prepare_features(df):

    df = df.copy()

    # Remove columns that were not used for training
    columns_to_drop = [
        "UDI",
        "Product ID",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ]

    df = df.drop(
        columns=[
            col for col in columns_to_drop
            if col in df.columns
        ],
        errors="ignore"
    )

    # Convert Type into numerical columns
    df = pd.get_dummies(
        df,
        columns=["Type"],
        drop_first=True
    )

    # Clean column names
    df.columns = (
        df.columns
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("<", "", regex=False)
        .str.replace(" ", "_")
    )

    # Make sure all expected columns exist
    df = df.reindex(
        columns=FEATURE_COLUMNS,
        fill_value=0
    )

    return df


# --------------------------------------------------
# Risk Category
# --------------------------------------------------

def get_risk_level(score):

    if score < 30:
        return "Low Risk"

    elif score < 70:
        return "Medium Risk"

    else:
        return "High Risk"


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# --------------------------------------------------
# Dashboard
# --------------------------------------------------

@app.route("/dashboard")
def dashboard():

    features = prepare_features(data)

    probabilities = model.predict_proba(
        features
    )[:, 1]

    predictions = model.predict(
        features
    )

    dashboard_data = data.copy()

    dashboard_data["Risk"] = (
        probabilities * 100
    ).round(2)

    dashboard_data["Prediction"] = predictions

    dashboard_data["Risk_Level"] = [
        get_risk_level(score)
        for score in dashboard_data["Risk"]
    ]

    total_machines = len(dashboard_data)

    high_risk = (
        dashboard_data["Risk_Level"] == "High Risk"
    ).sum()

    medium_risk = (
        dashboard_data["Risk_Level"] == "Medium Risk"
    ).sum()

    low_risk = (
        dashboard_data["Risk_Level"] == "Low Risk"
    ).sum()

    # Show highest-risk machines first
    machines = dashboard_data.sort_values(
        by="Risk",
        ascending=False
    ).head(100)

    machines = machines.to_dict(
        orient="records"
    )

    return render_template(
        "dashboard.html",
        machines=machines,
        total_machines=total_machines,
        high_risk=high_risk,
        medium_risk=medium_risk,
        low_risk=low_risk
    )


# --------------------------------------------------
# Individual Machine
# --------------------------------------------------

@app.route("/machine/<int:machine_id>")
def machine(machine_id):

    machine_data = data[
        data["UDI"] == machine_id
    ]

    if machine_data.empty:

        return "Machine not found", 404

    features = prepare_features(
        machine_data
    )

    probability = model.predict_proba(
        features
    )[0][1]

    prediction = model.predict(
        features
    )[0]

    risk = round(
        probability * 100,
        2
    )

    risk_level = get_risk_level(
        risk
    )

    machine_info = machine_data.iloc[0].to_dict()

    return render_template(
        "machine.html",
        machine=machine_info,
        risk=risk,
        prediction=prediction,
        risk_level=risk_level
    )


# --------------------------------------------------
# Manual Prediction
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    air_temperature = float(
        request.form["air_temperature"]
    )

    process_temperature = float(
        request.form["process_temperature"]
    )

    rotational_speed = float(
        request.form["rotational_speed"]
    )

    torque = float(
        request.form["torque"]
    )

    tool_wear = float(
        request.form["tool_wear"]
    )

    machine_type = request.form["machine_type"]


    # Create input dataframe
    input_data = pd.DataFrame([
        {
            "Air temperature [K]": air_temperature,
            "Process temperature [K]": process_temperature,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Type": machine_type
        }
    ])


    # Prepare exactly like training data
    features = prepare_features(
        input_data
    )


    # Prediction
    probability = model.predict_proba(
        features
    )[0][1]

    prediction = model.predict(
        features
    )[0]


    risk = round(
        probability * 100,
        2
    )

    risk_level = get_risk_level(
        risk
    )


    return render_template(
        "machine.html",
        machine=None,
        risk=risk,
        prediction=prediction,
        risk_level=risk_level,
        manual=True,
        input_data=input_data.iloc[0].to_dict()
    )


# --------------------------------------------------
# Run Application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )