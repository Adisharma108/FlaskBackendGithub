from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
from pymongo import MongoClient
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS

# MongoDB connection
mongo_user = os.getenv("MONGO_USER")
mongo_pass = os.getenv("MONGO_PASS")
mongo_uri = f"mongodb+srv://{mongo_user}:{mongo_pass}@clustersomeone.oho7pol.mongodb.net/"
client = MongoClient(mongo_uri)


db = client["medxtech"]
doctor_collection = db["doctors"]
patient_collection = db["patients"]

# Load ML model and scaler
model = load("./models/voting_classifier.joblib")
scaler = load("./models/scaler.joblib")

# Features expected by the model
feature_names = [
    "Heart Rate", "Respiratory Rate", "Body Temperature", "Oxygen Saturation",
    "Age", "Gender", "Derived_HRV", "Derived_Pulse_Pressure", "Derived_BMI", "Derived_MAP"
]


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get("features")

        if not features or len(features) != len(feature_names):
            return jsonify({"error": "Invalid input. Ensure all required fields are provided."}), 400

        input_df = pd.DataFrame([features], columns=feature_names)

        # Extract individual values
        heart_rate = float(input_df["Heart Rate"][0])
        resp_rate = float(input_df["Respiratory Rate"][0])
        temperature = float(input_df["Body Temperature"][0])
        oxygen = float(input_df["Oxygen Saturation"][0])
        hrv = float(input_df["Derived_HRV"][0])
        pulse_pressure = float(input_df["Derived_Pulse_Pressure"][0])
        bmi = float(input_df["Derived_BMI"][0])
        map_val = float(input_df["Derived_MAP"][0])

        # Rule-based check (from parameters.pdf)
        rule_unhealthy = (
            heart_rate > 105 or heart_rate < 60 or
            resp_rate > 21 or resp_rate < 10 or
            temperature > 40.2 or temperature < 35.1 or
            oxygen < 94 or
            hrv < 19 or
            pulse_pressure < 36 or pulse_pressure > 62 or
            bmi >= 30 or bmi < 18.5 or
            map_val < 70 or map_val > 100
        )

        # ML Prediction
        scaled_features = scaler.transform(input_df)
        prediction = model.predict(scaled_features)
        model_risk = "Healthy" if prediction[0] == 0 else "Unhealthy"
        # print("Input:", input_df.to_dict())
        # print("Rule Unhealthy:", rule_unhealthy)
        # print("model prediction:", prediction[0])
        # print("Model Prediction:", model_risk)

        # Final Decision
        final_risk = "Unhealthy" if rule_unhealthy else model_risk

        return jsonify({"risk_category": final_risk})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/register/doctor', methods=['POST'])
def register_doctor():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        doctor_collection.insert_one(data)
        return jsonify({"message": "Doctor registered successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/register/patient', methods=['POST'])
def register_patient():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        patient_collection.insert_one(data)
        return jsonify({"message": "Patient registered successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
