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
        scaled_features = scaler.transform(input_df)
        prediction = model.predict(scaled_features)
        risk_category = "Healthy" if prediction[0] == 0 else "Unhealthy"

        return jsonify({"risk_category": risk_category})
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
