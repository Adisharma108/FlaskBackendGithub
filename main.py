from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from joblib import load
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and scaler
model = load("./models/voting_classifier.joblib")
scaler = load("./models/scaler.joblib")

# Feature names as per the dataset
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

        # Convert input to DataFrame with correct feature names
        input_df = pd.DataFrame([features], columns=feature_names)

        # Scale input
        scaled_features = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_features)
        risk_category = "Healthy" if prediction[0] == 0 else "Unhealthy"

        return jsonify({"risk_category": risk_category})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

