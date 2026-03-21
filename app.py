from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# =========================
# Load Model & Scaler
# =========================
try:
    model = tf.keras.models.load_model('model_tf.h5')
    scaler = joblib.load('scaler.pkl')
    print("✅ Model and scaler loaded successfully")
except Exception as e:
    print("❌ Error loading model/scaler:", str(e))
    model = None
    scaler = None


# =========================
# Home Route (Fix 404)
# =========================
@app.route("/")
def home():
    return render_template("index.html")

# =========================
# Prediction Route
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded properly'}), 500

        data = request.get_json()

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        print("Incoming data:", data)

        # Required columns
        columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]

        # Validate all fields exist
        for col in columns:
            if col not in data:
                return jsonify({'error': f'Missing field: {col}'}), 400

        # Convert input to DataFrame
        input_data = [[float(data[col]) for col in columns]]
        input_df = pd.DataFrame(input_data, columns=columns)

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict (silent mode)
        prediction = model.predict(input_scaled, verbose=0)

        predicted_class = int(prediction[0][0] > 0.5)
        result = "Diabetic" if predicted_class == 1 else "Not Diabetic"

        print("Prediction:", result)

        return jsonify({
            'prediction': result,
            'confidence': float(prediction[0][0])
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({'error': str(e)}), 500


# =========================
# Run App
# =========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)