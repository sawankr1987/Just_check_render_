from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import joblib
import numpy as np

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
# Home Route
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

        # Ensure correct order
        features = [
            float(data['Pregnancies']),
            float(data['Glucose']),
            float(data['BloodPressure']),
            float(data['SkinThickness']),
            float(data['Insulin']),
            float(data['BMI']),
            float(data['DiabetesPedigreeFunction']),
            float(data['Age'])
        ]

        # Convert to numpy array (IMPORTANT FIX)
        input_array = np.array([features])

        # Scale input
        input_scaled = scaler.transform(input_array)

        # Predict
        prediction = model.predict(input_scaled)

        prob = float(prediction[0][0])
        predicted_class = int(prob > 0.5)

        result = "Diabetic" if predicted_class == 1 else "Not Diabetic"

        print("Prediction:", result, "| Confidence:", prob)

        return jsonify({
            'prediction': result,
            'confidence': prob
        })

    except Exception as e:
        print("🔥 ERROR:", str(e))
        return jsonify({'error': str(e)}), 500


# =========================
# Run App (local only)
# =========================
if __name__ == '__main__':
    app.run(debug=True)
