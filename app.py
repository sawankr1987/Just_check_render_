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
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.get_json()

        if not data:
            return jsonify({'error': 'No input data'}), 400

        print("Incoming data:", data)

        # Extract + validate
        features = []
        keys = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]

        for key in keys:
            if key not in data:
                return jsonify({'error': f'Missing {key}'}), 400
            features.append(float(data[key]))

        # Convert safely
        input_array = np.array(features, dtype=np.float32).reshape(1, -1)

        print("Input shape:", input_array.shape)

        # Scale
        input_scaled = scaler.transform(input_array)

        print("Scaled shape:", input_scaled.shape)

        # Predict safely
        prediction = model.predict(input_scaled, verbose=0)

        print("Raw prediction:", prediction)

        # Handle different output shapes
        if prediction.ndim == 2:
            prob = float(prediction[0][0])
        else:
            prob = float(prediction[0])

        predicted_class = int(prob > 0.5)

        result = "Diabetic" if predicted_class == 1 else "Not Diabetic"

        return jsonify({
            "prediction": result,
            "confidence": prob
        })

    except Exception as e:
        print("🔥 ERROR:", str(e))
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
