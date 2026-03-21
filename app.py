from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained Keras model
model = tf.keras.models.load_model('model_tf.h5')

# Load the scaler
scaler = joblib.load('scaler.pkl')

import pandas as pd

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Request received")

        data = request.get_json()
        print("Incoming data:", data)

        # Create DataFrame with correct column names
        columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                   'Insulin','BMI','DiabetesPedigreeFunction','Age']

        input_df = pd.DataFrame([[ 
            float(data['Pregnancies']),
            float(data['Glucose']),
            float(data['BloodPressure']),
            float(data['SkinThickness']),
            float(data['Insulin']),
            float(data['BMI']),
            float(data['DiabetesPedigreeFunction']),
            float(data['Age'])
        ]], columns=columns)

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)
        predicted_class = int(prediction[0][0] > 0.5)

        result = "Diabetic" if predicted_class == 1 else "Not Diabetic"

        print("Prediction:", result)

        return jsonify({'prediction': result})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



