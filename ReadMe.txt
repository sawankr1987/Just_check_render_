🧠 Diabetes Prediction Web App

A machine learning–powered web application for predicting diabetes risk based on user input data. This project integrates a trained ML model with a lightweight web interface for real-time predictions.

🚀 Features
🔍 Predict diabetes likelihood using trained ML model
🌐 Simple web interface for user input
⚡ Fast inference using pre-trained model (.h5)
📦 Ready for deployment (Render-compatible setup included)
🧪 Includes dataset and training notebook
📁 Project Structure
.
├── app.py                          # Main Flask application
├── templates/                      # HTML templates
│   └── index.html
├── model_tf.h5                     # Trained TensorFlow model
├── scaler.pkl                      # Feature scaler
├── diabetes.csv                    # Dataset used
├── training_model_and_save.ipynb   # Model training notebook
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version for deployment
├── .gitignore
└── ReadMe.txt
🛠️ Tech Stack
Python
Flask
TensorFlow / Keras
Scikit-learn
HTML (Frontend)
⚙️ Installation & Setup
1. Clone the repository
git clone https://github.com/your-username/just_check_render_.git
cd just_check_render_
2. Create virtual environment (recommended)
conda create -n diabetes_env python=3.10
conda activate diabetes_env
3. Install dependencies
pip install -r requirements.txt
▶️ Run the Application
python app.py

Then open your browser and go to:

http://127.0.0.1:5000/
📊 Model Details
Model type: Neural Network (TensorFlow/Keras)
Input: Patient health parameters
Output: Diabetes prediction (binary classification)
Preprocessing:
Feature scaling using scaler.pkl
🌍 Deployment

This project is configured for deployment on platforms like Render.

Ensure:

requirements.txt is updated
runtime.txt specifies Python version
Entry point is correctly set (app.py)
🧪 Training the Model

To retrain or modify the model:

Open training_model_and_save.ipynb
Run all cells
Save updated:
model_tf.h5
scaler.pkl
📌 Notes
Ensure consistent feature order during prediction
Use the same scaler used during training
HTML templates must remain inside /templates folder for Flask
🤝 Contributing

Contributions are welcome. Feel free to fork the repo and submit a pull request.

📄 License

This project is open-source and available under the MIT License.
