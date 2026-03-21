🧠 Diabetes Prediction Web App

🔗 Repository:

A machine learning–based web application that predicts whether a person is likely to have diabetes based on input health parameters. This project combines a trained deep learning model with a Flask web interface for real-time predictions.

🚀 Overview

This application allows users to input medical attributes (such as glucose level, BMI, age, etc.) and receive an instant prediction powered by a trained TensorFlow model.

✨ Features
🧾 User-friendly web interface (HTML + Flask)
🤖 Pre-trained deep learning model (model_tf.h5)
📊 Data preprocessing with scaler.pkl
⚡ Real-time prediction
🌐 Deployment-ready (Render-compatible setup included)
📁 Project Structure
.
├── app.py                          # Flask backend application
├── templates/
│   └── index.html                  # Frontend UI
├── model_tf.h5                     # Trained TensorFlow model
├── scaler.pkl                      # Data scaler for preprocessing
├── diabetes.csv                    # Dataset used for training
├── training_model_and_save.ipynb   # Model training notebook
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version for deployment
├── .gitignore
└── ReadMe.txt
🛠️ Tech Stack
Backend: Flask (Python)
Machine Learning: TensorFlow / Keras
Data Processing: NumPy, Pandas, Scikit-learn
Frontend: HTML, CSS
Deployment: Render
⚙️ Installation
1. Clone the repository
git clone https://github.com/sawankr1987/Just_check_render_.git
cd Just_check_render_
2. Create and activate virtual environment
conda create -n diabetes_env python=3.10
conda activate diabetes_env
3. Install dependencies
pip install -r requirements.txt
▶️ Running the Application
python app.py

Then open your browser at:

http://127.0.0.1:5000/
🧠 Model Details
Model Type: Deep Neural Network (TensorFlow/Keras)
Task: Binary Classification (Diabetes: Yes/No)
Dataset: diabetes.csv
Preprocessing: Standard scaling using scaler.pkl
🌍 Deployment (Render)

This project is configured for deployment on Render.

Key Requirements:
requirements.txt → dependencies list
runtime.txt → Python version
Entry point → app.py
🧪 Model Training

To retrain the model:

Open:

training_model_and_save.ipynb
Run all cells
Save updated files:
model_tf.h5
scaler.pkl
⚠️ Important Notes
Ensure feature order matches training data
Always use the same scaler for prediction
Flask requires templates inside /templates
🤝 Contributing

Contributions are welcome!

Fork the repo
Create a new branch
Make changes
Submit a Pull Request
📄 License

This project is licensed under the MIT License.

👨‍💻 Author

Sawan Kumar
GitHub: https://github.com/sawankr1987
