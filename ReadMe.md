# 🧠 Diabetes Prediction Web App

A machine learning–based web application that predicts whether a person is likely to have diabetes based on input health parameters. This project combines a trained deep learning model with a Flask web interface for real-time predictions.

---

## 🚀 Overview

This application allows users to input medical attributes (such as glucose level, BMI, age, etc.) and receive an instant prediction powered by a trained TensorFlow model.

---

## ✨ Features

- User-friendly web interface (HTML + Flask)
- Pre-trained deep learning model (`model_tf.h5`)
- Data preprocessing using `scaler.pkl`
- Real-time prediction
- Deployment-ready (Render compatible)

---

## 📁 Project Structure

.
├── app.py                          # Flask backend application  
├── templates/  
│   └── index.html                  # Frontend UI  
├── model_tf.h5                     # Trained TensorFlow model  
├── scaler.pkl                      # Data scaler  
├── diabetes.csv                    # Dataset  
├── training_model_and_save.ipynb   # Model training notebook  
├── requirements.txt                # Dependencies  
├── runtime.txt                     # Python version for deployment  
├── .gitignore  
└── ReadMe.txt  

---

## 🛠️ Tech Stack

- Python  
- Flask  
- TensorFlow / Keras  
- Scikit-learn  
- HTML / CSS  

---

## ⚙️ Installation

### 1. Clone the repository

git clone https://github.com/sawankr1987/Just_check_render_.git  
cd Just_check_render_  

### 2. Create virtual environment (optional but recommended)

conda create -n diabetes_env python=3.10  
conda activate diabetes_env  

### 3. Install dependencies

pip install -r requirements.txt  

---

## ▶️ Run the Application

python app.py  

Open your browser and go to:  
http://127.0.0.1:5000/

---

## 🧠 Model Details

- Model: Deep Neural Network (TensorFlow/Keras)  
- Task: Binary Classification (Diabetes Prediction)  
- Dataset: `diabetes.csv`  
- Preprocessing: Standard scaling (`scaler.pkl`)  

---

## 🌍 Deployment

This project is ready to deploy on platforms like Render.

Make sure:
- `requirements.txt` is updated  
- `runtime.txt` specifies Python version  
- Entry point is set to `app.py`  

---

## 🧪 Training the Model

1. Open `training_model_and_save.ipynb`  
2. Run all cells  
3. Save updated files:  
   - `model_tf.h5`  
   - `scaler.pkl`  

---

## ⚠️ Notes

- Maintain correct feature order during prediction  
- Always use the same scaler used during training  
- Keep HTML files inside the `/templates` folder  

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository  
2. Create a new branch  
3. Make changes  
4. Submit a pull request  

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Sawan Kumar  
https://github.com/sawankr1987
