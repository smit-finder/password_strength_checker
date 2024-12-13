import re
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import tensorflow as tf
from tensorflow import keras

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)
CORS(app)

model_path = r'C:\Users\smitr\OneDrive\Documents\python_projects\password_strength_checker_app\backend\password_strength_model.h5'
model = keras.models.load_model(model_path)

scaler_path = r'C:\Users\smitr\OneDrive\Documents\python_projects\password_strength_checker_app\backend\password_strength_scaler.pkl'
scaler = joblib.load(scaler_path)

def extract_password_features(password):
    if not isinstance(password, str):
        password = str(password)
    features = {
        'password_length': len(password),
        'has_special_char': int(bool(re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?/\\]', password))),
        'has_uppercase': int(bool(re.search(r'[A-Z]', password))),
        'has_lowercase': int(bool(re.search(r'[a-z]', password))),
        'has_digit': int(bool(re.search(r'\d', password))),
        'consecutive_digits': len(re.findall(r'\d{2,}', password)),
        'consecutive_letters': len(re.findall(r'[a-zA-Z]{2,}', password))
    }
    return features

feature_columns = ['password_length', 'has_special_char', 'has_uppercase', 'has_lowercase', 'has_digit', 'consecutive_digits', 'consecutive_letters']

def preprocess_password(password):
    features = extract_password_features(password)
    features_array = np.array([features[col] for col in feature_columns]).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    return features_scaled

@app.route('/check-password', methods=['POST'])
def check_password():
    data = request.json
    password = data.get('password', '')
    
    if not password:
        return jsonify({'error': 'Password is required'}), 400
    
    try:
        processed_password = preprocess_password(password)
        prediction = model.predict(processed_password)
        strength_class = np.argmax(prediction)
        if strength_class == 0:
            strength_label = "Weak"
        elif strength_class == 1:
            strength_label = "Moderate"
        else:
            strength_label = "Strong"
        probabilities = prediction[0]
        return jsonify({
            'strength': strength_label,
            'probabilities': {
                'Weak': float(probabilities[0]),
                'Moderate': float(probabilities[1]),
                'Strong': float(probabilities[2])
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)