from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and preprocessors ONCE
model = load_model('insurance_nn_model_no_child.keras')
scaler = joblib.load('scaler.pkl')
preprocessor = joblib.load('preprocessor.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Extract input features
    age = data.get('age')
    bmi = data.get('bmi')
    sex = data.get('sex', 1)  # Default: female
    smoker = data.get('smoker', 0)  # Default: non-smoker

    # Feature engineering (match training)
    input_df = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'smoker': smoker,
        'age_bmi': age * bmi / 100,
        'bmi_smoker': bmi * smoker,
        'age_smoker': age * smoker,
        'age_group': 2,  # You may want to automate binning
        'bmi_category': 'overweight'  # You may want to automate binning
    }])

    # Scale numerical features
    num_features = ['age', 'bmi', 'age_bmi', 'bmi_smoker', 'age_smoker']
    input_df[num_features] = scaler.transform(input_df[num_features])
    # Apply column transformer
    X_input = preprocessor.transform(input_df)

    # Predict
    reg_pred, _ = model.predict(X_input)
    predicted_premium = float(reg_pred.flatten()[0])
    return jsonify({'predicted_premium': predicted_premium})

if __name__ == '__main__':
    app.run(port=5050, debug=True)
