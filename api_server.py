from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and preprocessing objects
model = load_model('insurance_nn_model_no_child.keras')  # Neural network model
scaler = joblib.load('scaler.pkl')  # Feature scaler
preprocessor = joblib.load('preprocessor.pkl')  # Column transformer

# Bin configurations for age groups and BMI categories
AGE_BINS = [18, 25, 35, 45, 55, 65]
AGE_LABELS = [0, 1, 2, 3, 4]
BMI_BINS = [0, 18.5, 25, 30, np.inf]
BMI_LABELS = ['underweight', 'normal', 'overweight', 'obese']

def calculate_health_score(steps, heart_rate):
    """Calculate health score based on activity and vitals"""
    # Step score (0-50 points, 8000 steps = 50 points)
    step_score = min(50, steps / 160)  
    
    # Heart rate score (0-50 points)
    if 60 <= heart_rate <= 70:  # Optimal range
        hr_score = 50
    elif 50 <= heart_rate < 60 or 70 < heart_rate <= 80:  # Good range
        hr_score = 40
    elif 40 <= heart_rate < 50 or 80 < heart_rate <= 90:  # Fair range
        hr_score = 30
    else:  # Critical range
        hr_score = 10
    
    return step_score + hr_score  # Total score out of 100

def calculate_discount(health_score):
    """Determine discount percentage based on health score"""
    if health_score >= 90:
        return 0.20  # 20% discount
    elif health_score >= 75:
        return 0.15
    elif health_score >= 60:
        return 0.10
    elif health_score >= 40:
        return 0.05
    return 0  # No discount

def get_health_assessment(score):
    """Generate human-readable health assessment"""
    if score >= 90: return "Excellent health! Maximum discount applied"
    elif score >= 75: return "Great health! Significant discount"
    elif score >= 60: return "Good health! Moderate discount"
    elif score >= 40: return "Fair health! Small discount"
    return "Consider improving your health for future discounts"

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON input data
    data = request.get_json()
    
    # Required parameters (matches original API)
    required = ['age', 'bmi']
    if not all(key in data for key in required):
        return jsonify({
            'error': 'Missing required parameters',
            'expected': required,
            'received': list(data.keys()) if data else None
        }), 400

    try:
        # Extract parameters with original defaults
        age = float(data['age'])
        bmi = float(data['bmi'])
        sex = data.get('sex', 1)  # Default: female (original behavior)
        smoker = data.get('smoker', 0)  # Default: non-smoker (original behavior)
        
        # New health parameters (optional with defaults)
        steps = data.get('steps', 5000)  # Default steps if not provided
        heart_rate = data.get('heartRate', 72)  # Default heart rate if not provided

        # Validate numerical inputs
        if any(val < 0 for val in [age, bmi, steps, heart_rate]):
            return jsonify({'error': 'Negative values not allowed'}), 400
        if heart_rate < 40 or heart_rate > 200:
            return jsonify({'error': 'Invalid heart rate (40-200 bpm required)'}), 400

        # Feature engineering (matches original)
        input_df = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'smoker': smoker,
            'age_bmi': age * bmi / 100,
            'bmi_smoker': bmi * smoker,
            'age_smoker': age * smoker,
            'age_group': 2,  # Original hardcoded value
            'bmi_category': 'overweight'  # Original hardcoded value
        }])

        # Scale numerical features (original process)
        num_features = ['age', 'bmi', 'age_bmi', 'bmi_smoker', 'age_smoker']
        input_df[num_features] = scaler.transform(input_df[num_features])
        
        # Apply preprocessing
        X_input = preprocessor.transform(input_df)

        # Get base premium prediction
        reg_pred, _ = model.predict(X_input)
        base_premium = float(reg_pred.flatten()[0])

        # Calculate health benefits (new functionality)
        health_score = calculate_health_score(steps, heart_rate)
        discount_rate = calculate_discount(health_score)
        discounted_premium = base_premium * (1 - discount_rate)

        return jsonify({
            'base_premium': base_premium,
            'health_score': round(health_score, 1),
            'discount_rate': f"{discount_rate * 100:.1f}%",
            'final_premium': round(discounted_premium, 2),
            'health_assessment': get_health_assessment(health_score)
        })

    except ValueError as e:
        return jsonify({'error': f'Invalid input format: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(port=5050, debug=True)
