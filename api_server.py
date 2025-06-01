from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and preprocessing objects
model = load_model('PremiumPredictionModel/insurance_nn_model_no_child.keras')  # Neural network model
scaler = joblib.load('PremiumPredictionModel/scaler.pkl')  # Feature scaler
preprocessor = joblib.load('PremiumPredictionModel/preprocessor.pkl')  # Column transformer

# Bin configurations for age groups and BMI categories
AGE_BINS = [18, 25, 35, 45, 55, 65]
AGE_LABELS = [0, 1, 2, 3, 4]  # Must match training labels
BMI_BINS = [0, 18.5, 25, 30, np.inf]
BMI_LABELS = ['underweight', 'normal', 'overweight', 'obese']  # Must match training

def calculate_age_group(age):
    """Calculate age group matching training data preprocessing"""
    age_group = pd.cut([age], 
                      bins=AGE_BINS, 
                      labels=AGE_LABELS,
                      include_lowest=True)[0]
    # Handle NaN values (e.g., age > 65)
    return int(age_group) if not pd.isna(age_group) else AGE_LABELS[-1]

def calculate_bmi_category(bmi):
    """Calculate BMI category matching training data preprocessing"""
    bmi_category = pd.cut([bmi],
                         bins=BMI_BINS,
                         labels=BMI_LABELS,
                         include_lowest=True)[0]
    # Handle NaN values
    return str(bmi_category) if not pd.isna(bmi_category) else BMI_LABELS[-1]

def calculate_health_score(steps, heart_rate):
    """Calculate health score based on activity and vitals"""
    step_score = min(50, steps / 160)  
    if 60 <= heart_rate <= 70:
        hr_score = 50
    elif 50 <= heart_rate < 60 or 70 < heart_rate <= 80:
        hr_score = 40
    elif 40 <= heart_rate < 50 or 80 < heart_rate <= 90:
        hr_score = 30
    else:
        hr_score = 10
    return step_score + hr_score

def calculate_discount(health_score):
    """Determine discount percentage based on health score"""
    if health_score >= 90:
        return 0.20
    elif health_score >= 75:
        return 0.15
    elif health_score >= 60:
        return 0.10
    elif health_score >= 40:
        return 0.05
    return 0

def get_health_assessment(score):
    """Generate human-readable health assessment"""
    if score >= 90: return "Excellent health! Maximum discount applied"
    elif score >= 75: return "Great health! Significant discount"
    elif score >= 60: return "Good health! Moderate discount"
    elif score >= 40: return "Fair health! Small discount"
    return "Consider improving your health for future discounts"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Required parameters
    required = ['age', 'bmi']
    if not all(key in data for key in required):
        return jsonify({
            'error': 'Missing required parameters',
            'expected': required,
            'received': list(data.keys()) if data else None
        }), 400

    try:
        # Extract and validate parameters
        age = float(data['age'])
        bmi = float(data['bmi'])
        sex = int(data.get('sex', 1))  # Ensure int type
        smoker = int(data.get('smoker', 0))  # Ensure int type
        steps = int(data.get('steps', 5000))
        heart_rate = float(data.get('heartRate', 72))

        if any(val < 0 for val in [age, bmi, steps, heart_rate]):
            return jsonify({'error': 'Negative values not allowed'}), 400
        if heart_rate < 40 or heart_rate > 200:
            return jsonify({'error': 'Invalid heart rate (40-200 bpm required)'}), 400

        # Calculate dynamic categories (must match training)
        age_group = calculate_age_group(age)
        bmi_category = calculate_bmi_category(bmi)

        # Create input DataFrame with identical structure to training
        input_df = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'smoker': smoker,
            'age_bmi': age * bmi / 100,
            'bmi_smoker': bmi * smoker,
            'age_smoker': age * smoker,
            'age_group': age_group,  # Numeric value
            'bmi_category': bmi_category  # String value
        }])

        # Ensure proper data types
        input_df['age_group'] = input_df['age_group'].astype(float)  # Must match encoder
        input_df['bmi_category'] = input_df['bmi_category'].astype(str)

        # Scale numerical features
        num_features = ['age', 'bmi', 'age_bmi', 'bmi_smoker', 'age_smoker']
        input_df[num_features] = scaler.transform(input_df[num_features])

        # Apply preprocessing
        X_input = preprocessor.transform(input_df)

        # Get predictions
        reg_pred, _ = model.predict(X_input)
        base_premium = float(reg_pred.flatten()[0])

        # Calculate health benefits
        health_score = calculate_health_score(steps, heart_rate)
        discount_rate = calculate_discount(health_score)
        discounted_premium = base_premium * (1 - discount_rate)

        return jsonify({
            'base_premium': base_premium,
            'health_score': round(health_score, 1),
            'discount_rate': f"{discount_rate * 100:.1f}%",
            'final_premium': round(discounted_premium, 2),
            'health_assessment': get_health_assessment(health_score),
            'age_group': age_group,
            'bmi_category': bmi_category
        })

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(port=5050, debug=True)
