from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models and preprocessing objects
insurance_model = load_model('insurance_nn_model_no_child.keras')  # NN premium prediction model
health_score_model = load_model('health_score_nn_model.keras')  # NN health score model
health_scaler = joblib.load('health_score_scaler.pkl')  # Scaler for health score features
scaler = joblib.load('scaler.pkl')  # Feature scaler for insurance model
preprocessor = joblib.load('preprocessor.pkl')  # Column transformer for insurance model

# Bin configurations for age groups and BMI categories
AGE_BINS = [18, 25, 35, 45, 55, 65]
AGE_LABELS = [0, 1, 2, 3, 4]  # Must match training labels
BMI_BINS = [0, 18.5, 25, 30, np.inf]
BMI_LABELS = ['underweight', 'normal', 'overweight', 'obese']  # Must match training

def steps_to_exercise_frequency(steps):
    """Convert steps to Exercise_Frequency (0-6) as per requirements"""
    if steps < 2000:
        return 0
    elif steps < 4000:
        return 1
    elif steps < 6000:
        return 2
    elif steps < 8000:
        return 3
    elif steps < 10000:
        return 4
    elif steps < 12000:
        return 5
    else:
        return 6

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

def calculate_discount(health_score): # health_score is expected to be in 0-100 range
    """
    Determines the premium multiplier based on a health score (0-100).
    - Scores below 40: Reject coverage (return None)
    - Very poor health (40-50): 2.5x premium
    - Poor health (50-60): 2x premium
    - Below average (60-70): 1.5x premium
    - Average (70-80): Standard premium (1x)
    - Good health (80-90): 20% discount (0.8x premium)
    - Excellent health (90-95): 35% discount (0.65x premium)
    - Exceptional health (95+): 50% discount (0.5x premium)
    Returns the premium multiplier, or None if coverage is rejected.
    """
    if health_score < 40:
        return None  # Reject coverage
    elif health_score < 50:
        return 2.5   # Premium multiplier
    elif health_score < 60:
        return 2.0   # Premium multiplier
    elif health_score < 70:
        return 1.5   # Premium multiplier
    elif health_score < 80:
        return 1.0   # Standard premium, multiplier is 1.0
    elif health_score < 90:
        return 0.8   # 20% discount, multiplier is 0.8
    elif health_score < 95:
        return 0.65  # 35% discount, multiplier is 0.65
    else: # health_score >= 95
        return 0.5   # 50% discount, multiplier is 0.5

def get_health_assessment(health_score): # health_score is expected to be in 0-100 range
    """Generates a human-readable health assessment based on a health score (0-100)"""
    if health_score < 40:
        return "Reject! Application declined due to critical health risk"
    elif health_score < 50:
        return "Approved! High risk applicant - 250% premium applied"
    elif health_score < 60:
        return "Approved! Elevated risk applicant - 200% premium applied"
    elif health_score < 70:
        return "Approved! Moderate risk applicant - 150% premium applied"
    elif health_score < 80:
        return "Approved! Standard risk applicant"
    elif health_score < 90:
        return "Approved! Preferred risk applicant - 20% discount"
    elif health_score < 95:
        return "Approved! Excellent health - 35% discount"
    else: # health_score >= 95
        return "Approved! Exceptional health - 50% discount"

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
        sleep_hours = float(data.get('sleepHours', 7))
        heart_rate = float(data.get('heartRate', 72))

        if any(val < 0 for val in [age, bmi, steps, sleep_hours, heart_rate]):
            return jsonify({'error': 'Negative values not allowed'}), 400
        if heart_rate < 40 or heart_rate > 200:
            return jsonify({'error': 'Invalid heart rate (40-200 bpm required)'}), 400

        # Calculate dynamic categories
        age_group = calculate_age_group(age)
        bmi_category = calculate_bmi_category(bmi)
        exercise_freq = steps_to_exercise_frequency(steps)

        # Create input DataFrame for insurance model
        insurance_input_df = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'smoker': smoker,
            'age_bmi': age * bmi / 100,
            'bmi_smoker': bmi * smoker,
            'age_smoker': age * smoker,
            'age_group': age_group,
            'bmi_category': bmi_category
        }])

        # Ensure proper data types
        insurance_input_df['age_group'] = insurance_input_df['age_group'].astype(float)
        insurance_input_df['bmi_category'] = insurance_input_df['bmi_category'].astype(str)

        # Scale numerical features
        num_features = ['age', 'bmi', 'age_bmi', 'bmi_smoker', 'age_smoker']
        insurance_input_df[num_features] = scaler.transform(insurance_input_df[num_features])

        # Apply preprocessing
        X_insurance = preprocessor.transform(insurance_input_df)

        # Get insurance predictions
        reg_pred, _ = insurance_model.predict(X_insurance)
        base_premium = float(reg_pred.flatten()[0])

        # Create input for health score NN model
        health_input = pd.DataFrame([{
            'Age': age,
            'BMI': bmi,
            'Exercise_Frequency': exercise_freq,
            'Sleep_Hours': sleep_hours,
            'Smoking_Status': smoker
        }])

        # Scale features for NN health score model
        health_input_scaled = health_scaler.transform(health_input)

        # Predict health score using the NN model
        health_score = float(health_score_model.predict(health_input_scaled)[0][0])  # Convert to native Python float
        
        # Calculate discount based on predicted health score
        discount_rate = calculate_discount(health_score)
        discounted_premium = base_premium * (1 - discount_rate)

        return jsonify({
            'base_premium': base_premium,
            'health_score': health_score,  # Now a serializable float
            'discount_rate': f"{discount_rate * 100:.1f}%",
            'final_premium': round(discounted_premium, 2),
            'health_assessment': get_health_assessment(health_score),
            'age_group': int(age_group),  # Ensure serializable
            'bmi_category': bmi_category,
            'exercise_frequency': int(exercise_freq),
            'steps_converted': steps
        })

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 400


if __name__ == '__main__':
    app.run(port=5050, debug=True)
