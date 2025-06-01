from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import traceback # For more detailed error logging

app = Flask(__name__)

# --- Global Configuration ---
# Set to True to use the Neural Network health model, False for Random Forest
USE_NN_HEALTH_MODEL = True

# --- Load common models and preprocessing objects ---
insurance_model = load_model('insurance_nn_model_no_child.keras')  # NN premium prediction model
scaler = joblib.load('scaler.pkl')  # Feature scaler for insurance model
preprocessor = joblib.load('preprocessor.pkl')  # Column transformer for insurance model

# --- Conditionally load health score model and related objects ---
if USE_NN_HEALTH_MODEL:
    health_score_model = load_model('health_score_nn_model.keras')  # NN health score model
    health_scaler = joblib.load('health_score_scaler.pkl')  # Scaler for NN health score features
    # TRAIN_MIN_SCORE and TRAIN_MAX_SCORE are not used for the NN model in this setup
    TRAIN_MIN_SCORE = None # Not used by NN
    TRAIN_MAX_SCORE = None # Not used by NN
else:
    health_score_model = joblib.load('random_forest_health_score_model.joblib')  # RF Health score model
    health_scaler = None # Not used by RF model
    # Health score scaling parameters for RF model (MUST ADJUST THESE BASED ON YOUR RF MODEL's RAW OUTPUT)
    TRAIN_MIN_SCORE = 0    # Replace with actual minimum from RF training output
    TRAIN_MAX_SCORE = 100  # Replace with actual maximum from RF training output

# --- Bin configurations (common) ---
AGE_BINS = [18, 25, 35, 45, 55, 65] #
AGE_LABELS = [0, 1, 2, 3, 4]  # Must match training labels
BMI_BINS = [0, 18.5, 25, 30, np.inf] #
BMI_LABELS = ['underweight', 'normal', 'overweight', 'obese']  # Must match training

# --- Helper Functions ---

def steps_to_exercise_frequency(steps):
    """Convert steps to Exercise_Frequency (0-6) as per requirements""" #
    if steps < 2000: return 0
    elif steps < 4000: return 1
    elif steps < 6000: return 2
    elif steps < 8000: return 3
    elif steps < 10000: return 4
    elif steps < 12000: return 5
    else: return 6

def calculate_age_group(age):
    """Calculate age group matching training data preprocessing""" #
    age_group = pd.cut([age],
                      bins=AGE_BINS,
                      labels=AGE_LABELS,
                      include_lowest=True)[0]
    return int(age_group) if not pd.isna(age_group) else AGE_LABELS[-1]

def calculate_bmi_category(bmi):
    """Calculate BMI category matching training data preprocessing""" #
    bmi_category = pd.cut([bmi],
                         bins=BMI_BINS,
                         labels=BMI_LABELS,
                         include_lowest=True)[0]
    return str(bmi_category) if not pd.isna(bmi_category) else BMI_LABELS[-1]

def scale_health_score(raw_score):
    """Scale raw model output (primarily for RF model) to 0-100 range""" #
    # This function is primarily for the Random Forest model if its output isn't already 0-100.
    # For NN, we scale its 0-1 output by multiplying by 100.
    if TRAIN_MIN_SCORE is None or TRAIN_MAX_SCORE is None: # Should not happen if RF model is used with defined constants
        app.logger.warning("TRAIN_MIN_SCORE or TRAIN_MAX_SCORE not set for scale_health_score with RF model.")
        return max(0, min(100, raw_score)) # Basic clamping if constants are missing

    if TRAIN_MIN_SCORE == 0 and TRAIN_MAX_SCORE == 100: # If RF model already outputs 0-100
        return max(0, min(100, raw_score)) # Ensure it's strictly within bounds

    # Linear scaling to 0-100 range
    if (TRAIN_MAX_SCORE - TRAIN_MIN_SCORE) == 0: # Avoid division by zero
        app.logger.warning("TRAIN_MAX_SCORE and TRAIN_MIN_SCORE are equal, cannot scale.")
        return max(0, min(100, raw_score))

    scaled = ((raw_score - TRAIN_MIN_SCORE) / (TRAIN_MAX_SCORE - TRAIN_MIN_SCORE)) * 100 #
    return max(0, min(100, scaled))  # Ensure within bounds

def calculate_discount(health_score_0_100): # health_score is expected to be in 0-100 range
    """
    Determines the premium multiplier based on a health score (0-100).
    Returns the premium multiplier, or None if coverage is rejected.
    (Logic from api_server_nn_health_score.py / api_server_rf_health_score.py)
    """
    #
    if health_score_0_100 < 40:
        return None  # Reject coverage
    elif health_score_0_100 < 50:
        return 2.5   # Premium multiplier
    elif health_score_0_100 < 60:
        return 2.0   # Premium multiplier
    elif health_score_0_100 < 70:
        return 1.5   # Premium multiplier
    elif health_score_0_100 < 80:
        return 1.0   # Standard premium, multiplier is 1.0
    elif health_score_0_100 < 90:
        return 0.8   # 20% discount, multiplier is 0.8
    elif health_score_0_100 < 95:
        return 0.65  # 35% discount, multiplier is 0.65
    else: # health_score_0_100 >= 95
        return 0.5   # 50% discount, multiplier is 0.5

def get_health_assessment(health_score_0_100): # health_score is expected to be in 0-100 range
    """
    Generates a human-readable health assessment based on a health score (0-100).
    (Logic from api_server_nn_health_score.py - corrected version)
    """
    #
    if health_score_0_100 < 40:
        return "Reject! Application declined due to critical health risk"
    elif health_score_0_100 < 50:
        return "Approved! High risk applicant - 250% premium applied"
    elif health_score_0_100 < 60:
        return "Approved! Elevated risk applicant - 200% premium applied"
    elif health_score_0_100 < 70:
        return "Approved! Moderate risk applicant - 150% premium applied"
    elif health_score_0_100 < 80:
        return "Approved! Standard risk applicant"
    elif health_score_0_100 < 90:
        return "Approved! Preferred risk applicant - 20% discount"
    elif health_score_0_100 < 95:
        return "Approved! Excellent health - 35% discount"
    else: # health_score_0_100 >= 95
        return "Approved! Exceptional health - 50% discount"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

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
        sex = int(data.get('sex', 1))  # Defaulting to 1 if not provided
        smoker = int(data.get('smoker', 0))  # Defaulting to 0 if not provided
        steps = int(data.get('steps', 5000))  # Defaulting to 5000 if not provided
        sleep_hours = float(data.get('sleepHours', 7))  # Defaulting to 7 if not provided
        heart_rate = float(data.get('heartRate', 72))  # Defaulting to 72 if not provided

        if any(val < 0 for val in [age, bmi, steps, sleep_hours, heart_rate]):
            return jsonify({'error': 'Negative values not allowed'}), 400
        if heart_rate < 40 or heart_rate > 200: # Basic validation for heart rate
            return jsonify({'error': 'Invalid heart rate (40-200 bpm required)'}), 400

        # Calculate dynamic categories
        age_group = calculate_age_group(age)
        bmi_category = calculate_bmi_category(bmi)
        exercise_freq = steps_to_exercise_frequency(steps)

        # Create input DataFrame for insurance model (base premium calculation)
        insurance_input_df = pd.DataFrame([{
            'age': age, 'sex': sex, 'bmi': bmi, 'smoker': smoker,
            'age_bmi': age * bmi / 100, # Example feature engineering
            'bmi_smoker': bmi * smoker, # Example feature engineering
            'age_smoker': age * smoker, # Example feature engineering
            'age_group': age_group,
            'bmi_category': bmi_category
        }])
        insurance_input_df['age_group'] = insurance_input_df['age_group'].astype(float)
        insurance_input_df['bmi_category'] = insurance_input_df['bmi_category'].astype(str)
        num_features = ['age', 'bmi', 'age_bmi', 'bmi_smoker', 'age_smoker']
        insurance_input_df[num_features] = scaler.transform(insurance_input_df[num_features])
        X_insurance = preprocessor.transform(insurance_input_df)

        reg_pred, _ = insurance_model.predict(X_insurance)
        base_premium = float(reg_pred.flatten()[0])

        # Create input DataFrame for health score model
        health_input_df = pd.DataFrame([{
            'Age': age, 'BMI': bmi, 'Exercise_Frequency': exercise_freq,
            'Sleep_Hours': sleep_hours, 'Smoking_Status': smoker
        }])

        # Initialize variables for health score processing
        raw_model_output = None
        health_score_0_100 = None # This will be the 0-100 scaled score
        model_name_for_output = ""

        # Conditional health score prediction and processing
        if USE_NN_HEALTH_MODEL:
            model_name_for_output = 'Neural Network'
            if health_scaler is None:
                app.logger.error("Health scaler not loaded for NN model.")
                return jsonify({'error': 'Server configuration error: Health scaler missing for NN.'}), 500
            health_input_scaled = health_scaler.transform(health_input_df)
            raw_model_output = float(health_score_model.predict(health_input_scaled)[0][0])
            # Corrected processing for NN: use raw_model_output directly if it's already 0-100 scale, then clamp.
            health_score_0_100 = max(0, min(100, raw_model_output))
        else: # Use Random Forest Model
            model_name_for_output = 'Random Forest'
            raw_model_output = health_score_model.predict(health_input_df)[0]
            health_score_0_100 = scale_health_score(raw_model_output) # scale_health_score ensures 0-100

        # Get assessment and discount multiplier based on the 0-100 score
        health_assessment_message = get_health_assessment(health_score_0_100)
        premium_multiplier = calculate_discount(health_score_0_100)

        # Prepare the response data, ensuring field names match Swift struct where applicable
        response_data = {
            # Fields expected by Swift InsurancePrediction struct
            'base_premium': base_premium,
            'health_score': round(health_score_0_100, 2),

            # Additional fields (Swift will ignore these if not in its struct)
            'age_group': int(age_group),
            'bmi_category': bmi_category,
            'exercise_frequency': int(exercise_freq),
            'steps_converted': steps,
            'raw_model_output': round(raw_model_output, 4) if raw_model_output is not None else None,
            'health_model_used': model_name_for_output
        }

        if premium_multiplier is None:
            # Application is rejected
            response_data.update({
                'discount_rate': "N/A",
                'final_premium': 0.0,
                'health_assessment': health_assessment_message
            })
            return jsonify(response_data)

        # Application is approved, calculate final premium and discount string
        final_premium_calculated = base_premium * premium_multiplier
        actual_discount_value = (1 - premium_multiplier)
        discount_display_str = f"{actual_discount_value * 100:.1f}%"

        response_data.update({
            'discount_rate': discount_display_str,
            'final_premium': round(final_premium_calculated, 2),
            'health_assessment': health_assessment_message
        })
        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f'Processing failed: {str(e)}')
        app.logger.error(traceback.format_exc()) # Provides full stack trace to server logs
        return jsonify({'error': f'Processing failed: {str(e)}'}), 400


if __name__ == '__main__':
    app.run(port=5050, debug=True)