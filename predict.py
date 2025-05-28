import joblib
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

# User input variables
age = 40
bmi = 28.0

# Load model and preprocessors
model = load_model('insurance_nn_model_no_child.keras')
scaler = joblib.load('scaler.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Example input (replace with your actual data)
input_data = pd.DataFrame([{
    'age': age,
    'sex': 1,  # 1 for female, 0 for male
    'bmi': bmi,
    'smoker': 0,  # 1 for yes, 0 for no
    'age_bmi': age * bmi / 100,
    'bmi_smoker': bmi * 0,
    'age_smoker': age * 0,
    'age_group': 2,  # Use the same binning as in prepare_data
    'bmi_category': 'overweight'
}])

# Scale numerical features
num_features = ['age', 'bmi', 'age_bmi', 'bmi_smoker', 'age_smoker']
input_data[num_features] = scaler.transform(input_data[num_features])

# Apply column transformer
X_input = preprocessor.transform(input_data)

# Predict (returns [regression_output, classification_output])
reg_pred, _ = model.predict(X_input)
predicted_premium = reg_pred.flatten()[0]
print(f"Predicted insurance premium: ${predicted_premium:.2f}")
