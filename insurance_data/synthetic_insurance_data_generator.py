import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Read your original data
original_data = pd.read_csv('insurance.csv')

# Preprocess data
categorical_features = ['sex', 'smoker', 'region']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'
)

X = original_data.drop('charges', axis=1)
y = original_data['charges']

# Train a model to learn relationships
model = RandomForestRegressor(n_estimators=100, random_state=42)
X_processed = preprocessor.fit_transform(X)
model.fit(X_processed, y)

# Generate synthetic data
np.random.seed(42)
n_samples = 200000

# Generate features based on original data distributions
synthetic = pd.DataFrame({
    'age': np.random.normal(X['age'].mean(), X['age'].std(), n_samples).astype(int).clip(18, 65),
    'sex': np.random.choice(X['sex'].unique(), n_samples, p=X['sex'].value_counts(normalize=True).values),
    'bmi': np.random.normal(X['bmi'].mean(), X['bmi'].std(), n_samples).clip(15, 50).round(1),
    'children': np.random.poisson(X['children'].mean(), n_samples).clip(0, 5),
    'smoker': np.random.choice(X['smoker'].unique(), n_samples, p=X['smoker'].value_counts(normalize=True).values),
    'region': np.random.choice(X['region'].unique(), n_samples, p=X['region'].value_counts(normalize=True).values)
})

# Predict charges using the trained model
synthetic_processed = preprocessor.transform(synthetic)
synthetic_charges = model.predict(synthetic_processed)

# Add realistic noise
noise = np.random.normal(0, y.std()/10, n_samples)
synthetic['charges'] = np.round(synthetic_charges + noise, 2)

# Save synthetic data
synthetic.to_csv('synthetic_insurance_data.csv', index=False)
