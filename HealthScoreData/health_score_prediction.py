import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib 

file_path = "HealthScoreData/synthetic_expanded_health_data.csv"

print(f"Loading dataset: {file_path}...")
try:
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please ensure the file exists in the specified path.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    exit()

# --- 2. Define Features (X) and Target (y) ---
features = ['Age', 'BMI', 'Exercise_Frequency', 'Sleep_Hours', 'Smoking_Status']
target = 'Health_Score'

# Verify that all specified columns exist in the DataFrame
missing_features = [col for col in features if col not in df.columns]
if missing_features:
    print(f"Error: The following feature columns are missing in the dataset: {missing_features}")
    exit()
if target not in df.columns:
    print(f"Error: The target column '{target}' is missing in the dataset.")
    exit()

X = df[features]
y = df[target]
print(f"\nFeatures selected: {features}")
print(f"Target selected: {target}")

# --- 3. Data Preprocessing (Check for missing values) ---
print("\nChecking for missing values...")
if X.isnull().sum().any():
    print("Missing values found in features (X):")
    print(X.isnull().sum())
    print("Dropping rows with missing values for this demonstration...")
    temp_df = X.join(y)
    temp_df.dropna(inplace=True)
    X = temp_df[features]
    y = temp_df[target]
    print(f"Shape after dropping NaNs: X - {X.shape}, y - {y.shape}")
else:
    print("No missing values found in the selected features.")

if y.isnull().sum().any():
    print("Missing values found in target (y).")
    y.dropna(inplace=True)
    X = X.loc[y.index]
    print(f"Shape after dropping NaNs from y and realigning X: X - {X.shape}, y - {y.shape}")
else:
    print("No missing values found in the target variable.")


if X.empty or y.empty:
    print("Error: No data left after handling missing values. Cannot proceed.")
    exit()

# --- 4. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nData split into training and testing sets:")
print(f"X_train shape: {X_train.shape} ({len(X_train)/len(df)*100:.2f}%)")
print(f"X_test shape: {X_test.shape} ({len(X_test)/len(df)*100:.2f}%)")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# --- 5. Model Training ---

# Model 1: Linear Regression (Optional, you can comment this out if you only want Random Forest)
print("\n--- Training Linear Regression Model ---")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print("Linear Regression Model trained.")

# Model 2: Random Forest Regressor
print("\n--- Training Random Forest Regressor Model ---")
print("(This may take a moment for a large dataset)...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=50,
    min_samples_leaf=25,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("Random Forest Regressor Model trained.")

# --- 6. Model Evaluation (Optional, can be commented out if only saving the model) ---

print("\n--- Linear Regression Model Evaluation ---")
y_pred_lr = lr_model.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"MAE: {mae_lr:.4f}, MSE: {mse_lr:.4f}, RMSE: {rmse_lr:.4f}, R²: {r2_lr:.4f}")

print("\n--- Random Forest Regressor Model Evaluation ---")
y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"MAE: {mae_rf:.4f}, MSE: {mse_rf:.4f}, RMSE: {rmse_rf:.4f}, R²: {r2_rf:.4f}")

# --- 7. Feature Importances (Optional) ---
print("\nFeature Importances (from Random Forest Regressor):")
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

# --- 8. Export/Save the Random Forest Model ---
model_filename = 'random_forest_health_score_model.joblib'
print(f"\n--- Exporting Random Forest Model ---")
try:
    joblib.dump(rf_model, model_filename)
    print(f"Random Forest model saved successfully as '{model_filename}'")
except Exception as e:
    print(f"Error saving model: {e}")

print("\n--- Script Finished ---")



# (comp7705_env) macbookpro@Mac-mini comp7705_model % python health_score_data/health_score_prediction.py 
# Loading dataset: health_score_data/synthetic_expanded_health_data.csv...
# Dataset loaded successfully. Shape: (200000, 8)

# Features selected: ['Age', 'BMI', 'Exercise_Frequency', 'Sleep_Hours', 'Smoking_Status']
# Target selected: Health_Score

# Checking for missing values...
# No missing values found in the selected features.
# No missing values found in the target variable.

# Data split into training and testing sets:
# X_train shape: (160000, 5) (80.00%)
# X_test shape: (40000, 5) (20.00%)
# y_train shape: (160000,)
# y_test shape: (40000,)

# --- Training Linear Regression Model ---
# Linear Regression Model trained.

# --- Training Random Forest Regressor Model ---
# (This may take a moment for 200,000 rows)...
# Random Forest Regressor Model trained.

# --- Linear Regression Model Evaluation ---
# Mean Absolute Error (MAE): 8.6267
# Mean Squared Error (MSE): 116.9050
# Root Mean Squared Error (RMSE): 10.8123
# R-squared (R²): 0.3644

# --- Random Forest Regressor Model Evaluation ---
# Mean Absolute Error (MAE): 2.6624
# Mean Squared Error (MSE): 17.8098
# Root Mean Squared Error (RMSE): 4.2202
# R-squared (R²): 0.9032

# Feature Importances (from Random Forest Regressor):
#               Feature  Importance
# 1                 BMI    0.302404
# 3         Sleep_Hours    0.255755
# 0                 Age    0.211438
# 2  Exercise_Frequency    0.174631
# 4      Smoking_Status    0.055772

# --- Interpretation Notes ---
# R-squared (R²): Ranges from 0 to 1. Closer to 1 means the model explains more variance in the target variable.
# MAE, MSE, RMSE: Measure the error of predictions. Lower values are better.
# Feature Importances: Show the relative contribution of each feature to the Random Forest model's predictions.

# --- Script Finished ---
