import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

# Constants
MODEL_FILENAME = 'HealthScoreModel/health_score_nn_model.keras'
SCALER_FILENAME = 'HealthScoreModel/health_score_scaler.pkl'
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

def load_data(file_path):
    """Load and validate the dataset"""
    print(f"Loading dataset: {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        exit()

def validate_features(df, features, target):
    """Validate that required features exist in the dataset"""
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        print(f"Error: Missing feature columns: {missing_features}")
        exit()
    if target not in df.columns:
        print(f"Error: Target column '{target}' is missing.")
        exit()

def clean_data(df, features, target):
    """Handle missing values and clean data"""
    print("\nChecking for missing values...")
    if df[features].isnull().sum().any() or df[target].isnull().any():
        print("Missing values found. Dropping rows with missing values...")
        df_clean = df.dropna(subset=features+[target])
        print(f"Shape after cleaning: {df_clean.shape}")
        return df_clean
    print("No missing values found.")
    return df

def prepare_data(df, features, target):
    """Split data into training and test sets"""
    X = df[features]
    y = df[target]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print("\nData split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def build_model(input_shape):
    """Build and compile the neural network model"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,), 
              kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # Linear activation for regression
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model with early stopping"""
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test set"""
    y_pred = model.predict(X_test).flatten()
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Evaluation:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('Actual Health Score')
    plt.ylabel('Predicted Health Score')
    plt.title('Actual vs Predicted Health Scores')
    plt.savefig('health_score_predictions.png')
    plt.close()

def save_artifacts(model, scaler):
    """Save model and scaler to disk"""
    print("\nSaving model and scaler...")
    model.save(MODEL_FILENAME)
    joblib.dump(scaler, SCALER_FILENAME)
    print(f"Model saved as {MODEL_FILENAME}")
    print(f"Scaler saved as {SCALER_FILENAME}")

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Configuration
    file_path = "HealthScoreData/synthetic_expanded_health_data.csv"
    features = ['Age', 'BMI', 'Exercise_Frequency', 'Sleep_Hours', 'Smoking_Status']
    target = 'Health_Score'
    
    # Load and prepare data
    df = load_data(file_path)
    validate_features(df, features, target)
    df_clean = clean_data(df, features, target)
    X_train, X_test, y_train, y_test = prepare_data(df_clean, features, target)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Further split training set for validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE
    )
    
    # Build and train model
    model = build_model(X_train_scaled.shape[1])
    print("\nModel Summary:")
    model.summary()
    
    print("\nTraining model...")
    history = train_model(model, X_train_final, y_train_final, X_val, y_val)
    
    # Evaluate and save
    evaluate_model(model, X_test_scaled, y_test)
    plot_training_history(history)
    save_artifacts(model, scaler)
    
    print("\nTraining complete!")

if __name__ == '__main__':
    main()


# (comp7705_env) macbookpro@Mac-mini comp7705_model % python HealthScoreData/health_score_prediction.py 
# Loading dataset: HealthScoreData/synthetic_expanded_health_data.csv...
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
# (This may take a moment for a large dataset)...
# Random Forest Regressor Model trained.

# --- Linear Regression Model Evaluation ---
# MAE: 8.6267, MSE: 116.9050, RMSE: 10.8123, R²: 0.3644

# --- Random Forest Regressor Model Evaluation ---
# MAE: 2.6624, MSE: 17.8098, RMSE: 4.2202, R²: 0.9032

# Feature Importances (from Random Forest Regressor):
#               Feature  Importance
# 1                 BMI    0.302404
# 3         Sleep_Hours    0.255755
# 0                 Age    0.211438
# 2  Exercise_Frequency    0.174631
# 4      Smoking_Status    0.055772

# --- Exporting Random Forest Model ---
# Random Forest model saved successfully as 'random_forest_health_score_model.joblib'

# --- Script Finished ---
