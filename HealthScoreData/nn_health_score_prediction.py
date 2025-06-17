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


# (comp7705_env) macbookpro@Mac-mini comp7705_model % python HealthScoreData/nn_health_score_prediction.py 
# Loading dataset: HealthScoreData/synthetic_expanded_health_data.csv...
# Dataset loaded successfully. Shape: (200000, 8)

# Checking for missing values...
# No missing values found.

# Data split:
# Training set: 160000 samples
# Test set: 40000 samples
# /opt/anaconda3/envs/comp7705_env/lib/python3.11/site-packages/keras/src/layers/core/dense.py:93: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
#   super().__init__(activity_regularizer=activity_regularizer, **kwargs)

# Model Summary:
# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ dense (Dense)                        │ (None, 128)                 │             768 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dropout (Dropout)                    │ (None, 128)                 │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 64)                  │           8,256 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dropout_1 (Dropout)                  │ (None, 64)                  │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_2 (Dense)                      │ (None, 32)                  │           2,080 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_3 (Dense)                      │ (None, 1)                   │              33 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 11,137 (43.50 KB)
#  Trainable params: 11,137 (43.50 KB)
#  Non-trainable params: 0 (0.00 B)

# Training model...
# Epoch 1/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 430us/step - loss: 812.7598 - mae: 18.6492 - val_loss: 121.8208 - val_mae: 9.1192
# Epoch 2/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 416us/step - loss: 157.7004 - mae: 10.0738 - val_loss: 113.9480 - val_mae: 8.3929
# Epoch 3/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 416us/step - loss: 139.7124 - mae: 9.4706 - val_loss: 111.4528 - val_mae: 8.4244
# Epoch 4/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 428us/step - loss: 130.3746 - mae: 9.1214 - val_loss: 111.5638 - val_mae: 8.6070
# Epoch 5/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 424us/step - loss: 121.6174 - mae: 8.8165 - val_loss: 106.8964 - val_mae: 8.0873
# Epoch 6/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 439us/step - loss: 115.4875 - mae: 8.5992 - val_loss: 107.4807 - val_mae: 8.4785
# Epoch 7/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 428us/step - loss: 111.2941 - mae: 8.4293 - val_loss: 103.7995 - val_mae: 8.2975
# Epoch 8/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 415us/step - loss: 108.1315 - mae: 8.2865 - val_loss: 99.6304 - val_mae: 7.9196
# Epoch 9/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 415us/step - loss: 105.4360 - mae: 8.1393 - val_loss: 95.3717 - val_mae: 7.6077
# Epoch 10/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 414us/step - loss: 102.0657 - mae: 8.0037 - val_loss: 91.3057 - val_mae: 7.4509
# Epoch 11/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 426us/step - loss: 99.4412 - mae: 7.8757 - val_loss: 87.5442 - val_mae: 7.3316
# Epoch 12/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 420us/step - loss: 95.0594 - mae: 7.6906 - val_loss: 84.1387 - val_mae: 7.0860
# Epoch 13/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 424us/step - loss: 92.1807 - mae: 7.5123 - val_loss: 80.8537 - val_mae: 7.0515
# Epoch 14/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 415us/step - loss: 89.1054 - mae: 7.3700 - val_loss: 75.8203 - val_mae: 6.7384
# Epoch 15/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 416us/step - loss: 86.9786 - mae: 7.2707 - val_loss: 73.5293 - val_mae: 6.6154
# Epoch 16/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 415us/step - loss: 84.6659 - mae: 7.1411 - val_loss: 71.0979 - val_mae: 6.6319
# Epoch 17/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 422us/step - loss: 81.5412 - mae: 6.9914 - val_loss: 68.0007 - val_mae: 6.3697
# Epoch 18/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 416us/step - loss: 79.9366 - mae: 6.9065 - val_loss: 67.0099 - val_mae: 6.4028
# Epoch 19/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 420us/step - loss: 77.9283 - mae: 6.7945 - val_loss: 62.1612 - val_mae: 5.9700
# Epoch 20/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 415us/step - loss: 77.2849 - mae: 6.7602 - val_loss: 60.7426 - val_mae: 5.8985
# Epoch 21/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 74.7994 - mae: 6.6319 - val_loss: 60.3841 - val_mae: 5.9857
# Epoch 22/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 415us/step - loss: 74.1362 - mae: 6.5955 - val_loss: 56.6646 - val_mae: 5.7366
# Epoch 23/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 422us/step - loss: 73.1701 - mae: 6.5269 - val_loss: 56.3023 - val_mae: 5.7302
# Epoch 24/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 416us/step - loss: 72.0152 - mae: 6.4671 - val_loss: 55.0275 - val_mae: 5.5511
# Epoch 25/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 416us/step - loss: 71.7270 - mae: 6.4385 - val_loss: 53.9977 - val_mae: 5.6050
# Epoch 26/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 415us/step - loss: 70.2536 - mae: 6.3684 - val_loss: 52.4428 - val_mae: 5.3677
# Epoch 27/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 415us/step - loss: 70.0509 - mae: 6.3397 - val_loss: 52.6610 - val_mae: 5.5486
# Epoch 28/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 415us/step - loss: 69.4404 - mae: 6.3038 - val_loss: 52.5867 - val_mae: 5.4934
# Epoch 29/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 423us/step - loss: 68.6289 - mae: 6.2765 - val_loss: 50.0968 - val_mae: 5.3250
# Epoch 30/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 414us/step - loss: 68.0225 - mae: 6.2163 - val_loss: 51.1208 - val_mae: 5.4148
# Epoch 31/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 414us/step - loss: 68.6749 - mae: 6.2533 - val_loss: 48.9985 - val_mae: 5.2497
# Epoch 32/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 67.0673 - mae: 6.1756 - val_loss: 50.0790 - val_mae: 5.3446
# Epoch 33/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 414us/step - loss: 67.2036 - mae: 6.1667 - val_loss: 47.2451 - val_mae: 5.1382
# Epoch 34/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 415us/step - loss: 66.3169 - mae: 6.1190 - val_loss: 48.2632 - val_mae: 5.1887
# Epoch 35/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 421us/step - loss: 66.6136 - mae: 6.1231 - val_loss: 47.6648 - val_mae: 5.1867
# Epoch 36/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 65.8029 - mae: 6.0771 - val_loss: 48.4389 - val_mae: 5.2487
# Epoch 37/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 65.7958 - mae: 6.0777 - val_loss: 46.1490 - val_mae: 5.0785
# Epoch 38/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 65.9439 - mae: 6.0789 - val_loss: 47.0365 - val_mae: 5.1028
# Epoch 39/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 65.7931 - mae: 6.0492 - val_loss: 45.6261 - val_mae: 4.9511
# Epoch 40/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 416us/step - loss: 64.9652 - mae: 6.0077 - val_loss: 46.0096 - val_mae: 5.0425
# Epoch 41/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 420us/step - loss: 64.6867 - mae: 6.0034 - val_loss: 45.3831 - val_mae: 4.9423
# Epoch 42/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 64.7628 - mae: 6.0023 - val_loss: 44.7405 - val_mae: 4.9459
# Epoch 43/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 64.9283 - mae: 6.0029 - val_loss: 44.7749 - val_mae: 4.9205
# Epoch 44/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 64.3193 - mae: 5.9641 - val_loss: 47.4324 - val_mae: 5.0529
# Epoch 45/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 412us/step - loss: 64.2467 - mae: 5.9843 - val_loss: 46.2511 - val_mae: 4.9587
# Epoch 46/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 64.5165 - mae: 5.9744 - val_loss: 45.3056 - val_mae: 4.9628
# Epoch 47/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 419us/step - loss: 63.6110 - mae: 5.9253 - val_loss: 45.5824 - val_mae: 4.9809
# Epoch 48/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 64.4874 - mae: 5.9680 - val_loss: 45.4086 - val_mae: 5.0281
# Epoch 49/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 416us/step - loss: 63.6431 - mae: 5.9325 - val_loss: 45.2266 - val_mae: 4.8954
# Epoch 50/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 412us/step - loss: 63.2610 - mae: 5.9151 - val_loss: 44.6661 - val_mae: 4.9446
# Epoch 51/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 63.7686 - mae: 5.9218 - val_loss: 45.2567 - val_mae: 4.9318
# Epoch 52/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 64.1731 - mae: 5.9518 - val_loss: 44.7034 - val_mae: 4.9379
# Epoch 53/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 419us/step - loss: 63.2972 - mae: 5.9188 - val_loss: 43.8909 - val_mae: 4.8925
# Epoch 54/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 412us/step - loss: 63.1858 - mae: 5.8926 - val_loss: 44.8716 - val_mae: 4.8800
# Epoch 55/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 63.2397 - mae: 5.9054 - val_loss: 44.2740 - val_mae: 4.8282
# Epoch 56/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 63.6117 - mae: 5.9035 - val_loss: 43.5976 - val_mae: 4.8816
# Epoch 57/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 63.6155 - mae: 5.9201 - val_loss: 44.3096 - val_mae: 4.9202
# Epoch 58/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 62.4820 - mae: 5.8614 - val_loss: 45.0684 - val_mae: 4.9247
# Epoch 59/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 419us/step - loss: 63.0774 - mae: 5.8874 - val_loss: 45.1154 - val_mae: 4.9159
# Epoch 60/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 63.2973 - mae: 5.8965 - val_loss: 44.0754 - val_mae: 4.9008
# Epoch 61/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 412us/step - loss: 63.3004 - mae: 5.8880 - val_loss: 45.8563 - val_mae: 5.0759
# Epoch 62/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 63.4532 - mae: 5.9021 - val_loss: 43.2981 - val_mae: 4.8058
# Epoch 63/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 63.2996 - mae: 5.8774 - val_loss: 44.2561 - val_mae: 4.8627
# Epoch 64/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 62.5061 - mae: 5.8614 - val_loss: 44.4320 - val_mae: 4.9859
# Epoch 65/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 424us/step - loss: 63.1643 - mae: 5.8892 - val_loss: 44.1681 - val_mae: 4.9513
# Epoch 66/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 62.5083 - mae: 5.8535 - val_loss: 44.2682 - val_mae: 4.9761
# Epoch 67/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 61.7102 - mae: 5.8249 - val_loss: 44.4079 - val_mae: 4.9519
# Epoch 68/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 414us/step - loss: 63.0983 - mae: 5.8828 - val_loss: 43.4843 - val_mae: 4.9082
# Epoch 69/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 415us/step - loss: 62.7265 - mae: 5.8576 - val_loss: 43.1075 - val_mae: 4.8079
# Epoch 70/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 61.6327 - mae: 5.8092 - val_loss: 42.9501 - val_mae: 4.7710
# Epoch 71/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 62.3098 - mae: 5.8378 - val_loss: 43.9662 - val_mae: 4.9438
# Epoch 72/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 420us/step - loss: 61.8881 - mae: 5.8029 - val_loss: 43.2726 - val_mae: 4.8431
# Epoch 73/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 62.3361 - mae: 5.8357 - val_loss: 43.5120 - val_mae: 4.8174
# Epoch 74/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 414us/step - loss: 62.2356 - mae: 5.8173 - val_loss: 42.9178 - val_mae: 4.8347
# Epoch 75/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 61.9670 - mae: 5.8098 - val_loss: 42.8512 - val_mae: 4.7762
# Epoch 76/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 62.0018 - mae: 5.8096 - val_loss: 42.5312 - val_mae: 4.7601
# Epoch 77/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 414us/step - loss: 61.8028 - mae: 5.7934 - val_loss: 42.9679 - val_mae: 4.8088
# Epoch 78/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 426us/step - loss: 62.1576 - mae: 5.8169 - val_loss: 42.3658 - val_mae: 4.7335
# Epoch 79/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 414us/step - loss: 61.6516 - mae: 5.7781 - val_loss: 41.8817 - val_mae: 4.6909
# Epoch 80/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step - loss: 62.0965 - mae: 5.8023 - val_loss: 42.8127 - val_mae: 4.8404
# Epoch 81/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 415us/step - loss: 62.2222 - mae: 5.8127 - val_loss: 41.3168 - val_mae: 4.6557
# Epoch 82/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 414us/step - loss: 61.3393 - mae: 5.7711 - val_loss: 43.0731 - val_mae: 4.8010
# Epoch 83/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 415us/step - loss: 62.2842 - mae: 5.8140 - val_loss: 42.5759 - val_mae: 4.7717
# Epoch 84/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 422us/step - loss: 61.5493 - mae: 5.7664 - val_loss: 43.1690 - val_mae: 4.8516
# Epoch 85/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 414us/step - loss: 62.0256 - mae: 5.8032 - val_loss: 42.2727 - val_mae: 4.7664
# Epoch 86/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 415us/step - loss: 61.8328 - mae: 5.7842 - val_loss: 43.4195 - val_mae: 4.7658
# Epoch 87/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 418us/step - loss: 61.7035 - mae: 5.7836 - val_loss: 42.3390 - val_mae: 4.7358
# Epoch 88/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 415us/step - loss: 61.4786 - mae: 5.7688 - val_loss: 41.9075 - val_mae: 4.7281
# Epoch 89/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 417us/step - loss: 61.1656 - mae: 5.7452 - val_loss: 42.0490 - val_mae: 4.6530
# Epoch 90/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 423us/step - loss: 60.7774 - mae: 5.7398 - val_loss: 40.7410 - val_mae: 4.6203
# Epoch 91/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 417us/step - loss: 61.7006 - mae: 5.7679 - val_loss: 42.4580 - val_mae: 4.7806
# Epoch 92/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 417us/step - loss: 61.2632 - mae: 5.7523 - val_loss: 41.9191 - val_mae: 4.6814
# Epoch 93/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 417us/step - loss: 60.9911 - mae: 5.7419 - val_loss: 42.0133 - val_mae: 4.6946
# Epoch 94/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 417us/step - loss: 61.1920 - mae: 5.7405 - val_loss: 41.5773 - val_mae: 4.6869
# Epoch 95/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 419us/step - loss: 61.3286 - mae: 5.7676 - val_loss: 41.3281 - val_mae: 4.6696
# Epoch 96/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 424us/step - loss: 61.4575 - mae: 5.7471 - val_loss: 42.3606 - val_mae: 4.6893
# Epoch 97/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 418us/step - loss: 61.4660 - mae: 5.7562 - val_loss: 41.9736 - val_mae: 4.7669
# Epoch 98/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 417us/step - loss: 60.7418 - mae: 5.7209 - val_loss: 41.1796 - val_mae: 4.6403
# Epoch 99/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 417us/step - loss: 61.1579 - mae: 5.7465 - val_loss: 41.8091 - val_mae: 4.6706
# Epoch 100/100
# 4000/4000 ━━━━━━━━━━━━━━━━━━━━ 2s 417us/step - loss: 61.2220 - mae: 5.7490 - val_loss: 41.5742 - val_mae: 4.7191
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 0s 174us/step

# Model Evaluation:
# MAE: 4.6296
# MSE: 35.7729
# RMSE: 5.9810
# R²: 0.8055

# Saving model and scaler...
# Model saved as HealthScoreModel/health_score_nn_model.keras
# Scaler saved as HealthScoreModel/health_score_scaler.pkl

# Training complete!