# -*- coding: utf-8 -*-
"""
Updated Neural Network with NaN Fix on 2024-05-10
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LeakyReLU

# ======================
# 1. 增强数据预处理
# ======================

def preprocess_insurance(df):
    # 保留必要欄位
    df = df[['Gender', 'DrivAge', 'PremTotal']].copy()
    
    # 加強年齡映射處理
    age_mapping = {
        '18-25': 21.5,
        '26-35': 30.5,
        '36-45': 40.5,
        '46-55': 50.5,
        '>55': 65
    }
    # 處理未定義年齡類別
    df['Age'] = df['DrivAge'].map(age_mapping).fillna(age_mapping['>55'])  
    
    # 加強性別處理
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1, 'Corporate': 0}).fillna(0)
    
    # 改進異常值過濾（基於分位數）
    q_low = df['PremTotal'].quantile(0.05)
    q_high = df['PremTotal'].quantile(0.95)
    df = df[(df['PremTotal'] > q_low) & (df['PremTotal'] < q_high)]
    
    return df[['Age', 'Gender', 'PremTotal']]

# ======================
# 2. 改進的數據標準化
# ======================

def prepare_data(df):
    X = df[['Age', 'Gender']].values
    y = df['PremTotal'].values.reshape(-1, 1)

    # 使用RobustScaler處理特徵和目標值
    X_scaler = RobustScaler(quantile_range=(5, 95))
    y_scaler = RobustScaler(quantile_range=(5, 95))

    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    return train_test_split(
        X_scaled, y_scaled, 
        test_size=0.2, 
        random_state=42,
        shuffle=True
    )

# ======================
# 3. 防NaN神經網絡架構
# ======================

def build_robust_model(input_shape):
    model = Sequential([
        # 第一層
        Dense(64, kernel_regularizer=l2(0.001), input_shape=input_shape),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),
        
        # 第二層
        Dense(32, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),
        
        # 第三層
        Dense(16),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        
        # 輸出層
        Dense(1)
    ])
    
    # 帶梯度裁剪的優化器
    optimizer = Adam(
        learning_rate=0.0005,
        clipvalue=0.5
    )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    return model

# ======================
# 4. 安全訓練流程
# ======================

def train_model(X_train, y_train, X_test, y_test):
    # 數據完整性檢查
    assert not np.any(np.isnan(X_train)), "X_train contains NaN"
    assert not np.any(np.isinf(X_train)), "X_train contains Inf"
    assert not np.any(np.isnan(y_train)), "y_train contains NaN"
    
    model = build_robust_model((X_train.shape[1],))
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        min_delta=0.001
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=500,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )
    return model, history

# ======================
# 5. 完整執行流程
# ======================

if __name__ == "__main__":
    # 載入數據
    ins = pd.read_csv("CASdatasets_brvehins1a.csv")
    ins_clean = preprocess_insurance(ins)
    
    # 準備數據
    X_train, X_test, y_train, y_test = prepare_data(ins_clean)
    X_scaler = RobustScaler(quantile_range=(5, 95)).fit(ins_clean[['Age', 'Gender']])
    y_scaler = RobustScaler(quantile_range=(5, 95)).fit(ins_clean['PremTotal'].values.reshape(-1, 1))
    
    # 訓練模型
    model, history = train_model(X_train, y_train, X_test, y_test)
    
    # 評估模型
    preds_scaled = model.predict(X_test)
    preds = y_scaler.inverse_transform(preds_scaled)
    y_test_orig = y_scaler.inverse_transform(y_test)
    
    print("\n模型評估結果:")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_orig, preds)):.2f}")
    print(f"R²: {r2_score(y_test_orig, preds):.4f}")

    # 保存scaler
    import joblib
    joblib.dump(X_scaler, 'X_scaler.pkl')
    joblib.dump(y_scaler, 'y_scaler.pkl')
    model.save('premium_model.keras')

# ======================
# 6. 安全預測函數
# ======================

def safe_predict(input_data, model, X_scaler, y_scaler):
    """強化型預測函數"""
    try:
        # 輸入數據驗證
        required_cols = ['Age', 'Gender']
        if not all(col in input_data.columns for col in required_cols):
            raise ValueError("Missing required columns in input data")
            
        # 類型轉換
        if isinstance(input_data, pd.Series):
            input_df = input_data.to_frame().T
        else:
            input_df = input_data.copy()
        
        # 特徵處理
        features = input_df[required_cols].values.astype(np.float32)
        features_scaled = X_scaler.transform(features)
        
        # 範圍檢查
        if np.any(np.abs(features_scaled) > 5):
            raise ValueError("Input features out of scaled range")
            
        # 預測
        pred_scaled = model.predict(features_scaled, verbose=0)
        base_premium = float(y_scaler.inverse_transform(pred_scaled)[0])
        
        return {
            'base_premium': round(base_premium, 2),
            'currency': 'USD',
            'confidence': 'high'
        }
    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return {
            'error': str(e),
            'confidence': 'low'
        }


# (base) karl@JCV9Q9QRK2 COMP7705 % python CASDataScript/NN_V1.py
# /opt/anaconda3/lib/python3.12/site-packages/keras/src/layers/core/dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
#   super().__init__(activity_regularizer=activity_regularizer, **kwargs)
# /opt/anaconda3/lib/python3.12/site-packages/keras/src/layers/activations/leaky_relu.py:41: UserWarning: Argument `alpha` is deprecated. Use `negative_slope` instead.
#   warnings.warn(
# Epoch 1/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 4s 798us/step - loss: 0.2924 - mae: 0.3268 - val_loss: 0.1172 - val_mae: 0.2255
# Epoch 2/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 765us/step - loss: 0.1150 - mae: 0.2300 - val_loss: 0.1109 - val_mae: 0.2276
# Epoch 3/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 781us/step - loss: 0.1108 - mae: 0.2290 - val_loss: 0.1104 - val_mae: 0.2297
# Epoch 4/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 765us/step - loss: 0.1111 - mae: 0.2295 - val_loss: 0.1102 - val_mae: 0.2241
# Epoch 5/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 763us/step - loss: 0.1102 - mae: 0.2284 - val_loss: 0.1102 - val_mae: 0.2281
# Epoch 6/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 776us/step - loss: 0.1095 - mae: 0.2277 - val_loss: 0.1101 - val_mae: 0.2291
# Epoch 7/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 737us/step - loss: 0.1100 - mae: 0.2276 - val_loss: 0.1101 - val_mae: 0.2331
# Epoch 8/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 729us/step - loss: 0.1103 - mae: 0.2290 - val_loss: 0.1105 - val_mae: 0.2354
# Epoch 9/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 761us/step - loss: 0.1104 - mae: 0.2287 - val_loss: 0.1102 - val_mae: 0.2260
# Epoch 10/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 786us/step - loss: 0.1106 - mae: 0.2288 - val_loss: 0.1100 - val_mae: 0.2285
# Epoch 11/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 4s 789us/step - loss: 0.1099 - mae: 0.2278 - val_loss: 0.1101 - val_mae: 0.2265
# Epoch 12/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 772us/step - loss: 0.1096 - mae: 0.2272 - val_loss: 0.1100 - val_mae: 0.2245
# Epoch 13/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 765us/step - loss: 0.1111 - mae: 0.2291 - val_loss: 0.1103 - val_mae: 0.2228
# Epoch 14/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 645us/step - loss: 0.1101 - mae: 0.2283 - val_loss: 0.1100 - val_mae: 0.2257
# Epoch 15/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 566us/step - loss: 0.1095 - mae: 0.2278 - val_loss: 0.1099 - val_mae: 0.2236
# Epoch 16/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 2s 563us/step - loss: 0.1092 - mae: 0.2272 - val_loss: 0.1099 - val_mae: 0.2244
# Epoch 17/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 563us/step - loss: 0.1106 - mae: 0.2287 - val_loss: 0.1098 - val_mae: 0.2258
# Epoch 18/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 2s 562us/step - loss: 0.1113 - mae: 0.2292 - val_loss: 0.1099 - val_mae: 0.2326
# Epoch 19/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 582us/step - loss: 0.1094 - mae: 0.2275 - val_loss: 0.1100 - val_mae: 0.2247
# Epoch 20/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 563us/step - loss: 0.1105 - mae: 0.2285 - val_loss: 0.1099 - val_mae: 0.2262
# Epoch 21/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 591us/step - loss: 0.1097 - mae: 0.2282 - val_loss: 0.1098 - val_mae: 0.2275
# Epoch 22/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 575us/step - loss: 0.1099 - mae: 0.2283 - val_loss: 0.1098 - val_mae: 0.2246
# Epoch 23/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 564us/step - loss: 0.1102 - mae: 0.2285 - val_loss: 0.1100 - val_mae: 0.2325
# Epoch 24/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 2s 562us/step - loss: 0.1095 - mae: 0.2273 - val_loss: 0.1097 - val_mae: 0.2286
# Epoch 25/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 574us/step - loss: 0.1110 - mae: 0.2293 - val_loss: 0.1102 - val_mae: 0.2291
# Epoch 26/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 564us/step - loss: 0.1102 - mae: 0.2283 - val_loss: 0.1098 - val_mae: 0.2280
# Epoch 27/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 2s 563us/step - loss: 0.1101 - mae: 0.2280 - val_loss: 0.1099 - val_mae: 0.2255
# Epoch 28/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 2s 562us/step - loss: 0.1102 - mae: 0.2282 - val_loss: 0.1098 - val_mae: 0.2292
# Epoch 29/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 582us/step - loss: 0.1102 - mae: 0.2286 - val_loss: 0.1100 - val_mae: 0.2231
# Epoch 30/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 2s 561us/step - loss: 0.1103 - mae: 0.2280 - val_loss: 0.1098 - val_mae: 0.2250
# Epoch 31/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 3s 579us/step - loss: 0.1096 - mae: 0.2276 - val_loss: 0.1098 - val_mae: 0.2288
# Epoch 32/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 2s 559us/step - loss: 0.1099 - mae: 0.2280 - val_loss: 0.1105 - val_mae: 0.2363
# Epoch 33/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 2s 560us/step - loss: 0.1091 - mae: 0.2276 - val_loss: 0.1102 - val_mae: 0.2330
# Epoch 34/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 2s 557us/step - loss: 0.1108 - mae: 0.2293 - val_loss: 0.1099 - val_mae: 0.2242
# Epoch 35/500
# 4422/4422 ━━━━━━━━━━━━━━━━━━━━ 2s 557us/step - loss: 0.1098 - mae: 0.2279 - val_loss: 0.1098 - val_mae: 0.2264
# 2211/2211 ━━━━━━━━━━━━━━━━━━━━ 0s 186us/step 

# 模型評估結果:
# RMSE: 2459.31
# R²: 0.0016
