# -*- coding: utf-8 -*-
"""
Created on 2024-05-05
Updated with XGBoost (Regression only) on 2024-05-10
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# ======================
# 1. 数据预处理
# ======================

# 读取保险数据
ins = pd.read_csv("CASdatasets_brvehins1a.csv")

# 预处理保险数据
def preprocess_insurance(df):
    # 提取关键字段
    df = df[['Gender', 'DrivAge', 'PremTotal']].copy()  # 移除了VehGroup
    
    # 年龄区间转数值（示例处理）
    age_mapping = {
        '18-25': 21.5,
        '26-35': 30.5,
        '36-45': 40.5,
        '46-55': 50.5,
        '>55': 65
    }
    df['Age'] = df['DrivAge'].map(age_mapping)
    
    # 性别编码
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1, 'Corporate': 0}).fillna(0)
    
    return df[['Age', 'Gender', 'PremTotal']]  # 移除了VehGroup

ins_clean = preprocess_insurance(ins)

# 读取健康数据
health = pd.read_csv("synthetic_health_data_consistent.csv")

# 预处理健康数据
def preprocess_health(df):
    # 聚合特征
    user_health = df.groupby('user_id').agg(
        hear_rate_mean=('hear_rate', 'mean'),
        steps_mean=('steps', 'mean'),
        resting_heart_mean=('resting_heart', 'mean'),
        Age=('age', 'first'),
        Gender=('gender', 'first')
    ).reset_index()
    
    # 生成健康评分
    user_health['health_score'] = (
        (user_health['hear_rate_mean'] / 100) * 0.3 +
        (user_health['steps_mean'] / 1000) * 0.2 +
        (1 - (user_health['resting_heart_mean'] / 100)) * 0.5
    )
    
    return user_health[['Age', 'Gender', 'health_score']]

health_clean = preprocess_health(health)

# ======================
# 2. 模型训练 (XGBoost回归模型)
# ======================

# 保费预测模型
X_ins = ins_clean[['Age', 'Gender']]
y_ins = ins_clean['PremTotal']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_ins, y_ins, test_size=0.2, random_state=42
)

# 训练XGBoost回归模型
reg_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
reg_model.fit(X_train, y_train)

# 评估回归模型
preds = reg_model.predict(X_test)
print("\n回归模型评估:")
print(f"PremTotal预测RMSE: {np.sqrt(mean_squared_error(y_test, preds))}")
print(f"PremTotal预测R²: {r2_score(y_test, preds)}")

# ======================
# 3. 推理流程
# ======================
def predict_insurance(health_data):
    # 确保输入为DataFrame格式
    if isinstance(health_data, pd.Series):
        input_df = health_data.to_frame().T
    else:
        input_df = health_data.copy()
    
    # 提取必要特征
    features = input_df[['Age', 'Gender', 'health_score']]
    
    # 预测基础保费
    base_premium = reg_model.predict(features[['Age', 'Gender']])
    
    # 健康调整因子（示例：健康分每±0.1，保费±2%）
    health_adj = 1 + (features['health_score'].values[0] - 0.5) * 0.2
    
    return {
        'base_premium': round(base_premium[0], 2),
        'adjusted_premium': round(base_premium[0] * health_adj, 2),
        'health_score': round(features['health_score'].values[0], 3)
    }

# 正确调用方式（使用双括号保持DataFrame格式）
sample_user = health_clean.iloc[[0]]
print("\n预测结果示例:")
print(predict_insurance(sample_user))


# (base) karl@JCV9Q9QRK2 COMP7705 % python CASDataScript/XGBoost_V1.py

# 回归模型评估:
# PremTotal预测RMSE: 15437.491473376043
# PremTotal预测R²: 0.0018899191883491318

# 预测结果示例:
# {'base_premium': 1883.77, 'adjusted_premium': 1873.19, 'health_score': 0.472}