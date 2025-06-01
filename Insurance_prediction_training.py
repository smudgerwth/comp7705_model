import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

class SingleLineLogger(Callback):
    """Custom callback to display single-line training progress per epoch"""
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.epoch_width = len(str(total_epochs))  # Determine padding width for epoch numbers
        
    def on_train_begin(self, logs=None):
        """Print header at the start of training"""
        print(f"{'Epoch':>{self.epoch_width+1}}  {'Loss':<9}  {'RegLoss':<9}  {'ClsLoss':<9}  {'Acc':<6}  {'LR':<9}")
        print("-" * (self.epoch_width + 1 + 9*3 + 6 + 9 + 10))  # Underline header
        
    def on_epoch_end(self, epoch, logs=None):
        """Print training metrics at the end of each epoch"""
        lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        print(f"{epoch+1:>{self.epoch_width}d}  "
              f"{logs['loss']:9.4f}  "
              f"{logs['regression_loss']:9.4f}  "
              f"{logs['classification_loss']:9.4f}  "
              f"{logs['classification_accuracy']:6.3f}  "
              f"{lr:.2e}")

class WeightedMultiTaskTrainer:
    """Handles training for multi-task learning with weighted losses"""
    def __init__(self, model, class_weights, reg_weight=0.5, cls_weight=0.5):
        """
        Args:
            model: Compiled Keras model
            class_weights: Dictionary of weights for class imbalance
            reg_weight: Weight for regression loss
            cls_weight: Weight for classification loss
        """
        self.model = model
        self.class_weights = class_weights
        self.reg_weight = reg_weight
        self.cls_weight = cls_weight
        
        # Learning rate scheduler
        self.lr_scheduler = ReduceLROnPlateau(
            monitor='loss', 
            factor=0.5, 
            patience=3, 
            min_lr=1e-5,
            verbose=0  # Disable internal logging
        )
        
        # Early stopping callback
        self.early_stopping = EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True,
            verbose=0  # Disable internal logging
        )
    
    def train(self, X, y_reg, y_class, epochs=100, batch_size=128):
        """Train the multi-task model with sample weighting"""
        # Calculate sample weights based on class frequencies
        sample_weights = np.array([self.class_weights[np.argmax(y)] for y in y_class])
        
        # Create optimized dataset pipeline
        dataset = tf.data.Dataset.from_tensor_slices((X, {'regression': y_reg, 'classification': y_class}))
        dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Compile model with multi-task losses
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'regression': tf.keras.losses.Huber(),
                'classification': tf.keras.losses.CategoricalCrossentropy()
            },
            loss_weights={
                'regression': self.reg_weight,
                'classification': self.cls_weight
            },
            metrics={
                'classification': 'accuracy',
                'regression': 'mae'
            }
        )
        
        print("\nStarting training...")
        history = self.model.fit(
            dataset,
            epochs=epochs,
            callbacks=[
                self.early_stopping,
                self.lr_scheduler,
                SingleLineLogger(epochs)  # Our custom progress logger
            ],
            verbose=0  # Disable default progress bar
        )
        return history

def prepare_data(path):
    """Load and preprocess insurance data
    
    Args:
        path: Path to CSV file containing insurance data
        
    Returns:
        Tuple of (features, regression_target, classification_target, encoder)
    """
    df = pd.read_csv(path)
    
    # Create insurance product categories using quantile cuts
    df['insurance_product'] = pd.qcut(df['charges'], q=4, 
                                    labels=['Basic', 'Standard', 'Premium', 'VIP'])
    
    # Feature engineering
    df['sex'] = df['sex'].map({'female': 1, 'male': 0})
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
    df['age_bmi'] = df['age'] * df['bmi'] / 100  # Interaction feature
    df['bmi_smoker'] = df['bmi'] * df['smoker']  # Interaction feature
    df['age_smoker'] = df['age'] * df['smoker']  # Interaction feature
    df['age_group'] = pd.cut(df['age'], bins=[18, 25, 35, 45, 55, 65],
                           labels=[0, 1, 2, 3, 4])
    df['children_flag'] = (df['children'] > 0).astype(int)
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, np.inf],
                              labels=['underweight', 'normal', 'overweight', 'obese'])
    
    # Encode target categories
    product_encoder = OneHotEncoder(sparse_output=False)
    y_product = product_encoder.fit_transform(df[['insurance_product']])
    
    return df[['age', 'sex', 'bmi', 'smoker', 'children', 'age_bmi', 
              'bmi_smoker', 'age_smoker', 'age_group', 'children_flag', 'bmi_category']], \
           df['charges'].values, y_product, product_encoder

def preprocess_features(features):
    """Preprocess features with scaling and encoding
    
    Args:
        features: DataFrame containing raw features
        
    Returns:
        Tuple of (processed_features, scaler, preprocessor)
    """
    # Standardize numerical features
    num_features = ['age', 'bmi', 'age_bmi', 'bmi_smoker', 'age_smoker']
    scaler = StandardScaler()
    features[num_features] = scaler.fit_transform(features[num_features])
    
    # One-hot encode categorical features
    ct = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['age_group', 'bmi_category'])
        ],
        remainder='passthrough'  # Keep non-transformed columns
    )
    return ct.fit_transform(features), scaler, ct

def build_model(input_shape, num_classes):
    """Build multi-task neural network architecture
    
    Args:
        input_shape: Shape of input features
        num_classes: Number of target classes
        
    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=(input_shape,))
    
    # Shared feature extraction layers
    x = Dense(512, activation='swish')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Regression branch
    reg_branch = Dense(256, activation='swish')(x)
    reg_branch = BatchNormalization()(reg_branch)
    reg_branch = Dropout(0.3)(reg_branch)
    reg_branch = Dense(128, activation='swish')(reg_branch)
    regression = Dense(1, name='regression')(reg_branch)
    
    # Classification branch
    cls_branch = Dense(256, activation='swish')(x)
    cls_branch = BatchNormalization()(cls_branch)
    cls_branch = Dropout(0.3)(cls_branch)
    cls_branch = Dense(128, activation='swish')(cls_branch)
    classification = Dense(num_classes, activation='softmax', name='classification')(cls_branch)
    
    return Model(inputs=inputs, outputs=[regression, classification])

def evaluate_model(model, X_test, y_reg_test, y_class_test):
    """Evaluate model performance and generate reports
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_reg_test: Test regression targets
        y_class_test: Test classification targets
    """
    print("\nEvaluating model...")
    reg_pred, cls_pred = model.predict(X_test, batch_size=1024, verbose=0)
    
    # Regression metrics
    mae = np.mean(np.abs(reg_pred.flatten() - y_reg_test))
    mse = np.mean((reg_pred.flatten() - y_reg_test)**2)
    print(f"\nRegression Metrics:")
    print(f"MAE: ${mae:.2f}")
    print(f"MSE: ${mse:.2f}")
    print(f"RMSE: ${np.sqrt(mse):.2f}")
    
    # Classification report
    y_true = np.argmax(y_class_test, axis=1)
    y_pred = np.argmax(cls_pred, axis=1)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(y_reg_test, reg_pred, alpha=0.3)
    plt.plot([min(y_reg_test), max(y_reg_test)], [min(y_reg_test), max(y_reg_test)], 'r--')
    plt.xlabel('Actual Charges')
    plt.ylabel('Predicted Charges')
    plt.title('Regression Prediction')
    
    plt.subplot(1, 2, 2)
    conf_mat = confusion_matrix(y_true, y_pred)
    plt.imshow(conf_mat, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution pipeline"""
    print("Loading and preprocessing data...")
    features, y_reg, y_class, product_encoder = prepare_data('synthetic_insurance_data.csv')
    X_processed, scaler, preprocessor = preprocess_features(features)
    
    # Save preprocessing objects for future use
    joblib.dump(scaler, 'PremiumPredictionModel/scaler.pkl')
    joblib.dump(preprocessor, 'PremiumPredictionModel/preprocessor.pkl')
    joblib.dump(product_encoder, 'PremiumPredictionModel/product_encoder.pkl')
    
    # Train-test split
    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = \
        train_test_split(X_processed, y_reg, y_class, test_size=0.2, random_state=42)
    
    print("Building model...")
    model = build_model(X_train.shape[1], y_class.shape[1])
    
    # Calculate class weights for imbalance
    class_counts = np.sum(y_class_train, axis=0)
    total = np.sum(class_counts)
    class_weights = {i: total/(len(class_counts)*count) for i, count in enumerate(class_counts)}
    
    # Train model
    trainer = WeightedMultiTaskTrainer(model, class_weights)
    trainer.train(X_train, y_reg_train, y_class_train, epochs=100, batch_size=128)
    
    # Evaluate performance
    evaluate_model(model, X_test, y_reg_test, y_class_test)
    
    # Save trained model
    model.save('PremiumPredictionModel/insurance_nn_model.keras', save_format='keras')
    print("\nModel saved successfully in Keras format!")

if __name__ == "__main__":
    main()


# (base) karl@JCV9Q9QRK2 COMP7705 % python test.py
# Loading and preprocessing data...
# Building model...

# Starting training...
# Epoch  Loss       RegLoss    ClsLoss    Acc     LR       
# --------------------------------------------------------
#   1  6281.8262  12563.2979     0.3591   0.844  1.00e-03
#   2  1610.4840  3220.6411     0.3295   0.854  1.00e-03
#   3  1545.7224  3091.1255     0.3216   0.858  1.00e-03
#   4  1506.5339  3012.7495     0.3181   0.859  1.00e-03
#   5  1424.7815  2849.2505     0.3141   0.862  1.00e-03
#   6  1359.0262  2717.7373     0.3129   0.862  1.00e-03
#   7  1312.9370  2625.5615     0.3101   0.862  1.00e-03
#   8  1243.4980  2486.6858     0.3104   0.863  1.00e-03
#   9  1218.5634  2436.8174     0.3082   0.863  1.00e-03
#  10  1189.0352  2377.7620     0.3069   0.864  1.00e-03
#  11  1167.6799  2335.0544     0.3049   0.865  1.00e-03
#  12  1150.4730  2300.6431     0.3027   0.866  1.00e-03
#  13  1135.8202  2271.3376     0.3013   0.866  1.00e-03
#  14  1120.1785  2240.0537     0.2996   0.867  1.00e-03
#  15  1103.0889  2205.8765     0.2997   0.867  1.00e-03
#  16  1095.7725  2191.2461     0.2984   0.868  1.00e-03
#  17  1079.1289  2157.9614     0.2976   0.869  1.00e-03
#  18  1070.2885  2140.2810     0.2963   0.869  1.00e-03
#  19  1054.7219  2109.1470     0.2959   0.870  1.00e-03
#  20  1048.4646  2096.6367     0.2954   0.870  1.00e-03
#  21  1037.8557  2075.4180     0.2940   0.870  1.00e-03
#  22  1030.3574  2060.4219     0.2924   0.871  1.00e-03
#  23  1025.7225  2051.1541     0.2918   0.871  1.00e-03
#  24  1018.8802  2037.4694     0.2915   0.871  1.00e-03
#  25  1012.4574  2024.6270     0.2895   0.872  1.00e-03
#  26  1003.4262  2006.5668     0.2888   0.872  1.00e-03
#  27  1000.6375  2000.9875     0.2880   0.874  1.00e-03
#  28   994.5329  1988.7810     0.2869   0.874  1.00e-03
#  29   988.6369  1976.9871     0.2867   0.874  1.00e-03
#  30   983.8310  1967.3750     0.2849   0.874  1.00e-03
#  31   983.2815  1966.2786     0.2844   0.874  1.00e-03
#  32   977.1898  1954.0944     0.2836   0.876  1.00e-03
#  33   974.3307  1948.3782     0.2831   0.874  1.00e-03
#  34   966.8644  1933.4474     0.2819   0.875  1.00e-03
#  35   966.8304  1933.3798     0.2813   0.875  1.00e-03
#  36   964.3030  1928.3243     0.2803   0.877  1.00e-03
#  37   957.9154  1915.5492     0.2796   0.877  1.00e-03
#  38   958.7991  1917.3196     0.2795   0.878  1.00e-03
#  39   957.4690  1914.6594     0.2783   0.878  1.00e-03
#  40   953.5668  1906.8574     0.2771   0.879  1.00e-03
#  41   951.5245  1902.7729     0.2769   0.878  1.00e-03
#  42   946.3604  1892.4454     0.2766   0.878  1.00e-03
#  43   946.6716  1893.0698     0.2765   0.878  1.00e-03
#  44   945.2149  1890.1544     0.2754   0.879  1.00e-03
#  45   942.9267  1885.5762     0.2751   0.879  1.00e-03
#  46   940.7495  1881.2224     0.2753   0.879  1.00e-03
#  47   938.6590  1877.0444     0.2746   0.879  1.00e-03
#  48   937.2524  1874.2334     0.2734   0.880  1.00e-03
#  49   936.9282  1873.5826     0.2728   0.880  1.00e-03
#  50   934.2020  1868.1298     0.2732   0.880  1.00e-03
#  51   932.7049  1865.1370     0.2728   0.880  1.00e-03
#  52   932.9424  1865.6115     0.2725   0.880  1.00e-03
#  53   929.9380  1859.6050     0.2717   0.880  1.00e-03
#  54   929.7307  1859.1918     0.2708   0.880  1.00e-03
#  55   928.2604  1856.2494     0.2719   0.881  1.00e-03
#  56   925.9573  1851.6418     0.2716   0.880  1.00e-03
#  57   923.3314  1846.3900     0.2710   0.881  1.00e-03
#  58   924.9554  1849.6403     0.2712   0.880  1.00e-03
#  59   923.4503  1846.6296     0.2706   0.881  1.00e-03
#  60   923.1801  1846.0902     0.2698   0.882  1.00e-03
#  61   924.2414  1848.2106     0.2701   0.881  1.00e-03
#  62   919.3776  1838.4871     0.2703   0.881  1.00e-03
#  63   918.2943  1836.3192     0.2697   0.881  1.00e-03
#  64   913.0070  1825.7434     0.2693   0.882  1.00e-03
#  65   918.4834  1836.6962     0.2695   0.882  1.00e-03
#  66   914.0029  1827.7347     0.2698   0.882  1.00e-03
#  67   912.8522  1825.4344     0.2691   0.881  1.00e-03
#  68   915.7429  1831.2192     0.2689   0.881  1.00e-03
#  69   908.2853  1816.3011     0.2688   0.881  1.00e-03
#  70   910.9917  1821.7140     0.2678   0.882  1.00e-03
#  71   910.9814  1821.6967     0.2674   0.882  1.00e-03
#  72   909.9823  1819.6992     0.2673   0.882  5.00e-04
#  73   906.8308  1813.3976     0.2655   0.883  5.00e-04
#  74   898.9428  1797.6198     0.2645   0.883  5.00e-04
#  75   901.8765  1803.4900     0.2644   0.884  5.00e-04
#  76   899.5327  1798.8029     0.2640   0.883  5.00e-04
#  77   897.8022  1795.3398     0.2642   0.884  5.00e-04
#  78   898.5349  1796.8062     0.2637   0.883  5.00e-04
#  79   897.4925  1794.7209     0.2645   0.884  5.00e-04
#  80   897.6034  1794.9449     0.2639   0.883  5.00e-04
#  81   897.4413  1794.6182     0.2635   0.883  5.00e-04
#  82   892.6373  1785.0088     0.2640   0.884  5.00e-04
#  83   895.4880  1790.7124     0.2635   0.884  5.00e-04
#  84   896.6024  1792.9424     0.2629   0.884  5.00e-04
#  85   892.7545  1785.2469     0.2630   0.883  2.50e-04
#  86   890.5960  1780.9301     0.2619   0.885  2.50e-04
#  87   890.7668  1781.2710     0.2619   0.884  2.50e-04
#  88   890.1186  1779.9736     0.2615   0.884  2.50e-04
#  89   888.4287  1776.5978     0.2618   0.885  2.50e-04
#  90   887.8906  1775.5206     0.2617   0.884  2.50e-04
#  91   888.1124  1775.9630     0.2612   0.885  2.50e-04
#  92   888.9956  1777.7294     0.2609   0.885  2.50e-04
#  93   885.9929  1771.7274     0.2606   0.885  2.50e-04
#  94   884.1211  1767.9824     0.2609   0.885  2.50e-04
#  95   886.9828  1773.7054     0.2613   0.885  2.50e-04
#  96   885.5687  1770.8759     0.2608   0.884  2.50e-04
#  97   886.3401  1772.4204     0.2606   0.884  1.25e-04
#  98   885.4333  1770.6068     0.2597   0.884  1.25e-04
#  99   887.7508  1775.2386     0.2603   0.885  1.25e-04
# 100   885.4051  1770.5496     0.2600   0.885  6.25e-05

# Evaluating model...

# Regression Metrics:
# MAE: $1367.26
# MSE: $3265167.80
# RMSE: $1806.98

# Classification Report:
#               precision    recall  f1-score   support

#            0       0.93      0.91      0.92      9909
#            1       0.87      0.84      0.86     10106
#            2       0.84      0.89      0.87      9927
#            3       0.92      0.91      0.92     10058

#     accuracy                           0.89     40000
#    macro avg       0.89      0.89      0.89     40000
# weighted avg       0.89      0.89      0.89     40000

# 2025-05-05 21:24:26.072 python[26903:15437340] +[IMKClient subclass]: chose IMKClient_Modern
# 2025-05-05 21:24:26.072 python[26903:15437340] +[IMKInputSession subclass]: chose IMKInputSession_Modern
# WARNING:absl:The `save_format` argument is deprecated in Keras 3. We recommend removing this argument as it can be inferred from the file path. Received: save_format=keras

# Model saved successfully in Keras format!
