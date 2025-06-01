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
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, np.inf],
                              labels=['underweight', 'normal', 'overweight', 'obese'])
    
    # Encode target categories
    product_encoder = OneHotEncoder(sparse_output=False)
    y_product = product_encoder.fit_transform(df[['insurance_product']])
    
    return df[['age', 'sex', 'bmi', 'smoker', 'age_bmi', 
              'bmi_smoker', 'age_smoker', 'age_group', 'bmi_category']], \
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
    # regression = Dense(1, name='regression')(reg_branch)
    regression = Dense(1, activation='relu', name='regression')(reg_branch)

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
    plt.savefig('evaluation_plots.png')

def main():
    """Main execution pipeline"""
    print("Loading and preprocessing data...")
    features, y_reg, y_class, product_encoder = prepare_data('insurance_data/synthetic_insurance_data.csv')
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
    model.save('PremiumPredictionModel/insurance_nn_model_no_child.keras')
    print("\nModel saved successfully in Keras format!")

if __name__ == "__main__":
    main()


# (comp7705_env) macbookpro@Mac-mini comp7705_model % python insurance_model_no_children.py
# Loading and preprocessing data...
# Building model...

# Starting training...
# Epoch  Loss       RegLoss    ClsLoss    Acc     LR       
# --------------------------------------------------------
#   1  5943.2588  11886.1660     0.3648   0.843  1.00e-03
#   2  1635.0850  3269.8296     0.3386   0.851  1.00e-03
#   3  1190.1281  2379.9309     0.3279   0.856  1.00e-03
#   4  1167.9143  2335.5076     0.3211   0.858  1.00e-03
#   5  1160.9283  2321.5435     0.3162   0.861  1.00e-03
#   6  1140.1536  2279.9917     0.3132   0.862  1.00e-03
#   7  1144.0850  2287.8594     0.3104   0.863  1.00e-03
#   8  1120.6736  2241.0391     0.3072   0.864  1.00e-03
#   9  1114.0739  2227.8423     0.3048   0.866  1.00e-03
#  10  1111.0887  2221.8743     0.3029   0.866  1.00e-03
#  11  1100.9824  2201.6665     0.3001   0.868  1.00e-03
#  12  1088.8792  2177.4614     0.2988   0.868  1.00e-03
#  13  1083.5728  2166.8491     0.2963   0.869  1.00e-03
#  14  1081.0074  2161.7207     0.2952   0.869  1.00e-03
#  15  1075.1147  2149.9382     0.2933   0.871  1.00e-03
#  16  1068.5663  2136.8406     0.2916   0.872  1.00e-03
#  17  1050.7527  2101.2161     0.2905   0.872  1.00e-03
#  18  1047.1024  2093.9143     0.2890   0.872  1.00e-03
#  19  1027.4750  2054.6643     0.2880   0.873  1.00e-03
#  20  1015.3169  2030.3477     0.2871   0.873  1.00e-03
#  21   995.7980  1991.3068     0.2868   0.874  1.00e-03
#  22   986.6791  1973.0718     0.2863   0.874  1.00e-03
#  23   974.5140  1948.7432     0.2864   0.874  1.00e-03
#  24   974.4880  1948.6924     0.2861   0.874  1.00e-03
#  25   969.2437  1938.2034     0.2854   0.874  1.00e-03
#  26   965.0651  1929.8464     0.2842   0.875  1.00e-03
#  27   961.5524  1922.8188     0.2848   0.875  1.00e-03
#  28   958.9994  1917.7168     0.2838   0.874  1.00e-03
#  29   955.7497  1911.2166     0.2826   0.875  1.00e-03
#  30   953.0082  1905.7306     0.2838   0.875  1.00e-03
#  31   954.4883  1908.6918     0.2831   0.875  1.00e-03
#  32   950.2286  1900.1744     0.2820   0.876  1.00e-03
#  33   948.5139  1896.7448     0.2824   0.875  1.00e-03
#  34   946.8071  1893.3348     0.2812   0.876  1.00e-03
#  35   942.9738  1885.6694     0.2814   0.876  1.00e-03
#  36   939.5708  1878.8612     0.2805   0.876  1.00e-03
#  37   940.4218  1880.5636     0.2809   0.876  1.00e-03
#  38   939.0437  1877.8051     0.2801   0.877  1.00e-03
#  39   937.8137  1875.3490     0.2797   0.877  1.00e-03
#  40   935.7098  1871.1414     0.2799   0.876  1.00e-03
#  41   935.1073  1869.9344     0.2792   0.876  1.00e-03
#  42   933.4355  1866.5936     0.2790   0.876  1.00e-03
#  43   931.7168  1863.1552     0.2788   0.877  1.00e-03
#  44   933.1459  1866.0138     0.2783   0.877  1.00e-03
#  45   929.6580  1859.0380     0.2785   0.877  1.00e-03
#  46   928.3117  1856.3466     0.2771   0.877  1.00e-03
#  47   925.7382  1851.1976     0.2782   0.877  1.00e-03
#  48   925.0492  1849.8228     0.2773   0.877  1.00e-03
#  49   924.9934  1849.7087     0.2774   0.877  1.00e-03
#  50   923.8668  1847.4543     0.2780   0.877  1.00e-03
#  51   922.0467  1843.8145     0.2769   0.878  1.00e-03
#  52   923.6353  1846.9926     0.2775   0.878  1.00e-03
#  53   925.3155  1850.3546     0.2769   0.877  1.00e-03
#  54   920.4965  1840.7146     0.2764   0.877  1.00e-03
#  55   917.5312  1834.7860     0.2763   0.878  1.00e-03
#  56   915.5293  1830.7822     0.2762   0.878  1.00e-03
#  57   917.2172  1834.1556     0.2764   0.878  1.00e-03
#  58   916.1968  1832.1174     0.2757   0.878  1.00e-03
#  59   914.0746  1827.8718     0.2763   0.877  1.00e-03
#  60   911.8752  1823.4760     0.2750   0.879  1.00e-03
#  61   910.9618  1821.6484     0.2756   0.878  1.00e-03
#  62   911.8624  1823.4493     0.2758   0.878  1.00e-03
#  63   911.0047  1821.7344     0.2755   0.878  1.00e-03
#  64   912.4258  1824.5778     0.2754   0.878  5.00e-04
#  65   905.8901  1811.5072     0.2732   0.879  5.00e-04
#  66   902.8490  1805.4254     0.2726   0.878  5.00e-04
#  67   901.9700  1803.6680     0.2723   0.879  5.00e-04
#  68   901.6506  1803.0282     0.2718   0.879  5.00e-04
#  69   899.3882  1798.5050     0.2727   0.880  5.00e-04
#  70   897.1746  1794.0771     0.2724   0.879  5.00e-04
#  71   902.0830  1803.8932     0.2720   0.879  5.00e-04
#  72   897.1815  1794.0898     0.2716   0.879  5.00e-04
#  73   897.9667  1795.6617     0.2717   0.880  2.50e-04
#  74   896.4565  1792.6414     0.2708   0.879  2.50e-04
#  75   895.9022  1791.5348     0.2702   0.880  2.50e-04
#  76   896.7299  1793.1908     0.2696   0.880  2.50e-04
#  77   894.5577  1788.8462     0.2695   0.880  2.50e-04
#  78   894.9197  1789.5695     0.2692   0.881  2.50e-04
#  79   895.3501  1790.4294     0.2698   0.880  2.50e-04
#  80   893.5826  1786.8951     0.2697   0.881  2.50e-04
#  81   892.4568  1784.6420     0.2697   0.880  2.50e-04
#  82   891.1645  1782.0583     0.2701   0.880  2.50e-04
#  83   889.9951  1779.7198     0.2702   0.880  2.50e-04
#  84   891.6187  1782.9692     0.2691   0.880  2.50e-04
#  85   889.7548  1779.2375     0.2699   0.880  2.50e-04
#  86   891.3973  1782.5254     0.2695   0.881  2.50e-04
#  87   890.9658  1781.6630     0.2693   0.879  2.50e-04
#  88   890.9265  1781.5834     0.2701   0.880  1.25e-04
#  89   889.8452  1779.4218     0.2693   0.881  1.25e-04
#  90   888.6098  1776.9504     0.2683   0.881  1.25e-04
#  91   892.2253  1784.1798     0.2692   0.880  1.25e-04
#  92   889.9391  1779.6086     0.2683   0.880  1.25e-04
#  93   892.2249  1784.1810     0.2682   0.880  6.25e-05
#  94   891.1356  1782.0013     0.2688   0.880  6.25e-05
#  95   889.6686  1779.0674     0.2681   0.881  6.25e-05
#  96   888.2109  1776.1541     0.2679   0.881  6.25e-05
#  97   884.5495  1768.8322     0.2681   0.880  6.25e-05
#  98   885.3144  1770.3594     0.2687   0.881  6.25e-05
#  99   885.9778  1771.6858     0.2684   0.880  6.25e-05
# 100   886.1715  1772.0752     0.2682   0.880  3.13e-05

# Evaluating model...

# Regression Metrics:
# MAE: $1398.08
# MSE: $3609060.91
# RMSE: $1899.75

# Classification Report:
#               precision    recall  f1-score   support

#            0       0.93      0.89      0.91      9909
#            1       0.87      0.83      0.85     10106
#            2       0.82      0.91      0.86      9927
#            3       0.92      0.91      0.92     10058

#     accuracy                           0.88     40000
#    macro avg       0.89      0.88      0.89     40000
# weighted avg       0.89      0.88      0.89     40000


# Model saved successfully in Keras format!
