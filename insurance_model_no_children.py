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
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')
    joblib.dump(product_encoder, 'product_encoder.pkl')
    
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
    model.save('insurance_nn_model_no_child.keras')
    print("\nModel saved successfully in Keras format!")

if __name__ == "__main__":
    main()