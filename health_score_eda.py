import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style - UPDATED to use current style names
plt.style.use('seaborn-v0_8')  # Modern equivalent of 'seaborn'
sns.set_theme(style="whitegrid", palette="husl")  # Updated seaborn settings

def load_data(filepath):
    """Load and validate the dataset"""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def basic_eda(df, health_score_col='Health_Score'):
    """Perform basic exploratory analysis"""
    print("\n=== Basic Statistics ===")
    print(df[health_score_col].describe())
    
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    
    print("\n=== Value Counts ===")
    print(df[health_score_col].value_counts(bins=10, sort=False))

def plot_distribution(df, health_score_col='Health_Score'):
    """Visualize health score distribution"""
    plt.figure(figsize=(14, 8))
    
    # Histogram with KDE
    plt.subplot(2, 2, 1)
    sns.histplot(df[health_score_col], kde=True, bins=30)
    plt.title('Health Score Distribution')
    plt.xlabel('Health Score')
    plt.ylabel('Frequency')
    
    # Boxplot
    plt.subplot(2, 2, 2)
    sns.boxplot(x=df[health_score_col])
    plt.title('Health Score Boxplot')
    plt.xlabel('Health Score')
    
    # QQ-Plot
    plt.subplot(2, 2, 3)
    stats.probplot(df[health_score_col], plot=plt)
    plt.title('QQ-Plot')
    
    # Cumulative Distribution
    plt.subplot(2, 2, 4)
    sns.ecdfplot(df[health_score_col])
    plt.title('Cumulative Distribution')
    plt.xlabel('Health Score')
    plt.ylabel('Proportion')
    
    plt.tight_layout()
    plt.savefig('health_score_distribution.png')
    plt.close()  # Changed from show() to close() to prevent display issues

def analyze_percentiles(df, health_score_col='Health_Score'):
    """Analyze percentile distribution"""
    print("\n=== Percentile Analysis ===")
    percentiles = np.arange(0, 101, 5)
    percentile_values = np.percentile(df[health_score_col], percentiles)
    
    percentile_df = pd.DataFrame({
        'Percentile': percentiles,
        'Health_Score': percentile_values
    })
    
    print(percentile_df)
    
    # Plot percentile distribution
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Percentile', y='Health_Score', data=percentile_df, marker='o')
    plt.title('Health Score Percentile Distribution')
    plt.grid(True)
    plt.savefig('health_score_percentiles.png')
    plt.close()

def analyze_extremes(df, health_score_col='Health_Score'):
    """Analyze lowest and highest health scores"""
    print("\n=== Lowest 5% Health Scores ===")
    low_threshold = np.percentile(df[health_score_col], 5)
    low_scores = df[df[health_score_col] <= low_threshold]
    print(low_scores[health_score_col].describe())
    
    print("\n=== Highest 5% Health Scores ===")
    high_threshold = np.percentile(df[health_score_col], 95)
    high_scores = df[df[health_score_col] >= high_threshold]
    print(high_scores[health_score_col].describe())

def correlation_analysis(df, health_score_col='Health_Score'):
    """Analyze correlations with other features"""
    print("\n=== Correlation Analysis ===")
    
    # Select numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.savefig('health_score_correlations.png')
    plt.close()
    
    # Health score specific correlations
    health_corr = corr_matrix[health_score_col].sort_values(ascending=False)
    print("\nHealth Score Correlations:")
    print(health_corr)

def main():
    filepath = "./health_score_data/synthetic_expanded_health_data.csv"
    health_score_col = "Health_Score"
    
    # Load data
    df = load_data(filepath)
    if df is None:
        return
    
    # Basic EDA
    basic_eda(df, health_score_col)
    
    # Distribution analysis
    plot_distribution(df, health_score_col)
    
    # Percentile analysis
    analyze_percentiles(df, health_score_col)
    
    # Extreme value analysis
    analyze_extremes(df, health_score_col)
    
    # Correlation analysis
    correlation_analysis(df, health_score_col)
    
    # Save summary statistics
    summary_stats = df[health_score_col].describe().to_frame().T
    summary_stats.to_csv('health_score_summary_stats.csv', index=False)
    print("\nAnalysis complete. Results saved to files.")

if __name__ == "__main__":
    main()
