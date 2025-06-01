import pandas as pd
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

def generate_synthetic_data(num_users=None):
    # 1. Load and clean data
    original_df = pd.read_csv('aw_fb_data.csv')
    original_df = original_df.drop(columns=original_df.filter(regex='Unnamed').columns, errors='ignore')

    # 2. Create user_id column if missing
    if 'user_id' not in original_df.columns:
        original_df['user_id'] = [f"USER_{i:04d}" for i in range(1, len(original_df)+1)]

    # 3. Separate demographics and activity data
    static_columns = ['age', 'gender', 'height', 'weight', 'device']  # Added device to static columns
    activity_columns = list(set(original_df.columns) - set(static_columns) - {'user_id'})

    # 4. Generate unique users based on original data count
    demographics_df = original_df[static_columns + ['user_id']].drop_duplicates()
    
    # Determine number of users to generate
    if num_users is None:
        num_users = len(demographics_df)  # Default to original data's user count
    
    # Create demographics synthesizer
    demographics_metadata = SingleTableMetadata()
    demographics_metadata.detect_from_dataframe(demographics_df)
    demographics_metadata.set_primary_key('user_id')
    
    demo_synthesizer = GaussianCopulaSynthesizer(demographics_metadata)
    demo_synthesizer.fit(demographics_df)
    unique_users = demo_synthesizer.sample(num_users)

    # 5. Generate activity data
    activity_metadata = SingleTableMetadata()
    activity_metadata.detect_from_dataframe(original_df[activity_columns])
    activity_synthesizer = GaussianCopulaSynthesizer(activity_metadata)
    activity_synthesizer.fit(original_df[activity_columns])

    # 6. Create full dataset
    synthetic_data = []
    for _, user in unique_users.iterrows():
        # Generate random number of activity records between 15-80
        num_records = np.random.randint(15, 81)
        activity_data = activity_synthesizer.sample(num_records)
        activity_data['X1'] = range(1, num_records + 1)
        
        # Add consistent demographics and device
        for col in static_columns:
            activity_data[col] = user[col]
        activity_data['user_id'] = user['user_id']
        
        synthetic_data.append(activity_data)

    synthetic_df = pd.concat(synthetic_data, ignore_index=True)

    # 7. Validate and save
    # Validate device consistency
    assert synthetic_df.groupby('user_id')['device'].nunique().max() == 1
    # Validate demographic consistency
    assert synthetic_df.groupby('user_id')[static_columns].nunique().max().max() == 1
    
    synthetic_df.to_csv('synthetic_health_data_consistent.csv', index=False)

if __name__ == '__main__':
    generate_synthetic_data(5000)  # Call without parameter to use original data size
