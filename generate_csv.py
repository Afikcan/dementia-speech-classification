import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def generate_csvs(data_dir, output_dir, test_size=0.2, random_state=42):
    """
    Scans the data directory for .wav files and generates train/valid CSVs.
    Assumes structure:
        data_dir/
            dementia/
                PersonName/
                    file.wav
            nodementia/
                PersonName/
                    file.wav
    """
    records = []
    
    # Scan Dementia
    dem_path = os.path.join(data_dir, 'dementia')
    for person in os.listdir(dem_path):
        person_dir = os.path.join(dem_path, person)
        if not os.path.isdir(person_dir): continue
        
        for file in os.listdir(person_dir):
            if file.endswith('.wav'):
                # Format matches existing CSV: file, label, path
                # Note: valid_dm.csv uses absolute paths in the 'path' column
                full_path = os.path.abspath(os.path.join(person_dir, file))
                records.append({
                    'file': file.replace('.wav', ''),
                    'label': 'dementia',
                    'path': full_path,
                    'group': person  # For splitting
                })

    # Scan Nodementia
    nodem_path = os.path.join(data_dir, 'nodementia')
    for person in os.listdir(nodem_path):
        person_dir = os.path.join(nodem_path, person)
        if not os.path.isdir(person_dir): continue
        
        for file in os.listdir(person_dir):
            if file.endswith('.wav'):
                full_path = os.path.abspath(os.path.join(person_dir, file))
                records.append({
                    'file': file.replace('.wav', ''),
                    'label': 'nodementia',
                    'path': full_path,
                    'group': person
                })

    df = pd.DataFrame(records)
    
    print(f"Total files found: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Split by Person (Group) to avoid data leakage
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, valid_idx = next(splitter.split(df, groups=df['group']))
    
    train_df = df.iloc[train_idx].drop(columns=['group'])
    valid_df = df.iloc[valid_idx].drop(columns=['group'])
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    
    # Save CSVs
    train_csv = os.path.join(output_dir, 'train_dm_new.csv')
    valid_csv = os.path.join(output_dir, 'valid_dm_new.csv')
    
    # Use tab separator as implied by some previous notebooks, or comma?
    # valid_dm.csv viewed earlier looked tab-separated or comma? 
    # Viewing valid_dm.csv: "file\tlabel\tpath" -> It is TAB separated.
    
    train_df.to_csv(train_csv, sep='\t', index=False)
    valid_df.to_csv(valid_csv, sep='\t', index=False)
    
    print(f"Saved {train_csv}")
    print(f"Saved {valid_csv}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    generate_csvs(data_dir, data_dir)
