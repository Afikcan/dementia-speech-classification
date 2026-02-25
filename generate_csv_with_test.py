import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def generate_csvs_with_test(data_dir, output_dir, val_size=0.15, test_size=0.15, random_state=42):
    """
    Scans the data directory for .wav files and generates train/valid/test CSVs.
    Uses patient-level splitting to prevent data leakage.

    Args:
        data_dir: Directory containing dementia/ and nodementia/ folders
        output_dir: Directory to save CSV files
        val_size: Proportion of data for validation (default 0.15 = 15%)
        test_size: Proportion of data for test (default 0.15 = 15%)
        random_state: Random seed for reproducibility

    Returns:
        train_df, valid_df, test_df
    """
    records = []

    # Scan Dementia
    dem_path = os.path.join(data_dir, 'dementia')
    for person in os.listdir(dem_path):
        person_dir = os.path.join(dem_path, person)
        if not os.path.isdir(person_dir):
            continue

        for file in os.listdir(person_dir):
            if file.endswith('.wav'):
                full_path = os.path.abspath(os.path.join(person_dir, file))
                records.append({
                    'file': file.replace('.wav', ''),
                    'label': 'dementia',
                    'path': full_path,
                    'group': person  # For patient-level splitting
                })

    # Scan Nodementia
    nodem_path = os.path.join(data_dir, 'nodementia')
    for person in os.listdir(nodem_path):
        person_dir = os.path.join(nodem_path, person)
        if not os.path.isdir(person_dir):
            continue

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

    print("="*70)
    print("GENERATING TRAIN/VALIDATION/TEST SPLITS")
    print("="*70)
    print(f"Total files found: {len(df)}")
    print(f"Total unique speakers: {df['group'].nunique()}")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    print(f"\nSpeakers per class:")
    for label in df['label'].unique():
        n_speakers = df[df['label'] == label]['group'].nunique()
        print(f"  {label}: {n_speakers} speakers")

    # First split: separate out test set
    # We want test_size of the total data
    splitter1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(splitter1.split(df, groups=df['group']))

    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx].drop(columns=['group'])

    # Second split: split train_val into train and validation
    # val_size should be relative to the remaining data after removing test
    # If we want val_size% of original data, we need to adjust
    # remaining_data = (1 - test_size) of original
    # val_size_adjusted = val_size / (1 - test_size)
    val_size_adjusted = val_size / (1 - test_size)

    splitter2 = GroupShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_state)
    train_idx, valid_idx = next(splitter2.split(train_val_df, groups=train_val_df['group']))

    train_df = train_val_df.iloc[train_idx].drop(columns=['group'])
    valid_df = train_val_df.iloc[valid_idx].drop(columns=['group'])

    # Print statistics
    print("\n" + "="*70)
    print("SPLIT STATISTICS")
    print("="*70)

    def print_split_stats(df, name):
        print(f"\n{name} Set: {len(df)} samples ({100*len(df)/(len(train_df)+len(valid_df)+len(test_df)):.1f}% of total)")
        print(f"  Dementia: {len(df[df['label']=='dementia'])} ({100*len(df[df['label']=='dementia'])/len(df):.1f}%)")
        print(f"  No dementia: {len(df[df['label']=='nodementia'])} ({100*len(df[df['label']=='nodementia'])/len(df):.1f}%)")

    print_split_stats(train_df, "Training")
    print_split_stats(valid_df, "Validation")
    print_split_stats(test_df, "Test")

    print(f"\nTotal samples: {len(train_df) + len(valid_df) + len(test_df)}")
    print(f"Train:Val:Test ratio = {len(train_df)}:{len(valid_df)}:{len(test_df)}")
    print(f"Percentages = {100*len(train_df)/(len(train_df)+len(valid_df)+len(test_df)):.1f}%:"
          f"{100*len(valid_df)/(len(train_df)+len(valid_df)+len(test_df)):.1f}%:"
          f"{100*len(test_df)/(len(train_df)+len(valid_df)+len(test_df)):.1f}%")

    # Save CSVs
    train_csv = os.path.join(output_dir, 'train_dm_new.csv')
    valid_csv = os.path.join(output_dir, 'valid_dm_new.csv')
    test_csv = os.path.join(output_dir, 'test_dm_new.csv')

    train_df.to_csv(train_csv, sep='\t', index=False)
    valid_df.to_csv(valid_csv, sep='\t', index=False)
    test_df.to_csv(test_csv, sep='\t', index=False)

    print("\n" + "="*70)
    print("FILES SAVED")
    print("="*70)
    print(f"✓ {train_csv}")
    print(f"✓ {valid_csv}")
    print(f"✓ {test_csv}")
    print("="*70)

    return train_df, valid_df, test_df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')

    # Generate 70% train, 15% validation, 15% test
    train_df, valid_df, test_df = generate_csvs_with_test(
        data_dir,
        data_dir,
        val_size=0.15,
        test_size=0.15,
        random_state=42
    )

    print("\n✅ Train/Validation/Test split complete!")
    print("\nNote: This is a patient-level split - no speaker appears in multiple sets.")
