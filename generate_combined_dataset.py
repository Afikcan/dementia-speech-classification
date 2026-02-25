"""
Generate combined dataset from DementiaNet + ADReSSo21.

This script creates train/validation/test splits combining:
1. DementiaNet (455 samples) - celebrity speech
2. ADReSSo21 (166 samples) - clinical interviews

Total: 621 samples for binary dementia classification
"""

import os
import pandas as pd

def create_combined_dataset(output_dir='data'):
    """
    Combine DementiaNet and ADReSSo21 datasets.

    Strategy:
    - Use existing DementiaNet splits (train/val/test)
    - Add ADReSSo21 training data to DementiaNet training set
    - Keep DementiaNet val/test as-is for consistency
    """

    print("="*70)
    print("CREATING COMBINED DATASET: DementiaNet + ADReSSo21")
    print("="*70)

    # Load existing DementiaNet splits
    dnet_train = pd.read_csv(os.path.join(output_dir, 'train_dm_new.csv'), sep='\t')
    dnet_val = pd.read_csv(os.path.join(output_dir, 'valid_dm_new.csv'), sep='\t')
    dnet_test = pd.read_csv(os.path.join(output_dir, 'test_dm_new.csv'), sep='\t')

    print(f"\n📊 DementiaNet Dataset:")
    print(f"  Training: {len(dnet_train)} samples")
    print(f"  Validation: {len(dnet_val)} samples")
    print(f"  Test: {len(dnet_test)} samples")
    print(f"  Total: {len(dnet_train) + len(dnet_val) + len(dnet_test)} samples")

    # Load ADReSSo21 data
    adresso_root = os.path.join(output_dir, 'ADReSSo21_data')

    adresso_records = []

    # ADReSSo21 training - Dementia (AD)
    ad_dir = os.path.join(adresso_root, 'ADReSSo21-diagnosis-train/diagnosis/train/audio/ad')
    if os.path.exists(ad_dir):
        for file in os.listdir(ad_dir):
            if file.endswith('.wav'):
                full_path = os.path.abspath(os.path.join(ad_dir, file))
                adresso_records.append({
                    'file': file.replace('.wav', ''),
                    'label': 'dementia',
                    'path': full_path,
                    'source': 'ADReSSo21'
                })

    # ADReSSo21 training - Control (CN)
    cn_dir = os.path.join(adresso_root, 'ADReSSo21-diagnosis-train/diagnosis/train/audio/cn')
    if os.path.exists(cn_dir):
        for file in os.listdir(cn_dir):
            if file.endswith('.wav'):
                full_path = os.path.abspath(os.path.join(cn_dir, file))
                adresso_records.append({
                    'file': file.replace('.wav', ''),
                    'label': 'nodementia',
                    'path': full_path,
                    'source': 'ADReSSo21'
                })

    adresso_df = pd.DataFrame(adresso_records)

    print(f"\n📊 ADReSSo21 Dataset:")
    print(f"  Training: {len(adresso_df)} samples")
    print(f"    Dementia: {len(adresso_df[adresso_df['label']=='dementia'])}")
    print(f"    Control: {len(adresso_df[adresso_df['label']=='nodementia'])}")

    # Add source column to DementiaNet data
    dnet_train['source'] = 'DementiaNet'
    dnet_val['source'] = 'DementiaNet'
    dnet_test['source'] = 'DementiaNet'

    # Combine training sets
    combined_train = pd.concat([dnet_train, adresso_df], ignore_index=True)
    combined_val = dnet_val  # Keep DementiaNet validation as-is
    combined_test = dnet_test  # Keep DementiaNet test as-is

    print(f"\n📊 Combined Dataset:")
    print(f"  Training: {len(combined_train)} samples")
    print(f"    Dementia: {len(combined_train[combined_train['label']=='dementia'])} "
          f"({100*len(combined_train[combined_train['label']=='dementia'])/len(combined_train):.1f}%)")
    print(f"    Control: {len(combined_train[combined_train['label']=='nodementia'])} "
          f"({100*len(combined_train[combined_train['label']=='nodementia'])/len(combined_train):.1f}%)")
    print(f"  Validation: {len(combined_val)} samples (DementiaNet only)")
    print(f"  Test: {len(combined_test)} samples (DementiaNet only)")
    print(f"  Total: {len(combined_train) + len(combined_val) + len(combined_test)} samples")

    total_dementia = (len(combined_train[combined_train['label']=='dementia']) +
                      len(combined_val[combined_val['label']=='dementia']) +
                      len(combined_test[combined_test['label']=='dementia']))
    total_control = (len(combined_train[combined_train['label']=='nodementia']) +
                     len(combined_val[combined_val['label']=='nodementia']) +
                     len(combined_test[combined_test['label']=='nodementia']))

    print(f"\n  Overall class balance:")
    print(f"    Dementia: {total_dementia} ({100*total_dementia/(total_dementia+total_control):.1f}%)")
    print(f"    Control: {total_control} ({100*total_control/(total_dementia+total_control):.1f}%)")

    # Save combined datasets (drop source column for consistency)
    combined_train_csv = os.path.join(output_dir, 'train_dm_combined.csv')
    combined_val_csv = os.path.join(output_dir, 'valid_dm_combined.csv')
    combined_test_csv = os.path.join(output_dir, 'test_dm_combined.csv')

    combined_train.drop(columns=['source']).to_csv(combined_train_csv, sep='\t', index=False)
    combined_val.drop(columns=['source']).to_csv(combined_val_csv, sep='\t', index=False)
    combined_test.drop(columns=['source']).to_csv(combined_test_csv, sep='\t', index=False)

    # Also save a version WITH source labels for analysis
    combined_train_source_csv = os.path.join(output_dir, 'train_dm_combined_with_source.csv')
    combined_train.to_csv(combined_train_source_csv, sep='\t', index=False)

    print(f"\n📁 Files Saved:")
    print(f"  ✓ {combined_train_csv} ({len(combined_train)} samples)")
    print(f"  ✓ {combined_val_csv} ({len(combined_val)} samples)")
    print(f"  ✓ {combined_test_csv} ({len(combined_test)} samples)")
    print(f"  ✓ {combined_train_source_csv} (with source labels)")

    print(f"\n📈 Data Augmentation:")
    print(f"  Training samples increased: 314 → {len(combined_train)} (+{len(adresso_df)} samples, +{100*len(adresso_df)/314:.1f}%)")
    print(f"  This should improve model generalization!")

    print("\n" + "="*70)
    print("✅ Combined dataset created successfully!")
    print("="*70)

    print(f"\n💡 Usage:")
    print(f"  To use DementiaNet only: train_dm_new.csv, valid_dm_new.csv, test_dm_new.csv")
    print(f"  To use combined dataset: train_dm_combined.csv, valid_dm_combined.csv, test_dm_combined.csv")

    return combined_train, combined_val, combined_test

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')

    train_df, val_df, test_df = create_combined_dataset(data_dir)
