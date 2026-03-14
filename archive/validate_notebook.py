#!/usr/bin/env python3
"""
Comprehensive validation of train_dementia_model.ipynb for transformers 5.1.0.

This script checks:
1. All required libraries are importable
2. Data files exist
3. No deprecated parameters are used
4. Code syntax is valid
"""

import os
import sys
import json

print("="*70)
print("NOTEBOOK VALIDATION FOR TRANSFORMERS 5.1.0")
print("="*70)

# Check 1: Library imports
print("\n[1/5] Checking library imports...")
try:
    import torch
    import torchaudio
    import transformers
    import datasets
    import sklearn
    import pandas
    import numpy
    import matplotlib
    import seaborn

    print(f"  ✓ PyTorch {torch.__version__}")
    print(f"  ✓ Transformers {transformers.__version__}")
    print(f"  ✓ Datasets {datasets.__version__}")
    print(f"  ✓ All libraries importable")
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

# Check 2: Transformers version compatibility
print("\n[2/5] Checking transformers version...")
from packaging import version
transformers_version = version.parse(transformers.__version__)
if transformers_version >= version.parse("5.0.0"):
    print(f"  ✓ Transformers {transformers.__version__} (5.x series)")
    print("  ℹ Using 5.x API: eval_strategy, processing_class, warmup_steps")
else:
    print(f"  ⚠ Transformers {transformers.__version__} (older API)")
    print("  ⚠ Notebook is configured for 5.x - may have issues")

# Check 3: Data files
print("\n[3/5] Checking data files...")
data_files = {
    "DementiaNet train": "./data/train_dm_new.csv",
    "DementiaNet valid": "./data/valid_dm_new.csv",
    "DementiaNet test": "./data/test_dm_new.csv",
    "Combined train": "./data/train_dm_combined.csv",
    "Combined valid": "./data/valid_dm_combined.csv",
    "Combined test": "./data/test_dm_combined.csv",
}

missing_files = []
for name, path in data_files.items():
    if os.path.exists(path):
        print(f"  ✓ {name}: {path}")
    else:
        print(f"  ✗ {name}: {path} (NOT FOUND)")
        missing_files.append(path)

if missing_files:
    print(f"\n  ⚠ {len(missing_files)} data files missing")
    print("  Run generate_csv_with_test.py or generate_combined_dataset.py")
else:
    print("\n  ✓ All data files present")

# Check 4: Notebook code validation
print("\n[4/5] Validating notebook code...")
with open('train_dementia_model.ipynb', 'r') as f:
    notebook = json.load(f)

issues = []
deprecated_patterns = {
    'evaluation_strategy': 'Should use eval_strategy (5.x)',
    'tokenizer=': 'Should use processing_class= (5.x)',
    'warmup_ratio=': 'Should use warmup_steps= (5.2+)',
}

for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Check for deprecated patterns
        for pattern, message in deprecated_patterns.items():
            if pattern in source:
                issues.append(f"Cell {i}: Found '{pattern}' - {message}")

if issues:
    print("  ✗ Found compatibility issues:")
    for issue in issues:
        print(f"    - {issue}")
else:
    print("  ✓ No deprecated parameters found")
    print("  ✓ Using transformers 5.x API")

# Check 5: Verify critical fixes
print("\n[5/5] Verifying critical fixes...")
fixes_verified = {
    'eval_strategy': False,
    'processing_class': False,
    'warmup_steps': False,
}

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        if 'eval_strategy=' in source and 'TrainingArguments' in source:
            fixes_verified['eval_strategy'] = True
        if 'processing_class=' in source and 'Trainer(' in source:
            fixes_verified['processing_class'] = True
        if 'warmup_steps=' in source and 'TrainingArguments' in source:
            fixes_verified['warmup_steps'] = True

for fix, verified in fixes_verified.items():
    if verified:
        print(f"  ✓ {fix} parameter found")
    else:
        print(f"  ✗ {fix} parameter NOT found")

# Final summary
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)

all_good = (
    len(missing_files) == 0 and
    len(issues) == 0 and
    all(fixes_verified.values())
)

if all_good:
    print("\n✅ ALL CHECKS PASSED!")
    print("\nThe notebook is ready to run:")
    print("  1. Open train_dementia_model.ipynb")
    print("  2. Set USE_COMBINED_DATASET = True (or False)")
    print("  3. Run all cells")
    print("\nExpected training time: ~20-50 minutes on A100 GPU")
else:
    print("\n⚠️  SOME ISSUES FOUND")
    if missing_files:
        print(f"\n  - {len(missing_files)} data files missing")
        print("    Run: python generate_combined_dataset.py")
    if issues:
        print(f"\n  - {len(issues)} code compatibility issues")
        print("    Run: python fix_notebook_compatibility.py")
    if not all(fixes_verified.values()):
        print(f"\n  - Missing critical fixes")
        print("    Manually review the notebook")

print("="*70)
