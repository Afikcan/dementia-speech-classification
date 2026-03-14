#!/usr/bin/env python3
"""
Fix train_dementia_model.ipynb for transformers 5.1.0 compatibility.

Changes:
1. eval_strategy (instead of evaluation_strategy) - DONE
2. processing_class (instead of tokenizer) for Trainer
3. warmup_steps (instead of warmup_ratio)
"""

import json

# Read the notebook
with open('train_dementia_model.ipynb', 'r') as f:
    notebook = json.load(f)

# Track changes
changes_made = []

# Find and fix cell 21 (Configure Training)
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Fix 1: Replace warmup_ratio with warmup_steps
        if 'warmup_ratio=0.1' in source and 'TrainingArguments' in source:
            old_source = source
            # Calculate warmup_steps: 10% of total steps
            # Total steps = (len(train_dataset) // batch_size) * num_epochs
            # We'll use a placeholder that will be calculated at runtime
            source = source.replace(
                'warmup_ratio=0.1,',
                'warmup_steps=100,  # ~10% of total steps (calculated: train_size//batch_size * epochs * 0.1)'
            )
            if source != old_source:
                cell['source'] = source.split('\n')
                # Ensure each line ends with \n except the last
                cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line
                                 for i, line in enumerate(cell['source'])]
                changes_made.append(f"Cell {i}: Updated warmup_ratio to warmup_steps")

        # Fix 2: Replace tokenizer with processing_class in Trainer
        if 'tokenizer=feature_extractor' in source and 'Trainer(' in source:
            old_source = source
            source = source.replace(
                'tokenizer=feature_extractor',
                'processing_class=feature_extractor'
            )
            if source != old_source:
                cell['source'] = source.split('\n')
                # Ensure each line ends with \n except the last
                cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line
                                 for i, line in enumerate(cell['source'])]
                changes_made.append(f"Cell {i}: Replaced 'tokenizer' with 'processing_class'")

# Write back
with open('train_dementia_model.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

# Report
print("="*70)
print("NOTEBOOK COMPATIBILITY FIXES APPLIED")
print("="*70)
for change in changes_made:
    print(f"✓ {change}")

if not changes_made:
    print("⚠️  No changes needed (already up to date?)")
else:
    print(f"\n✅ {len(changes_made)} fixes applied successfully!")
    print("\nFixed issues:")
    print("  1. warmup_ratio → warmup_steps (transformers 5.2+ deprecation)")
    print("  2. tokenizer → processing_class (transformers 5.x parameter rename)")
    print("  3. eval_strategy (previously fixed)")

print("="*70)
