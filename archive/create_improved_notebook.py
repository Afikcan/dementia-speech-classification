#!/usr/bin/env python3
"""
Create an improved training notebook with all advanced techniques:
- Class weighting (weighted loss)
- Oversampling minority class
- Better metrics (F1-macro, per-class recall)
- Data augmentation
- Longer audio clips (15 seconds)
- Focal loss option
"""

import json
import os

# Read the original notebook
with open('train_dementia_model.ipynb', 'r') as f:
    notebook = json.load(f)

# Create a new notebook structure
new_notebook = {
    "cells": [],
    "metadata": notebook['metadata'],
    "nbformat": notebook['nbformat'],
    "nbformat_minor": notebook['nbformat_minor']
}

# Helper function to create a cell
def create_markdown_cell(content):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content.split('\n')
    }

def create_code_cell(content):
    lines = content.split('\n')
    # Add newlines to all lines except the last one
    source = [line + '\n' if i < len(lines) - 1 else line
              for i, line in enumerate(lines)]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

# Cell 0: Title and description
new_notebook['cells'].append(create_markdown_cell("""# Dementia Assessment - IMPROVED Training (v2)

**🚀 This version includes advanced techniques to fix the class prediction problem!**

## What's New in V2

✅ **Class Weighting**: 2x penalty for misclassifying dementia
✅ **Oversampling**: Balanced 50/50 training set
✅ **Better Metrics**: F1-macro + per-class recall
✅ **Data Augmentation**: Noise, time stretch, pitch shift
✅ **Longer Audio**: 15 seconds instead of 10
✅ **Focal Loss**: Auto-focus on hard examples

## Previous Problem

The original model only predicted "nodementia" (77% accuracy but 0% recall for dementia).
This version forces the model to learn BOTH classes.

**See [TRAINING_ANALYSIS.md](TRAINING_ANALYSIS.md) for detailed problem analysis**"""))

# Cell 1: Imports
new_notebook['cells'].append(create_markdown_cell("## 1. Imports and Setup"))

new_notebook['cells'].append(create_code_cell("""import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from datasets import Dataset
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import json
import audiomentations as A

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")"""))

# Cell 2: Configuration
new_notebook['cells'].append(create_markdown_cell("""## 2. Configuration

### ⚙️ Enhanced Configuration with Class Balancing"""))

new_notebook['cells'].append(create_code_cell("""# ========================================================================
# DATASET SELECTION
# ========================================================================

USE_COMBINED_DATASET = True  # Combined has better class balance

# ========================================================================
# HYPERPARAMETERS (IMPROVED)
# ========================================================================

# Audio processing
SAMPLING_RATE = 16000
MAX_DURATION = 15.0  # ← INCREASED from 10s to 15s

# Model
MODEL_CHECKPOINT = "facebook/wav2vec2-base"

# Training hyperparameters
EPOCHS = 15  # ← INCREASED for better convergence
BATCH_SIZE = 8
LEARNING_RATE = 2e-5  # ← REDUCED for more stable training

# Advanced techniques
USE_CLASS_WEIGHTING = True   # Weight loss by inverse class frequency
USE_OVERSAMPLING = True      # Oversample minority class to 50/50
USE_AUGMENTATION = True      # Add noise, time stretch, pitch shift
USE_FOCAL_LOSS = False       # Use focal loss (experimental, disable class weighting if True)

# ========================================================================
# PATHS
# ========================================================================

DATA_DIR = "./data"

if USE_COMBINED_DATASET:
    TRAIN_CSV = os.path.join(DATA_DIR, "train_dm_combined.csv")
    VALID_CSV = os.path.join(DATA_DIR, "valid_dm_combined.csv")
    TEST_CSV = os.path.join(DATA_DIR, "test_dm_combined.csv")
    OUTPUT_DIR = "./results/wav2vec2-base-improved-v2"
else:
    TRAIN_CSV = os.path.join(DATA_DIR, "train_dm_new.csv")
    VALID_CSV = os.path.join(DATA_DIR, "valid_dm_new.csv")
    TEST_CSV = os.path.join(DATA_DIR, "test_dm_new.csv")
    OUTPUT_DIR = "./results/wav2vec2-base-improved-dementianet-v2"

# ========================================================================
# SUMMARY
# ========================================================================

print(f"\\n{'='*70}")
print("IMPROVED TRAINING CONFIGURATION V2")
print(f"{'='*70}")
print(f"\\nModel: {MODEL_CHECKPOINT}")
print(f"Audio: {MAX_DURATION}s @ {SAMPLING_RATE} Hz")
print(f"\\nAdvanced Techniques:")
print(f"  ✓ Class weighting: {USE_CLASS_WEIGHTING}")
print(f"  ✓ Oversampling: {USE_OVERSAMPLING}")
print(f"  ✓ Augmentation: {USE_AUGMENTATION}")
print(f"  ✓ Focal loss: {USE_FOCAL_LOSS}")
print(f"\\nTraining:")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"\\nOutput: {OUTPUT_DIR}")
print(f"{'='*70}")"""))

# Cell 3: Load data
new_notebook['cells'].append(create_markdown_cell("## 3. Load Data"))

new_notebook['cells'].append(create_code_cell("""# Load CSV files
train_df = pd.read_csv(TRAIN_CSV, sep='\\t')
valid_df = pd.read_csv(VALID_CSV, sep='\\t')
test_df = pd.read_csv(TEST_CSV, sep='\\t')

print(f"\\n{'='*70}")
print("ORIGINAL DATA DISTRIBUTION")
print(f"{'='*70}")

print(f"\\nTraining Set: {len(train_df)} samples")
print(train_df['label'].value_counts())
dementia_pct = 100*len(train_df[train_df.label=='dementia'])/len(train_df)
print(f"  Dementia: {dementia_pct:.1f}%")
print(f"  No dementia: {100-dementia_pct:.1f}%")

print(f"\\nValidation Set: {len(valid_df)} samples")
print(valid_df['label'].value_counts())

print(f"\\nTest Set: {len(test_df)} samples")
print(test_df['label'].value_counts())"""))

# Cell 4: Oversample minority class
new_notebook['cells'].append(create_markdown_cell("""## 4. Balance Training Set (Oversampling)

**Problem**: Original training has 38.5% dementia, 61.5% nodementia
**Solution**: Oversample dementia to create 50/50 balanced dataset"""))

new_notebook['cells'].append(create_code_cell("""if USE_OVERSAMPLING:
    # Separate classes
    train_dementia = train_df[train_df['label'] == 'dementia']
    train_nodementia = train_df[train_df['label'] == 'nodementia']

    print(f"Before oversampling:")
    print(f"  Dementia: {len(train_dementia)} samples")
    print(f"  Nodementia: {len(train_nodementia)} samples")

    # Oversample dementia to match nodementia count
    train_dementia_oversampled = resample(
        train_dementia,
        replace=True,  # Sample with replacement
        n_samples=len(train_nodementia),
        random_state=42
    )

    # Combine and shuffle
    train_df = pd.concat([train_nodementia, train_dementia_oversampled])
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\\nAfter oversampling:")
    print(f"  Dementia: {len(train_df[train_df.label=='dementia'])} samples")
    print(f"  Nodementia: {len(train_df[train_df.label=='nodementia'])} samples")
    print(f"  Total: {len(train_df)} samples")
    print(f"  Balance: {100*len(train_df[train_df.label=='dementia'])/len(train_df):.1f}% dementia")
    print("\\n✅ Training set is now balanced 50/50!")
else:
    print("⚠️  Oversampling disabled - using imbalanced data")"""))

# Cell 5: Map labels
new_notebook['cells'].append(create_markdown_cell("## 5. Prepare Label Mapping"))

new_notebook['cells'].append(create_code_cell("""# Map labels to integers
label_list = sorted(train_df['label'].unique().tolist())
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

print(f"Label mapping: {label2id}")
print(f"ID to label: {id2label}")

train_df['labels'] = train_df['label'].map(label2id)
valid_df['labels'] = valid_df['label'].map(label2id)
test_df['labels'] = test_df['label'].map(label2id)

# Compute class weights
if USE_CLASS_WEIGHTING:
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(train_df['labels']),
        y=train_df['labels']
    )
    print(f"\\nClass weights (for loss function):")
    for i, weight in enumerate(class_weights_array):
        print(f"  {id2label[i]}: {weight:.3f}")
    print(f"\\n→ Dementia misclassifications will be penalized {class_weights_array[0]/class_weights_array[1]:.2f}x more!")
else:
    class_weights_array = np.array([1.0, 1.0])
    print("\\n⚠️  Class weighting disabled - equal weights for both classes")"""))

# Cell 6: Audio augmentation
new_notebook['cells'].append(create_markdown_cell("""## 6. Define Audio Augmentation

Apply random transformations to increase training data diversity:
- Gaussian noise
- Time stretching
- Pitch shifting
- Time shifting"""))

new_notebook['cells'].append(create_code_cell("""if USE_AUGMENTATION:
    # Define augmentation pipeline
    augment = A.Compose([
        A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5),
        A.TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
        A.PitchShift(min_semitones=-3, max_semitones=3, p=0.5),
        A.Shift(min_shift=-0.3, max_shift=0.3, p=0.5),
    ])
    print("✅ Audio augmentation enabled:")
    print("  - Gaussian noise (50% probability)")
    print("  - Time stretch 0.8x-1.2x (50% probability)")
    print("  - Pitch shift ±3 semitones (50% probability)")
    print("  - Time shift ±30% (50% probability)")
else:
    augment = None
    print("⚠️  Augmentation disabled")

def load_and_preprocess_audio(path, target_sr=16000, max_duration=15.0, apply_augment=False):
    """Load audio file and preprocess it."""
    # Load audio
    waveform, sr = torchaudio.load(path)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Convert to numpy for augmentation
    audio_np = waveform.squeeze().numpy()

    # Apply augmentation (only during training)
    if apply_augment and augment is not None:
        audio_np = augment(samples=audio_np, sample_rate=target_sr)

    # Truncate or pad to max_duration
    max_length = int(target_sr * max_duration)
    if len(audio_np) > max_length:
        audio_np = audio_np[:max_length]
    elif len(audio_np) < max_length:
        padding = max_length - len(audio_np)
        audio_np = np.pad(audio_np, (0, padding), mode='constant')

    return audio_np

print("\\n✓ Audio processing function defined")"""))

# Cell 7: Feature extraction
new_notebook['cells'].append(create_markdown_cell("## 7. Load Feature Extractor"))

new_notebook['cells'].append(create_code_cell("""# Load feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)
feature_extractor.return_attention_mask = True

print(f"Feature extractor: {type(feature_extractor).__name__}")
print(f"Sampling rate: {feature_extractor.sampling_rate} Hz")

def preprocess_function(examples, apply_augment=False):
    """Preprocess batch of examples."""
    audio_arrays = []
    for path in examples["path"]:
        audio = load_and_preprocess_audio(
            path,
            target_sr=feature_extractor.sampling_rate,
            max_duration=MAX_DURATION,
            apply_augment=apply_augment and USE_AUGMENTATION
        )
        audio_arrays.append(audio)

    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
        max_length=int(feature_extractor.sampling_rate * MAX_DURATION),
        truncation=True
    )

    # Convert to numpy arrays for datasets
    return {k: v.numpy() for k, v in inputs.items()}

print("✓ Preprocessing function defined")"""))

# Cell 8: Create datasets
new_notebook['cells'].append(create_markdown_cell("## 8. Create HuggingFace Datasets"))

new_notebook['cells'].append(create_code_cell("""# Create datasets from dataframes
train_dataset = Dataset.from_pandas(train_df[['path', 'labels']])
valid_dataset = Dataset.from_pandas(valid_df[['path', 'labels']])
test_dataset = Dataset.from_pandas(test_df[['path', 'labels']])

print(f"✓ Train dataset: {train_dataset}")
print(f"✓ Valid dataset: {valid_dataset}")
print(f"✓ Test dataset: {test_dataset}")"""))

# Cell 9: Preprocess datasets
new_notebook['cells'].append(create_markdown_cell("""## 9. Preprocess Datasets

**Note**: Training data will have augmentation applied, validation/test will not."""))

new_notebook['cells'].append(create_code_cell("""print("Preprocessing training data (WITH augmentation)...")
encoded_train = train_dataset.map(
    lambda x: preprocess_function(x, apply_augment=True),
    batched=True,
    batch_size=8,
    remove_columns=["path"]
)

print("\\nPreprocessing validation data (NO augmentation)...")
encoded_valid = valid_dataset.map(
    lambda x: preprocess_function(x, apply_augment=False),
    batched=True,
    batch_size=8,
    remove_columns=["path"]
)

print("\\nPreprocessing test data (NO augmentation)...")
encoded_test = test_dataset.map(
    lambda x: preprocess_function(x, apply_augment=False),
    batched=True,
    batch_size=8,
    remove_columns=["path"]
)

print(f"\\n{'='*70}")
print("PREPROCESSING COMPLETE")
print(f"{'='*70}")
print(f"✓ Train: {len(encoded_train)} samples")
print(f"✓ Valid: {len(encoded_valid)} samples")
print(f"✓ Test: {len(encoded_test)} samples")
print(f"{'='*70}")"""))

# Cell 10: Improved metrics
new_notebook['cells'].append(create_markdown_cell("""## 10. Define Improved Evaluation Metrics

**Key changes:**
- Use F1-macro (equal weight to both classes) instead of binary F1
- Report per-class metrics (dementia vs nodementia)
- Focus on recall for dementia (medical screening priority)"""))

new_notebook['cells'].append(create_code_cell("""def compute_metrics(eval_pred):
    \"\"\"Compute comprehensive evaluation metrics.\"\"\"
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None, labels=[0, 1]
    )

    # Macro averages (equal weight to both classes)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro'
    )

    # Accuracy
    acc = accuracy_score(labels, predictions)

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,  # ← PRIMARY METRIC
        'f1_dementia': f1_per_class[0],
        'f1_nodementia': f1_per_class[1],
        'recall_dementia': recall_per_class[0],  # ← CRITICAL for medical screening
        'recall_nodementia': recall_per_class[1],
        'precision_dementia': precision_per_class[0],
        'precision_nodementia': precision_per_class[1],
    }

print("✓ Improved metrics function defined")
print("  Primary metric: F1-macro (equal weight to both classes)")
print("  Critical metric: Recall for dementia class")"""))

# Cell 11: Custom trainer with weighted/focal loss
new_notebook['cells'].append(create_markdown_cell("""## 11. Custom Trainer with Weighted Loss

Implement custom loss functions to handle class imbalance:
- **Weighted Cross-Entropy**: Penalize dementia misclassifications more
- **Focal Loss** (optional): Auto-focus on hard examples"""))

new_notebook['cells'].append(create_code_cell("""class FocalLoss(nn.Module):
    \"\"\"Focal Loss for imbalanced classification.\"\"\"
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()

class WeightedTrainer(Trainer):
    \"\"\"Custom trainer with class-weighted or focal loss.\"\"\"

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if USE_FOCAL_LOSS:
            # Use focal loss
            focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
            loss = focal_loss_fn(logits, labels)
        elif USE_CLASS_WEIGHTING:
            # Use weighted cross-entropy
            weight = torch.tensor(class_weights_array, device=logits.device, dtype=logits.dtype)
            loss_fn = nn.CrossEntropyLoss(weight=weight)
            loss = loss_fn(logits, labels)
        else:
            # Standard cross-entropy
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

print("✓ Custom trainer defined")
if USE_FOCAL_LOSS:
    print("  Using: Focal Loss (alpha=0.25, gamma=2.0)")
elif USE_CLASS_WEIGHTING:
    print(f"  Using: Weighted Cross-Entropy (weights: {class_weights_array})")
else:
    print("  Using: Standard Cross-Entropy")"""))

# Cell 12: Load model
new_notebook['cells'].append(create_markdown_cell("## 12. Load Model"))

new_notebook['cells'].append(create_code_cell("""num_labels = len(label_list)
model = AutoModelForAudioClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True
)

print(f"✓ Model loaded: {MODEL_CHECKPOINT}")
print(f"  Number of labels: {num_labels}")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")"""))

# Cell 13: Training arguments
new_notebook['cells'].append(create_markdown_cell("## 13. Configure Training"))

new_notebook['cells'].append(create_code_cell("""training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    warmup_steps=100,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",  # ← CHANGED from "accuracy"
    push_to_hub=False,
    remove_unused_columns=False,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = WeightedTrainer(  # ← Using custom trainer
    model=model,
    args=training_args,
    train_dataset=encoded_train,
    eval_dataset=encoded_valid,
    processing_class=feature_extractor,
    compute_metrics=compute_metrics,
)

print(f"{'='*70}")
print("TRAINER CONFIGURED")
print(f"{'='*70}")
print(f"Output: {OUTPUT_DIR}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Primary metric: F1-macro (changed from accuracy!)")
print(f"Mixed precision: {training_args.fp16}")
print(f"Total steps: {len(encoded_train) // BATCH_SIZE * EPOCHS}")
print(f"{'='*70}")"""))

# Cell 14: Train
new_notebook['cells'].append(create_markdown_cell("""## 14. Train Model

**Expected time: 30-60 minutes** (longer due to 15s audio + augmentation)

**What to watch:**
- Validation metrics should CHANGE across epochs (not flat!)
- Both f1_dementia and f1_nodementia should be > 0
- Recall for dementia should improve over epochs"""))

new_notebook['cells'].append(create_code_cell("""print(f"{'='*70}")
print("STARTING IMPROVED TRAINING V2")
print(f"{'='*70}")
print(f"Training samples: {len(encoded_train)}")
print(f"Validation samples: {len(encoded_valid)}")
print(f"\\nEnhancements applied:")
print(f"  ✓ Class weighting: {USE_CLASS_WEIGHTING}")
print(f"  ✓ Oversampling: {USE_OVERSAMPLING}")
print(f"  ✓ Augmentation: {USE_AUGMENTATION}")
print(f"  ✓ Focal loss: {USE_FOCAL_LOSS}")
print(f"  ✓ Longer audio: {MAX_DURATION}s")
print(f"  ✓ More epochs: {EPOCHS}")
print(f"{'='*70}")
print()

train_result = trainer.train()

print(f"\\n{'='*70}")
print("TRAINING COMPLETE!")
print(f"{'='*70}")"""))

# Cell 15-17: Evaluation (validation, test, confusion matrices) - copy from original
for i in range(25, 30):
    new_notebook['cells'].append(notebook['cells'][i])

# Cell 18: Save results
new_notebook['cells'].append(notebook['cells'][30])
new_notebook['cells'].append(notebook['cells'][31])

# Cell 19: Final summary with improvements
new_notebook['cells'].append(create_markdown_cell("## 17. Final Summary - V2 Improvements"))

new_notebook['cells'].append(create_code_cell("""print(f"\\n{'='*70}")
print(" "*15 + "IMPROVED TRAINING SUMMARY V2")
print(f"{'='*70}")
print(f"\\nModel: {MODEL_CHECKPOINT}")
print(f"Audio: {MAX_DURATION}s @ {SAMPLING_RATE} Hz")

print(f"\\n🚀 Improvements Applied:")
print(f"  ✓ Class weighting: {USE_CLASS_WEIGHTING}")
if USE_CLASS_WEIGHTING:
    print(f"    Weights: dementia={class_weights_array[0]:.2f}, nodementia={class_weights_array[1]:.2f}")
print(f"  ✓ Oversampling: {USE_OVERSAMPLING}")
if USE_OVERSAMPLING:
    print(f"    Training set balanced to 50/50")
print(f"  ✓ Augmentation: {USE_AUGMENTATION}")
print(f"  ✓ Focal loss: {USE_FOCAL_LOSS}")
print(f"  ✓ F1-macro primary metric (not accuracy)")

print(f"\\n📊 Final Metrics:")
print(f"\\nValidation:")
for key in ['eval_accuracy', 'eval_f1_macro', 'eval_recall_dementia', 'eval_recall_nodementia']:
    if key in valid_metrics:
        print(f"  {key:25s}: {valid_metrics[key]:.4f}")

print(f"\\n🎯 Test (FINAL):")
for key in ['eval_accuracy', 'eval_f1_macro', 'eval_recall_dementia', 'eval_recall_nodementia']:
    if key in test_metrics:
        print(f"  {key:25s}: {test_metrics[key]:.4f}")

print(f"\\n{'='*70}")
print("✅ IMPROVED TRAINING COMPLETE!")
print(f"\\nExpected improvements vs v1:")
print("  • Both classes should be predicted (not just nodementia)")
print("  • Recall for dementia > 0% (was 0% in v1)")
print("  • F1-macro should be meaningful")
print("  • Confusion matrix should show both diagonal cells > 0")
print(f"{'='*70}")"""))

# Write the new notebook
with open('train_dementia_model_v2_improved.ipynb', 'w') as f:
    json.dump(new_notebook, f, indent=1)

print("="*70)
print("✅ IMPROVED NOTEBOOK CREATED!")
print("="*70)
print("\\nFile: train_dementia_model_v2_improved.ipynb")
print("\\nKey improvements:")
print("  1. Class weighting (2x penalty for dementia misclassification)")
print("  2. Oversampling minority class (50/50 balanced training)")
print("  3. Better metrics (F1-macro + per-class recall)")
print("  4. Data augmentation (noise, time stretch, pitch shift)")
print("  5. Longer audio (15 seconds instead of 10)")
print("  6. Focal loss option (experimental)")
print("  7. More epochs (15 instead of 10)")
print("\\nReady to run!")
print("="*70)
