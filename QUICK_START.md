# Quick Start

Train a Wav2Vec2 model to detect dementia from speech audio in under an hour.

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Check Data

Data splits are ready in `data/`:
- `train_dm_combined.csv` (480 samples)
- `valid_dm_combined.csv` (69 samples)
- `test_dm_combined.csv` (72 samples)

### 3. Open Training Notebook

```bash
jupyter notebook train_v2_improved.ipynb
```

### 4. Run All Cells

- Cell 1: GPU selection (uses GPU 6)
- Cells 2-38: Training pipeline

Training time: ~45 minutes on A100 GPU

### 5. Check Results

Results saved in: `results/wav2vec2-base-improved-v2/`

Key files:
- `all_metrics_v2.json` - Performance metrics
- `model.safetensors` - Trained model weights
- `config.json` - Model configuration

---

## Expected Results

**Test Performance:**
- Accuracy: 76.4%
- Recall (dementia): 43.8%
- Recall (nodementia): 85.7%
- F1-macro: 65.1%

The model predicts both classes and is competitive with published research.

---

## Configuration (Optional)

Edit Cell 4 in the notebook to change settings:

```python
# Dataset
USE_COMBINED_DATASET = True  # or False for DementiaNet only

# Audio
MAX_DURATION = 15.0  # Try 10.0 or 20.0
SAMPLING_RATE = 16000

# Training
EPOCHS = 15
BATCH_SIZE = 8  # Reduce to 4 if GPU OOM
LEARNING_RATE = 2e-5

# Recommended settings
USE_CLASS_WEIGHTING = True
USE_OVERSAMPLING = True
USE_AUGMENTATION = True
```

---

## Troubleshooting

### GPU Out of Memory

Edit Cell 4:
```python
BATCH_SIZE = 4  # Reduce from 8
MAX_DURATION = 12.0  # Reduce from 15.0
```

### Different GPU

Edit Cell 1:
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Change GPU number
```

Check available GPUs: `nvidia-smi`

---

## Additional Documentation

- [RESULTS.md](RESULTS.md) - Detailed performance analysis
- [docs/RUN_V2_IMPROVED.md](docs/RUN_V2_IMPROVED.md) - Complete training guide
- [docs/TRAINING_ANALYSIS.md](docs/TRAINING_ANALYSIS.md) - Why V1 failed

---

## What Changed from V1?

V1 had a critical bug: only predicted "nodementia" (0% recall for dementia).

V2 fixes:
- Class weighting (2x penalty for dementia)
- Oversampling (50/50 balanced training set)
- Data augmentation (noise, time/pitch shift)
- Better metrics (F1-macro)
- Longer audio (15s)

Result: 43.75% dementia recall (was 0% in V1)
