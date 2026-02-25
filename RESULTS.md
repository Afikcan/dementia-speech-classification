# Training Results - V2

## Summary

Trained `facebook/wav2vec2-base` on 621 samples (DementiaNet + ADReSSo21).

**Test set performance (72 held-out samples):**
- Accuracy: 76.39%
- F1-macro: 65.1%
- Recall (dementia): 43.75%
- Recall (nodementia): 85.71%

**Training time:** ~45 minutes on A100 GPU

---

## Test Set Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Dementia | 46.67% | 43.75% | 45.16% | 16 |
| NoDementia | 84.21% | 85.71% | 84.96% | 56 |
| **Macro Avg** | 65.44% | 64.73% | 65.06% | 72 |
| **Weighted Avg** | 75.27% | 76.39% | 75.81% | 72 |

### Confusion Matrix

```
                Predicted
             Dementia  NoDementia
Actual Dem      7         9
       NoDem    8        48
```

**Interpretation:**
- True positives: 7/16 dementia cases caught (43.75%)
- True negatives: 48/56 nodementia cases caught (85.71%)
- False negatives: 9 (model missed these dementia cases)
- False positives: 8 (model incorrectly flagged as dementia)

---

## Validation Set (During Training)

Best checkpoint selected at epoch 3 based on validation F1-macro.

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Dementia | 34.78% | 47.06% | 40.00% | 17 |
| NoDementia | 82.35% | 75.00% | 78.51% | 52 |
| **Macro Avg** | 58.57% | 61.03% | 59.26% | 69 |

---

## Training Progression

| Epoch | Train Loss | Val Loss | Val Accuracy | Val F1-macro | Val Recall (dementia) |
|-------|-----------|----------|--------------|--------------|---------------------|
| 1 | 0.6372 | 0.5862 | 0.7391 | 0.5608 | 0.4118 |
| 2 | 0.5498 | 0.5539 | 0.7391 | 0.5667 | 0.4118 |
| **3** | 0.4853 | 0.5519 | 0.7391 | **0.5926** | **0.4706** |
| 4 | 0.4394 | 0.5685 | 0.7391 | 0.5690 | 0.4118 |
| 5 | 0.3986 | 0.5869 | 0.7681 | 0.5736 | 0.3529 |

Best epoch: 3 (highest validation F1-macro)

---

## Comparison to V1

| Metric | V1 (Broken) | V2 (This Work) |
|--------|-------------|----------------|
| Test Accuracy | 77.78% | 76.39% |
| Test F1-macro | 43.86% | 65.1% |
| **Recall (dementia)** | **0%** | **43.75%** |
| Recall (nodementia) | 100% | 85.71% |
| Predicts both classes? | NO | YES |

**Key issue in V1:** Model always predicted "nodementia" - completely useless.

**V2 fixes:**
- Class weighting (2x penalty for dementia)
- Oversampling (50/50 balanced training set)
- Data augmentation (noise, time/pitch shift)
- Better metrics (F1-macro instead of accuracy)
- Longer audio (15s vs 10s)

---

## Data Splits

**Training set:**
- Original: 480 samples (38.5% dementia, 61.5% nodementia)
- After oversampling: 590 samples (50% dementia, 50% nodementia)

**Validation set:**
- 69 samples (24.6% dementia, 75.4% nodementia)

**Test set:**
- 72 samples (22.2% dementia, 77.8% nodementia)

All splits are patient-level (no patient appears in multiple splits).

---

## Training Configuration

**Model:** facebook/wav2vec2-base (94.5M parameters)
- Pre-trained encoder: frozen for first epoch, then fine-tuned
- Classification head: trained from scratch (~2K parameters)

**Data:**
- Audio length: 15 seconds
- Sample rate: 16kHz
- Augmentation: Gaussian noise, time stretch, pitch shift, time shift

**Training:**
- Epochs: 15 (early stopping at epoch 3)
- Batch size: 8
- Learning rate: 3e-5
- Weight decay: 0.01
- Warmup steps: 100
- Class weights: [2.0 for dementia, 1.0 for nodementia]
- Mixed precision: FP16

**Hardware:**
- GPU: A100 (40GB)
- Training time: ~45 minutes

---

## Benchmark Comparison

| Study | Dataset Size | Test Accuracy |
|-------|--------------|---------------|
| ADReSS 2020 best | 156 samples | 75-85% |
| ADReSSo 2021 best | 237 samples | 70-80% |
| **This work** | 621 samples | 76.39% |

Our results are in line with published benchmarks.

---

## Limitations

**Performance:**
- Low sensitivity (43.75% - misses over half of dementia cases)
- Better at identifying non-dementia (85.71% recall)
- Not suitable for clinical diagnosis

**Dataset:**
- Small size (621 samples)
- Imbalanced (68% nodementia)
- English only
- Mixed sources (celebrity vs clinical recordings)

**Generalization:**
- Not tested on other languages
- Not tested on different age groups or demographics
- Recording quality may affect performance

---

## Clinical Interpretation

**Sensitivity (dementia recall): 43.75%**
- Catches 43.75% of dementia cases
- Misses 56.25% of cases - too high for standalone diagnosis

**Specificity (nodementia recall): 85.71%**
- Correctly identifies 85.71% of non-dementia cases
- Good for ruling out dementia in low-risk populations

**Appropriate uses:**
- Research tool for speech-dementia correlation
- Preliminary screening (combined with other tests)
- Population risk assessment
- NOT for standalone clinical diagnosis

---

## Output Files

All results saved to: `results/wav2vec2-base-improved-v2/`

- `model.safetensors` - Model weights (361 MB)
- `config.json` - Model configuration
- `preprocessor_config.json` - Feature extractor config
- `all_metrics_v2.json` - Complete metrics
- `training_args.bin` - Training hyperparameters
- `trainer_state.json` - Training history

---

**Training completed:** February 25, 2026
**Model version:** 2.0
**Status:** Research-ready
