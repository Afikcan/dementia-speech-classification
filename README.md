# Dementia Assessment from Speech Audio

Fine-tuned Wav2Vec2 model for binary dementia classification from speech.

**Performance**: 76.4% accuracy, 43.8% recall for dementia on test set (72 samples)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Open training notebook
jupyter notebook train_v2_improved.ipynb

# Run all cells - training takes ~45 minutes on A100
```

---

## Results

Model: `facebook/wav2vec2-base` fine-tuned (94.5M parameters)
Dataset: DementiaNet + ADReSSo21 (621 samples total)

### Test Set (72 held-out samples)

| Metric | Value |
|--------|-------|
| Accuracy | 76.39% |
| F1-macro | 65.1% |
| Recall (dementia) | 43.75% |
| Recall (nodementia) | 85.71% |
| Precision (dementia) | 46.67% |
| Precision (nodementia) | 84.21% |

### Confusion Matrix

```
                Predicted
             Dementia  NoDementia
Actual Dem      7         9
       NoDem    8        48
```

**Key point**: The model predicts both classes. Previous version (V1) only predicted "nodementia" - completely broken.

See [RESULTS.md](RESULTS.md) for details.

---

## What Makes It Work

- **Class weighting**: Penalizes dementia misclassification 2x more
- **Oversampling**: Balances training set to 50/50
- **Data augmentation**: Noise, time stretch, pitch shift, time shift
- **15-second audio clips**: More context than typical 10s
- **F1-macro metric**: Better than accuracy for imbalanced data
- **Patient-level splitting**: Prevents data leakage

---

## Dataset

**Combined from two sources:**
1. DementiaNet: 455 samples (celebrity speech from YouTube)
2. ADReSSo21: 166 samples (clinical interview data)

**Total**: 621 samples (199 dementia, 422 nodementia)

**Splits** (patient-level):
- Training: 480 → 590 with oversampling (balanced 50/50)
- Validation: 69 samples
- Test: 72 samples

---

## Installation

### Requirements
- Python 3.12+
- CUDA 12.4
- ~5GB disk space
- ~8GB GPU memory

### Setup

```bash
git clone <your-repo-url>
cd dementia_assessment

python3.12 -m venv venv
source venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Download Data

```bash
python download_audio.py
```

Or download manually from Google Drive (link provided separately).

---

## Usage

### Training

**Jupyter Notebook (recommended):**
```bash
jupyter notebook train_v2_improved.ipynb
# Run all cells
```

**Python Script:**
```bash
python train_v2_improved.py
```

Training time: ~45 minutes on A100 GPU

### Inference

```python
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch
import torchaudio

# Load model
model_path = "results/wav2vec2-base-improved-v2"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
model = AutoModelForAudioClassification.from_pretrained(model_path)

# Load audio (15s, 16kHz, mono)
waveform, sr = torchaudio.load("audio.wav")

# Preprocess
inputs = feature_extractor(waveform.squeeze().numpy(),
                           sampling_rate=16000,
                           return_tensors="pt")

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=-1).item()

label = "dementia" if predicted_class == 0 else "nodementia"
print(f"Prediction: {label}")
```

---

## Project Structure

```
dementia_assessment/
├── train_v2_improved.ipynb       # Main training notebook
├── train_v2_improved.py          # Python script version
├── RESULTS.md                    # Detailed results
├── MODEL_CARD.md                 # Model documentation
├── QUICK_START.md                # Getting started
├── requirements.txt              # Dependencies
│
├── generate_csv_with_test.py    # Generate data splits
├── generate_combined_dataset.py # Combine datasets
├── download_audio.py             # Download data
│
├── docs/                         # Additional documentation
│   ├── RUN_V2_IMPROVED.md
│   ├── TRAINING_ANALYSIS.md
│   └── GPU_SELECTION_FIX.md
│
├── data/                         # Data files
│   ├── train_dm_combined.csv
│   ├── valid_dm_combined.csv
│   └── test_dm_combined.csv
│
├── results/
│   └── wav2vec2-base-improved-v2/  # Trained model
│
└── archive/                      # Old scripts
```

---

## How It Works

### Transfer Learning Approach

**Pre-trained encoder** (94.5M params):
- Trained on 960 hours of speech
- Already understands phonetics, prosody, rhythm
- We keep these weights and fine-tune them

**Classification head** (~2K params):
- Small layer on top
- Maps audio features to dementia/nodementia
- Trained from scratch on our data

This approach works well with small datasets because we're not learning speech understanding from scratch.

---

## Limitations

### Technical
- Small dataset (621 samples)
- Low sensitivity (43.75% - misses over half of dementia cases)
- English only
- Fixed 15-second audio length

### Clinical
- **Not diagnostic-grade** - sensitivity too low for clinical use
- No FDA approval
- Not validated on diverse populations
- Should not replace clinical assessment

### Appropriate Use
- Research tool
- Preliminary screening (combined with other tests)
- Population risk assessment
- **NOT** standalone clinical diagnosis

---

## Acknowledgments

**Datasets:**
- DementiaNet corpus
- ADReSSo Challenge 2021

**Course Project:**
- Instructor: Zongxing Xie
- Institution: Kennesaw State University

**Tools:**
- HuggingFace Transformers
- PyTorch
- Facebook AI Research (Wav2Vec2)

---

**Last Updated**: February 25, 2026
**Model Version**: 2.0
**Status**: Research-ready, not clinical-grade
