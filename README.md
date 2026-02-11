# Dementia Assessment from Speech Audio

A deep learning project for binary classification of dementia from speech audio using fine-tuned transformer-based audio models (Wav2Vec2, AST, HuBERT, Whisper).

## Overview

This project uses the **DementiaNet** corpus to classify speech recordings as either dementia or nodementia. The goal is to improve upon a baseline 37.5% accuracy (from a 4-class task) by:

1. Simplifying to binary classification (dementia vs nodementia)
2. Adding explainability via attention map visualization
3. Optimizing for deployment via INT8 quantization
4. Implementing audio augmentation techniques

**Course Project** - Instructor: Zongxing Xie

## Dataset

**DementiaNet Corpus:**
- 455 audio clips (.wav files)
- 215 speakers (84 dementia, 131 nodementia)
- Class distribution: ~29% dementia, ~71% nodementia (imbalanced)
- Organized in `data/dementia/` and `data/nodementia/` folders by speaker

**Data Structure:**
```
data/
├── dementia/
│   ├── PersonName1/
│   │   ├── audio1.wav
│   │   └── audio2.wav
│   └── PersonName2/
│       └── audio1.wav
└── nodementia/
    ├── PersonName3/
    │   └── audio1.wav
    └── PersonName4/
        └── audio1.wav
```

## Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd dementia_assessment
```

### 2. Create virtual environment
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

**Option A: With CUDA 12.4 (recommended for training)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**Option B: CPU-only**
```bash
pip install -r requirements.txt
```

### 4. Download the data
```bash
python download_audio.py
```

**Note:** If automatic download fails (Google Drive >50 files limit), manually download the data:
1. Download `dementia.zip` and `nodementia.zip` from the Google Drive link (provided separately)
2. Extract to `data/dementia/` and `data/nodementia/`

## Usage

### 1. Generate train/validation splits
```bash
python generate_csv.py
```

This creates:
- `data/train_dm_new.csv` (359 samples: 110 dementia, 249 nodementia)
- `data/valid_dm_new.csv` (96 samples: 21 dementia, 75 nodementia)

**Important:** Uses `GroupShuffleSplit` to prevent data leakage (same speaker never appears in both train and validation).

### 2. Train a model
```bash
python train_model.py --model_type wav2vec2 --epochs 5 --batch_size 4 --lr 3e-5
```

**Supported models:**
- `wav2vec2` (default) - facebook/wav2vec2-base (95M params)
- `ast` - MIT Audio Spectrogram Transformer
- `hubert` - facebook/hubert-base
- `whisper` - OpenAI Whisper encoder (falls back to wav2vec2)

**Training parameters:**
- Audio resampled to 16kHz
- Clips truncated/padded to 10 seconds
- Checkpoints saved to `./results/{model}-finetuned/`

## Project Structure

```
dementia_assessment/
├── train_model.py           # Main training script
├── generate_csv.py          # Create train/valid splits
├── download_audio.py        # Download DementiaNet data
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
├── CLAUDE.md               # Detailed project documentation
├── README.md               # This file
├── data/                   # Audio data (not in git)
│   ├── dementia/
│   ├── nodementia/
│   ├── train_dm_new.csv
│   └── valid_dm_new.csv
├── code_examples/          # Reference notebooks (original 4-class baseline)
│   ├── 01-prepare-data.ipynb
│   ├── 02-finetune.ipynb
│   └── 03-eval.ipynb
└── files/
    └── project_7357_zx.pdf  # Project documentation
```

## Results

**Baseline (4-class task):**
- Model: wav2vec2-xls-r-300m (300M params)
- Accuracy: 37.5% on 32 test samples
- Classes: no dementia, 5 years to dementia, 10 years to dementia, zero years to dementia
- Issues: Model mostly predicted "five years" and "no dementia" classes

**Current (binary task):**
- Binary baseline not yet established
- Work in progress...

## Known Issues & Future Work

### Issues
1. **Class imbalance**: 71% nodementia / 29% dementia - models may bias toward majority class
2. **No test set**: Currently only train/valid splits exist (need 3-way split)
3. **Small dataset**: Only 455 clips total - augmentation critical

### Planned Improvements
1. **Audio augmentation**: Time stretch, pitch shift, noise injection, SpecAugment
2. **Multiple crops per file**: Use overlapping 10s windows to effectively multiply data
3. **Class weighting**: Use `CrossEntropyLoss(weight=...)` to address imbalance
4. **Stratified splitting**: Ensure train/valid have similar class distributions
5. **Segment length experiments**: Compare 10s vs 15s vs 20s vs 30s clips
6. **Attention visualization**: Extract and visualize attention maps over audio waveforms
7. **INT8 quantization**: Post-training quantization for deployment efficiency
8. **Test set creation**: Proper held-out evaluation set

## Environment

- **Hardware**: A100 GPU with CUDA 12.4
- **Python**: 3.12.3
- **Key packages**:
  - PyTorch 2.6.0+cu124
  - transformers 5.1.0
  - torchaudio 2.6.0+cu124
  - datasets 4.5.0
  - librosa 0.11.0
  - audiomentations 0.43.1

## Citation & Attribution

Dataset: **DementiaNet corpus** (attribution details to be added)

Project developed as part of coursework under the instruction of **Zongxing Xie**.

## License

(To be determined - add appropriate license for your coursework)

## Contact

(Add your contact information or leave blank)
