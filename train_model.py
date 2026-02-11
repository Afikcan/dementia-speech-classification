
import os
import argparse
import sys
import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import Dataset, Audio, ClassLabel
from transformers import (
    AutoConfig, 
    AutoFeatureExtractor, 
    AutoModelForAudioClassification, 
    TrainingArguments, 
    Trainer,
    ASTFeatureExtractor
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- Constants ---
SAMPLING_RATE = 16000
MAX_DURATION = 10.0 # seconds, to avoid OOM. Original was 32s?
# Wav2Vec2 usually trained on shorter clips. AST requires specific length.

def parse_args():
    parser = argparse.ArgumentParser(description="Train Audio Classification Model")
    parser.add_argument("--model_type", type=str, default="wav2vec2", choices=["wav2vec2", "ast", "hubert", "whisper"], help="Model architecture to use")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing CSVs and audio folders")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    return parser.parse_args()

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def preprocess_function(examples, feature_extractor, max_duration=10.0):
    audio_arrays = [x["array"] for x in examples["audio"]]
    # Pad or truncate to max_duration
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        truncation=True, 
        padding=True # Will pad in collator usually, but here for safety
    )
    return inputs

def main():
    args = parse_args()
    
    # 1. Select Model checkpoint
    if args.model_type == "wav2vec2":
        model_checkpoint = "facebook/wav2vec2-base" # or "facebook/wav2vec2-large-xls-r-300m" for multilingual
    elif args.model_type == "ast":
        model_checkpoint = "MIT/ast-finetuned-audioset-10-10-0.4593"
    elif args.model_type == "hubert":
        model_checkpoint = "facebook/hubert-base-ls960"
    elif args.model_type == "whisper":
         # Whisper is Seq2Seq usually, but we can use Encoder for classification? 
         # Or use "openai/whisper-tiny" and extract features.
         # For simplicity, let's stick to AutoModelForAudioClassification compatible models first.
         # There isn't a direct AutoModelForAudioClassification for Whisper in standardized way yet without custom head.
         # We will skip Whisper for this initial script or implement custom if needed.
         print("Whisper classification requires custom head implementation. Switching to wav2vec2 for now.")
         model_checkpoint = "facebook/wav2vec2-base"
         args.model_type = "wav2vec2"
    
    print(f"Using model: {model_checkpoint}")

    # 2. Load Data
    train_csv = os.path.join(args.data_dir, "train_dm_new.csv")
    valid_csv = os.path.join(args.data_dir, "valid_dm_new.csv")
    
    # Read CSVs directly into Dataset
    # We need to load audio. The 'path' column has absolute paths.
    train_df = pd.read_csv(train_csv, sep='\t')
    valid_df = pd.read_csv(valid_csv, sep='\t')
    
    # Map labels to integers
    label_list = train_df['label'].unique().tolist()
    label_list.sort() # Ensure consistent order
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    
    print(f"Labels: {label2id}")
    
    train_df['label_id'] = train_df['label'].map(label2id)
    valid_df['label_id'] = valid_df['label'].map(label2id)
    
    # Create Dataset objects
    train_dataset = Dataset.from_pandas(train_df[['path', 'label_id']])
    valid_dataset = Dataset.from_pandas(valid_df[['path', 'label_id']])
    
    # Cast 'path' to Audio feature
    train_dataset = train_dataset.cast_column("path", Audio(sampling_rate=SAMPLING_RATE))
    valid_dataset = valid_dataset.cast_column("path", Audio(sampling_rate=SAMPLING_RATE))
    
    # Rename 'path' to 'audio' for clarity? No, cast column handles it.
    train_dataset = train_dataset.rename_column("path", "audio")
    valid_dataset = valid_dataset.rename_column("path", "audio")
    
    # 3. Feature Extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    
    # Wav2Vec2 feature extractor might default to do_normalize=True.
    if args.model_type == "wav2vec2":
         feature_extractor.return_attention_mask = True
    
    # Preprocess datasets
    encoded_train = train_dataset.map(lambda x: preprocess_function(x, feature_extractor), batched=True, remove_columns=["audio"])
    encoded_valid = valid_dataset.map(lambda x: preprocess_function(x, feature_extractor), batched=True, remove_columns=["audio"])
    
    # Rename label_id to labels
    encoded_train = encoded_train.rename_column("label_id", "labels")
    encoded_valid = encoded_valid.rename_column("label_id", "labels")

    # 4. Model
    num_labels = len(label_list)
    model = AutoModelForAudioClassification.from_pretrained(
        model_checkpoint, 
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label
    )

    # 5. Trainer
    model_name = model_checkpoint.split("/")[-1]
    
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, f"{model_name}-finetuned"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        remove_unused_columns=False # Important/Required for audio usually
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train,
        eval_dataset=encoded_valid,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    
    print("Evaluating...")
    metrics = trainer.evaluate()
    print(metrics)

if __name__ == "__main__":
    main()
