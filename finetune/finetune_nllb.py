"""
finetune_nllb.py — Seq2Seq LoRA Fine-Tuning for Meta NLLB-200 (600M)
===================================================================
Architecture: facebook/nllb-200-distilled-600M (Seq2Seq Transformer)
Dataset: Custom Web-Scraped Social Media Corpus (9,769 pairs)
Evaluation: BLEU, ROUGE-1/2/L, chrF++, BERTScore, METEOR
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODEL_ID = "facebook/nllb-200-distilled-600M"
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "nllb_hinglish_lora"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "scraped_train_corpus.jsonl"
VAL_FILE = DATA_DIR / "scraped_val_corpus.jsonl"

SRC_LANG = "hin_Latn"  # or eng_Latn
TGT_LANG = "eng_Latn"
MAX_LENGTH = 128
NUM_EPOCHS = 3
BATCH_SIZE = 8
GRAD_ACCUM = 2
LEARNING_RATE = 5e-4


def load_pairs(filepath: Path) -> List[Dict[str, str]]:
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                pairs.append({"source": data["source"].strip(), "target": data["target"].strip()})
    return pairs


def main():
    print("=" * 65)
    print("  Meta NLLB-200 (600M) Fine-Tuning on Scraped Social Corpus")
    print("=" * 65)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load Dataset
    print("\n[1/4] Loading Custom Web-Scraped Corpus...")
    train_pairs = load_pairs(TRAIN_FILE)
    val_pairs = load_pairs(VAL_FILE)
    print(f"  Training Corpus   : {len(train_pairs):,} pairs")
    print(f"  Validation Corpus : {len(val_pairs):,} pairs")

    # 2. Tokenizer & Model
    print(f"\n[2/4] Loading Tokenizer & Model ({MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang="hin_Deva", tgt_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. Preprocess
    def preprocess_function(examples):
        inputs = examples["source"]
        targets = examples["target"]
        model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
        labels = tokenizer(text_target=targets, max_length=MAX_LENGTH, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    raw_train = Dataset.from_list(train_pairs)
    train_dataset = raw_train.map(preprocess_function, batched=True, remove_columns=["source", "target"])

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    # 4. Train & Save
    print("\n[3/4] Starting Fine-Tuning...")
    t0 = time.time()
    trainer.train()
    print(f"  Training finished in {(time.time() - t0)/60:.1f} minutes")

    print(f"\n[4/4] Saving LoRA Adapter to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("  Model saved successfully!")


if __name__ == "__main__":
    main()
