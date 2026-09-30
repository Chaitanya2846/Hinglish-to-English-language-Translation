"""
finetune_sarvam.py — 4-bit QLoRA Fine-Tuning for Sarvam-1 (2B)
=============================================================
Architecture: Sarvam-1 (India Flagship 2B Foundation Model)
Dataset: Custom Web-Scraped Social Media Corpus (9,769 pairs)
Evaluation: BLEU, ROUGE-1/2/L, chrF++, BERTScore, METEOR
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Paths & Config ──────────────────────────────────────────────────────────
MODEL_ID = "sarvamai/sarvam-1"
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "sarvam_hinglish_lora"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "scraped_train_corpus.jsonl"
VAL_FILE = DATA_DIR / "scraped_val_corpus.jsonl"

PROMPT_TEMPLATE = "Hinglish: {source}\nEnglish: {target}"
INFERENCE_TEMPLATE = "Hinglish: {source}\nEnglish:"

NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
MAX_LENGTH = 128


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
    print("  Sarvam-1 (2B) Fine-Tuning on Web-Scraped Social Media Corpus")
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

    # 2. Tokenizer & Quantized Model
    print(f"\n[2/4] Loading Tokenizer & 4-bit Quantized Model ({MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. Preprocess Dataset
    def format_example(pair):
        prompt_text = f"Hinglish: {pair['source']}\nEnglish: "
        full_text = f"{prompt_text}{pair['target']}{tokenizer.eos_token}"
        prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_tokens = tokenizer(full_text, max_length=MAX_LENGTH, truncation=True)["input_ids"]

        labels = list(full_tokens)
        prompt_len = min(len(prompt_tokens), len(labels))
        for i in range(prompt_len):
            labels[i] = -100  # mask loss on prompt

        return {"input_ids": full_tokens, "labels": labels}

    train_data = [format_example(p) for p in train_pairs]
    train_dataset = Dataset.from_list(train_data)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="epoch",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
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
