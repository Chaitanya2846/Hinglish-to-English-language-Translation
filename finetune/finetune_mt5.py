"""
finetune_mt5.py — Seq2Seq LoRA Fine-Tuning for Google mT5-Small (300M)
=====================================================================
Architecture: google/mt5-small (Multilingual Text-to-Text Transformer)
              Trained on mC4 corpus which includes massive Romanized Hindi internet text.
              Proven in literature: Agarwal et al. (RANLP 2021) achieved 29.5 BLEU with mT5.

Dataset: Custom Web-Scraped Social Media Corpus (9,769 train / 996 val pairs)
Evaluation: BLEU, ROUGE-1/2/L, chrF++, BERTScore, METEOR

Usage (Colab T4 GPU):
  python finetune_mt5.py
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List

import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODEL_ID = "google/mt5-small"
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "mt5_hinglish_lora"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "scraped_train_corpus.jsonl"
VAL_FILE = DATA_DIR / "scraped_val_corpus.jsonl"

# mT5 uses text-to-text format — prefix tells the model what task to do
PREFIX = "translate Hinglish to English: "
MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 128
NUM_EPOCHS = 5          # More epochs for smaller model to converge
BATCH_SIZE = 8
GRAD_ACCUM = 2          # Effective batch = 16
LEARNING_RATE = 3e-4    # Higher LR works well for mT5-small with LoRA
NUM_BEAMS = 5
LORA_R = 32
LORA_ALPHA = 64


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    data = []
    if not path.exists():
        print(f"  [WARNING] File not found: {path}")
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    src = item.get("source", "").strip()
                    tgt = item.get("target", "").strip()
                    if src and tgt:
                        data.append({"source": src, "target": tgt})
                except json.JSONDecodeError:
                    continue
    return data


def compute_all_metrics(predictions, references):
    import evaluate as hf_evaluate
    results = {}
    try:
        import sacrebleu
        results["BLEU"] = round(sacrebleu.corpus_bleu(predictions, [[r] for r in references]).score, 2)
        results["chrF++"] = round(sacrebleu.corpus_chrf(predictions, [[r] for r in references], word_order=2).score, 2)
    except Exception:
        results["BLEU"] = results["chrF++"] = 0.0
    try:
        r = hf_evaluate.load("rouge").compute(predictions=predictions, references=references)
        results["ROUGE-1"] = round(r["rouge1"] * 100, 2)
        results["ROUGE-2"] = round(r["rouge2"] * 100, 2)
        results["ROUGE-L"] = round(r["rougeL"] * 100, 2)
    except Exception:
        results["ROUGE-1"] = results["ROUGE-2"] = results["ROUGE-L"] = 0.0
    try:
        results["METEOR"] = round(hf_evaluate.load("meteor").compute(predictions=predictions, references=references)["meteor"] * 100, 2)
    except Exception:
        results["METEOR"] = 0.0
    try:
        bs = hf_evaluate.load("bertscore").compute(predictions=predictions, references=references, lang="en")
        results["BERTScore_F1"] = round(sum(bs["f1"]) / len(bs["f1"]), 4)
    except Exception:
        results["BERTScore_F1"] = 0.0
    return results


def main():
    print("=" * 70)
    print("  Google mT5-Small (300M) LoRA Fine-Tuning — Scraped Social Corpus")
    print("=" * 70)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB)")

    # 1. Load Data
    print("\n[1/5] Loading Web-Scraped Corpus...")
    train_pairs = load_jsonl(TRAIN_FILE)
    val_pairs = load_jsonl(VAL_FILE)
    print(f"  Train: {len(train_pairs):,} | Val: {len(val_pairs):,}")

    # 2. Tokenizer & Model
    print(f"\n[2/5] Loading Tokenizer & Model ({MODEL_ID})...")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("\n[3/5] Loading Model...")
    t0 = time.time()
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,  # mT5-small fits in fp32 on T4 (~1.2 GB)
    )
    if device == "cuda":
        model = model.cuda()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # LoRA on encoder + decoder attention layers
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"],  # T5-style layer names
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. Preprocess with text-to-text format
    print("\n[4/5] Tokenizing dataset (text-to-text format)...")

    def preprocess(examples):
        inputs = [PREFIX + src for src in examples["source"]]
        targets = examples["target"]
        model_inputs = tokenizer(inputs, max_length=MAX_SOURCE_LENGTH, truncation=True, padding=False)
        labels = tokenizer(text_target=targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding=False)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    raw_train = Dataset.from_list(train_pairs)
    raw_val = Dataset.from_list(val_pairs)
    train_dataset = raw_train.map(preprocess, batched=True, remove_columns=["source", "target"])
    val_dataset = raw_val.map(preprocess, batched=True, remove_columns=["source", "target"])

    # 4. Training
    print(f"\n[5/5] Training for {NUM_EPOCHS} epochs...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        generation_num_beams=NUM_BEAMS,
        generation_max_length=MAX_TARGET_LENGTH,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    t0 = time.time()
    trainer.train()
    train_time = (time.time() - t0) / 60
    print(f"  Training finished in {train_time:.1f} minutes")

    # Save
    print(f"\nSaving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 5. Full 7-metric evaluation
    print("\n[EVAL] Running 7-metric evaluation on validation set...")
    model.eval()
    eval_samples = val_pairs[:150]
    predictions, references, latencies = [], [], []

    for i, item in enumerate(eval_samples):
        input_text = PREFIX + item["source"]
        inputs = tokenizer(input_text, return_tensors="pt", max_length=MAX_SOURCE_LENGTH, truncation=True)
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=96,
                num_beams=NUM_BEAMS,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        latencies.append((time.time() - t0) * 1000)

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        predictions.append(pred)
        references.append(item["target"])

        if i < 5:
            print(f"[{i+1}] Src : {item['source']}")
            print(f"    Ref : {item['target']}")
            print(f"    Pred: {pred}")
            print("-" * 50)

    metrics = compute_all_metrics(predictions, references)
    metrics["Latency_ms"] = round(sum(latencies) / len(latencies), 1)
    metrics["Train_time_min"] = round(train_time, 1)

    print(f"\n{'─' * 70}")
    print(f"  mT5-Small RESULTS:")
    print(f"    BLEU       : {metrics['BLEU']}")
    print(f"    ROUGE-1    : {metrics['ROUGE-1']}")
    print(f"    ROUGE-2    : {metrics['ROUGE-2']}")
    print(f"    ROUGE-L    : {metrics['ROUGE-L']}")
    print(f"    chrF++     : {metrics['chrF++']}")
    print(f"    METEOR     : {metrics['METEOR']}")
    print(f"    BERTScore  : {metrics['BERTScore_F1']}")
    print(f"    Latency    : {metrics['Latency_ms']} ms/sentence")
    print(f"    Train Time : {metrics['Train_time_min']} minutes")
    print(f"{'─' * 70}")

    with open(OUTPUT_DIR / "eval_metrics.json", "w") as f:
        json.dump({"model": "Google mT5-Small (300M)", "metrics": metrics}, f, indent=2)
    print("\n[+] Done! Metrics saved.")


if __name__ == "__main__":
    main()
