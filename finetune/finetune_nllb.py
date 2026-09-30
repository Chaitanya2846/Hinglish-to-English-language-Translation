"""
finetune_nllb.py — Seq2Seq LoRA Fine-Tuning for Meta NLLB-200 (600M)
===================================================================
Architecture: facebook/nllb-200-distilled-600M (Encoder-Decoder Seq2Seq)
Dataset: Custom Web-Scraped Social Media Corpus (9,769 train / 996 val pairs)
Evaluation: BLEU, ROUGE-1/2/L, chrF++, BERTScore, METEOR

Usage (Colab T4 GPU):
  python finetune_nllb.py
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

MODEL_ID = "facebook/nllb-200-distilled-600M"
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "nllb_hinglish_lora"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "scraped_train_corpus.jsonl"
VAL_FILE = DATA_DIR / "scraped_val_corpus.jsonl"

# NLLB uses hin_Deva for Hindi, but our data is romanized Hinglish
# We treat it as hin_Deva -> eng_Latn and let fine-tuning adapt
SRC_LANG = "hin_Deva"
TGT_LANG = "eng_Latn"
MAX_LENGTH = 128
NUM_EPOCHS = 3
BATCH_SIZE = 8
GRAD_ACCUM = 2
LEARNING_RATE = 5e-4
NUM_BEAMS = 4


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
    print("  Meta NLLB-200 (600M) LoRA Fine-Tuning — Web-Scraped Corpus")
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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang=SRC_LANG, tgt_lang=TGT_LANG)

    print("\n[3/5] Loading Model...")
    t0 = time.time()
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    if device == "cuda":
        model = model.cuda()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.05, bias="none")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. Preprocess
    print("\n[4/5] Tokenizing dataset...")
    def preprocess(examples):
        inputs = examples["source"]
        targets = examples["target"]
        model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True, padding=False)
        labels = tokenizer(text_target=targets, max_length=MAX_LENGTH, truncation=True, padding=False)
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
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        generation_num_beams=NUM_BEAMS,
        generation_max_length=MAX_LENGTH,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model))

    t0 = time.time()
    trainer.train()
    print(f"  Training finished in {(time.time() - t0)/60:.1f} minutes")

    # Save
    print(f"\nSaving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 5. Evaluate with all 6 metrics
    print("\n[EVAL] Running 6-metric evaluation on validation set...")
    model.eval()
    eval_samples = val_pairs[:150]
    predictions, references, latencies = [], [], []

    forced_bos_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

    for i, item in enumerate(eval_samples):
        inputs = tokenizer(item["source"], return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=96, num_beams=NUM_BEAMS,
                forced_bos_token_id=forced_bos_id, pad_token_id=tokenizer.pad_token_id)
        latencies.append((time.time() - t0) * 1000)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        predictions.append(pred)
        references.append(item["target"])
        if i < 5:
            print(f"[{i+1}] Src: {item['source']}\n    Ref: {item['target']}\n    Pred: {pred}\n" + "-" * 50)

    metrics = compute_all_metrics(predictions, references)
    metrics["Latency_ms"] = round(sum(latencies) / len(latencies), 1)
    print(f"\n  BLEU: {metrics['BLEU']} | ROUGE-L: {metrics['ROUGE-L']} | chrF++: {metrics['chrF++']} | METEOR: {metrics['METEOR']} | BERTScore: {metrics['BERTScore_F1']} | Latency: {metrics['Latency_ms']}ms")

    with open(OUTPUT_DIR / "eval_metrics.json", "w") as f:
        json.dump({"model": "Meta NLLB-200 (600M)", "metrics": metrics}, f, indent=2)
    print("\n[+] Done! Metrics saved.")


if __name__ == "__main__":
    main()
