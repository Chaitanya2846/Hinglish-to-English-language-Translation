"""
finetune_sarvam.py — 4-bit QLoRA Fine-Tuning for Sarvam-1 (2B)
=============================================================
Architecture: Sarvam-1 (India's Flagship 2B Foundation Model by Sarvam AI)
Dataset: Custom Web-Scraped Social Media Corpus (9,769 train / 996 val pairs)
Evaluation: BLEU, ROUGE-1/2/L, chrF++, BERTScore, METEOR

Usage (Colab T4 GPU):
  python finetune_sarvam.py
"""

import os
import sys
import json
import time
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any

import torch

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

PROMPT_TEMPLATE = "Translate Hinglish to English.\nHinglish: {source}\nEnglish: {target}"
INFERENCE_TEMPLATE = "Translate Hinglish to English.\nHinglish: {source}\nEnglish:"

NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
MAX_LENGTH = 192
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
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


def format_examples(data, tokenizer, max_len=192):
    input_ids_list, labels_list, attention_masks = [], [], []
    for item in data:
        prompt = INFERENCE_TEMPLATE.format(source=item["source"])
        full_text = PROMPT_TEMPLATE.format(source=item["source"], target=item["target"])
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        full_ids = tokenizer.encode(full_text, add_special_tokens=True)
        if len(full_ids) == 0 or full_ids[-1] != tokenizer.eos_token_id:
            full_ids.append(tokenizer.eos_token_id)
        if len(full_ids) > max_len:
            full_ids = full_ids[:max_len]
        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100
        input_ids_list.append(full_ids)
        labels_list.append(labels)
        attention_masks.append([1] * len(full_ids))
    return {"input_ids": input_ids_list, "labels": labels_list, "attention_mask": attention_masks}


class HinglishDataset(torch.utils.data.Dataset):
    def __init__(self, d):
        self.input_ids = d["input_ids"]
        self.labels = d["labels"]
        self.attention_mask = d["attention_mask"]
    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx], "attention_mask": self.attention_mask[idx]}


@dataclass
class CausalLMDataCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8
    def __call__(self, features):
        batch_max = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            batch_max = math.ceil(batch_max / self.pad_to_multiple_of) * self.pad_to_multiple_of
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        b_ids, b_lab, b_att = [], [], []
        for f in features:
            p = batch_max - len(f["input_ids"])
            b_ids.append(f["input_ids"] + [pad_id] * p)
            b_lab.append(f["labels"] + [-100] * p)
            b_att.append(f["attention_mask"] + [0] * p)
        return {"input_ids": torch.tensor(b_ids), "labels": torch.tensor(b_lab), "attention_mask": torch.tensor(b_att)}


def compute_all_metrics(predictions, references):
    import evaluate as hf_evaluate
    results = {}
    try:
        import sacrebleu
        results["BLEU"] = round(sacrebleu.corpus_bleu(predictions, [references], smooth_method="exp").score, 2)
        results["chrF++"] = round(sacrebleu.corpus_chrf(predictions, [references], word_order=2).score, 2)
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


def evaluate_model(model, tokenizer, eval_data, split_name, device, max_samples=150, num_beams=4):
    model.eval()
    samples = eval_data[:max_samples] if max_samples else eval_data
    predictions, references, latencies = [], [], []
    print(f"\n{'=' * 65}")
    print(f"  EVALUATING: {split_name} ({len(samples)} samples | Beams={num_beams})")
    print(f"{'=' * 65}")
    input_device = model.get_input_embeddings().weight.device if device == "cuda" else torch.device("cpu")
    for i, item in enumerate(samples):
        prompt = INFERENCE_TEMPLATE.format(source=item["source"])
        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(input_device) for k, v in inputs.items()}
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=96, num_beams=num_beams, length_penalty=1.0, no_repeat_ngram_size=3, early_stopping=True, pad_token_id=tokenizer.eos_token_id)
        latencies.append((time.time() - t0) * 1000)
        gen = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred = gen.split("English:")[-1].strip() if "English:" in gen else gen.replace(prompt, "").strip()
        predictions.append(pred)
        references.append(item["target"])
        if i < 5:
            print(f"[{i+1}] Src: {item['source']}\n    Ref: {item['target']}\n    Pred: {pred}\n" + "-" * 50)

    metrics = compute_all_metrics(predictions, references)
    metrics["Latency_ms"] = round(sum(latencies) / len(latencies), 1)
    print(f"\n  BLEU: {metrics['BLEU']} | ROUGE-L: {metrics['ROUGE-L']} | chrF++: {metrics['chrF++']} | METEOR: {metrics['METEOR']} | BERTScore: {metrics['BERTScore_F1']} | Latency: {metrics['Latency_ms']}ms")
    return metrics, predictions


def main():
    print("=" * 70)
    print("  Sarvam-1 (2B) QLoRA Fine-Tuning — Web-Scraped Social Corpus")
    print("=" * 70)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB)")

    print("\n[1/5] Loading Web-Scraped Corpus...")
    train_data = load_jsonl(TRAIN_FILE)
    val_data = load_jsonl(VAL_FILE)
    print(f"  Train: {len(train_data):,} | Val: {len(val_data):,}")

    print(f"\n[2/5] Loading Tokenizer & Model ({MODEL_ID})...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import inspect

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("\n[3/5] Loading 4-bit Quantized Model...")
    t0 = time.time()
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.config.use_cache = False
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print("\n[4/5] Tokenizing with prompt loss masking...")
    train_tokens = format_examples(train_data, tokenizer, MAX_LENGTH)
    val_tokens = format_examples(val_data, tokenizer, MAX_LENGTH)
    train_dataset = HinglishDataset(train_tokens)
    val_dataset = HinglishDataset(val_tokens)

    sig = inspect.signature(TrainingArguments.__init__).parameters
    args_dict = {"output_dir": str(OUTPUT_DIR / "checkpoints"), "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE, "gradient_accumulation_steps": GRAD_ACCUM,
        "learning_rate": LEARNING_RATE, "logging_steps": 25, "save_strategy": "epoch",
        "fp16": not torch.cuda.is_bf16_supported(), "bf16": torch.cuda.is_bf16_supported(),
        "report_to": "none", "optim": "adamw_torch", "remove_unused_columns": False}
    if "eval_strategy" in sig: args_dict["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in sig: args_dict["evaluation_strategy"] = "epoch"
    if "lr_scheduler_type" in sig: args_dict["lr_scheduler_type"] = "cosine"
    if "weight_decay" in sig: args_dict["weight_decay"] = 0.01
    if "warmup_ratio" in sig: args_dict["warmup_ratio"] = 0.10

    checkpoint_dir = OUTPUT_DIR / "checkpoints"
    resume_checkpoint = None
    if checkpoint_dir.exists():
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0
        )
        if checkpoints:
            resume_checkpoint = str(checkpoints[-1])
            print(f"  [+] Resuming training from checkpoint: {resume_checkpoint}")

    print(f"\n[5/5] Training for {NUM_EPOCHS} epochs...")
    trainer = Trainer(model=model, args=TrainingArguments(**args_dict),
        train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=CausalLMDataCollator(tokenizer=tokenizer))
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    print(f"\nSaving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Evaluate
    metrics, _ = evaluate_model(model, tokenizer, val_data, "Sarvam-1 Validation", device, 150, NUM_BEAMS)
    with open(OUTPUT_DIR / "eval_metrics.json", "w") as f:
        json.dump({"model": "Sarvam-1 (2B)", "metrics": metrics}, f, indent=2)
    print("\n[+] Done! Metrics saved.")


if __name__ == "__main__":
    main()
