"""
finetune_rlm_v2.py — Optimized 3-Epoch QLoRA Fine-Tuning for SOTA Hinglish Translation
====================================================================================
Targets & Benchmarks:
  - Base Paper Benchmark (Srivastava & Singh, W-NUT 2020): 15.3 BLEU (PPGT)
  - SOTA Multilingual Benchmark (Agarwal et al., RANLP 2021): 29.5 BLEU (mT5)
  - Previous Run (v1): 20.8 BLEU on PHINC Val | 11.52 on YouTube Gold (greedy)

Enhancements in v2:
  1. LoRA capacity boost: r=32, alpha=64 across all 7 projection matrices (q, k, v, o, gate, up, down).
  2. Prompt loss masking: Gradients strictly optimize English target tokens (labels = -100).
  3. YouTube domain upsampling: 5x weighting on conversational pairs (extra_train.jsonl)
     so the model masters colloquial social media slang without overfitting PHINC.
  4. Cosine annealing LR: 1e-4 peak, 10% warmup, weight decay 0.01 for fast, stable 3-epoch convergence.
  5. 4-Beam search decoding: Replaces raw greedy decoding with num_beams=4 and length penalty,
     yielding higher fluency and significant BLEU gains.
  6. Dual-split benchmark evaluation: Automatically evaluates and reports metrics on both
     PHINC validation and the YouTube Gold set.

Usage:
  python finetune_rlm_v2.py
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

# Force UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Paths & Configuration ───────────────────────────────────────────────────
BASE_MODEL_NAME = "rudrashah/RLM-hinglish-translator"
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "rlm_hinglish_lora_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "phinc_train.jsonl"
EXTRA_TRAIN_FILE = DATA_DIR / "extra_train.jsonl"
VAL_FILE = DATA_DIR / "phinc_val.jsonl"
EVAL_FILE = DATA_DIR / "youtube_eval_gold.jsonl"

PROMPT_TEMPLATE = "Hinglish:\n{source}\n\nEnglish:\n{target}"
INFERENCE_TEMPLATE = "Hinglish:\n{source}\n\nEnglish:\n"

# ── Optimized Hyperparameters for High Quality in 3 Epochs ──────────────────
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4          # Effective batch size = 16
LEARNING_RATE = 1e-4          # Optimal peak learning rate for QLoRA rank 32
WARMUP_RATIO = 0.10           # 10% warmup steps
WEIGHT_DECAY = 0.01
MAX_LENGTH = 128
LORA_R = 32                   # Doubled rank for higher expressivity
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
YOUTUBE_UPSAMPLE_FACTOR = 5   # 5x oversample conversational social media pairs
NUM_BEAMS = 4                 # Beam search decoding for test evaluation


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    """Load JSONL data with 'source' and 'target' keys."""
    data = []
    if not path.exists():
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = item.get("source", "").strip()
            tgt = item.get("target", "").strip()
            if src and tgt:
                data.append({"source": src, "target": tgt})
    return data


def format_examples(data: List[Dict[str, str]], tokenizer, max_len: int = 128):
    """
    Format and tokenize examples for causal LM instruction fine-tuning.
    Masks the prompt tokens with -100 so loss is calculated ONLY on the English translation.
    """
    input_ids_list = []
    labels_list = []
    attention_masks = []

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

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_masks,
    }


class HinglishDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "attention_mask": self.attention_mask[idx],
        }


@dataclass
class CausalLMDataCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            batch_max_len = math.ceil(batch_max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for f in features:
            cur_len = len(f["input_ids"])
            pad_len = batch_max_len - cur_len

            batch_input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            batch_labels.append(f["labels"] + [-100] * pad_len)
            batch_attention_mask.append(f["attention_mask"] + [0] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }


def evaluate_split(model, tokenizer, eval_data, split_name: str, device: str, max_samples: int = 100, num_beams: int = 4):
    """Run beam-search generation on an evaluation split and compute BLEU / chrF."""
    model.eval()
    samples = eval_data[:max_samples] if max_samples else eval_data
    predictions = []
    references = []

    print("\n" + "=" * 65)
    print(f"  EVALUATING: {split_name} ({len(samples)} samples | Beams={num_beams})")
    print("=" * 65)

    input_device = model.get_input_embeddings().weight.device if device == "cuda" else torch.device("cpu")

    for i, item in enumerate(samples):
        src = item["source"]
        ref = item["target"]
        prompt = INFERENCE_TEMPLATE.format(source=src)

        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(input_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=num_beams,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "English:\n" in generated_text:
            pred = generated_text.split("English:\n")[-1].strip()
        else:
            pred = generated_text.replace(prompt, "").strip()

        predictions.append(pred)
        references.append(ref)

        if i < 5:
            print(f"[{i+1}] Source : {src}")
            print(f"    Target : {ref}")
            print(f"    Pred   : {pred}")
            print("-" * 50)

    metrics = {}
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, [[r] for r in references])
        chrf = sacrebleu.corpus_chrf(predictions, [[r] for r in references])
        metrics = {
            "samples": len(samples),
            "bleu": round(bleu.score, 2),
            "chrf": round(chrf.score, 2),
            "num_beams": num_beams,
        }
        print(f"\n[{split_name}] Scores -> BLEU: {metrics['bleu']}  |  chrF++: {metrics['chrf']}")
    except ImportError:
        print("[!] sacrebleu not installed; skipping metric calculation.")

    return metrics, predictions


def main():
    print("=" * 70)
    print("  RLM-Hinglish-Translator Fine-Tuning v2 (Targeting SOTA)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device.upper()}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU    : {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("WARNING: CUDA not detected! Running on CPU will be extremely slow.")

    # 1. Load Data with Targeted Domain Balancing
    print("\n[1/5] Loading & Balancing Datasets...")
    phinc_train = load_jsonl(TRAIN_FILE)
    extra_train = load_jsonl(EXTRA_TRAIN_FILE)
    val_data = load_jsonl(VAL_FILE)
    eval_gold = load_jsonl(EVAL_FILE)

    print(f"  PHINC Train        : {len(phinc_train):,} pairs")
    print(f"  Extra YouTube Train: {len(extra_train):,} pairs (Original)")

    # 5x oversample conversational social media data
    train_data = list(phinc_train)
    if extra_train:
        upsampled_extra = extra_train * YOUTUBE_UPSAMPLE_FACTOR
        train_data.extend(upsampled_extra)
        print(f"  Extra Upsampled 5x : {len(upsampled_extra):,} pairs added to training")

    print(f"  Total Training Set : {len(train_data):,} pairs")
    print(f"  PHINC Val Set      : {len(val_data):,} pairs")
    print(f"  YouTube Gold Eval  : {len(eval_gold):,} pairs")

    # 2. Tokenizer Setup
    print("\n[2/5] Loading Tokenizer...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 3. Model Loading with 4-bit Quantization (QLoRA)
    print("\n[3/5] Loading Model with 4-bit Quantization (NF4)...")
    t0 = time.time()

    if device == "cuda":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model = prepare_model_for_kbit_training(model)
        model.gradient_checkpointing_enable()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float32,
        )

    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # Setup Enhanced LoRA (Rank 32, Alpha 64 across all linear projections)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False

    # 4. Tokenize & Prepare Datasets
    print("\n[4/5] Tokenizing data with target-only loss masking...")
    train_tokens = format_examples(train_data, tokenizer, max_len=MAX_LENGTH)
    val_tokens = format_examples(val_data, tokenizer, max_len=MAX_LENGTH)

    train_dataset = HinglishDataset(train_tokens)
    val_dataset = HinglishDataset(val_tokens)
    data_collator = CausalLMDataCollator(tokenizer=tokenizer)

    # 5. Trainer Configuration (3 Epochs with Cosine Warmup)
    print(f"\n[5/5] Configuring Trainer for {NUM_EPOCHS} epochs with Cosine Annealing...")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        eval_strategy="steps",
        eval_steps=150,
        save_steps=150,
        save_total_limit=2,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=(device == "cuda"),
        logging_steps=25,
        report_to="none",
        optim="adamw_torch",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("\n>>> Launching training loop...")
    trainer.train()

    # Save final LoRA adapter
    print(f"\nSaving fine-tuned LoRA adapter v2 to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[+] Model adapter and tokenizer saved successfully!")

    # 6. Benchmark Evaluation against Research Papers
    eval_results = {}

    if val_data:
        val_metrics, _ = evaluate_split(
            model, tokenizer, val_data,
            split_name="PHINC Validation Set (Research Paper Benchmark)",
            device=device,
            max_samples=200,
            num_beams=NUM_BEAMS,
        )
        eval_results["phinc_val"] = val_metrics

    if eval_gold:
        yt_metrics, _ = evaluate_split(
            model, tokenizer, eval_gold,
            split_name="YouTube Gold Evaluation Set",
            device=device,
            max_samples=None,
            num_beams=NUM_BEAMS,
        )
        eval_results["youtube_gold"] = yt_metrics

    results_file = OUTPUT_DIR / "eval_metrics_v2.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n[+] Full metrics saved to: {results_file}")

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPARISON SUMMARY")
    print("=" * 70)
    print("  Paper 1 Baseline (Srivastava & Singh, 2020) : 15.30 BLEU")
    print("  Paper 2 SOTA (Agarwal et al., 2021)          : 29.50 BLEU")
    if "phinc_val" in eval_results and "bleu" in eval_results["phinc_val"]:
        print(f"  Our Model v2 (PHINC Val, Beams=4)            : {eval_results['phinc_val']['bleu']} BLEU")
    if "youtube_gold" in eval_results and "bleu" in eval_results["youtube_gold"]:
        print(f"  Our Model v2 (YouTube Gold, Beams=4)         : {eval_results['youtube_gold']['bleu']} BLEU")
    print("=" * 70)


if __name__ == "__main__":
    main()
