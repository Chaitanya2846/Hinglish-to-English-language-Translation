"""
finetune_rlm_v3.py — Enhanced QLoRA Fine-Tuning for RLM-Gemma-2B (Targeting 29+ BLEU)
======================================================================================
Improvements over v2:
  1. LoRA rank doubled: r=64, alpha=128 (4x more adapter capacity)
  2. 5 epochs instead of 3 (more training signal)
  3. Beam search = 6 with length_penalty=1.2 (better decoding)
  4. Gradient checkpointing enabled (saves VRAM)
  5. Max sequence length 192 (captures longer sentences)
  6. Full 6-metric evaluation: BLEU, ROUGE-1/2/L, chrF++, BERTScore, METEOR
  7. Unified dataset naming (Web-Scraped Social Media Corpus)

Usage (Colab T4 GPU):
  python finetune_rlm_v3.py
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

# Force UTF-8 on Windows
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
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "rlm_hinglish_lora_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "scraped_train_corpus.jsonl"
VAL_FILE = DATA_DIR / "scraped_val_corpus.jsonl"

PROMPT_TEMPLATE = "Hinglish:\n{source}\n\nEnglish:\n{target}"
INFERENCE_TEMPLATE = "Hinglish:\n{source}\n\nEnglish:\n"

# ── Enhanced Hyperparameters for 29+ BLEU ────────────────────────────────────
NUM_EPOCHS = 5
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4          # Effective batch size = 16
LEARNING_RATE = 8e-5          # Slightly lower for more stable 5-epoch training
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.01
MAX_LENGTH = 192              # Longer context window
LORA_R = 64                   # Doubled rank for higher expressivity
LORA_ALPHA = 128              # 2x alpha for stronger adaptation
LORA_DROPOUT = 0.05
NUM_BEAMS = 6                 # Higher beam search


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    data = []
    if not path.exists():
        print(f"  [WARNING] File not found: {path}")
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


def format_examples(data: List[Dict[str, str]], tokenizer, max_len: int = 192):
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


def compute_all_metrics(predictions, references):
    """Compute all 6 evaluation metrics."""
    import evaluate as hf_evaluate

    results = {}

    # 1. BLEU
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, [[r] for r in references])
        chrf = sacrebleu.corpus_chrf(predictions, [[r] for r in references], word_order=2)
        results["BLEU"] = round(bleu.score, 2)
        results["chrF++"] = round(chrf.score, 2)
    except ImportError:
        bleu_m = hf_evaluate.load("sacrebleu")
        res = bleu_m.compute(predictions=predictions, references=[[r] for r in references])
        results["BLEU"] = round(res["score"], 2)
        chrf_m = hf_evaluate.load("chrf")
        res2 = chrf_m.compute(predictions=predictions, references=[[r] for r in references], word_order=2)
        results["chrF++"] = round(res2["score"], 2)

    # 2. ROUGE
    try:
        rouge_m = hf_evaluate.load("rouge")
        r_res = rouge_m.compute(predictions=predictions, references=references)
        results["ROUGE-1"] = round(r_res["rouge1"] * 100, 2)
        results["ROUGE-2"] = round(r_res["rouge2"] * 100, 2)
        results["ROUGE-L"] = round(r_res["rougeL"] * 100, 2)
    except Exception as e:
        print(f"  [WARN] ROUGE computation failed: {e}")
        results["ROUGE-1"] = results["ROUGE-2"] = results["ROUGE-L"] = 0.0

    # 3. METEOR
    try:
        meteor_m = hf_evaluate.load("meteor")
        m_res = meteor_m.compute(predictions=predictions, references=references)
        results["METEOR"] = round(m_res["meteor"] * 100, 2)
    except Exception as e:
        print(f"  [WARN] METEOR computation failed: {e}")
        results["METEOR"] = 0.0

    # 4. BERTScore
    try:
        bert_m = hf_evaluate.load("bertscore")
        bs_res = bert_m.compute(predictions=predictions, references=references, lang="en")
        results["BERTScore_F1"] = round(sum(bs_res["f1"]) / len(bs_res["f1"]), 4)
    except Exception as e:
        print(f"  [WARN] BERTScore computation failed: {e}")
        results["BERTScore_F1"] = 0.0

    return results


def evaluate_model(model, tokenizer, eval_data, split_name, device, max_samples=150, num_beams=6):
    model.eval()
    samples = eval_data[:max_samples] if max_samples else eval_data
    predictions = []
    references = []
    latencies = []

    print(f"\n{'=' * 65}")
    print(f"  EVALUATING: {split_name} ({len(samples)} samples | Beams={num_beams})")
    print(f"{'=' * 65}")

    input_device = model.get_input_embeddings().weight.device if device == "cuda" else torch.device("cpu")

    for i, item in enumerate(samples):
        src = item["source"]
        ref = item["target"]
        prompt = INFERENCE_TEMPLATE.format(source=src)

        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(input_device) for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=96,
                num_beams=num_beams,
                length_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        latencies.append((time.time() - t0) * 1000)

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

    # Compute all 6 metrics
    metrics = compute_all_metrics(predictions, references)
    metrics["Latency_ms"] = round(sum(latencies) / len(latencies), 1)
    metrics["samples"] = len(samples)

    print(f"\n{'─' * 65}")
    print(f"  [{split_name}] RESULTS:")
    print(f"    BLEU       : {metrics['BLEU']}")
    print(f"    ROUGE-1    : {metrics.get('ROUGE-1', 0)}")
    print(f"    ROUGE-2    : {metrics.get('ROUGE-2', 0)}")
    print(f"    ROUGE-L    : {metrics.get('ROUGE-L', 0)}")
    print(f"    chrF++     : {metrics['chrF++']}")
    print(f"    METEOR     : {metrics.get('METEOR', 0)}")
    print(f"    BERTScore  : {metrics.get('BERTScore_F1', 0)}")
    print(f"    Latency    : {metrics['Latency_ms']} ms/sentence")
    print(f"{'─' * 65}")

    return metrics, predictions


def main():
    print("=" * 70)
    print("  RLM-Hinglish-Translator Fine-Tuning v3 (Targeting 29+ BLEU)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device.upper()}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU    : {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("WARNING: CUDA not detected! Training on CPU will be extremely slow.")

    # 1. Load Data
    print("\n[1/5] Loading Web-Scraped Social Media Corpus...")
    train_data = load_jsonl(TRAIN_FILE)
    val_data = load_jsonl(VAL_FILE)
    print(f"  Training Corpus   : {len(train_data):,} pairs")
    print(f"  Validation Corpus : {len(val_data):,} pairs")

    # 2. Tokenizer
    print("\n[2/5] Loading Tokenizer...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 3. Model Loading with 4-bit Quantization
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
            BASE_MODEL_NAME, torch_dtype=torch.float32,
        )

    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # Enhanced LoRA (r=64, alpha=128)
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

    # 4. Tokenize
    print("\n[4/5] Tokenizing data with target-only loss masking...")
    train_tokens = format_examples(train_data, tokenizer, max_len=MAX_LENGTH)
    val_tokens = format_examples(val_data, tokenizer, max_len=MAX_LENGTH)

    train_dataset = HinglishDataset(train_tokens)
    val_dataset = HinglishDataset(val_tokens)
    data_collator = CausalLMDataCollator(tokenizer=tokenizer)

    # 5. Trainer
    print(f"\n[5/5] Configuring Trainer for {NUM_EPOCHS} epochs...")
    import inspect
    sig = inspect.signature(TrainingArguments.__init__).parameters

    args_dict = {
        "output_dir": str(OUTPUT_DIR / "checkpoints"),
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "eval_steps": 200,
        "save_steps": 200,
        "save_total_limit": 2,
        "learning_rate": LEARNING_RATE,
        "fp16": (device == "cuda"),
        "logging_steps": 25,
        "report_to": "none",
        "optim": "adamw_torch",
        "remove_unused_columns": False,
    }

    if "eval_strategy" in sig:
        args_dict["eval_strategy"] = "steps"
    elif "evaluation_strategy" in sig:
        args_dict["evaluation_strategy"] = "steps"

    if "lr_scheduler_type" in sig:
        args_dict["lr_scheduler_type"] = "cosine"
    if "weight_decay" in sig:
        args_dict["weight_decay"] = WEIGHT_DECAY
    if "warmup_steps" in sig:
        args_dict["warmup_steps"] = 150
    elif "warmup_ratio" in sig:
        args_dict["warmup_ratio"] = WARMUP_RATIO

    training_args = TrainingArguments(**args_dict)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("\n>>> Launching training loop...")
    trainer.train()

    # Save
    print(f"\nSaving fine-tuned LoRA adapter v3 to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[+] Model adapter and tokenizer saved!")

    # 6. Full 6-Metric Evaluation
    eval_results = {}
    if val_data:
        metrics, _ = evaluate_model(
            model, tokenizer, val_data,
            split_name="Web-Scraped Social Media Corpus (Validation)",
            device=device, max_samples=200, num_beams=NUM_BEAMS,
        )
        eval_results["validation"] = metrics

    results_file = OUTPUT_DIR / "eval_metrics_v3.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n[+] Full metrics saved to: {results_file}")

    print(f"\n{'=' * 70}")
    print("  BENCHMARK COMPARISON")
    print(f"{'=' * 70}")
    print("  RCMT (LREC-COLING 2024 SOTA Base)    : ~14.00 BLEU")
    print("  PACMANtrans (ACL/ICON 2023 Base)     : ~18.66 BLEU")
    if "validation" in eval_results:
        print(f"  Our Model v3 (5 epochs, r=64, beam=6) :  {eval_results['validation'].get('BLEU', '?')} BLEU")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
