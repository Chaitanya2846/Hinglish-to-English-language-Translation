"""
finetune_llama3.py — 4-bit QLoRA Fine-Tuning for Meta-Llama-3-8B
================================================================
Task: Romanized Hinglish-to-English Neural Machine Translation
Architecture: Meta-Llama-3-8B (8 Billion Parameters) via QLoRA
Target Hardware: Google Colab Free T4 GPU (16GB VRAM) or A100 / L4
Dataset: Web-Scraped Social Media Corpus (9,769 Train / 996 Val pairs)

Key Technical Features:
  1. 4-bit NormalFloat (NF4) quantization + double quantization via bitsandbytes
  2. Full linear module LoRA adaptation: q, k, v, o, gate, up, down projections
  3. Strict loss masking on prompt tokens (loss computed only on English target)
  4. Memory-optimized for T4: Gradient Checkpointing + Paged AdamW 8-bit
  5. Built-in evaluation computing BLEU, chrF++, ROUGE, METEOR, BERTScore, and latency

Usage (Google Colab GPU):
  export HF_TOKEN="your_hf_token_here"
  python finetune/finetune_llama3.py

Or with arguments:
  python finetune/finetune_llama3.py --hf_token "hf_xxx" --epochs 3 --lr 2e-4
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import torch

# Force UTF-8 encoding on Windows / Linux terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Paths & Defaults ────────────────────────────────────────────────────────
DEFAULT_BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").exists() else SCRIPT_DIR
DATA_DIR = SCRIPT_DIR / "data" if (SCRIPT_DIR / "data").exists() else PROJECT_ROOT / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "llama3_hinglish_lora"

def find_data_file(filename: str) -> Path:
    """Robust multi-path resolution for data files across different working directories."""
    candidates = [
        SCRIPT_DIR / "data" / filename,
        PROJECT_ROOT / "finetune" / "data" / filename,
        PROJECT_ROOT / "data" / filename,
        Path("finetune") / "data" / filename,
        Path("data") / filename,
        Path("/content/NLP/finetune/data") / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    return SCRIPT_DIR / "data" / filename


DEFAULT_TRAIN_FILE = find_data_file("scraped_train_corpus.jsonl")
DEFAULT_VAL_FILE = find_data_file("scraped_val_corpus.jsonl")

PROMPT_TEMPLATE = "Translate the following Romanized Hinglish text to fluent English.\nHinglish: {source}\nEnglish: {target}"
INFERENCE_TEMPLATE = "Translate the following Romanized Hinglish text to fluent English.\nHinglish: {source}\nEnglish:"


def get_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    """Retrieve Hugging Face token from CLI argument, environment, or Colab secrets."""
    if cli_token:
        return cli_token

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token

    try:
        from google.colab import userdata
        colab_token = userdata.get("HF_TOKEN")
        if colab_token:
            return colab_token
    except Exception:
        pass

    try:
        from huggingface_hub import HfFolder
        saved_token = HfFolder.get_token()
        if saved_token:
            return saved_token
    except Exception:
        pass

    return None


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    """Loads parallel sentence pairs from a JSONL file."""
    data = []
    if not path.exists():
        print(f"  [!] Warning: File not found: {path}")
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                src = item.get("source", "").strip()
                tgt = item.get("target", "").strip()
                if src and tgt:
                    data.append({"source": src, "target": tgt})
            except json.JSONDecodeError:
                continue
    return data


def format_examples(data: List[Dict[str, str]], tokenizer, max_len: int = 192) -> Dict[str, list]:
    """
    Formats training examples with strict prompt loss masking:
    Labels for all prompt tokens are set to -100 so cross-entropy loss is computed
    strictly on the English translation tokens.
    """
    input_ids_list = []
    labels_list = []
    attention_masks = []

    eos_id = tokenizer.eos_token_id

    for item in data:
        prompt = INFERENCE_TEMPLATE.format(source=item["source"])
        full_text = PROMPT_TEMPLATE.format(source=item["source"], target=item["target"])

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        full_ids = tokenizer.encode(full_text, add_special_tokens=True)

        if len(full_ids) == 0 or full_ids[-1] != eos_id:
            full_ids.append(eos_id)

        if len(full_ids) > max_len:
            full_ids = full_ids[:max_len]

        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(labels))

        # Mask prompt tokens from loss calculation
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
    def __init__(self, data_dict: Dict[str, list]):
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
        batch_max = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            batch_max = math.ceil(batch_max / self.pad_to_multiple_of) * self.pad_to_multiple_of

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for f in features:
            pad_len = batch_max - len(f["input_ids"])
            batch_input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            batch_labels.append(f["labels"] + [-100] * pad_len)
            batch_attention_mask.append(f["attention_mask"] + [0] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }


def compute_evaluation_metrics(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    """Computes BLEU, chrF++, ROUGE, METEOR, and BERTScore on test predictions."""
    results = {}
    try:
        import sacrebleu
        results["BLEU"] = round(sacrebleu.corpus_bleu(predictions, [references], smooth_method="exp").score, 2)
        results["chrF++"] = round(sacrebleu.corpus_chrf(predictions, [references], word_order=2).score, 2)
    except Exception as e:
        print(f"  [WARN] sacrebleu error: {e}")
        results["BLEU"] = results["chrF++"] = 0.0

    try:
        import evaluate
        rouge = evaluate.load("rouge")
        r = rouge.compute(predictions=predictions, references=references)
        results["ROUGE-1"] = round(r["rouge1"] * 100, 2)
        results["ROUGE-2"] = round(r["rouge2"] * 100, 2)
        results["ROUGE-L"] = round(r["rougeL"] * 100, 2)
    except Exception as e:
        print(f"  [WARN] ROUGE error: {e}")
        results["ROUGE-1"] = results["ROUGE-2"] = results["ROUGE-L"] = 0.0

    try:
        import nltk
        for pkg in ["wordnet", "punkt", "punkt_tab", "omw-1.4"]:
            nltk.download(pkg, quiet=True)
        import evaluate
        meteor = evaluate.load("meteor")
        m = meteor.compute(predictions=predictions, references=references)
        results["METEOR"] = round(m["meteor"] * 100, 2)
    except Exception as e:
        print(f"  [WARN] METEOR error: {e}")
        results["METEOR"] = 0.0

    try:
        import evaluate
        bertscore = evaluate.load("bertscore")
        bs = bertscore.compute(predictions=predictions, references=references, lang="en")
        results["BERTScore_F1"] = round(sum(bs["f1"]) / len(bs["f1"]), 4)
    except Exception as e:
        print(f"  [WARN] BERTScore error: {e}")
        results["BERTScore_F1"] = 0.0

    return results


def run_post_training_evaluation(model, tokenizer, val_pairs: List[Dict[str, str]], device: str, num_samples: int = 150) -> Dict[str, Any]:
    """Generates translations on a subset of the validation set and logs empirical metrics."""
    test_set = val_pairs[:num_samples] if num_samples > 0 else val_pairs
    print(f"\n[Evaluation] Generating predictions for {len(test_set)} samples with Beam Search = 4...")

    model.eval()
    model.config.use_cache = True
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None

    input_device = next(model.parameters()).device
    tokenizer.padding_side = "left"

    predictions = []
    references = []
    latencies = []

    for i, item in enumerate(test_set):
        prompt = INFERENCE_TEMPLATE.format(source=item["source"])
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=4,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed_ms = (time.time() - t0) * 1000
        latencies.append(elapsed_ms)

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        # Clean potential residual prefixes
        if "English:" in pred_text:
            pred_text = pred_text.split("English:")[-1].strip()
        if not pred_text:
            full_decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            pred_text = full_decoded.split("English:")[-1].strip() if "English:" in full_decoded else full_decoded

        predictions.append(pred_text)
        references.append(item["target"])

        if i < 3:
            print(f"\n  Sample #{i+1}:")
            print(f"    Hinglish (Src) : {item['source']}")
            print(f"    Llama-3  (Pred): {pred_text}")
            print(f"    Gold     (Ref) : {item['target']}")

    tokenizer.padding_side = "right"
    metrics = compute_evaluation_metrics(predictions, references)
    metrics["Latency_ms"] = round(sum(latencies) / len(latencies), 1)

    print("\n" + "=" * 60)
    print("  🏆 Meta-Llama-3-8B Fine-Tuned Evaluation Summary")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"    {k:<16} : {v}")
    print("=" * 60)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="4-bit QLoRA Fine-Tuning for Meta-Llama-3-8B on Hinglish Translation")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL,
                        help="Base HuggingFace model repo (default: meta-llama/Meta-Llama-3-8B)")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace access token for gated Llama-3 access")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size (default: 2 for T4)")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps (default: 8, effective batch=16)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank dimension (default: 32)")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha scaling factor (default: 64)")
    parser.add_argument("--max_length", type=int, default=192, help="Max token sequence length (default: 192)")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Output directory for saved adapter")
    parser.add_argument("--combine_phinc", action="store_true", help="Also include PHINC dataset for augmented training volume")
    parser.add_argument("--eval_samples", type=int, default=150, help="Number of validation samples to evaluate post-training (default: 150)")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  🚀 Meta-Llama-3-8B (8B Parameters) Hinglish-to-English QLoRA Fine-Tuning")
    print("=" * 80)

    # 1. HuggingFace Authentication
    hf_token = get_hf_token(args.hf_token)
    if not hf_token:
        print("[!] ERROR: Meta-Llama-3-8B is a gated model requiring Hugging Face authorization.")
        print("    Please obtain an access token from: https://huggingface.co/settings/tokens")
        print("    and request access at: https://huggingface.co/meta-llama/Meta-Llama-3-8B")
        print("\n    Pass it via: python finetune/finetune_llama3.py --hf_token 'hf_xxx'")
        print("    or export HF_TOKEN='hf_xxx'")
        sys.exit(1)

    # 2. Hardware Verification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[System Info]")
    print(f"  Device           : {device.upper()}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU Hardware     : {gpu_name} ({vram_gb:.1f} GB VRAM)")
        if vram_gb < 12.0:
            print("  ⚠️  Notice: Low VRAM detected. Running with batch_size=2 and Paged AdamW 8-bit to fit within memory.")
    else:
        print("  ⚠️  Warning: CUDA not detected! 8B fine-tuning is impractical on CPU. Please run on Google Colab T4 GPU.")
        sys.exit(1)

    # 3. Load Datasets
    print(f"\n[1/5] Loading Training & Validation Corpora...")
    train_pairs = load_jsonl(DEFAULT_TRAIN_FILE)
    val_pairs = load_jsonl(DEFAULT_VAL_FILE)

    if args.combine_phinc:
        phinc_file = find_data_file("phinc_train.jsonl")
        phinc_pairs = load_jsonl(phinc_file)
        train_pairs.extend(phinc_pairs)
        print(f"  [+] Augmented with PHINC corpus (+{len(phinc_pairs)} pairs)")

    print(f"  [+] Loaded {len(train_pairs):,} Training Pairs")
    print(f"  [+] Loaded {len(val_pairs):,} Validation Pairs")

    if not train_pairs:
        print(f"[!] Error: Training corpus is empty. Verify {DEFAULT_TRAIN_FILE}")
        sys.exit(1)

    # 4. Tokenizer Setup
    print(f"\n[2/5] Initializing Llama-3 Tokenizer from {args.base_model}...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # 5. Tokenize Datasets with Prompt Masking
    print("      Tokenizing & formatting with prompt loss masking...")
    train_encoded = format_examples(train_pairs, tokenizer, max_len=args.max_length)
    train_dataset = HinglishDataset(train_encoded)

    # 6. Load Quantized Base Model
    print(f"\n[3/5] Loading {args.base_model} in 4-bit NormalFloat (NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=hf_token,
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False  # CRITICAL: must be False during training with gradient checkpointing

    # 7. Configure LoRA for all Linear Layers
    print(f"\n[4/5] Injecting LoRA Adapter (r={args.lora_r}, alpha={args.lora_alpha})...")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    print("\n  Trainable Parameters Summary:")
    model.print_trainable_parameters()

    # 8. Training Setup
    collator = CausalLMDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    import inspect
    sig = inspect.signature(TrainingArguments.__init__).parameters

    training_kwargs = {
        "output_dir": str(output_path / "checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "fp16": True,
        "report_to": "none",
        "remove_unused_columns": False,
    }

    if "optim" in sig:
        training_kwargs["optim"] = "paged_adamw_8bit"
    if "lr_scheduler_type" in sig:
        training_kwargs["lr_scheduler_type"] = "cosine"
    if "warmup_ratio" in sig:
        training_kwargs["warmup_ratio"] = 0.05
    elif "warmup_steps" in sig:
        training_kwargs["warmup_steps"] = 50
    if "weight_decay" in sig:
        training_kwargs["weight_decay"] = 0.01
    if "gradient_checkpointing" in sig:
        training_kwargs["gradient_checkpointing"] = True

    valid_args = {k: v for k, v in training_kwargs.items() if k in sig}
    training_args = TrainingArguments(**valid_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    # 9. Train Model
    print(f"\n[5/5] Commencing QLoRA Fine-Tuning for {args.epochs} Epochs...")
    t_start = time.time()
    trainer.train()
    training_time_min = (time.time() - t_start) / 60
    print(f"\n  [+] Training successfully completed in {training_time_min:.1f} minutes!")

    # 10. Save Adapter Weights & Tokenizer
    print(f"\n  Saving fine-tuned LoRA adapter to: {output_path}...")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print("  [+] Model weights and tokenizer configuration saved successfully.")

    # 11. Run Multi-Metric Benchmark on Validation Set
    if val_pairs:
        metrics = run_post_training_evaluation(
            model=model,
            tokenizer=tokenizer,
            val_pairs=val_pairs,
            device=device,
            num_samples=args.eval_samples,
        )
        metrics["training_time_minutes"] = round(training_time_min, 1)
        metrics["base_model"] = args.base_model
        metrics["lora_r"] = args.lora_r

        metrics_file = output_path / "eval_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"  [+] Evaluation report written to: {metrics_file}")

    print("\n" + "=" * 80)
    print("  🎉 Meta-Llama-3-8B Fine-Tuning Pipeline Complete!")
    print(f"  Adapter Location: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
