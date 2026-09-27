"""
finetune_rlm.py — Fine-tune rudrashah/RLM-hinglish-translator with QLoRA
========================================================================
Fine-tunes the 2B Gemma-based Hinglish translator on code-mixed social media
pairs (PHINC corpus) using 4-bit QLoRA.

Target Hardware:
    - GPU: NVIDIA RTX 3050 (4GB/6GB VRAM) or better
    - VRAM consumption: ~3.0 - 3.8 GB
    - Expected runtime: ~30-45 minutes on RTX 3050 (1-2 epochs)

Usage:
    python finetune_rlm.py
"""

import os
import sys
import json
import time
import math
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Configuration ────────────────────────────────────────────────────────────
BASE_MODEL_NAME = "rudrashah/RLM-hinglish-translator"
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "rlm_hinglish_lora"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "phinc_train.jsonl"
EXTRA_TRAIN_FILE = DATA_DIR / "extra_train.jsonl"
VAL_FILE   = DATA_DIR / "phinc_val.jsonl"
EVAL_FILE  = DATA_DIR / "youtube_eval_gold.jsonl"

# Prompt template matching the base model's exact format
PROMPT_TEMPLATE = "Hinglish:\n{source}\n\nEnglish:\n{target}"
INFERENCE_TEMPLATE = "Hinglish:\n{source}\n\nEnglish:\n"

# Training Hyperparameters optimized for 6GB RTX 3050
NUM_EPOCHS = 5
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4  # Effective batch size = 16
LEARNING_RATE = 2e-4
MAX_LENGTH = 128
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    """Load JSONL data with 'source' and 'target' keys."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "source" in item and "target" in item:
                data.append({"source": item["source"].strip(), "target": item["target"].strip()})
    return data


def format_examples(data: List[Dict[str, str]], tokenizer, max_len: int = 128):
    """
    Format and tokenize examples for causal LM training.
    Masks the prompt tokens with -100 so loss is calculated ONLY on the English translation.
    """
    input_ids_list = []
    labels_list = []
    attention_masks = []

    for item in data:
        prompt = INFERENCE_TEMPLATE.format(source=item["source"])
        full_text = PROMPT_TEMPLATE.format(source=item["source"], target=item["target"])

        # Tokenize prompt and full text
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        full_ids = tokenizer.encode(full_text, add_special_tokens=True)

        # Append EOS token if missing
        if full_ids[-1] != tokenizer.eos_token_id:
            full_ids.append(tokenizer.eos_token_id)

        # Truncate if necessary
        if len(full_ids) > max_len:
            full_ids = full_ids[:max_len]

        # Labels: mask prompt with -100 so loss is only on target
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


def run_evaluation(model, tokenizer, eval_data, device, max_samples: int = 30):
    """Run generation on eval_data and print sample predictions."""
    model.eval()
    samples = eval_data[:max_samples]
    predictions = []
    references = []

    print("\n" + "=" * 60)
    print("  EVALUATING MODEL ON YOUTUBE GOLD SAMPLES")
    print("=" * 60)

    for i, item in enumerate(samples):
        src = item["source"]
        ref = item["target"]
        prompt = INFERENCE_TEMPLATE.format(source=src)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract response after 'English:\n'
        if "English:\n" in generated_text:
            pred = generated_text.split("English:\n")[-1].strip()
        else:
            pred = generated_text.replace(prompt, "").strip()

        predictions.append(pred)
        references.append(ref)

        if i < 5:
            print(f"Sample [{i+1}]")
            print(f"  Input  : {src}")
            print(f"  Target : {ref}")
            print(f"  Pred   : {pred}")
            print("-" * 40)

    # Compute BLEU if sacrebleu is available
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, [[r] for r in references])
        chrf = sacrebleu.corpus_chrf(predictions, [[r] for r in references])
        print(f"\nFinal Eval Metrics on YouTube Gold ({len(samples)} samples):")
        print(f"  BLEU: {bleu.score:.2f}")
        print(f"  chrF: {chrf.score:.2f}")
        return {"bleu": round(bleu.score, 2), "chrf": round(chrf.score, 2)}
    except ImportError:
        print("sacrebleu not installed; skipping metric calculation.")
        return {}


def main():
    print("=" * 60)
    print("  RLM-Hinglish-Translator Fine-Tuning (QLoRA)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device.upper()}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU    : {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("WARNING: CUDA not detected! Running on CPU will be extremely slow.")

    # 1. Load Data
    print("\n[1/5] Loading datasets...")
    train_data = load_jsonl(TRAIN_FILE)
    if EXTRA_TRAIN_FILE.exists():
        extra_data = load_jsonl(EXTRA_TRAIN_FILE)
        print(f"  Extra : {len(extra_data):,} pairs ({EXTRA_TRAIN_FILE.name})")
        train_data.extend(extra_data)
    val_data   = load_jsonl(VAL_FILE)
    eval_data  = load_jsonl(EVAL_FILE) if EVAL_FILE.exists() else []

    print(f"  Train : {len(train_data):,} pairs (total)")
    print(f"  Val   : {len(val_data):,} pairs ({VAL_FILE.name})")
    print(f"  Eval  : {len(eval_data):,} pairs ({EVAL_FILE.name})")

    # 2. Tokenizer
    print("\n[2/5] Loading Tokenizer...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 3. Model Loading with 4-bit Quantization (QLoRA)
    print("\n[3/5] Loading Model with 4-bit Quantization...")
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
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float32,
        )

    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # Setup LoRA
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

    # 4. Tokenize datasets
    print("\n[4/5] Preprocessing and Tokenizing data...")
    train_tokens = format_examples(train_data, tokenizer, max_len=MAX_LENGTH)
    val_tokens   = format_examples(val_data, tokenizer, max_len=MAX_LENGTH)

    train_dataset = HinglishDataset(train_tokens)
    val_dataset   = HinglishDataset(val_tokens)
    data_collator = CausalLMDataCollator(tokenizer=tokenizer)

    # 5. Trainer Configuration
    print("\n[5/5] Starting Training...")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        learning_rate=LEARNING_RATE,
        fp16=(device == "cuda"),
        logging_steps=50,
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

    trainer.train()

    # Save final LoRA adapter
    print(f"\nSaving fine-tuned LoRA adapter to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model saved successfully!")

    # 6. Evaluation
    if eval_data:
        metrics = run_evaluation(model, tokenizer, eval_data, device)
        results_file = OUTPUT_DIR / "eval_metrics.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {results_file}")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print(f"  Adapter checkpoint: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
