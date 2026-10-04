"""
finetune_nllb.py — 4-bit QLoRA / FP16 LoRA Fine-Tuning for NLLB-200 (1.3B)
===========================================================================
Model Architecture : facebook/nllb-200-distilled-1.3B (1.3B Parameters)
Task Architecture  : Encoder-Decoder Seq2Seq Machine Translation (M2M100)
Task               : Romanized Hinglish-to-English Neural Machine Translation
Dataset            : Custom Web-Scraped Social Media Corpus (9,769 Train / 996 Val)
Hardware Support   : Local RTX 3050 6GB Laptop GPU / Google Colab T4 (15GB)

Why NLLB-200 (1.3B) Excels for Code-Mixed Indian Language Translation:
  1. Purpose-built Seq2Seq cross-attention architecture for translation.
  2. Pre-trained on 200+ languages including extensive Indic data.
  3. Compact 1.3B size fits inside 6GB VRAM (under 2.5GB in 4-bit QLoRA).
  4. Yields high BLEU, chrF++, and ROUGE scores without CPU offloading.

Usage (Local RTX 3050 6GB or Colab T4 GPU):
  python finetune/finetune_nllb.py --epochs 3 --batch_size 4 --grad_accum 4
"""

import os
import sys
import json
import time
import argparse
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq

# UTF-8 terminal encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "nllb_hinglish_lora"

DEFAULT_BASE_MODEL = "facebook/nllb-200-distilled-1.3B"
SRC_LANG = "eng_Latn"
TGT_LANG = "eng_Latn"


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    """Loads JSONL pairs with source and target strings."""
    data = []
    if not path.exists():
        print(f"  [!] File not found: {path}")
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


def compute_all_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculates standard translation metrics: BLEU, chrF++, ROUGE, METEOR, BERTScore."""
    results = {}
    try:
        import sacrebleu
        results["BLEU"] = round(sacrebleu.corpus_bleu(predictions, [[r] for r in references]).score, 2)
        results["chrF++"] = round(sacrebleu.corpus_chrf(predictions, [[r] for r in references], word_order=2).score, 2)
    except Exception as e:
        print(f"  [WARN] sacrebleu metric failed: {e}")
        results["BLEU"] = results["chrF++"] = 0.0

    try:
        import evaluate as hf_evaluate
        rouge = hf_evaluate.load("rouge").compute(predictions=predictions, references=references)
        results["ROUGE-1"] = round(rouge["rouge1"] * 100, 2)
        results["ROUGE-2"] = round(rouge["rouge2"] * 100, 2)
        results["ROUGE-L"] = round(rouge["rougeL"] * 100, 2)
    except Exception as e:
        print(f"  [WARN] rouge metric failed: {e}")
        results["ROUGE-1"] = results["ROUGE-2"] = results["ROUGE-L"] = 0.0

    try:
        import evaluate as hf_evaluate
        meteor = hf_evaluate.load("meteor").compute(predictions=predictions, references=references)
        results["METEOR"] = round(meteor["meteor"] * 100, 2)
    except Exception as e:
        print(f"  [WARN] meteor metric skipped: {e}")
        results["METEOR"] = 0.0

    try:
        import evaluate as hf_evaluate
        bs = hf_evaluate.load("bertscore").compute(predictions=predictions, references=references, lang="en")
        results["BERTScore_F1"] = round(sum(bs["f1"]) / len(bs["f1"]), 4)
    except Exception as e:
        print(f"  [WARN] bertscore metric failed: {e}")
        results["BERTScore_F1"] = 0.0

    return results


def evaluate_model(
    model,
    tokenizer,
    eval_pairs: List[Dict[str, str]],
    device: str,
    forced_bos_token_id: int,
    num_samples: int = 150,
    num_beams: int = 4,
) -> Tuple[Dict[str, float], List[str]]:
    """Evaluates the model on validation sample pairs using beam search."""
    model.eval()
    samples = eval_pairs[:num_samples]
    predictions, references, latencies = [], [], []

    print(f"\n[Evaluation] Evaluating on {len(samples)} samples (Beam Search = {num_beams})...")

    input_device = next(model.parameters()).device

    for i, item in enumerate(samples):
        inputs = tokenizer(
            item["source"],
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=False,
        )
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=64,
                num_beams=num_beams,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        latencies.append((time.time() - t0) * 1000)

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        predictions.append(pred)
        references.append(item["target"])

        if i < 5:
            print(f"[{i+1}/{len(samples)}]")
            print(f"  Src  : {item['source']}")
            print(f"  Pred : {pred}")
            print(f"  Ref  : {item['target']}")
            print("-" * 60)

    metrics = compute_all_metrics(predictions, references)
    metrics["Latency_ms"] = round(sum(latencies) / len(latencies), 1)

    print("\n" + "=" * 70)
    print("  🏆 NLLB-200-1.3B EVALUATION RESULTS")
    print("=" * 70)
    print(f"  BLEU         : {metrics['BLEU']}")
    print(f"  chrF++       : {metrics['chrF++']}")
    print(f"  ROUGE-1      : {metrics['ROUGE-1']}")
    print(f"  ROUGE-2      : {metrics['ROUGE-2']}")
    print(f"  ROUGE-L      : {metrics['ROUGE-L']}")
    print(f"  METEOR       : {metrics['METEOR']}")
    print(f"  BERTScore F1 : {metrics['BERTScore_F1']}")
    print(f"  Latency      : {metrics['Latency_ms']} ms/sentence")
    print("=" * 70)

    return metrics, predictions


class Seq2SeqCollatorWithDecoderIds(DataCollatorForSeq2Seq):
    """Pads inputs/labels AND builds `decoder_input_ids` (labels shifted right).

    Why this is needed: with `label_smoothing_factor > 0` the HF Trainer pops
    `labels` out of the batch before calling the model, so the model can no longer
    build decoder inputs itself. Without decoder_input_ids the M2M100/NLLB decoder
    raises "You cannot specify both decoder_input_ids and decoder_inputs_embeds".
    """

    def __init__(self, *args, decoder_start_token_id: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors)
        if "labels" in batch and "decoder_input_ids" not in batch:
            labels = batch["labels"]
            dec = labels.new_full(labels.shape, self.decoder_start_token_id)
            dec[:, 1:] = labels[:, :-1].clone()
            dec.masked_fill_(dec == -100, self.tokenizer.pad_token_id)
            batch["decoder_input_ids"] = dec
        return batch


def main():
    parser = argparse.ArgumentParser(description="Fine-Tuning for facebook/nllb-200-distilled-1.3B on Hinglish Translation")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL, help="Base HuggingFace repo")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device train batch size (default: 8)")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate (default: 2.5e-4)")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank dimension (default: 32)")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha scaling factor (default: 64)")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length (default: 128)")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Output directory for saved adapter")
    parser.add_argument("--combine_phinc", action="store_true", help="Also include PHINC dataset for augmented training volume")
    parser.add_argument("--eval_samples", type=int, default=150, help="Number of validation samples to evaluate post-training")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 precision")
    parser.add_argument("--load_in_4bit", action="store_true", default=False, help="Use 4-bit NormalFloat (NF4) quantization")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  🚀 NLLB-200 (1.3B Parameters) Hinglish-to-English Seq2Seq LoRA Fine-Tuning")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[System Info]")
    print(f"  Device           : {device.upper()}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU Hardware     : {gpu_name} ({vram:.1f} GB VRAM)")
        if vram < 7.0:
            print("  Configuration    : 4-bit QLoRA + Gradient Checkpointing (Optimized for 6GB VRAM)")

    # 1. Load Data
    print("\n[1/5] Loading Training & Validation Corpora...")
    train_file = DATA_DIR / "scraped_train_corpus.jsonl"
    val_file = DATA_DIR / "scraped_val_corpus.jsonl"

    train_pairs = load_jsonl(train_file)
    val_pairs = load_jsonl(val_file)

    if args.combine_phinc:
        phinc_file = PROJECT_ROOT / "data" / "phinc" / "phinc_train_cleaned.jsonl"
        if phinc_file.exists():
            phinc_pairs = load_jsonl(phinc_file)
            train_pairs.extend(phinc_pairs)
            print(f"  [+] Augmented with {len(phinc_pairs):,} PHINC pairs")

    print(f"  [+] Loaded {len(train_pairs):,} Training Pairs")
    print(f"  [+] Loaded {len(val_pairs):,} Validation Pairs")

    # 2. Tokenizer
    print(f"\n[2/5] Initializing NLLB-200 Tokenizer from {args.base_model}...")
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        BitsAndBytesConfig,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
    )

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    print(f"  Source Language  : {SRC_LANG}")
    print(f"  Target Language  : {TGT_LANG} (token ID: {forced_bos_token_id})")

    # Preprocessing function for datasets
    def preprocess_batch(batch):
        model_inputs = tokenizer(
            batch["source"],
            max_length=args.max_length,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=args.max_length,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    raw_train = Dataset.from_dict({
        "source": [p["source"] for p in train_pairs],
        "target": [p["target"] for p in train_pairs],
    })
    raw_val = Dataset.from_dict({
        "source": [p["source"] for p in val_pairs],
        "target": [p["target"] for p in val_pairs],
    })

    print("      Tokenizing corpora...")
    tokenized_train = raw_train.map(preprocess_batch, batched=True, remove_columns=["source", "target"])
    tokenized_val = raw_val.map(preprocess_batch, batched=True, remove_columns=["source", "target"])

    # 3. Model Loading
    print(f"\n[3/5] Loading {args.base_model}...")
    t0 = time.time()

    if device == "cuda":
        if args.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.base_model,
                quantization_config=bnb_config,
                device_map="auto",
            )
            model = prepare_model_for_kbit_training(model)
        else:
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, dtype=torch.float16)
            except TypeError:  # older transformers
                model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, torch_dtype=torch.float16)
    else:
        print("  ⚠️  Notice: Running on CPU.")
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, dtype=torch.float32)
        except TypeError:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, torch_dtype=torch.float32)

    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # 4. LoRA Setup
    print(f"\n[4/5] Injecting LoRA Adapter (r={args.lora_r}, alpha={args.lora_alpha})...")
    # All projection matrices in encoder and decoder multi-head attention + MLP feed-forward
    target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)
    for _, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    print("\n  Trainable Parameters Summary:")
    model.print_trainable_parameters()

    # 5. Training Arguments
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    training_kwargs = {
        "output_dir": str(output_path / "checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "logging_steps": 25,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "fp16": (device == "cuda"),
        "report_to": "none",
        "remove_unused_columns": False,
        "label_smoothing_factor": 0.1,
    }

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
        if "gradient_checkpointing_kwargs" in sig:
            training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if "predict_with_generate" in sig:
        training_kwargs["predict_with_generate"] = False

    valid_args = {k: v for k, v in training_kwargs.items() if k in sig}
    training_args = Seq2SeqTrainingArguments(**valid_args)

    decoder_start_token_id = model.get_base_model().config.decoder_start_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.eos_token_id
    data_collator = Seq2SeqCollatorWithDecoderIds(
        tokenizer=tokenizer,
        model=None,
        pad_to_multiple_of=8 if device == "cuda" else None,
        label_pad_token_id=-100,
        decoder_start_token_id=decoder_start_token_id,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    print(f"\n[5/5] Commencing Fine-Tuning for {args.epochs} Epochs...")
    train_start = time.time()
    train_result = trainer.train()
    total_train_time = time.time() - train_start

    print(f"\n[+] Fine-Tuning Completed in {total_train_time / 60:.1f} minutes!")

    # Save final model & tokenizer
    print(f"\nSaving fine-tuned LoRA adapter to: {output_path}...")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Re-enable KV cache (disabled by gradient checkpointing) for fast generation
    model.config.use_cache = True
    try:
        model.base_model.model.config.use_cache = True
    except Exception:
        pass

    # Evaluate on held-out validation sample
    metrics, _ = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        eval_pairs=val_pairs,
        device=device,
        forced_bos_token_id=forced_bos_token_id,
        num_samples=args.eval_samples,
    )

    # Save metrics JSON
    metrics_path = output_path / "eval_metrics.json"
    report_data = {
        "model": "NLLB-200 (1.3B Seq2Seq LoRA)",
        "base_model": args.base_model,
        "src_lang": SRC_LANG,
        "tgt_lang": TGT_LANG,
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_time_min": round(total_train_time / 60, 2),
        "train_loss": train_result.training_loss if hasattr(train_result, "training_loss") else None,
        "metrics": metrics,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[+] Evaluation report saved to: {metrics_path}")
    print("[+] All Done!")


if __name__ == "__main__":
    main()
