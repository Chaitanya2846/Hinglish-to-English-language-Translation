"""
evaluate_all_models.py — Unified Multi-Metric Benchmark for All Fine-Tuned Models
==================================================================================
Run this AFTER training all models to generate the final comparison table.

Evaluates each model on the SAME validation set with these 7 metrics:
  1. BLEU          — n-gram Precision
  2. ROUGE-1       — Unigram Recall
  3. ROUGE-2       — Bigram Recall
  4. ROUGE-L       — Longest Common Subsequence Recall
  5. chrF++        — Character n-gram F-score (best for code-mixed text)
  6. METEOR        — Synonym-aware alignment score
  7. BERTScore F1  — Semantic similarity via contextual embeddings

Outputs:
  - Console: Formatted comparison table ready for PPT
  - File: finetune/outputs/final_comparison.json

Usage (Colab T4 GPU):
  python evaluate_all_models.py
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
VAL_FILE = DATA_DIR / "scraped_val_corpus.jsonl"
OUTPUT_FILE = SCRIPT_DIR / "outputs" / "final_comparison.json"


# ── Model Registry ──────────────────────────────────────────────────────────
MODELS = {
    "RLM-Gemma-2B (QLoRA)": {
        "type": "causal",
        "base": "rudrashah/RLM-hinglish-translator",
        "hf_adapter": "Nickhasntlost/rlm-gemma-2b-hinglish-lora",
        "adapter": SCRIPT_DIR / "outputs" / "rlm_hinglish_lora_v3",
        "prompt_template": "Hinglish:\n{source}\n\nEnglish:\n",
        "extract_key": "English:\n",
        "num_beams": 6,
    },
    "Sarvam-1 (QLoRA)": {
        "type": "causal",
        "base": "sarvamai/sarvam-1",
        "hf_adapter": "Nickhasntlost/sarvam-1-hinglish-lora",
        "adapter": SCRIPT_DIR / "outputs" / "sarvam_hinglish_lora",
        "prompt_template": "Translate Hinglish to English.\nHinglish: {source}\nEnglish:",
        "extract_key": "English:",
        "num_beams": 4,
    },
    "mT5-Small (LoRA)": {
        "type": "seq2seq_mt5",
        "base": "google/mt5-small",
        "hf_adapter": "Nickhasntlost/mt5-small-hinglish-lora",
        "adapter": SCRIPT_DIR / "outputs" / "mt5_hinglish_lora",
        "prefix": "translate Hinglish to English: ",
        "num_beams": 5,
    },
}


def load_val_pairs(filepath: Path, limit: int = 150) -> List[Dict[str, str]]:
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                d = json.loads(line)
                pairs.append({"source": d["source"].strip(), "target": d["target"].strip()})
    return pairs


def compute_all_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    import evaluate as hf_evaluate
    results = {}

    # BLEU + chrF++
    try:
        import sacrebleu
        results["BLEU"] = round(sacrebleu.corpus_bleu(predictions, [references], smooth_method="exp").score, 2)
        results["chrF++"] = round(sacrebleu.corpus_chrf(predictions, [references], word_order=2).score, 2)
    except Exception as e:
        print(f"  [WARN] BLEU/chrF++ failed: {e}")
        results["BLEU"] = results["chrF++"] = 0.0

    # ROUGE
    try:
        r = hf_evaluate.load("rouge").compute(predictions=predictions, references=references)
        results["ROUGE-1"] = round(r["rouge1"] * 100, 2)
        results["ROUGE-2"] = round(r["rouge2"] * 100, 2)
        results["ROUGE-L"] = round(r["rougeL"] * 100, 2)
    except Exception as e:
        print(f"  [WARN] ROUGE failed: {e}")
        results["ROUGE-1"] = results["ROUGE-2"] = results["ROUGE-L"] = 0.0

    # METEOR
    try:
        results["METEOR"] = round(hf_evaluate.load("meteor").compute(predictions=predictions, references=references)["meteor"] * 100, 2)
    except Exception as e:
        print(f"  [WARN] METEOR failed: {e}")
        results["METEOR"] = 0.0

    # BERTScore
    try:
        bs = hf_evaluate.load("bertscore").compute(predictions=predictions, references=references, lang="en")
        results["BERTScore_F1"] = round(sum(bs["f1"]) / len(bs["f1"]), 4)
    except Exception as e:
        print(f"  [WARN] BERTScore failed: {e}")
        results["BERTScore_F1"] = 0.0

    return results


def evaluate_causal_model(config: dict, val_pairs: List[Dict], device: str) -> Tuple[Dict, List[str]]:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(config["base"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(config["base"], quantization_config=bnb_config,
        device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

    adapter_path = Path(config["adapter"])
    actual_adapter = None
    if (adapter_path / "adapter_config.json").exists():
        actual_adapter = str(adapter_path)
    else:
        chk_dirs = sorted((adapter_path / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0)
        if chk_dirs:
            actual_adapter = str(chk_dirs[-1])
        elif config.get("hf_adapter"):
            actual_adapter = config["hf_adapter"]

    if actual_adapter:
        model = PeftModel.from_pretrained(model, actual_adapter)
        print(f"  [+] Loaded LoRA adapter from {actual_adapter}")
    else:
        print(f"  [!] No adapter found at {adapter_path} or on Hugging Face, using base model")

    model.eval()
    predictions, references, latencies = [], [], []
    input_device = model.get_input_embeddings().weight.device

    for i, item in enumerate(val_pairs):
        prompt = config["prompt_template"].format(source=item["source"])
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=96, num_beams=config["num_beams"],
                length_penalty=1.0, no_repeat_ngram_size=3, early_stopping=True, pad_token_id=tokenizer.eos_token_id)
        latencies.append((time.time() - t0) * 1000)
        gen = tokenizer.decode(outputs[0], skip_special_tokens=True)
        key = config["extract_key"]
        pred = gen.split(key)[-1].strip() if key in gen else gen.replace(prompt, "").strip()
        predictions.append(pred)
        references.append(item["target"])
        if i < 3:
            print(f"  [{i+1}] Src: {item['source'][:60]}... -> {pred[:60]}...")

    metrics = compute_all_metrics(predictions, references)
    metrics["Latency_ms"] = round(sum(latencies) / len(latencies), 1)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    return metrics, predictions


def evaluate_seq2seq_model(config: dict, val_pairs: List[Dict], device: str) -> Tuple[Dict, List[str]]:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(config["base"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["base"], torch_dtype=torch.float32)

    adapter_path = Path(config["adapter"])
    actual_adapter = None
    if (adapter_path / "adapter_config.json").exists():
        actual_adapter = str(adapter_path)
    else:
        chk_dirs = sorted((adapter_path / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0)
        if chk_dirs:
            actual_adapter = str(chk_dirs[-1])
        elif config.get("hf_adapter"):
            actual_adapter = config["hf_adapter"]

    if actual_adapter:
        model = PeftModel.from_pretrained(model, actual_adapter)
        print(f"  [+] Loaded LoRA adapter from {actual_adapter}")
    else:
        print(f"  [!] No adapter found at {adapter_path} or on Hugging Face, using base model")

    if device == "cuda":
        model = model.cuda()
    model.eval()

    prefix = config.get("prefix", "")
    predictions, references, latencies = [], [], []

    for i, item in enumerate(val_pairs):
        input_text = prefix + item["source"]
        inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=96, num_beams=config["num_beams"],
                length_penalty=1.0, no_repeat_ngram_size=3, early_stopping=True)
        latencies.append((time.time() - t0) * 1000)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        predictions.append(pred)
        references.append(item["target"])
        if i < 3:
            print(f"  [{i+1}] Src: {item['source'][:60]}... -> {pred[:60]}...")

    metrics = compute_all_metrics(predictions, references)
    metrics["Latency_ms"] = round(sum(latencies) / len(latencies), 1)

    del model
    torch.cuda.empty_cache()
    return metrics, predictions


def print_comparison_table(all_results: Dict[str, Dict]):
    print("\n")
    print("=" * 110)
    print("    FINAL COMPARATIVE ANALYSIS — WEB-SCRAPED CODE-MIXED SOCIAL MEDIA CORPUS")
    print("=" * 110)
    header = f"{'Model':<28} | {'BLEU':<6} | {'ROUGE-1':<8} | {'ROUGE-2':<8} | {'ROUGE-L':<8} | {'chrF++':<7} | {'METEOR':<7} | {'BERTSc':<7} | {'Lat(ms)':<8}"
    print(header)
    print("-" * 110)
    for name, m in all_results.items():
        row = (f"{name:<28} | {m.get('BLEU',0):<6} | {m.get('ROUGE-1',0):<8} | {m.get('ROUGE-2',0):<8} | "
               f"{m.get('ROUGE-L',0):<8} | {m.get('chrF++',0):<7} | {m.get('METEOR',0):<7} | "
               f"{m.get('BERTScore_F1',0):<7} | {m.get('Latency_ms',0):<8}")
        print(row)
    print("=" * 110)
    best = max(all_results.items(), key=lambda x: x[1].get("BLEU", 0))
    print(f"\n  >>> BEST MODEL: {best[0]} (BLEU: {best[1]['BLEU']})")
    print(f"  >>> vs RCMT (LREC-COLING 2024 SOTA Base): ~14.00 BLEU (+{best[1]['BLEU'] - 14.00:.2f})")
    print(f"  >>> vs PACMANtrans (ACL/ICON 2023 Base) : ~18.66 BLEU (+{best[1]['BLEU'] - 18.66:.2f})")


def main():
    print("=" * 70)
    print("  Unified Multi-Metric Model Benchmark")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")

    val_pairs = load_val_pairs(VAL_FILE, limit=150)
    print(f"Validation samples: {len(val_pairs)}")

    all_results = {}

    for model_name, config in MODELS.items():
        print(f"\n{'─' * 70}")
        print(f"  Evaluating: {model_name}")
        print(f"{'─' * 70}")

        adapter_path = config["adapter"]
        has_local = adapter_path.exists() and (adapter_path / "adapter_config.json").exists()
        has_hf = bool(config.get("hf_adapter"))
        if not has_local and not has_hf:
            print(f"  [SKIP] Adapter not found at {adapter_path} or on Hugging Face")
            print(f"         Train this model first: python finetune_*.py")
            continue

        try:
            if config["type"] == "causal":
                metrics, _ = evaluate_causal_model(config, val_pairs, device)
            elif config["type"] in ("seq2seq", "seq2seq_mt5"):
                metrics, _ = evaluate_seq2seq_model(config, val_pairs, device)
            all_results[model_name] = metrics
        except Exception as e:
            print(f"  [ERROR] {model_name} failed: {e}")
            continue

    if all_results:
        print_comparison_table(all_results)

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[+] Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
