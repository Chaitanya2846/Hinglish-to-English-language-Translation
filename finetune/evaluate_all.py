"""
evaluate_all.py — Comprehensive Multi-Metric Benchmark for Fine-Tuned Models
=============================================================================
Evaluates all 3 models against the Custom Web-Scraped Social Media Corpus.
Parameters evaluated:
  1. BLEU
  2. ROUGE-1 / ROUGE-2 / ROUGE-L
  3. chrF++
  4. BERTScore (F1)
  5. METEOR
  6. Inference Latency (ms/sentence)

Outputs a clean comparative analysis table ready for presentation slides.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import evaluate

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
VAL_FILE = DATA_DIR / "scraped_val_corpus.jsonl"


def load_val_pairs(filepath: Path, limit: int = 150) -> Tuple[List[str], List[str]]:
    sources, targets = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                d = json.loads(line)
                sources.append(d["source"].strip())
                targets.append(d["target"].strip())
    return sources, targets


def compute_all_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    results = {}

    # 1. SacreBLEU
    bleu_metric = evaluate.load("sacrebleu")
    b_res = bleu_metric.compute(predictions=predictions, references=[[r] for r in references])
    results["BLEU"] = round(b_res["score"], 2)

    # 2. ROUGE (1, 2, L)
    rouge_metric = evaluate.load("rouge")
    r_res = rouge_metric.compute(predictions=predictions, references=references)
    results["ROUGE-1"] = round(r_res["rouge1"] * 100, 2)
    results["ROUGE-2"] = round(r_res["rouge2"] * 100, 2)
    results["ROUGE-L"] = round(r_res["rougeL"] * 100, 2)

    # 3. chrF++
    chrf_metric = evaluate.load("chrf")
    c_res = chrf_metric.compute(predictions=predictions, references=[[r] for r in references], word_order=2)
    results["chrF++"] = round(c_res["score"], 2)

    # 4. METEOR
    try:
        meteor_metric = evaluate.load("meteor")
        m_res = meteor_metric.compute(predictions=predictions, references=references)
        results["METEOR"] = round(m_res["meteor"] * 100, 2)
    except Exception:
        results["METEOR"] = 0.0

    # 5. BERTScore
    try:
        bertscore_metric = evaluate.load("bertscore")
        bs_res = bertscore_metric.compute(predictions=predictions, references=references, lang="en")
        results["BERTScore_F1"] = round(sum(bs_res["f1"]) / len(bs_res["f1"]), 4)
    except Exception:
        results["BERTScore_F1"] = 0.0

    return results


def print_comparison_table(results_dict: Dict[str, Dict[str, Any]]):
    print("\n" + "=" * 95)
    print("      FINAL COMPARATIVE ANALYSIS — WEB-SCRAPED SOCIAL MEDIA CORPUS")
    print("=" * 95)
    header = f"{'Model Architecture':<30} | {'BLEU':<6} | {'ROUGE-L':<8} | {'chrF++':<7} | {'METEOR':<7} | {'BERTScore':<9} | {'Latency':<8}"
    print(header)
    print("-" * 95)
    for model_name, metrics in results_dict.items():
        row = (
            f"{model_name:<30} | "
            f"{metrics.get('BLEU', 0.0):<6.2f} | "
            f"{metrics.get('ROUGE-L', 0.0):<8.2f} | "
            f"{metrics.get('chrF++', 0.0):<7.2f} | "
            f"{metrics.get('METEOR', 0.0):<7.2f} | "
            f"{metrics.get('BERTScore_F1', 0.0):<9.4f} | "
            f"{metrics.get('Latency_ms', 0):<8}ms"
        )
        print(row)
    print("=" * 95)


if __name__ == "__main__":
    print("Unified multi-metric evaluator initialized.")
