"""
evaluate_sarvam_quick.py — Quick evaluation for Sarvam-1 on validation split
"""
import json
import sys
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from evaluate_all_models import evaluate_causal_model, MODELS, load_val_pairs

def main():
    script_dir = Path(__file__).resolve().parent
    val_file = script_dir / "data" / "scraped_val_corpus.jsonl"
    
    val_pairs = load_val_pairs(val_file, limit=150)
    cfg = MODELS["Sarvam-1 (QLoRA)"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating Sarvam-1 on {len(val_pairs)} samples using device: {device.upper()}...")
    
    metrics, _ = evaluate_causal_model(cfg, val_pairs, device=device)
    
    print("\n" + "=" * 50)
    print("  SARVAM-1 EVALUATION RESULTS:")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<15} : {v}")
    print("=" * 50)
    
    out_file = script_dir / "outputs" / "sarvam_quick_eval.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[+] Saved results to {out_file}")

if __name__ == "__main__":
    main()
