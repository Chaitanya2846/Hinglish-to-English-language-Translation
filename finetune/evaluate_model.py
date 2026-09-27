"""
evaluate_model.py -- Evaluate Hinglish Translation Model (Extended)
====================================================================
Metrics: BLEU, chrF++, BERTScore F1, Inference Latency
Usage:
    python evaluate_model.py
"""

import sys
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EVAL_FILE = DATA_DIR / "youtube_eval_gold.jsonl"
ADAPTER_PATH = BASE_DIR / "outputs" / "rlm_hinglish_lora"
BASE_MODEL_NAME = "rudrashah/RLM-hinglish-translator"
INFERENCE_TEMPLATE = "Hinglish:\n{source}\n\nEnglish:\n"


def load_eval_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            src = item.get("source", "").strip()
            tgt = item.get("target", "").strip()
            if src and tgt:
                data.append({"id": item.get("id", len(data)+1), "source": src, "target": tgt})
    return data


def generate_translation(model, tokenizer, source, device):
    prompt = INFERENCE_TEMPLATE.format(source=source)
    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda":
        input_device = model.get_input_embeddings().weight.device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - t0) * 1000
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "English:\n" in full_output:
        pred = full_output.split("English:\n")[-1].strip()
    else:
        pred = full_output.replace(prompt, "").strip()
    return pred.split("\n")[0].strip(), latency_ms


def print_qualitative_table(eval_data, predictions):
    sep = "=" * 65
    dash = "-" * 63
    print("\n" + sep)
    print("  QUALITATIVE COMPARISON (first 10 samples)")
    print(sep)
    for i, (item, pred) in enumerate(zip(eval_data[:10], predictions[:10])):
        print("")
        print("  [" + str(i+1) + "] Source    : " + item["source"])
        print("      Reference : " + item["target"])
        print("      Predicted : " + pred)
        print("  " + dash)


def main():
    sep = "=" * 65
    print(sep)
    print("  Hinglish -> English Evaluation (Extended)")
    print(sep)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: " + device.upper())

    print("\nLoading tokenizer for " + BASE_MODEL_NAME + "...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if ADAPTER_PATH.exists() and (ADAPTER_PATH / "adapter_config.json").exists():
        print("\n[+] LoRA adapter found. Loading...")
        model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
        eval_label = "Fine-Tuned (RLM + Custom QLoRA)"
    else:
        print("\n[-] No adapter found. Evaluating base model.")
        eval_label = "Base Model (rudrashah/RLM-hinglish-translator)"

    model.eval()

    if not EVAL_FILE.exists():
        print("ERROR: " + str(EVAL_FILE) + " not found")
        return

    eval_data = load_eval_data(EVAL_FILE)
    print("\nLoaded " + str(len(eval_data)) + " gold pairs from " + EVAL_FILE.name)

    predictions, references, latencies = [], [], []
    for idx, item in enumerate(eval_data):
        pred, lat = generate_translation(model, tokenizer, item["source"], device)
        predictions.append(pred)
        references.append(item["target"])
        latencies.append(lat)
        if idx < 5:
            print("[" + str(idx+1) + "] " + item["source"] + " -> " + pred)

    avg_lat = sum(latencies) / len(latencies)
    total_s = sum(latencies) / 1000
    print("\nInference: " + str(round(total_s, 1)) + "s total | " + str(round(avg_lat)) + " ms/sentence")

    bleu_score = chrf_score = bertscore_f1 = None

    try:
        import sacrebleu
        bleu_score = round(sacrebleu.corpus_bleu(predictions, [[r] for r in references]).score, 2)
        chrf_score = round(sacrebleu.corpus_chrf(predictions, [[r] for r in references]).score, 2)
        print("BLEU: " + str(bleu_score) + "  |  chrF: " + str(chrf_score))
    except ImportError:
        print("[!] pip install sacrebleu")

    try:
        from bert_score import score as bscore
        print("\nComputing BERTScore (may take ~1 min on CPU)...")
        _, _, F1 = bscore(predictions, references, lang="en", rescale_with_baseline=True, verbose=False)
        bertscore_f1 = round(F1.mean().item(), 4)
        print("BERTScore F1: " + str(bertscore_f1))
    except ImportError:
        print("[!] pip install bert-score")

    print_qualitative_table(eval_data, predictions)

    print("\n" + sep)
    print("  FINAL SUMMARY")
    print(sep)
    print("  Model        : " + eval_label)
    print("  Samples      : " + str(len(eval_data)))
    if bleu_score   is not None: print("  BLEU         : " + str(bleu_score))
    if chrf_score   is not None: print("  chrF++       : " + str(chrf_score))
    if bertscore_f1 is not None: print("  BERTScore F1 : " + str(bertscore_f1))
    print("  Avg Latency  : " + str(round(avg_lat)) + " ms/sentence")
    print("  Device       : " + device.upper())
    print(sep)

    out = BASE_DIR / "outputs" / "evaluation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "model": eval_label,
            "dataset": EVAL_FILE.name,
            "num_samples": len(eval_data),
            "bleu": bleu_score,
            "chrf": chrf_score,
            "bertscore_f1": bertscore_f1,
            "avg_latency_ms": round(avg_lat, 1),
            "device": device,
            "samples": [
                {
                    "id": eval_data[i]["id"],
                    "source": eval_data[i]["source"],
                    "reference": references[i],
                    "prediction": predictions[i],
                    "latency_ms": round(latencies[i], 1),
                }
                for i in range(len(eval_data))
            ],
        }, f, indent=2, ensure_ascii=False)
    print("\nFull report saved -> " + str(out))


if __name__ == "__main__":
    main()
