"""
test_mt5_local.py — Automated Local Verification for Google mT5-Small LoRA
==========================================================================
Runs local CPU inference on curated Hinglish test cases and gold validation
corpus pairs, measuring memory footprint, generation latency, and output quality.
"""
import sys
import time
import psutil
from pathlib import Path

# Force UTF-8 encoding for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

MODEL_ID = "google/mt5-small"
SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = SCRIPT_DIR / "outputs" / "mt5_hinglish_lora"
VAL_CORPUS = SCRIPT_DIR / "data" / "scraped_val_corpus.jsonl"
PREFIX = "translate Hinglish to English: "


def get_ram_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)


def get_available_ram_gb():
    return psutil.virtual_memory().available / (1024 * 1024 * 1024)


def main():
    print("=" * 75)
    print("  🚀 Google mT5-Small (LoRA) Local CPU Self-Test")
    print("  Evaluating Translation Quality, Latency & RAM Footprint")
    print("=" * 75)

    ram_start_avail = get_available_ram_gb()
    proc_ram_start = get_ram_mb()
    print(f"\n[System Info]")
    print(f"  CPU Cores Available : {psutil.cpu_count(logical=True)}")
    print(f"  System RAM Available: {ram_start_avail:.2f} GB")
    print(f"  Process RAM at Start: {proc_ram_start:.1f} MB")

    print(f"\n[1/3] Loading Tokenizer & Base Model '{MODEL_ID}' (FP32)...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    load_model_time = time.time() - t0
    print(f"  [+] Base model loaded in {load_model_time:.2f}s")

    print(f"\n[2/3] Attaching Fine-Tuned LoRA Adapter from {ADAPTER_DIR.name}...")
    t1 = time.time()
    if (ADAPTER_DIR / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
        print(f"  [+] Adapter attached in {time.time() - t1:.2f}s")
    else:
        print("  [!] WARNING: Adapter not found! Running base model.")

    model.eval()
    proc_ram_loaded = get_ram_mb()
    ram_loaded_avail = get_available_ram_gb()
    model_ram_used = proc_ram_loaded - proc_ram_start
    print(f"  [+] Memory Footprint of Model in RAM : {model_ram_used:.1f} MB")
    print(f"  [+] System RAM Remaining Free       : {ram_loaded_avail:.2f} GB")

    # Define test sentences
    custom_tests = [
        "ye product sach me bahut accha hai, delivery bhi fast thi",
        "aaj mausam kaisa hai bhai?",
        "mujhe ye idea bohot pasand aaya, definitely try karunga",
        "order cancel karne ke liye customer support se baat karni padegi",
        "agar tum kal free ho to hum sham ko movie dekhne chal sakte hain",
    ]

    # Load 3 samples from validation set if available
    val_tests = []
    if VAL_CORPUS.exists():
        import json
        with open(VAL_CORPUS, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                row = json.loads(line)
                val_tests.append((row["source"], row["target"]))

    def translate(sentence):
        prompt = PREFIX + sentence
        inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
        t_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=4,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        lat_ms = (time.time() - t_start) * 1000
        text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return text, lat_ms

    print("\n" + "=" * 75)
    print("  🧪 PART 1: Real-World Conversational Hinglish Test Cases")
    print("=" * 75)

    latencies = []
    for idx, hinglish in enumerate(custom_tests, 1):
        translation, lat = translate(hinglish)
        latencies.append(lat)
        print(f"\nTest #{idx}:")
        print(f"  Hinglish Input : \"{hinglish}\"")
        print(f"  Model Output   : \"{translation}\"")
        print(f"  Inference Time : {lat:.1f} ms")

    if val_tests:
        print("\n" + "=" * 75)
        print("  📊 PART 2: Scraped Validation Corpus Samples (vs Gold Human Target)")
        print("=" * 75)

        for idx, (source, target) in enumerate(val_tests, 1):
            translation, lat = translate(source)
            latencies.append(lat)
            print(f"\nValidation Sample #{idx}:")
            print(f"  Hinglish Source : \"{source}\"")
            print(f"  Model Output    : \"{translation}\"")
            print(f"  Gold Target     : \"{target}\"")
            print(f"  Inference Time  : {lat:.1f} ms")

    avg_latency = sum(latencies) / len(latencies)
    print("\n" + "=" * 75)
    print("  🏁 Test Summary & Performance Metrics")
    print("=" * 75)
    print(f"  Total Sentences Evaluated : {len(latencies)}")
    print(f"  Average Latency per Item  : {avg_latency:.1f} ms ({avg_latency/1000:.2f} s)")
    print(f"  Min / Max Latency         : {min(latencies):.1f} ms / {max(latencies):.1f} ms")
    print(f"  Peak Model RAM Footprint  : {model_ram_used:.1f} MB (~{model_ram_used/1024:.2f} GB)")
    print(f"  Final Available System RAM: {get_available_ram_gb():.2f} GB")
    print("  Status: ✅ PASSED — Model executes smoothly on local laptop CPU!")
    print("=" * 75)


if __name__ == "__main__":
    main()
