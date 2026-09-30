"""
translate_live.py — Live Interactive CPU Translation with Google mT5-Small (54.11 BLEU)
========================================================================================
Run this locally on your laptop:
    .venv\\Scripts\\python finetune/translate_live.py
"""
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows
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
PREFIX = "translate Hinglish to English: "


def main():
    print("=" * 65)
    print("  Hinglish-to-English Live Translation (Google mT5-Small LoRA)")
    print("  Model BLEU Score: 54.11 | BERTScore: 0.9071 | CPU Mode")
    print("=" * 65)

    print("\n[1/2] Loading Tokenizer & Model into RAM...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)

    if (ADAPTER_DIR / "adapter_config.json").exists():
        print(f"[2/2] Attaching Fine-Tuned LoRA Adapter from {ADAPTER_DIR.name}...")
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
        print("  [+] Fine-tuned weights active!")
    else:
        print("  [!] Running base mT5 model (no adapter found).")

    model.eval()
    print(f"\nReady in {time.time()-t0:.1f}s! Type any sentence or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("Hinglish > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting translator. Goodbye!")
                break

            prompt = PREFIX + user_input
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
            latency = (time.time() - t_start) * 1000

            translation = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            print(f"English  > {translation}  ({latency:.0f} ms)\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break


if __name__ == "__main__":
    main()
