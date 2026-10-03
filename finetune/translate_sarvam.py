"""
translate_sarvam.py — Live Interactive Translation with Sarvam-1 (2B QLoRA)
=============================================================================
Flagship 2B Indic Foundation Model by Sarvam AI.
Scored: 17.30 BLEU | 41.56 chrF++ | 49.18 ROUGE-1 | 47.16 METEOR | 0.9020 BERTScore

Run:
    .venv\\Scripts\\python finetune/translate_sarvam.py
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
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "sarvamai/sarvam-1"
SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = SCRIPT_DIR / "outputs" / "sarvam_hinglish_lora"
PROMPT_TEMPLATE = "Translate Hinglish to English.\nHinglish: {source}\nEnglish:"


def main():
    print("=" * 70)
    print("  🇮🇳 Sarvam-1 (2B) — Live Hinglish-to-English Translation")
    print("  Benchmark: 17.30 BLEU | 41.56 chrF++ | 49.18 ROUGE-1 | 47.16 METEOR")
    print("=" * 70)

    has_cuda = torch.cuda.is_available()
    device_label = f"CUDA GPU ({torch.cuda.get_device_name(0)})" if has_cuda else "CPU (Laptop)"
    print(f"\n[Hardware] Target Device: {device_label}")
    if not has_cuda:
        print("  ⚠️  Notice: 2B models run best on GPU. Running on CPU may take 10-25s per sentence.")

    print(f"\n[1/2] Loading Tokenizer & Base Model ({MODEL_ID})...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if has_cuda:
        try:
            import bitsandbytes
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        except (ImportError, Exception) as e:
            print("  [i] bitsandbytes not found. Running natively in FP16 on GPU (~4GB VRAM)...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
    else:
        # Load in float32 or bfloat16 on CPU
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    HF_REPO = "Nickhasntlost/sarvam-1-hinglish-lora"
    if (ADAPTER_DIR / "adapter_config.json").exists():
        print(f"[2/2] Attaching Fine-Tuned LoRA Adapter from {ADAPTER_DIR.name}...")
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
        print("  [+] LoRA Adapter attached successfully!")
    else:
        print(f"[2/2] Attaching Fine-Tuned LoRA Adapter from Hugging Face ({HF_REPO})...")
        model = PeftModel.from_pretrained(model, HF_REPO)
        print("  [+] LoRA Adapter attached successfully from Hugging Face!")

    model.eval()
    print(f"\nReady in {time.time()-t0:.1f}s! Type any Hinglish sentence or 'exit' to quit.\n")

    input_device = next(model.parameters()).device
    beam_count = 4 if has_cuda else 1

    while True:
        try:
            user_input = input("Hinglish > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting Sarvam-1 translator. Goodbye!")
                break

            prompt = PROMPT_TEMPLATE.format(source=user_input)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(input_device) for k, v in inputs.items()}

            t_start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    num_beams=beam_count,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            latency = (time.time() - t_start) * 1000

            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "English:" in full_text:
                translation = full_text.split("English:")[-1].strip()
            else:
                translation = full_text.replace(prompt, "").strip()

            print(f"English  > {translation}  ({latency:.0f} ms)\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break


if __name__ == "__main__":
    main()
