"""
translate_rlm.py — Live Interactive Translation with RLM-Gemma-2B (v3 QLoRA)
=============================================================================
Fine-Tuned Causal Decoder Model on Web-Scraped Social Media Corpus.
Parameters: 2.0B | LoRA Rank r=64 | Trained with Masked Causal Cross-Entropy

Run:
    .venv\\Scripts\\python finetune/translate_rlm.py
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

MODEL_ID = "rudrashah/RLM-hinglish-translator"
SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = SCRIPT_DIR / "outputs" / "rlm_hinglish_lora_v3"
PROMPT_TEMPLATE = "Hinglish:\n{source}\n\nEnglish:\n"


def main():
    print("=" * 70)
    print("  💎 RLM-Gemma-2B (v3 QLoRA) — Live Hinglish-to-English Translation")
    print("  LoRA Capacity: r=64, alpha=128 | Causal Decoder Architecture")
    print("=" * 70)

    has_cuda = torch.cuda.is_available()
    device_label = f"CUDA GPU ({torch.cuda.get_device_name(0)})" if has_cuda else "CPU (Laptop)"
    print(f"\n[Hardware] Target Device: {device_label}")
    if not has_cuda:
        print("  ⚠️  Notice: 2B models run best on GPU. Running on CPU may take 10-25s per sentence.")

    print(f"\n[1/2] Loading Tokenizer & Base Model ({MODEL_ID})...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if has_cuda:
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
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

    HF_REPO = "Nickhasntlost/rlm-gemma-2b-hinglish-lora"
    if (ADAPTER_DIR / "adapter_config.json").exists():
        print(f"[2/2] Attaching Fine-Tuned LoRA Adapter from {ADAPTER_DIR.name}...")
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
        print("  [+] LoRA Adapter (v3) attached successfully!")
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
                print("Exiting RLM translator. Goodbye!")
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
            if "English:\n" in full_text:
                translation = full_text.split("English:\n")[-1].strip()
            elif "English:" in full_text:
                translation = full_text.split("English:")[-1].strip()
            else:
                translation = full_text.replace(prompt, "").strip()

            print(f"English  > {translation}  ({latency:.0f} ms)\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break


if __name__ == "__main__":
    main()
