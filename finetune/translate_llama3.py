"""
translate_llama3.py — Interactive Hinglish-to-English Translator (Meta-Llama-3-8B)
================================================================================
Loads the fine-tuned 8-Billion parameter Meta-Llama-3 model with its custom
Hinglish-to-English QLoRA adapter for interactive inference.

Usage:
  # Interactive console:
  python finetune/translate_llama3.py

  # Single sentence translation:
  python finetune/translate_llama3.py --text "bhai kya scene hai kal ka?"
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

import torch

# Force UTF-8 encoding on Windows / Linux terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_ADAPTER_DIR = SCRIPT_DIR / "outputs" / "llama3_hinglish_lora"
DEFAULT_BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
HF_FALLBACK_REPO = "Nickhasntlost/llama-3-8b-hinglish-lora"

INFERENCE_TEMPLATE = "Translate the following Romanized Hinglish text to fluent English.\nHinglish: {source}\nEnglish:"


def get_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    if cli_token:
        return cli_token
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token()
    except Exception:
        return None


class Llama3Translator:
    def __init__(self, base_model: str = DEFAULT_BASE_MODEL, adapter_path: Path = DEFAULT_ADAPTER_DIR, token: Optional[str] = None):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.token = token
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        print("=" * 70)
        print("  🦙 Meta-Llama-3-8B Hinglish-to-English Neural Translator")
        print(f"  Base Model : {self.base_model}")
        print(f"  Device     : {self.device.upper()}")
        print("=" * 70)

        t0 = time.time()
        print("  [1/2] Loading Tokenizer...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, token=self.token)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        print(f"  [2/2] Loading Model Weights on {self.device.upper()}...", flush=True)
        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                token=self.token,
            )
        else:
            print("  ⚠️  Notice: Running 8B model on CPU. Inference will take ~10-20 seconds per sentence.")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                token=self.token,
            )

        # Check adapter weights
        adapter_source = None
        if self.adapter_path.exists() and ((self.adapter_path / "adapter_model.safetensors").exists() or (self.adapter_path / "adapter_config.json").exists()):
            adapter_source = str(self.adapter_path)
        else:
            adapter_source = HF_FALLBACK_REPO

        try:
            print(f"  Attaching LoRA Adapter from: {adapter_source}...", flush=True)
            self.model = PeftModel.from_pretrained(self.model, adapter_source)
            print("  [+] LoRA Adapter successfully mounted!")
        except Exception as e:
            print(f"  [!] Notice: Could not load LoRA adapter ({e}). Running base model.")

        self.model.eval()
        self.model.config.use_cache = True
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.max_length = None

        print(f"  [+] Translator ready in {time.time() - t0:.1f}s!\n")

    def translate(self, text: str, num_beams: int = 4) -> str:
        clean_text = text.strip()
        if not clean_text:
            return ""

        prompt = INFERENCE_TEMPLATE.format(source=clean_text)
        input_device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=num_beams,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        latency_ms = (time.time() - t0) * 1000

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        pred_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        if "English:" in pred_text:
            pred_text = pred_text.split("English:")[-1].strip()

        if not pred_text:
            full_decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            pred_text = full_decoded.split("English:")[-1].strip() if "English:" in full_decoded else full_decoded

        return pred_text, latency_ms


def main():
    parser = argparse.ArgumentParser(description="Interactive Meta-Llama-3-8B Hinglish Translator")
    parser.add_argument("--text", type=str, default=None, help="Single Hinglish sentence to translate")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL, help="Base Hugging Face model repository")
    parser.add_argument("--adapter", type=str, default=str(DEFAULT_ADAPTER_DIR), help="Path to local adapter directory")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face access token")
    parser.add_argument("--beams", type=int, default=4, help="Beam search width (default: 4)")
    args = parser.parse_args()

    token = get_hf_token(args.hf_token)
    translator = Llama3Translator(base_model=args.base_model, adapter_path=Path(args.adapter), token=token)

    if args.text:
        pred, latency = translator.translate(args.text, num_beams=args.beams)
        print(f"Hinglish   : {args.text}")
        print(f"Translation: {pred}")
        print(f"Latency    : {latency:.1f} ms")
        return

    print("Type a Hinglish sentence to translate (or 'exit' / 'q' to quit):")
    print("-" * 70)
    while True:
        try:
            user_input = input("\nHinglish > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            translation, latency = translator.translate(user_input, num_beams=args.beams)
            print(f"English     : {translation}")
            print(f"Latency     : {latency:.1f} ms")

        except KeyboardInterrupt:
            print("\nExiting translator.")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
