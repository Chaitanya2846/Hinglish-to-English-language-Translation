"""
translate_nllb.py — Interactive CLI Translator for NLLB-200 (1.3B)
====================================================================
Architecture : facebook/nllb-200-distilled-1.3B (1.3B Parameters)
Task         : Romanized Hinglish-to-English Neural Translation
Decoding     : Beam Search (beams=4, length_penalty=1.0, no_repeat_ngram=3)
Device       : CUDA (RTX 3050 6GB Laptop GPU in 4-bit / FP16) or CPU

Usage:
  # Interactive mode:
  python finetune/translate_nllb.py

  # Single sentence translation:
  python finetune/translate_nllb.py --text "kya chal raha hai bhai"
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

import torch

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ADAPTER_DIR = SCRIPT_DIR / "outputs" / "nllb_hinglish_lora"
DEFAULT_BASE_MODEL = "facebook/nllb-200-distilled-1.3B"
HF_FALLBACK_REPO = "Nickhasntlost/nllb-200-1.3b-hinglish-lora"

SRC_LANG = "eng_Latn"
TGT_LANG = "eng_Latn"


class NLLBTranslator:
    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        adapter_path: Path = DEFAULT_ADAPTER_DIR,
        load_in_4bit: bool = True,
    ):
        self.base_model = base_model
        self.adapter_path = Path(adapter_path)
        self.load_in_4bit = load_in_4bit
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.forced_bos_token_id = None
        self._load_model()

    def _load_model(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
        from peft import PeftModel

        print("=" * 70)
        print("  🌍 NLLB-200 (1.3B) Hinglish-to-English Neural Translator")
        print(f"  Base Model : {self.base_model}")
        print(f"  Device     : {self.device.upper()}")
        print("=" * 70)

        t0 = time.time()
        print("  [1/2] Loading Tokenizer...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            src_lang=SRC_LANG,
            tgt_lang=TGT_LANG,
        )
        self.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(TGT_LANG)

        print(f"  [2/2] Loading Model Weights on {self.device.upper()}...", flush=True)
        if self.device == "cuda":
            if self.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.base_model,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.base_model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
        else:
            print("  ⚠️  Notice: Running on CPU. Inference may take ~2-5s per sentence.")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )

        # Check for fine-tuned LoRA adapter
        adapter_source = None
        has_local_weights = self.adapter_path.exists() and any(
            (self.adapter_path / fn).exists()
            for fn in ("adapter_model.safetensors", "adapter_model.bin")
        )
        if has_local_weights:
            adapter_source = str(self.adapter_path)
        else:
            # Check checkpoint subdirectories
            chk_dirs = sorted(
                (self.adapter_path / "checkpoints").glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
            ) if self.adapter_path.exists() else []
            if chk_dirs:
                latest = chk_dirs[-1]
                if (latest / "adapter_model.safetensors").exists() or (latest / "adapter_model.bin").exists():
                    adapter_source = str(latest)

        if adapter_source:
            try:
                print(f"  Attaching Fine-Tuned LoRA Adapter from: {adapter_source}...", flush=True)
                self.model = PeftModel.from_pretrained(self.model, adapter_source)
                print("  [+] LoRA Adapter successfully mounted!")
            except Exception as e:
                print(f"  [!] Notice: Could not attach adapter ({e}). Running base model.")
        else:
            print("  [i] Notice: No local LoRA adapter found yet.")
            print(f"      (Run 'python finetune/finetune_nllb.py' to train the Hinglish adapter).")
            print("      Running NLLB-200 base zero-shot translation mode.\n")

        self.model.eval()
        print(f"  [+] Ready in {time.time() - t0:.1f}s!\n")

    def translate(self, text: str, num_beams: int = 4) -> dict:
        """Translates a Romanized Hinglish sentence to English."""
        text = text.strip()
        if not text:
            return {"source": "", "translation": "", "latency_ms": 0.0}

        input_device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
        )
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.forced_bos_token_id,
                max_new_tokens=64,
                num_beams=num_beams,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        latency_ms = round((time.time() - t0) * 1000, 1)

        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return {
            "source": text,
            "translation": translation,
            "latency_ms": latency_ms,
        }


def main():
    parser = argparse.ArgumentParser(description="NLLB-200 (1.3B) Hinglish Translator")
    parser.add_argument("--text", type=str, default=None, help="Translate a single input string")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL, help="Base HF model")
    parser.add_argument("--adapter", type=str, default=str(DEFAULT_ADAPTER_DIR), help="Path to adapter")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()

    translator = NLLBTranslator(
        base_model=args.base_model,
        adapter_path=Path(args.adapter),
        load_in_4bit=not args.no_4bit,
    )

    if args.text:
        res = translator.translate(args.text)
        print(f"Hinglish    : {res['source']}")
        print(f"English     : {res['translation']}")
        print(f"Latency     : {res['latency_ms']} ms")
        return

    print("=" * 70)
    print("  Type any Romanized Hinglish sentence to translate into English.")
    print("  Type 'q', 'exit', or press Ctrl+C to quit.")
    print("=" * 70 + "\n")

    while True:
        try:
            line = input("Hinglish > ").strip()
            if not line:
                continue
            if line.lower() in ("q", "quit", "exit"):
                print("\nGoodbye!")
                break

            result = translator.translate(line)
            print(f"English  > {result['translation']}  ({result['latency_ms']} ms)\n")
        except KeyboardInterrupt:
            print("\nTranslator stopped.")
            break


if __name__ == "__main__":
    main()
