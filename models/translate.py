"""
translate.py — Hinglish -> English Inference Wrapper (Model 3)
===============================================================
Loads the 2B Gemma-based Hinglish translation model (`rudrashah/RLM-hinglish-translator`)
with automatic detection for custom fine-tuned LoRA weights.

Exposes:
    translate(text: str) -> str

Architecture:
    Input (Roman Hinglish)  -->  [RLM-Hinglish-Translator (+ LoRA if available)]  -->  Output (English)

Automatic LoRA Detection:
    Checks for fine-tuned LoRA adapter in:
      1. finetune/outputs/rlm_hinglish_lora/
      2. models/lora_adapter/
    If found, seamlessly overlays adapter weights onto the base model.
    Otherwise, runs the high-quality base model directly.
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

# Silence non-critical transformers warnings
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
warnings.filterwarnings("ignore", message="You are using a model of type")

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_NAME = "rudrashah/RLM-hinglish-translator"
INFERENCE_TEMPLATE = "Hinglish:\n{source}\n\nEnglish:\n"

# Search paths for custom trained LoRA adapters
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_CANDIDATE_PATHS = [
    _PROJECT_ROOT / "finetune" / "outputs" / "rlm_hinglish_lora",
    _PROJECT_ROOT / "models" / "lora_adapter",
]

# Module-level singletons (lazy loaded on first inference)
_model = None
_tokenizer = None
_device = None
_active_model_desc = "uninitialized"


def _get_adapter_path() -> Optional[Path]:
    """Return path to fine-tuned LoRA adapter if it exists and is valid."""
    for candidate in ADAPTER_CANDIDATE_PATHS:
        if candidate.exists() and (candidate / "adapter_config.json").exists():
            return candidate
    return None


def _load():
    """
    Load tokenizer and causal language model.
    Lazy-loaded on first translate() call.
    """
    global _model, _tokenizer, _device, _active_model_desc

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  Model 3: Hinglish -> English Translation Engine")
    print(f"  Base Model : {MODEL_NAME}")
    print(f"  Device     : {_device.upper()}")
    if _device == "cpu":
        print("  NOTE: Running on CPU. Inference takes ~2-5s per sentence.")
    print("=" * 60)

    t0 = time.time()
    print("  [1/2] Loading tokenizer...", flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token_id = _tokenizer.eos_token_id
    print(f"        Tokenizer ready ({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    print("  [2/2] Loading model weights...", flush=True)

    if _device == "cuda":
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

    # Check for fine-tuned LoRA adapter
    adapter_path = _get_adapter_path()
    if adapter_path:
        try:
            from peft import PeftModel
            print(f"        [+] Custom LoRA adapter found: {adapter_path}")
            print("        Overlaying fine-tuned weights...", flush=True)
            _model = PeftModel.from_pretrained(_model, str(adapter_path))
            _active_model_desc = f"{MODEL_NAME} + LoRA ({adapter_path.name})"
        except Exception as e:
            print(f"        [!] Warning: Failed to load LoRA adapter: {e}")
            print("        Falling back to base model.")
            _active_model_desc = f"{MODEL_NAME} (base)"
    else:
        _active_model_desc = f"{MODEL_NAME} (base)"

    _model.eval()
    print(f"        Model ready: {_active_model_desc} ({time.time() - t0:.1f}s)", flush=True)
    print("=" * 60)


def translate(text: str) -> str:
    """
    Translate a Roman Hinglish, Devanagari Hindi, or code-mixed string to English.

    Args:
        text: Input text (e.g., 'ye project aisa bana hai')

    Returns:
        Translated English sentence string.
    """
    global _model, _tokenizer, _device

    if not text or not text.strip():
        return ""

    if _model is None:
        _load()

    import torch

    clean_text = text.strip()
    prompt = INFERENCE_TEMPLATE.format(source=clean_text)

    inputs = _tokenizer(prompt, return_tensors="pt")
    input_device = _model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )

    full_output = _tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract target following 'English:\n'
    if "English:\n" in full_output:
        raw_pred = full_output.split("English:\n")[-1].strip()
    else:
        raw_pred = full_output.replace(prompt, "").strip()

    # Take first paragraph/line and clean any stray quotes
    pred_line = raw_pred.split("\n")[0].strip()
    if pred_line.startswith('"') and pred_line.endswith('"'):
        pred_line = pred_line[1:-1].strip()

    return pred_line


# ── Backward-Compatibility Helpers ───────────────────────────────────────────

def translate_devanagari(text: str) -> str:
    """Alias for translate() to maintain full compatibility with existing callers."""
    return translate(text)


def is_roman_script(text: str) -> bool:
    """Return True if text is predominantly Latin/Roman alphabet."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    latin = sum(1 for c in alpha if c.isascii())
    return latin / len(alpha) > 0.5


def roman_to_deva(text: str) -> str:
    """Optional phonetic converter retained for utility fallback."""
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        return transliterate(text.lower(), sanscript.ITRANS, sanscript.DEVANAGARI)
    except ImportError:
        return text


def get_model_status() -> dict:
    """Return current model metadata and adapter status."""
    adapter_path = _get_adapter_path()
    return {
        "base_model": MODEL_NAME,
        "is_loaded": _model is not None,
        "device": _device or "not loaded",
        "has_lora_adapter": adapter_path is not None,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "active_description": _active_model_desc,
    }


if __name__ == "__main__":
    test_phrases = [
        "ye project aisa bana hai",
        "Bhai kya kar raha hai aajkal?",
        "sir aap bahut achha padhate ho, mujhe samajh aa gaya",
        "video bahut informative thi bhai",
    ]
    print("Testing Translation Module directly:")
    for phrase in test_phrases:
        res = translate(phrase)
        print(f"Hinglish : {phrase}")
        print(f"English  : {res}")
        print("-" * 50)
