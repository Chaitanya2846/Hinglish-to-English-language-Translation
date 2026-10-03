"""
translate_any.py — Multi-Model Interactive Translation Hub
==========================================================
Switch effortlessly between all fine-tuned Hinglish-to-English models:
  [1] Google mT5-Small (300M Seq2Seq LoRA) — 19.94 BLEU (Sub-second inference)
  [2] Sarvam-1 (2B QLoRA)                 — 17.30 BLEU, 41.56 chrF++, 49.18 ROUGE-1
  [3] RLM-Gemma-2B (QLoRA)                — 14.64 BLEU, rank r=64

Run:
    .venv\\Scripts\\python finetune/translate_any.py
"""
import sys
import subprocess
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable

MODELS = {
    "1": ("Google mT5-Small (300M Seq2Seq)", SCRIPT_DIR / "translate_mt5.py"),
    "2": ("Sarvam-1 (2B QLoRA - Best Fluency)", SCRIPT_DIR / "translate_sarvam.py"),
    "3": ("RLM-Gemma-2B (QLoRA - Causal LM)", SCRIPT_DIR / "translate_rlm.py"),
    "4": ("NLLB-200 (1.3B Seq2Seq LoRA - SOTA Translation)", SCRIPT_DIR / "translate_nllb.py"),
}


def main():
    print("=" * 70)
    print("  🌐 Multi-Model Hinglish-to-English Translation Hub")
    print("=" * 70)
    print("Select a model to launch:\n")
    print("  [1] Google mT5-Small (300M)  — 🥇 19.94 BLEU (Fastest, CPU-friendly)")
    print("  [2] Sarvam-1 (2B)            — 🥈 17.30 BLEU, 41.56 chrF++ (Best fluency)")
    print("  [3] RLM-Gemma-2B             — 🥉 14.64 BLEU (Causal LM)")
    print("  [4] NLLB-200-1.3B (1.3B)     — 🌍 Meta SOTA Machine Translation (GPU & Colab)")
    print("  [q] Quit")
    print("-" * 70)

    choice = input("Enter choice (1/2/3/4/q): ").strip().lower()
    if choice in ("q", "quit", "exit"):
        print("Goodbye!")
        return

    if choice in MODELS:
        name, script = MODELS[choice]
        print(f"\nLaunching {name} via {script.name}...\n")
        try:
            subprocess.run([PYTHON_EXE, str(script)], check=True)
        except KeyboardInterrupt:
            print("\nTranslator stopped.")
        except subprocess.CalledProcessError as exc:
            print(f"\n{name} exited with code {exc.returncode}. See the message above for details.")
    else:
        print(f"Invalid option '{choice}'. Please select 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
