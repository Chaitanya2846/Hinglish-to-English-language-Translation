"""
upload_to_hf.py — Upload Fine-Tuned LoRA Adapters to Hugging Face Hub
====================================================================
Uploads your trained adapters directly to your Hugging Face account so you can
load them anywhere (Google Colab, another laptop, cloud VM) with zero file transfers.

Usage:
    .venv\\Scripts\\python finetune/upload_to_hf.py --token YOUR_HF_WRITE_TOKEN
"""
import sys
import argparse
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from huggingface_hub import HfApi, whoami

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"

MODELS_TO_UPLOAD = [
    {
        "local_dir": OUTPUTS_DIR / "sarvam_hinglish_lora",
        "repo_name": "sarvam-1-hinglish-lora",
        "desc": "Sarvam-1 (2B) QLoRA Adapter for Romanized Hinglish-to-English NMT",
    },
    {
        "local_dir": OUTPUTS_DIR / "rlm_hinglish_lora_v3",
        "repo_name": "rlm-gemma-2b-hinglish-lora",
        "desc": "RLM-Gemma-2B (v3) QLoRA Adapter for Romanized Hinglish-to-English NMT",
    },
    {
        "local_dir": OUTPUTS_DIR / "mt5_hinglish_lora",
        "repo_name": "mt5-small-hinglish-lora",
        "desc": "Google mT5-Small LoRA Adapter for Romanized Hinglish-to-English NMT",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Upload trained adapters to Hugging Face Hub")
    parser.add_argument("--token", type=str, required=True, help="Your Hugging Face write token (from huggingface.co/settings/tokens)")
    args = parser.parse_args()

    api = HfApi(token=args.token)

    try:
        user_info = whoami(token=args.token)
        username = user_info["name"]
        print(f"Logged in to Hugging Face as: {username}\n")
    except Exception as e:
        print(f"[!] Authentication failed: {e}")
        print("Please check your token from https://huggingface.co/settings/tokens")
        sys.exit(1)

    # Files needed for inference (skip heavy optimizer/checkpoint folders)
    allowed_extensions = {".safetensors", ".json", ".jinja", ".txt", ".md", ".model"}

    for item in MODELS_TO_UPLOAD:
        local_path = item["local_dir"]
        repo_id = f"{username}/{item['repo_name']}"

        if not local_path.exists():
            print(f"[-] Skipping {item['repo_name']} (folder not found at {local_path})")
            continue

        print(f"[*] Uploading {item['desc']} -> https://huggingface.co/{repo_id}...")

        # Create repo if it doesn't exist
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

        # Upload essential adapter files
        for f in local_path.iterdir():
            if f.is_file() and f.suffix in allowed_extensions:
                print(f"    Uploading {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)...")
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=f.name,
                    repo_id=repo_id,
                    repo_type="model",
                )

        print(f"[+] Successfully published: https://huggingface.co/{repo_id}\n")

    print("=" * 60)
    print("All models uploaded! You can now load them anywhere:")
    print(f"  model = PeftModel.from_pretrained(base_model, '{username}/sarvam-1-hinglish-lora')")
    print("=" * 60)


if __name__ == "__main__":
    main()
