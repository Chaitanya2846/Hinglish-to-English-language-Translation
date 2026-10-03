# 🚀 Fine-Tuning & Multi-Model Evaluation Guide
## Romanized Hinglish-to-English Neural Machine Translation

This directory contains the training engines, dataset pipelines, multi-model evaluation benchmarks, and interactive translation CLIs across all evaluated architectures:
1. **NLLB-200-distilled-1.3B** (1.3B Parameters — Meta AI SOTA Multilingual NMT) 🌍
2. **Google mT5-Small** (300M Parameters — 19.94 BLEU, 648ms Latency) 🥇
3. **Sarvam-1** (2B Parameters — 41.56 chrF++, 47.16 METEOR, 49.18 ROUGE-1) 🥈
4. **RLM-Gemma-2B** (2B Parameters — 14.64 BLEU, Causal LM) 🥉

---

## 🌍 Fine-Tuning NLLB-200 (1.3B) on GPU or Colab

### Why NLLB-200 (1.3B) for Hinglish-to-English Translation?
* **Purpose-Built Architecture**: Unlike general causal decoder-only LMs, NLLB-200 is an **Encoder-Decoder (Seq2Seq)** translation model with bidirectional source attention and dedicated cross-attention.
* **Pre-trained on 200+ Languages**: Features deep multilingual knowledge including extensive Indic languages, allowing it to generalize swiftly to Romanized Hinglish.
* **Lightweight & Efficient**: Uses **under 2.5 GB VRAM** in 4-bit QLoRA and ~3.5 GB in FP16.
* **Dual Execution**: Runs effortlessly on:
  1. **Local Laptop GPU** (RTX 3050 6GB Laptop GPU) with zero out-of-memory errors!
  2. **Google Colab Free GPU** (Tesla T4) training in just **~20-30 minutes**!

---

### Step-by-Step Training Execution:

#### Option A: Fine-Tune on Google Colab (Recommended)
1. Open [`finetune/FINETUNE_NLLB_COLAB.ipynb`](FINETUNE_NLLB_COLAB.ipynb) in Google Colab.
2. Select Runtime: **T4 GPU** (`Runtime` > `Change runtime type` > `T4 GPU`).
3. Run all cells:
```bash
!python finetune/finetune_nllb.py \
    --base_model facebook/nllb-200-distilled-1.3B \
    --epochs 3 \
    --batch_size 8 \
    --grad_accum 2 \
    --lr 2e-4 \
    --lora_r 32 \
    --max_length 128
```
4. Step 7 in the notebook will automatically package and download `nllb_hinglish_lora.zip`.

#### Option B: Fine-Tune Locally on RTX 3050 6GB Laptop GPU
Open your PowerShell terminal and run:
```powershell
python finetune/finetune_nllb.py --epochs 3 --batch_size 4 --grad_accum 4
```

---

## 💻 Local Testing & Multi-Model Interactive Hub

Once your adapter is trained or downloaded into `finetune/outputs/nllb_hinglish_lora/`:

### Launch Interactive Hub:
```bash
.\venv\Scripts\python.exe finetune/translate_any.py
```
This presents an interactive menu to switch between:
* `[1]` Google mT5-Small (300M Seq2Seq — 19.94 BLEU)
* `[2]` Sarvam-1 (2B QLoRA — 17.30 BLEU, 41.56 chrF++)
* `[3]` RLM-Gemma-2B (2B QLoRA — 14.64 BLEU)
* `[4]` NLLB-200-1.3B (1.3B Seq2Seq LoRA — SOTA Translation)

### Individual Model Runners:
```bash
# NLLB-200 (1.3B):
python finetune/translate_nllb.py

# Single sentence test:
python finetune/translate_nllb.py --text "kya chal raha hai bhai"

# Google mT5-Small (runs locally on CPU in ~650ms):
python finetune/translate_mt5.py

# Sarvam-1:
python finetune/translate_sarvam.py

# RLM-Gemma-2B:
python finetune/translate_rlm.py
```

---

## 📁 Directory Layout

```
finetune/
├── FINETUNE_NLLB_COLAB.ipynb    ← One-click Google Colab notebook for NLLB-200
├── EVALUATE_IN_COLAB.ipynb     ← Multi-model evaluation notebook
├── finetune_nllb.py            ← Seq2Seq LoRA fine-tuning engine for NLLB-200 (1.3B)
├── finetune_sarvam.py          ← QLoRA fine-tuning for Sarvam-1 (2B)
├── finetune_mt5.py             ← LoRA fine-tuning for Google mT5-Small (300M)
├── finetune_rlm_v3.py          ← QLoRA fine-tuning for RLM-Gemma-2B
├── translate_any.py            ← Interactive multi-model console hub
├── translate_nllb.py           ← Interactive CLI for NLLB-200 (1.3B)
├── translate_mt5.py            ← Interactive CLI for Google mT5-Small
├── translate_sarvam.py         ← Interactive CLI for Sarvam-1
├── translate_rlm.py            ← Interactive CLI for RLM-Gemma-2B
├── evaluate_colab.py           ← Multi-model GPU evaluation script
├── evaluate_all_models.py      ← Consolidated evaluation suite
├── upload_to_hf.py             ← One-click upload of trained adapters to Hugging Face Hub
├── benchmark_dashboard.html    ← Interactive visual dashboard & charts
├── BENCHMARK_RESULTS.md        ← Verified empirical metrics table
└── data/
    ├── scraped_train_corpus.jsonl  ← 9,769 curated social media training pairs
    ├── scraped_val_corpus.jsonl    ← 996 validation pairs
    ├── phinc_train.jsonl           ← 8,695 Twitter Hinglish-English pairs
    ├── phinc_val.jsonl             ← 967 Twitter validation pairs
    └── youtube_eval_gold.jsonl     ← 29-sample gold reference evaluation set
```
