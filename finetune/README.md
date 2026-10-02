# 🚀 Fine-Tuning & Multi-Model Evaluation Guide
## Romanized Hinglish-to-English Neural Machine Translation

This directory contains the training engines, dataset pipelines, multi-model evaluation benchmarks, and interactive translation CLIs across all evaluated architectures:
1. **Meta-Llama-3-8B** (8B Parameters — Flagship Open LLM) 🦙
2. **Google mT5-Small** (300M Parameters — 19.94 BLEU, 648ms Latency) 🥇
3. **Sarvam-1** (2B Parameters — 41.56 chrF++, 47.16 METEOR, 49.18 ROUGE-1) 🥈
4. **RLM-Gemma-2B** (2B Parameters — 14.64 BLEU, Causal LM) 🥉

---

## 🦙 Fine-Tuning Meta-Llama-3-8B on Google Colab (Recommended)

### Why Google Colab for Meta-Llama-3-8B?
* **Model Size**: An 8-billion parameter model requires ~16 GB in FP16 or ~5.5 GB in 4-bit just for base weights.
* **VRAM Requirements**: With activations, optimizer states (Paged AdamW), LoRA adapters ($r=32$), and sequence length 192, training requires **~7.5 to 8.5 GB VRAM**.
* **Local Laptop GPU**: A laptop GPU (e.g. RTX 3050 with 4GB/6GB VRAM) will hit CUDA Out-Of-Memory (OOM).
* **Google Colab Free Tier**: Provides a **Tesla T4 GPU with 15.3–16.0 GB VRAM**, which fits Llama-3 4-bit QLoRA with zero memory pressure.

---

### Step-by-Step Colab Execution for Llama-3:

#### 1. Hugging Face Access & Token Setup
Meta-Llama-3-8B is a gated model. Before training:
1. Request access on Hugging Face: [https://huggingface.co/meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
2. Generate an access token with read permissions: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

#### 2. Open Notebook in Google Colab
Open [`finetune/FINETUNE_LLAMA3_COLAB.ipynb`](FINETUNE_LLAMA3_COLAB.ipynb) directly in Google Colab:
* Set Runtime to **GPU** (`Runtime` > `Change runtime type` > `T4 GPU`).
* In Colab's left sidebar, click the **Secrets** (🔑) icon, add `HF_TOKEN`, and paste your Hugging Face token.

#### 3. Run Fine-Tuning Command
```bash
# Clone repository
!git clone https://github.com/Nickhasntlost/NLP.git
%cd NLP

# Install requirements
!pip install -q -U torch transformers peft accelerate bitsandbytes sacrebleu evaluate rouge_score bert_score

# Run 4-bit QLoRA fine-tuning
!python finetune/finetune_llama3.py \
    --base_model meta-llama/Meta-Llama-3-8B \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 2e-4 \
    --lora_r 32 \
    --max_length 192
```

#### 4. Evaluate Against Benchmark Baselines
```bash
!python finetune/evaluate_colab.py --model llama3 --limit 150
```

#### 5. Interactive Translation Testing
```bash
!python finetune/translate_llama3.py --text "bhaaaai kyaaa scene hai kal ka? plzz batao"
```

---

## 💻 Local Testing & Multi-Model Interactive Hub

Once your adapter is trained or downloaded into `finetune/outputs/`:

### Launch Interactive Hub:
```bash
.\venv\Scripts\python.exe finetune/translate_any.py
```
This presents an interactive menu to switch between:
* `[1]` Google mT5-Small (300M Seq2Seq — 19.94 BLEU)
* `[2]` Sarvam-1 (2B QLoRA — 17.30 BLEU, 41.56 chrF++)
* `[3]` RLM-Gemma-2B (2B QLoRA — 14.64 BLEU)
* `[4]` Meta-Llama-3-8B (8B QLoRA — Flagship LLM)

### Individual Model Runners:
```bash
# Google mT5-Small (runs locally on CPU in ~650ms):
python finetune/translate_mt5.py

# Sarvam-1:
python finetune/translate_sarvam.py

# RLM-Gemma-2B:
python finetune/translate_rlm.py

# Meta-Llama-3-8B:
python finetune/translate_llama3.py
```

---

## 📁 Directory Layout

```
finetune/
├── FINETUNE_LLAMA3_COLAB.ipynb  ← One-click Google Colab notebook for Meta-Llama-3-8B
├── EVALUATE_IN_COLAB.ipynb     ← Multi-model evaluation notebook
├── finetune_llama3.py          ← 4-bit QLoRA fine-tuning engine for Llama-3-8B
├── finetune_sarvam.py          ← QLoRA fine-tuning for Sarvam-1 (2B)
├── finetune_mt5.py             ← LoRA fine-tuning for Google mT5-Small (300M)
├── finetune_rlm_v3.py          ← QLoRA fine-tuning for RLM-Gemma-2B
├── translate_any.py            ← Interactive multi-model console hub
├── translate_llama3.py         ← Interactive CLI for Meta-Llama-3-8B
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
