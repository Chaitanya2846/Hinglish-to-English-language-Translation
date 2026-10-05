# 🌍 Beyond Words: Context-Aware Translation of Code-Mixed Indian Languages
### Romanized Hinglish-to-English Neural Machine Translation (NMT) Suite

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![Tests Passed](https://img.shields.io/badge/Tests-31%2F31%20Passing-brightgreen.svg)]()
[![SOTA BLEU](https://img.shields.io/badge/NLLB--200-33.65%20BLEU%20(SOTA)-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

---

## 📌 1. Project Overview

Social media discourse across South Asia frequently occurs in **Hinglish** (Hindi blended with English, written primarily in the Roman/Latin alphabet). Standard machine translation systems designed for formal monolingual text collapse on Hinglish due to:
* **Non-Standard Phonetic Spelling**: *"bohot"* vs *"bahut"*, *"kya krrhe ho"* vs *"kya kar rahe ho"*.
* **Colloquial Youth Slang & Idioms**: *"scene sort hai"*, *"jugaad"*, *"badhiya"*.
* **Polysemy & Homophones**: *"kam"* (less) vs *"kaam"* (work); *"chalna"* (walking vs running software).
* **Syntactic Discrepancy**: Hindi Subject-Object-Verb (SOV) order transitioning to English Subject-Verb-Object (SVO).

**Beyond Words** is an end-to-end, production-ready NLP system featuring **4 fine-tuned neural models**, dual-track preprocessing, a unified evaluation benchmark, a FastAPI service, a React UI, and a Chrome Extension.

Our flagship model, **NLLB-200 (1.3B Seq2Seq LoRA)**, achieves **33.65 BLEU** and **0.9359 BERTScore F1**, outperforming prior published literature baselines (*Agarwal et al. @ 29.50 BLEU*, *PACMANtrans @ 18.66 BLEU*, *RCMT @ 14.00 BLEU*).

---

## 🏆 2. Multi-Model Benchmark Results

Evaluated on the held-out blind validation corpus ([`finetune/data/scraped_val_corpus.jsonl`](finetune/data/scraped_val_corpus.jsonl)) using Beam Search decoding:

| Model Architecture | Parameter Size | Adaptation Type | BLEU ↑ | chrF++ ↑ | ROUGE-L ↑ | METEOR ↑ | BERTScore F1 ↑ | Latency ↓ | Hardware Target |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 **NLLB-200 (Distilled)** | **1.38B** | **Seq2Seq LoRA ($r=32$, MLP)** | **33.65** | **55.00** | **58.21** | **59.63** | **0.9359** | ~2,239 ms | **Tesla T4 / RTX 3050** |
| 🥈 **Google mT5-Small** | 300M | Seq2Seq LoRA ($r=32$, FP32) | **19.94** | 43.12 | 42.10 | 38.20 | 0.8841 | **~180 ms** | **CPU / Edge Device** |
| 🥉 **Sarvam-1** | 2.0B | QLoRA ($r=32$, 4-bit NF4) | **17.30** | 41.56 | 39.80 | 35.10 | 0.8710 | ~120 ms | GPU (CUDA FP16) |
| 4️⃣ **RLM-Gemma-2B** | 2.0B | QLoRA ($r=64$, 4-bit NF4) | **14.64** | 36.80 | 34.20 | 30.40 | 0.8420 | ~140 ms | GPU (CUDA FP16) |
| 📚 *Agarwal et al. (2021)* | — | *Prior Published SOTA* | 29.50 | — | — | — | — | — | *Literature Baseline* |
| 📚 *PACMANtrans (2023)* | — | *Benchmark Baseline* | 18.66 | — | — | — | — | — | *Literature Baseline* |
| 📚 *RCMT (2024)* | 48M | *LREC-COLING 2024* | 14.00 | — | — | 14.00 | — | — | *Literature Baseline* |

---

## ⚡ 3. Quick Start (Run Locally in 3 Steps)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Nickhasntlost/NLP.git
cd NLP
```

### Step 2: Create a Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Linux / macOS:
source venv/bin/activate

# Upgrade pip & install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Launch the Interactive Translation Hub
```bash
python finetune/translate_any.py
```
This presents an interactive menu allowing you to switch between all 4 fine-tuned models seamlessly!

---

## 💻 4. Running Individual Models

All CLI scripts feature automatic device detection (CUDA GPU $\to$ 4-bit/FP16, CPU $\to$ FP32), beam search decoding, and remote Hugging Face fallback if local adapter weights are not found.

### 🥇 1. Flagship Model: NLLB-200 (1.3B Seq2Seq LoRA)
```bash
# Interactive conversation loop:
python finetune/translate_nllb.py

# Single-sentence one-shot translation:
python finetune/translate_nllb.py --text "sridhar sir bohot aacha padhate haii"
# Output > "sridhar sir teaches very well" (Latency: ~1.2s)
```

### 🥈 2. Ultra-Fast CPU Model: Google mT5-Small (300M)
```bash
python finetune/translate_mt5.py
# Runs in ~180ms per sentence on modern multi-core CPUs without requiring a GPU!
```

### 🥉 3. Indic Foundation Model: Sarvam-1 (2B)
```bash
python finetune/translate_sarvam.py
```

### 4️⃣ 4. Causal Model: RLM-Gemma-2B (2B)
```bash
python finetune/translate_rlm.py
```

---

## 🌐 5. Running the Full Pipeline, REST API & Web UI

### A. End-to-End Pipeline Smoke Test
Executes the full 4-stage sequential pipeline (*Preprocessing $\to$ Language Identification $\to$ Normalization $\to$ Translation $\to$ Grammar Polish*):
```bash
python evaluate_pipeline.py
```

### B. Launch FastAPI Backend Server
```bash
uvicorn api.main:app --reload --port 8000
```
* **Interactive API Docs (Swagger UI)**: `http://localhost:8000/docs`
* **Health Check**: `GET http://localhost:8000/health`
* **Translate Endpoint**: `POST http://localhost:8000/translate`
  ```json
  {
    "text": "Bhai kal presentation ke liye model ready hai kya?",
    "use_neural_grammar": false
  }
  ```

### C. Launch React Frontend
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 🧪 6. Automated Test Suite

We maintain a strict 100% test pass rate across unit, preprocessing, schema validation, and API tests.
```bash
pytest
```
*Status: **31 passed**, 0 failed.*

---

## ☁️ 7. Training & Evaluation on Google Colab (Free T4 GPU)

If you wish to re-train the models or run multi-metric evaluation benchmarks on a free GPU:

| Task | Notebook Link | Hardware | Time |
| :--- | :--- | :---: | :---: |
| **Fine-Tune NLLB-200 (1.3B)** | [`finetune/FINETUNE_NLLB_COLAB.ipynb`](finetune/FINETUNE_NLLB_COLAB.ipynb) | Tesla T4 (15GB) | ~88 min |
| **Evaluate All 4 Models** | [`finetune/EVALUATE_IN_COLAB.ipynb`](finetune/EVALUATE_IN_COLAB.ipynb) | Tesla T4 (15GB) | ~15 min |

---

## 📂 8. Repository Structure

```
NLP/
├── api/                         # FastAPI backend & test suites
│   ├── main.py                  # REST API application with CORS
│   └── test_api.py              # TestClient integration tests
├── data/                        # Curated training & validation corpora
│   ├── raw/                     # Filtered social media comments (6,583 comments)
│   └── phinc/                   # PHINC social baseline dataset
├── finetune/                    # Fine-tuning engines, benchmarks & CLIs
│   ├── finetune_nllb.py         # NLLB-200 (1.3B) Seq2Seq LoRA trainer
│   ├── finetune_mt5.py          # Google mT5-Small (300M) trainer
│   ├── finetune_sarvam.py       # Sarvam-1 (2B) QLoRA trainer
│   ├── finetune_rlm_v3.py       # RLM-Gemma-2B (2B) trainer
│   ├── translate_any.py         # Multi-model interactive translation hub
│   ├── translate_nllb.py        # NLLB-200 CLI translator
│   ├── translate_mt5.py         # Google mT5-Small CLI translator
│   ├── translate_sarvam.py      # Sarvam-1 CLI translator
│   ├── translate_rlm.py         # RLM-Gemma CLI translator
│   ├── evaluate_colab.py        # Multi-model GPU evaluation benchmark
│   └── BENCHMARK_RESULTS.md     # Detailed empirical benchmark tables
├── frontend/                    # React + Tailwind Web Interface
│   ├── src/                     # React components
│   └── package.json             # Vite configuration
├── models/                      # Pipeline stages (LID, Normalization, Grammar)
│   ├── lid_colab.py             # Language Identification (XLM-RoBERTa)
│   ├── normalize.py             # Rule-based spelling variant normalizer
│   └── grammar.py               # Post-processing grammar correction
├── preprocessing/               # Dual-track text preprocessing
│   └── pipeline.py              # Track A (Syllabus) & Track B (Model Input)
├── reference_papers/            # Academic publications & conference PDF references
│   └── README.md                # Literature survey & bibliographic index
├── evaluate_pipeline.py         # End-to-end integration test runner
├── pipeline.py                  # Core orchestrator interface
├── requirements.txt             # Pinned Python package dependencies
└── README.md                    # Root project documentation
```

---

## 🔬 9. Academic Foundation & Baseline Citations

1. **RCMT (Primary 2024 Base Paper)**:  
   *Kartik, Soni, Kunchukuttan, Chakraborty, Akhtar.* **"Synthetic Data Generation and Joint Learning for Robust Code-Mixed Translation"**. *Proceedings of LREC-COLING 2024*, pages 15480–15492.
2. **PACMANtrans (Benchmark Baseline)**:  
   *Chatterjee, Sharma, Yashwanth, Kumar, Raj, Ekbal.* **"Fine-tuned transformer-based models for CodeMix to English Machine Translation"**. *Proceedings of ICON 2023*, pages 226–235.
3. **Agarwal et al. (Architectural Foundation)**:  
   *Agarwal, Rao, Jayagopi.* **"Hinglish to English Machine Translation using Multilingual Transformers"**. *Proceedings of RANLP 2021*, pages 16–21.
4. **LoRA**:  
   *Hu et al.* **"LoRA: Low-Rank Adaptation of Large Language Models"**. *ICLR 2022*.
5. **QLoRA**:  
   *Dettmers et al.* **"QLoRA: Efficient Finetuning of Quantized LLMs"**. *NeurIPS 2023*.

---

## 👥 Contributors & Contact

* Developed for the **Natural Language Processing (NLP)** Course Project.
* Team: Beyond Words Research & Engineering Group.
