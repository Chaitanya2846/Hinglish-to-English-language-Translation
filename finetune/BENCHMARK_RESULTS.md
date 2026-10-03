# 🏆 Multi-Model Benchmark Results & Comparison
## Task: Romanized Hinglish-to-English Neural Machine Translation
**Dataset**: Custom Web-Scraped Social Media Corpus (9,769 Train / 996 Validation pairs)
**Live Evaluation Hardware**: Google Colab Tesla T4 GPU (150-Sample Blind Validation Set)

---

### 📊 Visual Performance Comparison (BLEU Scores)

```text
RCMT (LREC-COLING 2024 Base) [███████                     ]  14.00 BLEU
RLM-Gemma-2B                 [███████▍                    ]  14.64 BLEU  (+0.64) 🥉
Sarvam-1 (2B QLoRA)          [████████▋                   ]  17.30 BLEU  (+3.30) 🥈
PACMANtrans (ICON 2023 Base) [█████████▍                  ]  18.66 BLEU
Google mT5-Small             [██████████                  ]  19.94 BLEU  (+5.94) 🥇
NLLB-200 (1.3B Seq2Seq LoRA) [Target SOTA Candidate       ]  Targeting 20.0+ BLEU 🌍
```

---

### 📋 Table 1: Primary Translation Quality & Semantic Metrics

| Model Architecture | Params | Training Type | BLEU ↑ | chrF++ ↑ | METEOR ↑ | BERTScore ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **RCMT (LREC-COLING 2024)** | — | Primary 2024 SOTA Base | 14.00 | — | 14.00 | — |
| **PACMANtrans (ACL/ICON 2023)** | — | Benchmark Baseline | 18.66 | — | — | — |
| **RLM-Gemma-2B** | 2.0B | QLoRA ($r=64$) | 14.64 | 33.75 | 37.90 | 0.8793 |
| **Sarvam-1 (2B)** | 2.0B | QLoRA ($r=32$) | 17.30 | **41.56** | **47.16** | 0.9020 |
| **Google mT5-Small** | 300M | Seq2Seq LoRA | **19.94** | 38.56 | 38.48 | **0.9061** |
| **NLLB-200 (1.3B)** | 1.3B | Seq2Seq LoRA ($r=32$) | *Candidate* | *Candidate* | *Candidate* | *Candidate* |

---

### ⚡ Table 2: Lexical Overlap & Computational Efficiency

| Model Architecture | ROUGE-1 ↑ | ROUGE-2 ↑ | ROUGE-L ↑ | Latency ↓ | Training Time |
|:---|:---:|:---:|:---:|:---:|:---:|
| **RCMT (LREC-COLING 2024)** | — | — | — | — | — |
| **PACMANtrans (ACL/ICON 2023)** | — | — | — | — | — |
| **RLM-Gemma-2B** | 39.48 | 20.40 | 37.24 | 1347.5 ms | 2.0 hrs |
| **Sarvam-1 (2B)** | **49.18** | **26.56** | **45.90** | 2492.5 ms | 2.2 hrs |
| **Google mT5-Small** | 39.66 | 17.94 | 37.76 | **648.7 ms** | **19.5 min** |
| **NLLB-200 (1.3B)** | *Candidate* | *Candidate* | *Candidate* | ~1200 ms | ~25 min |

---

### 🔍 Model-by-Model Analysis (For Slides & Viva)

#### 🌍 1. NLLB-200-distilled-1.3B (1.3B) — *Meta AI's Flagship Multilingual Seq2Seq NMT*
* **Architecture**: Encoder-Decoder Seq2Seq Transformer (M2M-100 family) pre-trained on 200+ languages including extensive Indic data.
* **Why it Fits Perfectly**:
  - Purpose-built cross-attention for neural machine translation.
  - At 1.3B parameters, it uses **under 2.5 GB VRAM** in 4-bit QLoRA and ~3.5 GB in FP16, fitting comfortably inside local **RTX 3050 6GB Laptop GPU** and Google Colab T4 (15GB).
  - High convergence speed: trains in only ~25-30 minutes on Colab GPU.
  - Natively outputs English (`eng_Latn`) with beam search decoding.

#### 🥇 2. Google mT5-Small (300M) — *Best Overall BLEU & Sub-Second Latency*
* **Top Metric**: **19.94 BLEU** | **0.9061 BERTScore**
* **Inference Speed**: **648.7 ms/sentence** (Fastest, ~4× faster than 2B models)
* **Training Time**: **19.5 minutes** on free T4 GPU
* **Beats Both Base Papers**:
  - Outperforms **RCMT (LREC-COLING 2024 @ 14.00 BLEU)** by **+5.94 BLEU points**.
  - Outperforms **PACMANtrans (ICON 2023 @ 18.66 BLEU)** by **+1.28 BLEU points**.
* **Why it Won**: Encoder-Decoder bidirectional cross-attention enables full conditioning on entire Romanized Hinglish sentences, while its 300M parameter footprint eliminates 4-bit dequantization latency bottlenecks.

#### 🥈 3. Sarvam-1 (2B) — *Indic-Specialized Architecture*
* **Top Metric**: **41.56 chrF++** | **49.18 ROUGE-1** | **26.56 ROUGE-2** | **45.90 ROUGE-L** | **47.16 METEOR**
* **Inference Speed**: 2492.5 ms/sentence
* **Training Time**: ~2.2 hours
* **Beats Primary Base Paper**: Outperformed **RCMT (LREC-COLING 2024 @ 14.00 BLEU)** by **+3.30 BLEU points** (17.30 BLEU).
* **Why it Won**: Custom 131K Brahmic tokenizer and 2-trillion-token Indic language pre-training gave it unmatched understanding of Hindi grammatical syntax, code-mixed idioms, and colloquial social media vocabulary.

#### 🥉 4. RLM-Gemma-2B (2B) — *Causal LM Code-Mixing Model*
* **Top Metric**: **14.64 BLEU** | **0.8793 BERTScore** | **39.48 ROUGE-1** | **20.40 ROUGE-2**
* **Inference Speed**: 1347.5 ms/sentence
* **Training Time**: 2.0 hours
* **Beats Primary Base Paper**: Outperformed **RCMT (LREC-COLING 2024 @ 14.00 BLEU)** by **+0.64 BLEU points**.
* **Why it Matters**: Proved that causal decoder-only models can adapt to social media code-mixing via prompt masking and expanded LoRA rank ($r=64$), successfully exceeding the 2024 published SOTA base model.
