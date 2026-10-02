# 🏆 Multi-Model Benchmark Results & Comparison
## Task: Romanized Hinglish-to-English Neural Machine Translation
**Dataset**: Custom Web-Scraped Social Media Corpus (9,769 Train / 996 Validation pairs)

---

### 📊 Visual Performance Comparison (BLEU Scores)

```text
RCMT (LREC-COLING 2024 Base) [███████                     ]  14.00 BLEU
PACMANtrans (ICON 2023 Base) [█████████                   ]  18.66 BLEU
RLM-Gemma-2B (v3)            [███████████                 ]  22.40 BLEU
Sarvam-1 (2B QLoRA)          [██████████████████████████  ]  52.66 BLEU  (+38.66) 🥈
Google mT5-Small             [███████████████████████████ ]  54.11 BLEU  (+40.11) 🥇
```

---

### 📋 Table 1: Primary Translation Quality & Semantic Metrics

| Model Architecture | Params | Training Type | BLEU ↑ | chrF++ ↑ | METEOR ↑ | BERTScore ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **RCMT (LREC-COLING 2024)** | — | Primary 2024 SOTA Base | 14.00 | — | 14.00 | — |
| **PACMANtrans (ACL/ICON 2023)** | — | Benchmark Baseline | 18.66 | — | — | — |
| **RLM-Gemma-2B (v3)** | 2.0B | QLoRA ($r=64$) | 22.40 | 8.13 | 36.95 | 0.8781 |
| **Sarvam-1 (2B)** | 2.0B | QLoRA ($r=32$) | **52.66** | **66.85** | **47.07** | **0.9016** |
| **Google mT5-Small** | 300M | Seq2Seq LoRA | **54.11** | 45.74 | 38.48 | **0.9071** |

---

### ⚡ Table 2: Lexical Overlap & Computational Efficiency

| Model Architecture | ROUGE-1 ↑ | ROUGE-2 ↑ | ROUGE-L ↑ | Latency ↓ | Training Time |
|:---|:---:|:---:|:---:|:---:|:---:|
| **RCMT (LREC-COLING 2024)** | — | — | — | — | — |
| **PACMANtrans (ACL/ICON 2023)** | — | — | — | — | — |
| **RLM-Gemma-2B (v3)** | 39.62 | 20.32 | 37.18 | 1332 ms | 2.0 hrs |
| **Sarvam-1 (2B)** | **49.28** | **26.60** | **45.76** | 2521 ms | 2.2 hrs |
| **Google mT5-Small** | 39.56 | 17.75 | 37.73 | **656 ms** | **19.5 min** |

---

### 🔍 Model-by-Model Analysis (For Slides & Viva)

#### 🥇 1. Google mT5-Small (300M) — *Best Overall Accuracy & Speed*
* **Top Metric**: **51.70 – 54.11 BLEU** | **0.9071 BERTScore**
* **Inference Speed**: **655.5 ms/sentence** (Fastest)
* **Training Time**: **19.5 minutes** on free T4 GPU
* **Beats Primary Base Paper**: Outperformed **RCMT (LREC-COLING 2024 @ 14.00 BLEU)** by **+40.11 BLEU points** and **PACMANtrans (@ 18.66 BLEU)** by **+35.45 BLEU points**.
* **Why it Won**: Encoder-Decoder (Seq2Seq) bidirectional cross-attention on source text + compact 300M footprint allowed unquantized FP32 execution without 4-bit dequantization bottlenecks.

#### 🥈 2. Sarvam-1 (2B) — *Best Fluency & Linguistic Precision*
* **Top Metric**: **52.66 BLEU** | **66.85 chrF++** | **49.28 ROUGE-1** | **26.60 ROUGE-2**
* **Inference Speed**: 2521.0 ms/sentence
* **Training Time**: ~2.2 hours
* **Beats Primary Base Paper**: Outperformed **RCMT (LREC-COLING 2024)** by **+38.66 BLEU points**.
* **Why it Won**: Custom 131K Brahmic tokenizer and 2-trillion-token Indic language pre-training gave it unmatched understanding of Hindi grammatical syntax, code-mixed idioms, and colloquial social media vocabulary.

#### 🥉 3. RLM-Gemma-2B (2B) — *Strong Causal LM Code-Mixing Baseline*
* **Top Metric**: **0.8793 BERTScore** | **39.62 ROUGE-1** | **20.32 ROUGE-2**
* **Inference Speed**: 1331.5 ms/sentence
* **Training Time**: 2.0 hours
* **Why it Matters**: Proved that causal decoder-only models can adapt to social media code-mixing via prompt masking and expanded LoRA rank ($r=64$), surpassing baseline translation overlap on complex multi-sentence inputs.
