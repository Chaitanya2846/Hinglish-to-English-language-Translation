# 🏆 Multi-Model Benchmark Results & Comparison
## Task: Romanized Hinglish-to-English Neural Machine Translation
**Dataset**: Custom Web-Scraped Social Media Corpus (9,769 Train / 996 Validation pairs)

---

### 📊 Visual Performance Comparison (BLEU Scores)

```text
PACMANtrans (Baseline)  [█████████                     ]  18.66 BLEU
RLM-Gemma-2B (v2)       [██████████                    ]  20.80 BLEU
RLM-Gemma-2B (v3)       [███████████                   ]  22.40 BLEU
Sarvam-1 (2B)           [██████████████████████████    ]  52.66 BLEU  (+34.00) 🥈
Google mT5-Small        [███████████████████████████   ]  54.11 BLEU  (+35.45) 🥇
```

---

### 📋 Table 1: Primary Translation Quality & Semantic Metrics

| Model Architecture | Params | Training Type | BLEU ↑ | chrF++ ↑ | METEOR ↑ | BERTScore ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **PACMANtrans (ACL 2024)** | — | Baseline | 18.66 | — | — | — |
| **RLM-Gemma-2B (v2)** | 2.0B | QLoRA ($r=32$) | 20.80 | — | 31.20 | 0.8520 |
| **RLM-Gemma-2B (v3)** | 2.0B | QLoRA ($r=64$) | 22.40 | 8.13 | 36.95 | 0.8781 |
| **Sarvam-1 (2B)** | 2.0B | QLoRA ($r=32$) | **52.66** | **66.85** | **47.00** | **0.9017** |
| **Google mT5-Small** | 300M | Seq2Seq LoRA | **54.11** | 40.23 | 38.55 | **0.9071** |

---

### ⚡ Table 2: Lexical Overlap & Computational Efficiency

| Model Architecture | ROUGE-1 ↑ | ROUGE-2 ↑ | ROUGE-L ↑ | Latency ↓ | Training Time |
|:---|:---:|:---:|:---:|:---:|:---:|
| **PACMANtrans (ACL 2024)** | — | — | — | — | — |
| **RLM-Gemma-2B (v2)** | 34.20 | 15.10 | 32.50 | ~1400 ms | 1.5 hrs |
| **RLM-Gemma-2B (v3)** | 39.17 | 19.72 | 36.79 | 1377 ms | 2.0 hrs |
| **Sarvam-1 (2B)** | **49.04** | **26.56** | **45.83** | 2511 ms | 2.2 hrs |
| **Google mT5-Small** | 40.31 | 18.68 | 38.20 | **645 ms** | **19.5 min** |

---

### 🔍 Model-by-Model Analysis (For Slides & Viva)

#### 🥇 1. Google mT5-Small (300M) — *Best Overall Accuracy & Speed*
* **Top Metric**: **54.11 BLEU** | **0.9071 BERTScore**
* **Inference Speed**: **645.4 ms/sentence** (Fastest)
* **Training Time**: **19.5 minutes** on free T4 GPU
* **Why it Won**: Encoder-Decoder (Seq2Seq) bidirectional attention on source text + compact 300M footprint allowed unquantized FP32 execution without 4-bit dequantization bottlenecks.

#### 🥈 2. Sarvam-1 (2B) — *Best Fluency & Linguistic Precision*
* **Top Metric**: **66.85 chrF++** | **49.04 ROUGE-1** | **26.56 ROUGE-2**
* **Inference Speed**: 2511.4 ms/sentence
* **Training Time**: ~2.2 hours
* **Why it Won**: Custom 131K Brahmic tokenizer and 2-trillion-token Indic language pre-training gave it unmatched understanding of Hindi grammatical syntax and vocabulary.

#### 🥉 3. RLM-Gemma-2B (2B) — *Strong Causal LM Code-Mixing Baseline*
* **Top Metric**: **0.8781 BERTScore** | **39.17 ROUGE-1**
* **Inference Speed**: 1377.5 ms/sentence
* **Training Time**: 2.0 hours
* **Why it Matters**: Proved that causal decoder-only models can adapt to social media code-mixing via prompt masking and expanded LoRA rank ($r=64$).
