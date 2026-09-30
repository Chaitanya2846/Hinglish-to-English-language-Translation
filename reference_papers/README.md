# Reference Research Papers

This folder contains the complete academic publications and technical reports serving as theoretical foundations and comparative benchmarks for our Hinglish-to-English Neural Machine Translation project.

---

### 1. `01_Agarwal_2021_Hinglish_to_English_MT.pdf`
* **Title**: *Hinglish to English Machine Translation using Multilingual Transformers*
* **Authors**: Vibhav Agarwal, Pooja S. B. Rao, Dinesh Babu Jayagopi
* **Conference**: Proceedings of the Student Research Workshop Associated with RANLP 2021 (pages 1–7)
* **Relevance**: The seminal paper demonstrating that multilingual text-to-text models (mT5 and mBART) effectively translate Romanized code-mixed Hinglish into English, serving as the primary academic justification for our mT5-Small experiment.

---

### 2. `02_PACMANtrans_ICON2023_CodeMix_to_English.pdf`
* **Title**: *Lost in Translation No More: Fine-tuned transformer-based models for CodeMix to English Machine Translation*
* **Authors**: ICON 2023 / ACL Anthology
* **Relevance**: Introduces the **PACMANtrans** benchmark dataset and evaluation suite (~18.66 BLEU baseline) which our fine-tuned models comprehensively outperformed by **+35.45 BLEU points**.

---

### 3. `03_Google_mT5_Multilingual_Transformer.pdf`
* **Title**: *mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer*
* **Authors**: Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, Colin Raffel
* **Conference**: NAACL-HLT 2021
* **Relevance**: Details the architecture of Google's mT5, pre-trained on 101 languages using the mC4 corpus. Demonstrates why its unquantized Seq2Seq encoder-decoder structure achieves 54.11 BLEU at sub-second inference speeds.

---

### 4. `04_Hu_2021_LoRA_Low_Rank_Adaptation.pdf`
* **Title**: *LoRA: Low-Rank Adaptation of Large Language Models*
* **Authors**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (Microsoft)
* **Conference**: ICLR 2022
* **Relevance**: Theoretical foundation for Low-Rank Adaptation (LoRA), decomposing weight update matrices into rank $r$ decompositions ($W = W_0 + B \cdot A$), enabling full-parameter adaptation while training only 1.6% to 2.2% of model parameters.

---

### 5. `05_Dettmers_2023_QLoRA_Quantized_LLMs.pdf`
* **Title**: *QLoRA: Efficient Finetuning of Quantized LLMs*
* **Authors**: Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (University of Washington)
* **Conference**: NeurIPS 2023
* **Relevance**: Foundation for training 2B-parameter foundation models (RLM-Gemma-2B and Sarvam-1) on a single commodity 16GB Nvidia T4 GPU using 4-bit NormalFloat (NF4) quantization and double quantization.

---

### 6. `06_IndicTrans2_Indian_Languages_MT.pdf`
* **Title**: *IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages*
* **Authors**: Jay Gala, Pranjal A. Chitale, Raghavan Ashok, et al. (AI4Bharat)
* **Conference**: ACL 2023
* **Relevance**: Benchmark Indian NLP translation architecture providing standardized data formatting and tokenization techniques for Indic languages.

---

### 7. `07_Sarvam_BrahmicTokenizer131K.pdf`
* **Title**: *BrahmicTokenizer-131K: A Comprehensive Tokenizer for Indian Languages*
* **Authors**: Sarvam AI Research Team (2024)
* **Relevance**: Details the design of the custom tokenizer powering **Sarvam-1 (2B)**, explaining why Sarvam-1 achieves outstanding character and bigram fluency (**66.85 chrF++**, **49.04 ROUGE-1**) across Indian code-mixed constructs.
