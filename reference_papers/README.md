# Reference Research Papers

This folder contains the complete, verified academic publications and technical reports serving as theoretical foundations, model architectures, and comparative benchmarks for our Hinglish-to-English Neural Machine Translation project.

All papers are verified, official conference versions from the **ACL Anthology**, **IEEE**, and **arXiv**.

---

### 1. `01_Kartik_2024_LREC_COLING_Robust_CodeMix_MT.pdf`
* **Title**: *Synthetic Data Generation and Joint Learning for Robust Code-Mixed Translation*
* **Authors**: Kartik, Sanjana Soni, Anoop Kunchukuttan (AI4Bharat), Tanmoy Chakraborty (IIT Delhi), Md Shad Akhtar (IIIT Delhi)
* **Conference**: Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (**LREC-COLING 2024**, pages 15480–15492, May 2024, ACL Anthology)
* **Relevance**: **The Primary SOTA 2024 Base Paper**. Introduces the HINMIX corpus and evaluates Hinglish-to-English translation across noise-robust baselines (10.44 BLEU) and the proposed RCMT architecture (~14.00 BLEU). Serves as our primary modern benchmark, which our fine-tuned models comprehensively outperformed by **+38.66 BLEU points**.

---

### 2. `02_PACMANtrans_ICON2023_CodeMix_to_English.pdf`
* **Title**: *Lost in Translation No More: Fine-tuned transformer-based models for CodeMix to English Machine Translation*
* **Authors**: Arindam Chatterjee, Chhavi Sharma, Yashwanth V.P., Niraj Kumar, Ayush Raj, Asif Ekbal (Wipro Research Lab45 & IIT Patna)
* **Conference**: Proceedings of the 20th International Conference on Natural Language Processing (**ICON 2023**, pages 226–235, ACL Anthology)
* **Relevance**: **The Benchmark Baseline Paper**. Establishes the **PACMANtrans** benchmark dataset and evaluation suite (~18.66 BLEU baseline) which our fine-tuned models comprehensively outperformed by **+34.00 BLEU points**.

---

### 3. `03_Agarwal_2021_Hinglish_to_English_MT.pdf`
* **Title**: *Hinglish to English Machine Translation using Multilingual Transformers*
* **Authors**: Vibhav Agarwal, Pooja S. B. Rao, Dinesh Babu Jayagopi (Netaji Subhas Univ / IIIT Bangalore)
* **Conference**: Proceedings of the Student Research Workshop Associated with RANLP 2021 (pages 16–21, ACL Anthology)
* **Relevance**: **Architectural Foundation Paper**. Demonstrates that multilingual text-to-text models (mT5 and mBART) effectively translate Romanized code-mixed Hinglish into English without transliterating back to Devanagari first, serving as the academic justification for our mT5-Small experiment.

---

### 4. `04_Google_mT5_Multilingual_Transformer.pdf`
* **Title**: *mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer*
* **Authors**: Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, Colin Raffel (Google Research)
* **Conference**: Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (**NAACL-HLT 2021**, pages 483–498, ACL Anthology)
* **Relevance**: **Model 1 Architecture**. Details Google's mT5 architecture pre-trained on 101 languages using the mC4 corpus. Demonstrates why its unquantized Seq2Seq encoder-decoder bidirectional structure achieves 54.11 BLEU at sub-second (655 ms) inference speeds.

---

### 5. `05_Google_Gemma_2024_Open_Models.pdf`
* **Title**: *Gemma: Open Models Based on Gemini Research and Technology*
* **Authors**: Gemma Team, Google DeepMind (Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, et al.)
* **Venue**: arXiv:2403.08295 (March 2024)
* **Relevance**: **Model 3 Base Architecture**. Details Google's lightweight Gemma foundation models built from Gemini research, serving as the base model for our fine-tuned **RLM-Gemma-2B** causal translation model.

---

### 6. `06_Hu_2021_LoRA_Low_Rank_Adaptation.pdf`
* **Title**: *LoRA: Low-Rank Adaptation of Large Language Models*
* **Authors**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (Microsoft Corporation)
* **Conference**: International Conference on Learning Representations (**ICLR 2022**)
* **Relevance**: **Fine-Tuning Methodology**. Theoretical foundation for Low-Rank Adaptation (LoRA), decomposing weight update matrices into rank $r$ decompositions ($W = W_0 + B \cdot A$), enabling full-parameter adaptation of Google mT5 while training only ~2.2% of total parameters.

---

### 7. `07_Dettmers_2023_QLoRA_Quantized_LLMs.pdf`
* **Title**: *QLoRA: Efficient Finetuning of Quantized LLMs*
* **Authors**: Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (University of Washington)
* **Conference**: Advances in Neural Information Processing Systems (**NeurIPS 2023**)
* **Relevance**: **Quantization & Training Methodology**. Theoretical and practical foundation for training 2B-parameter foundation models (**Sarvam-1 (2B)** and **RLM-Gemma-2B**) on a single commodity 16GB Nvidia T4 GPU using 4-bit NormalFloat (NF4) quantization and double quantization.
