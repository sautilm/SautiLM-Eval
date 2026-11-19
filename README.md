# SautiLM-Eval (Bbadha Edition)

**SautiLM-Eval** is the official scientific evaluation benchmark and metrics suite for the **SautiLM** family of offline Edge-AI language models (**Radi**, **Nuru**, **Jua**, **Wazo**, **Kuu**, **Mali**), engineered in **Bunia, Ituri, Democratic Republic of the Congo** by **LosingTek Research** for **SautiLM**.

This benchmark currently focuses on **Bbadha (Kilendu - ISO 639-3: `led`)**, evaluating tonal orthographic accuracy, implosive consonant preservation (`bb`, `dd`, `dy`), trilingual lexical bridging (**Bbadha <-> Kiswahili <-> French/English**), and Bbadha-grounded domain reasoning across `SautiLM-Lex`, `SautiLM-Edu`, `SautiLM-Med`, `SautiLM-Math`, `SautiLM-Ius`, and `SautiLM-Voice`.

- **Official Website**: [https://sautilm.com](https://sautilm.com)
- **Foundation Model (`SautiLM Jua`)**: [https://huggingface.co/sautilm/SautiLM-Jua](https://huggingface.co/sautilm/SautiLM-Jua)
- **Training Pipeline (`SautiLM-Forge`)**: [https://github.com/sautilm/SautiLM-Forge](https://github.com/sautilm/SautiLM-Forge)

---

## 1. Evaluation Metrics

`SautiLM-Eval` implements three complementary evaluation metrics tailored for low-resource tonal African languages:

1. **Bbadha Lexical & Tonal Keyword Accuracy (`%`)** : Evaluates exact and tonally normalized preservation of Bbadha headwords (`Dda`, `Ddi`, `'Ábba`, `Dyà`, `Kawa`, `Ke`, `Dyi`, `Zzà`, `Rra`, `Chà`, `Bbi`) and their Swahili/French equivalents.
2. **`chrF++` (Character 6-gram + Word 2-gram F-Score)** : The international scientific standard for morphologically rich and tonal African languages.
3. **`BLEU-4` (Smoothed Sentence & Corpus Precision)** : Measures phrase-level alignment against human linguistic references.

---

## 2. Empirical Results: Zero-Leakage v5 Suite (`litert_lm.Engine` CPU XNNPACK)

### 2.1. `bbadha_eval_v1` (`SautiLM-Jua`, `283.48 MB` LiteRT-LM INT8, `n=15` real Bbadha prompts)

| Configuration | Backend | Language | Bbadha Keyword Accuracy | Semantic Accuracy | Mean `chrF++` | Mean `BLEU-4` | Mean Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Parametric Weights Only (`Zero-RAG`) | CPU (XNNPACK) | Bbadha (`led`) | **31.67%** | **70.00%** | **20.85 / 100** | **4.91 / 100** | `1,913.2 ms` |
| **Hybrid Edge-AI** (Weights + Bbadha RAG + MathSolver) | CPU (XNNPACK) | Bbadha (`led`) | **58.89%** | **76.67%** | **27.60 / 100** | **9.06 / 100** | `2,602.1 ms` |

### 2.2. `domain_eval_v2` Multi-Tier Held-Out & Epistemic Abstention Suite (`n=7` across `Lex`, `Ius`, `Med`, `Agri`, `Kuu-STEM`)

| Suite | Models Evaluated (`litert_lm.Engine` + `sautilm_sdk` Guardrails) | Mean Keyword Accuracy | Mean Semantic Accuracy |
| :--- | :--- | :--- | :--- |
| `domain_eval_v2` | `SautiLM-Jua-Lex`, `SautiLM-Jua-Ius`, `SautiLM-Jua-Med`, `SautiLM-Jua-Agri`, `SautiLM-Kuu-STEM` (`1B`) | **92.86%** | **100.00% (`7/7`)** |

Full machine-readable and human-readable reports are generated in:
- [`reports/sautilm_jua_bbadha_report.json`](reports/sautilm_jua_bbadha_report.json)
- [`reports/sautilm_jua_bbadha_report.md`](reports/sautilm_jua_bbadha_report.md)
- [`reports/sautilm_domain_eval_v2_report.json`](reports/sautilm_domain_eval_v2_report.json)

---

## 3. Quick Start

```bash
# 1. Run live evaluation against the real .litertlm container
python -m src.evaluator --model ../SautiLM-Forge/kaggle_output/sautilm-jua.litertlm --lexicon ../SautiLM-Forge/data/bbadha_full_lexicon.json

# 2. Run unit tests
python -m unittest discover -s tests -v
```

---

## 4. Founding & Research Team (SautiLM / LosingTek Research)

- **Georges Byona** - Co-Founder, Software Engineer & Edge-AI Researcher (Catholic University of Eastern Africa / LosingTek)
- **Abraham Imani** - Co-Founder, MSc Student & AI/ML Lead (Université Catholique de Bukavu)
- **Jacques Uwonda** - Co-Founder, MSc Student & Software Engineer (Université Shalom de Bunia)

---

## 5. License

Licensed under the **Apache License 2.0** (see [LICENSE](LICENSE)).
