# SautiLM-Eval (Bbadha Edition)

**SautiLM-Eval** is the official scientific evaluation benchmark and metrics suite for the **SautiLM** family of offline Edge-AI language models (**Jua**, **Nuru**, **Wazo**, **Moto**, **Kuu**, **Mali**), engineered in **Bunia, Ituri, Democratic Republic of the Congo** by **LosingTek Research** for **SautiLM**.

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

## 2. Baseline Results : `SautiLM Jua` (`92.48 MB` LiteRT-LM INT4)

| Model | Language | Benchmark | Bbadha Keyword Accuracy | Mean `chrF++` | Mean `BLEU-4` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`SautiLM Jua` (`92.48 MB`)** | Bbadha (`led`) | `bbadha_eval_v1` | **100.0%** | **84.42 / 100** | **67.85 / 100** |

Full machine-readable and human-readable reports are generated in:
- [`reports/sautilm_jua_bbadha_report.json`](reports/sautilm_jua_bbadha_report.json)
- [`reports/sautilm_jua_bbadha_report.md`](reports/sautilm_jua_bbadha_report.md)

---

## 3. Quick Start

```bash
# 1. Run the Bbadha evaluation suite and regenerate reports
python -m src.evaluator

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
