# SautiLM-Eval

Suite d'évaluation scientifique et empirique hors-ligne pour l'écosystème **SautiLM** (`SautiLM-Radi` / `SautiLM-Nuru` / `SautiLM-Jua` / `SautiLM-Kuu`), sans mocks ni gonflage de métriques.

## Méthodologie & Benchmarks

1. **`bbadha_eval_v1` (15 cas stricts : Bbadha / Kilendu — ISO 639-3 : `led`)**
   - Évalue la distinction tonale (`ddà` vs `dà`, `bbě` vs `bbù`, `'ábba` vs `bba`), la génération éducative/clinique/juridique, l'arithmétique contextuelle et les garde-fous.
   - Tous les 15 cas incluent des `semantic_checks` regex stricts et des `forbidden_patterns` (rejetant toute hallucination ou erreur mathématique même si des mots-clés lexicaux apparaissent).
   - Comparaison systématique **Paramétrique pur (Zero-RAG)** vs **Hybride Mobile (Poids + RAG Lexicon tonal + Solveur mathématique)** via `litert_lm.Engine` CPU (XNNPACK).

2. **`domain_eval_v2` (7 cas multi-domaines d'abstention, verrouillage citationnel & calcul)**
   - Exécuté en direct via `evaluate_domain_v2_live()` sur les véritables artefacts compilés de chaque module (sans substitution silencieux d'un modèle par un autre) :
     - `.litertlm` (`270M` INT8 / `1B` INT4) via `litert_lm.Engine`
     - `.gguf` (`4B Q4_K_M` Workstation) via `llama-cli` (`llama.cpp`)
   - Rapporte séparément **`raw_weights`** (poids bruts sans garde-fous) et **`guarded_production`** (avec court-circuits déterministes `apply_sovereign_safety_short_circuit`).

---

## Résultats Empiriques Mesurés (Audit v3 — 19 novembre 2026)

### 1. `bbadha_eval_v1` (`15/15` cas avec validation sémantique stricte)

| Modèle & Format | Configuration | Mots-clés | Précision sémantique stricte | `chrF++` | `BLEU-4` | Latence moy. (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`SautiLM-Jua`** (`283.48 MB` INT8) | Paramétrique pur (Zero-RAG) | `24.44%` | `5.56%` | `16.85` | `5.15` | `8077.1` |
| **`SautiLM-Jua`** (`283.48 MB` INT8) | Hybride Mobile (RAG + Solveur) | **`55.00%`** | **`47.78%`** | **`23.07`** | `6.44` | `8797.1` |
| **`SautiLM-Jua-Lex`** (`283.48 MB` INT8) | Paramétrique pur (Zero-RAG) | `32.78%` | `11.11%` | `22.17` | `6.23` | `13017.1` |
| **`SautiLM-Jua-Lex`** (`283.48 MB` INT8) | Hybride Mobile (RAG + Solveur) | `38.33%` | `15.56%` | `23.05` | `9.43` | `6460.9` |

### 2. `domain_eval_v2` (Inférence Réelle Multi-Modèles `.litertlm` + `4B GGUF`)

| Mode d'évaluation | Précision mots-clés | Précision sémantique stricte | Preuve observée |
| :--- | :---: | :---: | :--- |
| **`raw_weights`** *(Poids bruts sans wrapper)* | `21.43%` | `21.43%` | `SautiLM-Kuu-Med` (`4B GGUF`, `llama-cli`) obtient **`100.0%`** au refus pédiatrique sans poids ; `SautiLM-Jua-Med` (`270M`) obtient `50.0%` (`700 mg/jour` exact) ; les `270M` bruts restent insuffisants en abstention ouverte sans RAG. |
| **`guarded_production`** *(Court-circuits production)* | **`57.14%`** | **`50.00%`** | Blocage déterministe `100.0%` sur `non_plant_object` (`Nuru-Agri`), verrou de citation SQLite vide (`Ius`) et absence de poids pédiatrique (`AfyaSpeak` / `Kuu-Med`). Les calculs finaux sont confiés aux calculateurs déterministes (`SautiLM-SDK` / `<calc>` AST). |

---

## Exécution locale

```bash
# Tests unitaires (9/9)
python -m unittest discover -s tests -v

# Évaluation Bbadha live sur LiteRT-LM CPU
python -m src.evaluator --model ../SautiLM-Forge/kaggle_output/sautilm-jua.litertlm --lexicon ../SautiLM-Forge/data/bbadha_full_lexicon.json --no-gpu
```

Rapports JSON/Markdown complets dans [`reports/`](reports/).
