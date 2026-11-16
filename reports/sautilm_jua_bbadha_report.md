# SautiLM-Eval : Rapport d'Évaluation Empirique Réel (Bbadha Edition)

- **Modèle évalué** : `sautilm-jua.litertlm` (`272.36 MB`, LiteRT-LM `dynamic_wi8_afp32` INT8)
- **Langue cible** : `Bbadha (Kilendu - ISO 639-3: led)`
- **Benchmark** : `bbadha_eval_v1` (15 cas de test réels exécutés via `litert_lm.Engine`)

## Comparaison Empirique (Sans Mock)

| Configuration | Backend | Mots-clés Bbadha | `chrF++` | `BLEU-4` | Latence moyenne (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Paramétrique pur (Zero-RAG) | CPU (XNNPACK) | **38.89%** | 24.69 | 4.81 | 4235.0 |
| Hybride Mobile (Poids + RAG Lexicon + MathSolver) | CPU (XNNPACK) | **68.33%** | **34.15** | **9.58** | 4267.2 |
| Hybride Mobile (Poids + RAG Lexicon + MathSolver) | GPU (WebGPU) | **68.33%** | 31.4 | 7.32 | 2841.4 |

## Détail par Cas de Test (Pipeline Hybride CPU)

| ID | Module | Catégorie | chrF++ | BLEU-4 | Mots-clés | Latence (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `BBADHA-LEX-001` | `SautiLM-Lex` | `polysemy_and_tones` | 32.63 | 5.18 | 100.0% | 3849.0 |
| `BBADHA-LEX-002` | `SautiLM-Lex` | `tonal_minimal_pair` | 25.82 | 3.96 | 66.67% | 5133.8 |
| `BBADHA-LEX-003` | `SautiLM-Lex` | `glottal_initial` | 50.28 | 33.41 | 100.0% | 3029.9 |
| `BBADHA-LEX-004` | `SautiLM-Lex` | `glottal_initial` | 34.13 | 12.18 | 100.0% | 2095.0 |
| `BBADHA-LEX-005` | `SautiLM-Lex` | `kinship_lexicon` | 37.48 | 4.98 | 100.0% | 4159.1 |
| `BBADHA-LEX-006` | `SautiLM-Lex` | `nature_lexicon` | 31.93 | 7.59 | 66.67% | 4891.7 |
| `BBADHA-LEX-007` | `SautiLM-Lex` | `climate_lexicon` | 22.54 | 2.56 | 33.33% | 6220.5 |
| `BBADHA-LEX-008` | `SautiLM-Lex` | `health_lexicon` | 20.41 | 2.75 | 25.0% | 5940.2 |
| `BBADHA-LEX-009` | `SautiLM-Lex` | `polysemy_and_tones` | 23.59 | 6.3 | 50.0% | 3050.7 |
| `BBADHA-LEX-010` | `SautiLM-Lex` | `family_lexicon` | 46.84 | 12.73 | 100.0% | 3800.7 |
| `BBADHA-EDU-011` | `SautiLM-Edu` | `science_in_bbadha` | 38.29 | 8.49 | 100.0% | 4569.6 |
| `BBADHA-MED-012` | `SautiLM-Med` | `health_in_bbadha` | 40.19 | 11.52 | 66.67% | 4943.8 |
| `BBADHA-MATH-013` | `SautiLM-Math` | `contextual_math` | 32.74 | 5.75 | 0.0% | 2703.9 |
| `BBADHA-IUS-014` | `SautiLM-Ius` | `legal_linguistic_rights` | 57.92 | 24.43 | 66.67% | 4670.1 |
| `BBADHA-VOICE-015` | `SautiLM-Voice` | `phonetics_and_tones` | 17.41 | 1.88 | 50.0% | 4950.6 |
