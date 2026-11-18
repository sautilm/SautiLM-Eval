# SautiLM-Eval : Rapport d'Évaluation Empirique Réel (Bbadha Edition)

- **Modèle évalué** : `sautilm-jua.litertlm` (`283.48 MB`, LiteRT-LM `dynamic_wi8_afp32` INT8)
- **Langue cible** : `Bbadha (Kilendu - ISO 639-3: led)`
- **Benchmark** : `bbadha_eval_v1` (15 cas de test réels exécutés via `litert_lm.Engine`)

## Comparaison Empirique (Sans Mock)

| Configuration | Backend | Mots-clés Bbadha | `chrF++` | `BLEU-4` | Latence moyenne (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Paramétrique pur (Zero-RAG) | CPU (XNNPACK) | **31.67%** | 20.85 | 4.91 | 1913.2 |
| Hybride Mobile (Poids + RAG Lexicon + MathSolver) | CPU (XNNPACK) | **58.89%** | **27.6** | **9.06** | 2602.1 |

## Détail par Cas de Test (Pipeline Hybride CPU)

| ID | Module | Catégorie | chrF++ | BLEU-4 | Mots-clés | Latence (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `BBADHA-LEX-001` | `SautiLM-Lex` | `polysemy_and_tones` | 20.92 | 4.15 | 25.0% | 2498.6 |
| `BBADHA-LEX-002` | `SautiLM-Lex` | `tonal_minimal_pair` | 21.24 | 9.53 | 66.67% | 1848.9 |
| `BBADHA-LEX-003` | `SautiLM-Lex` | `glottal_initial` | 25.67 | 2.68 | 100.0% | 1371.6 |
| `BBADHA-LEX-004` | `SautiLM-Lex` | `glottal_initial` | 31.84 | 8.77 | 100.0% | 1363.7 |
| `BBADHA-LEX-005` | `SautiLM-Lex` | `kinship_lexicon` | 34.67 | 13.32 | 66.67% | 1699.1 |
| `BBADHA-LEX-006` | `SautiLM-Lex` | `nature_lexicon` | 27.29 | 18.07 | 66.67% | 1982.8 |
| `BBADHA-LEX-007` | `SautiLM-Lex` | `climate_lexicon` | 28.53 | 5.73 | 66.67% | 4521.4 |
| `BBADHA-LEX-008` | `SautiLM-Lex` | `health_lexicon` | 16.58 | 3.34 | 25.0% | 2063.9 |
| `BBADHA-LEX-009` | `SautiLM-Lex` | `polysemy_and_tones` | 22.9 | 10.02 | 75.0% | 2102.8 |
| `BBADHA-LEX-010` | `SautiLM-Lex` | `family_lexicon` | 45.52 | 12.36 | 100.0% | 3574.9 |
| `BBADHA-EDU-011` | `SautiLM-Edu` | `science_in_bbadha` | 16.54 | 2.28 | 66.67% | 5025.4 |
| `BBADHA-MED-012` | `SautiLM-Med` | `health_in_bbadha` | 21.73 | 6.44 | 66.67% | 3527.8 |
| `BBADHA-MATH-013` | `SautiLM-Math` | `contextual_math` | 37.44 | 12.69 | 0.0% | 2889.9 |
| `BBADHA-IUS-014` | `SautiLM-Ius` | `legal_linguistic_rights` | 42.68 | 21.51 | 33.33% | 2425.7 |
| `BBADHA-VOICE-015` | `SautiLM-Voice` | `phonetics_and_tones` | 20.4 | 4.96 | 25.0% | 2134.5 |
