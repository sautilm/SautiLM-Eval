# SautiLM-Eval : Rapport d'Évaluation Empirique Réel (Bbadha Edition)

- **Modèle évalué** : `sautilm-jua.litertlm` (`283.48 MB`, LiteRT-LM `dynamic_wi8_afp32` INT8)
- **Langue cible** : `Bbadha (Kilendu - ISO 639-3: led)`
- **Benchmark** : `bbadha_eval_v1` (15 cas de test réels exécutés via `litert_lm.Engine`)

## Comparaison Empirique (Sans Mock)

| Configuration | Backend | Mots-clés Bbadha | `chrF++` | `BLEU-4` | Latence moyenne (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Paramétrique pur (Zero-RAG) | CPU (XNNPACK) | **24.44%** | 16.85 | 5.15 | 8077.1 |
| Hybride Mobile (Poids + RAG Lexicon + MathSolver) | CPU (XNNPACK) | **55.0%** | **23.07** | **6.44** | 8797.1 |

## Détail par Cas de Test (Pipeline Hybride CPU)

| ID | Module | Catégorie | chrF++ | BLEU-4 | Mots-clés | Latence (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `BBADHA-LEX-001` | `SautiLM-Lex` | `polysemy_and_tones` | 32.52 | 16.69 | 75.0% | 10989.8 |
| `BBADHA-LEX-002` | `SautiLM-Lex` | `tonal_minimal_pair` | 9.85 | 1.37 | 16.67% | 16087.9 |
| `BBADHA-LEX-003` | `SautiLM-Lex` | `glottal_initial` | 28.61 | 6.79 | 100.0% | 3702.1 |
| `BBADHA-LEX-004` | `SautiLM-Lex` | `glottal_initial` | 30.09 | 9.37 | 33.33% | 2445.1 |
| `BBADHA-LEX-005` | `SautiLM-Lex` | `kinship_lexicon` | 13.31 | 2.38 | 66.67% | 10203.5 |
| `BBADHA-LEX-006` | `SautiLM-Lex` | `nature_lexicon` | 26.0 | 9.64 | 66.67% | 7205.6 |
| `BBADHA-LEX-007` | `SautiLM-Lex` | `climate_lexicon` | 18.38 | 2.07 | 33.33% | 11835.3 |
| `BBADHA-LEX-008` | `SautiLM-Lex` | `health_lexicon` | 33.1 | 6.4 | 75.0% | 7678.5 |
| `BBADHA-LEX-009` | `SautiLM-Lex` | `polysemy_and_tones` | 22.82 | 4.73 | 100.0% | 8882.4 |
| `BBADHA-LEX-010` | `SautiLM-Lex` | `family_lexicon` | 15.69 | 2.13 | 50.0% | 10389.3 |
| `BBADHA-EDU-011` | `SautiLM-Edu` | `science_in_bbadha` | 29.53 | 7.82 | 33.33% | 7228.0 |
| `BBADHA-MED-012` | `SautiLM-Med` | `health_in_bbadha` | 11.7 | 4.43 | 66.67% | 11605.3 |
| `BBADHA-MATH-013` | `SautiLM-Math` | `contextual_math` | 27.97 | 11.51 | 0.0% | 3860.9 |
| `BBADHA-IUS-014` | `SautiLM-Ius` | `legal_linguistic_rights` | 28.95 | 7.06 | 33.33% | 10835.7 |
| `BBADHA-VOICE-015` | `SautiLM-Voice` | `phonetics_and_tones` | 17.52 | 4.18 | 75.0% | 9007.1 |
