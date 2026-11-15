# SautiLM-Eval : Rapport d'Évaluation Officiel (Bbadha Edition)

- **Modèle évalué** : `SautiLM Jua (92.48 MB LiteRT INT4)`
- **Langue cible** : `Bbadha (Kilendu - ISO 639-3: led)`
- **Benchmark** : `bbadha_eval_v1` (15 cas de test vérifiés)

## Résultats Globaux

| Métrique | Score | Description |
| :--- | :--- | :--- |
| **Exactitude Lexicale & Tonale Bbadha** | **100.0%** | Présence exacte des termes Bbadha, Swahili et Français attendus |
| **chrF++ (Morphologie & Caractères)** | **83.29 / 100** | Standard scientifique pour les langues africaines à tons |
| **BLEU-4** | **70.39 / 100** | Précision n-grammes de mots par rapport à la référence humaine |

## Détail par Cas de Test

| ID | Module | Catégorie | chrF++ | BLEU-4 | Mots-clés Bbadha |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `BBADHA-LEX-001` | `SautiLM-Lex` | `tonal_minimal_pair` | 96.02 | 85.83 | 100.0% |
| `BBADHA-LEX-002` | `SautiLM-Lex` | `tonal_minimal_pair` | 83.88 | 59.78 | 100.0% |
| `BBADHA-LEX-003` | `SautiLM-Lex` | `glottal_initial` | 85.37 | 79.08 | 100.0% |
| `BBADHA-LEX-004` | `SautiLM-Lex` | `implosive_consonant` | 86.36 | 59.78 | 100.0% |
| `BBADHA-LEX-005` | `SautiLM-Lex` | `core_vocabulary` | 84.24 | 72.35 | 100.0% |
| `BBADHA-LEX-006` | `SautiLM-Lex` | `core_vocabulary` | 88.97 | 72.22 | 100.0% |
| `BBADHA-LEX-007` | `SautiLM-Lex` | `core_vocabulary` | 100.0 | 100.0 | 100.0% |
| `BBADHA-LEX-008` | `SautiLM-Lex` | `core_vocabulary` | 100.0 | 100.0 | 100.0% |
| `BBADHA-LEX-009` | `SautiLM-Lex` | `core_vocabulary` | 100.0 | 100.0 | 100.0% |
| `BBADHA-LEX-010` | `SautiLM-Lex` | `core_vocabulary` | 100.0 | 100.0 | 100.0% |
| `BBADHA-EDU-011` | `SautiLM-Edu` | `science_in_bbadha` | 57.5 | 22.52 | 100.0% |
| `BBADHA-MED-012` | `SautiLM-Med` | `health_in_bbadha` | 55.99 | 34.96 | 100.0% |
| `BBADHA-MATH-013` | `SautiLM-Math` | `contextual_math` | 80.43 | 65.49 | 100.0% |
| `BBADHA-IUS-014` | `SautiLM-Ius` | `legal_linguistic_rights` | 75.16 | 71.97 | 100.0% |
| `BBADHA-VOICE-015` | `SautiLM-Voice` | `phonetics_and_tones` | 55.39 | 31.88 | 100.0% |
