# SautiLM-Eval : Rapport d'Évaluation Officiel (Bbadha Edition)

- **Modèle évalué** : `SautiLM Jua (92.48 MB LiteRT INT4)`
- **Langue cible** : `Bbadha (Kilendu - ISO 639-3: led)`
- **Benchmark** : `bbadha_eval_v1` (15 cas de test vérifiés issus de `a.json` - `z.json`)

## Résultats Globaux

| Métrique | Score | Description |
| :--- | :--- | :--- |
| **Exactitude Lexicale & Tonale Bbadha** | **100.0%** | Présence exacte des termes Bbadha, Swahili et Français attendus |
| **chrF++ (Morphologie & Caractères)** | **97.13 / 100** | Standard scientifique pour les langues africaines à tons |
| **BLEU-4** | **95.75 / 100** | Précision n-grammes de mots par rapport à la référence humaine |

## Détail par Cas de Test

| ID | Module | Catégorie | chrF++ | BLEU-4 | Mots-clés Bbadha |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `BBADHA-LEX-001` | `SautiLM-Lex` | `polysemy_and_tones` | 100.0 | 100.0 | 100.0% |
| `BBADHA-LEX-002` | `SautiLM-Lex` | `tonal_minimal_pair` | 85.02 | 78.92 | 100.0% |
| `BBADHA-LEX-003` | `SautiLM-Lex` | `glottal_initial` | 100.0 | 100.0 | 100.0% |
| `BBADHA-LEX-004` | `SautiLM-Lex` | `glottal_initial` | 100.0 | 100.0 | 100.0% |
| `BBADHA-LEX-005` | `SautiLM-Lex` | `kinship_lexicon` | 100.0 | 100.0 | 100.0% |
| `BBADHA-LEX-006` | `SautiLM-Lex` | `nature_lexicon` | 100.0 | 100.0 | 100.0% |
| `BBADHA-LEX-007` | `SautiLM-Lex` | `climate_lexicon` | 100.0 | 100.0 | 100.0% |
| `BBADHA-LEX-008` | `SautiLM-Lex` | `health_lexicon` | 80.26 | 71.05 | 100.0% |
| `BBADHA-LEX-009` | `SautiLM-Lex` | `polysemy_and_tones` | 100.0 | 100.0 | 100.0% |
| `BBADHA-LEX-010` | `SautiLM-Lex` | `family_lexicon` | 100.0 | 100.0 | 100.0% |
| `BBADHA-EDU-011` | `SautiLM-Edu` | `science_in_bbadha` | 91.65 | 86.28 | 100.0% |
| `BBADHA-MED-012` | `SautiLM-Med` | `health_in_bbadha` | 100.0 | 100.0 | 100.0% |
| `BBADHA-MATH-013` | `SautiLM-Math` | `contextual_math` | 100.0 | 100.0 | 100.0% |
| `BBADHA-IUS-014` | `SautiLM-Ius` | `legal_linguistic_rights` | 100.0 | 100.0 | 100.0% |
| `BBADHA-VOICE-015` | `SautiLM-Voice` | `phonetics_and_tones` | 100.0 | 100.0 | 100.0% |
