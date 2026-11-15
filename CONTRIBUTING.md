# Contributing to SautiLM-Eval

We welcome linguists, native Bbadha (Kilendu) speakers, and NLP researchers to contribute new verified evaluation test cases to `SautiLM-Eval`.

## How to Submit Bbadha Test Cases

1. Each test case in `benchmarks/bbadha_eval_v1.jsonl` follows this JSON schema:
   ```json
   {
     "id": "BBADHA-LEX-001",
     "module": "SautiLM-Lex",
     "category": "tonal_minimal_pair",
     "prompt": "Que signifie le mot Bbadha \"Dda\" ?",
     "reference": "Le mot Bbadha **Dda** [ddà] signifie **Eau** en français et **Maji** en kiswahili.",
     "expected_keywords": ["Dda", "Maji", "Eau"]
   }
   ```
2. Run `python -m unittest discover -s tests -v` to verify all evaluation checks pass before opening a Pull Request.
