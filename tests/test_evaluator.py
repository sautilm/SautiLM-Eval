import unittest
from src.evaluator import (
    build_benchmark_file,
    build_rag_system_prompt,
    evaluate_predictions,
    get_bbadha_benchmark_items,
)
from src.metrics import (
    compute_chrf_plus_plus,
    compute_sentence_bleu,
    evaluate_target_keywords,
    normalize_bbadha,
)


class TestSautiLMEvalBbadha(unittest.TestCase):
    def test_normalize_bbadha_handles_tones_and_glottal(self):
        self.assertEqual(normalize_bbadha("’Ábba", strip_tones=True), "abba")
        self.assertEqual(normalize_bbadha("Dyà", strip_tones=True), "dya")
        self.assertEqual(normalize_bbadha("Dda", strip_tones=False), "dda")

    def test_chrf_and_bleu_metrics(self):
        ref = "Le mot Bbadha Dda signifie Eau en français et Maji en kiswahili."
        hyp = "Le mot Bbadha Dda signifie Eau en français et Maji en kiswahili."
        self.assertEqual(compute_chrf_plus_plus(hyp, ref), 100.0)
        self.assertEqual(compute_sentence_bleu(hyp, ref), 100.0)

    def test_evaluate_target_keywords(self):
        resp = "En Bbadha, 'Ábba veut dire en ville (mjini)."
        score, matched = evaluate_target_keywords(resp, ["'Ábba", "mjini", "ville"])
        self.assertEqual(score, 100.0)
        self.assertEqual(len(matched), 3)

    def test_benchmark_and_evaluator_contract(self):
        items = get_bbadha_benchmark_items()
        self.assertEqual(len(items), 15)
        b_path = build_benchmark_file()
        self.assertTrue(b_path.exists())

        sample_preds = {item["id"]: item["reference"] for item in items}
        report = evaluate_predictions(sample_preds, model_name="Reference Self-Check")
        self.assertEqual(report["metrics"]["mean_keyword_accuracy"], 100.0)
        self.assertEqual(report["metrics"]["mean_chrf_pp"], 100.0)

    def test_build_rag_system_prompt_injects_dictionary_and_math(self):
        mock_entries = [
            {
                "key": "ddà_221",
                "bbadha": "ddà",
                "grammar": "n.",
                "swahili": "maji, kuwa refu",
                "french": "l'eau, la rivière",
                "examples": [],
            }
        ]
        rag_prompt = build_rag_system_prompt(
            'Que signifie le mot Bbadha "ddà" ?',
            "SautiLM-Lex",
            lexicon_entries=mock_entries,
        )
        self.assertIsNotNone(rag_prompt)
        self.assertIn("DICTIONNAIRE BBADHA VERIFIE", rag_prompt)
        self.assertIn("maji", rag_prompt)

        math_prompt = build_rag_system_prompt(
            "Dans une coopérative, 20 membres ont 3 parts à 2000 FC.",
            "SautiLM-Math",
            lexicon_entries=[],
        )
        self.assertIsNotNone(math_prompt)
        self.assertIn("120000", math_prompt)


if __name__ == "__main__":
    unittest.main()
