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

    def test_rag_tonal_first_and_all_homographs_retrieved(self):
        mock_entries = [
            {
                "key": "bbe_wrong_tone",
                "bbadha": "bbè",
                "grammar": "adv.",
                "swahili": "faux ton",
                "french": "mauvais ton",
                "examples": [],
            },
            {
                "key": "bbe_lift",
                "bbadha": "bbě",
                "grammar": "v.",
                "swahili": "kuinua",
                "french": "lever ; jeter en l'air",
                "examples": [],
            },
            {
                "key": "bbe_rain",
                "bbadha": "bbě",
                "grammar": "n.",
                "swahili": "wakati wa mvua nyingi",
                "french": "période de grandes pluies",
                "english": "season of heavy rains",
                "examples": [],
            },
            {
                "key": "abba_village",
                "bbadha": "'ábba",
                "grammar": "n.",
                "swahili": "mjini",
                "french": "au village",
                "english": "in the village",
                "examples": [],
            },
        ]
        rag_bbe = build_rag_system_prompt(
            'Que signifie le mot Bbadha "bbě" dans le domaine du climat ?',
            "SautiLM-Lex",
            lexicon_entries=mock_entries,
        )
        self.assertNotIn("mauvais ton", rag_bbe)
        self.assertIn("lever", rag_bbe)
        self.assertIn("période de grandes pluies", rag_bbe)
        self.assertIn("season of heavy rains", rag_bbe)

        rag_abba = build_rag_system_prompt(
            'Que veut dire "\'ábba" au village ?',
            "SautiLM-Lex",
            lexicon_entries=mock_entries,
        )
        self.assertIn("'ábba", rag_abba)
        self.assertIn("au village", rag_abba)

    def test_semantic_accuracy_rejects_hallucinations_and_wrong_math(self):
        from src.evaluator import get_bbadha_benchmark_items, get_domain_abstention_benchmark_items
        from src.metrics import evaluate_semantic_accuracy

        items = {it["id"]: it for it in get_bbadha_benchmark_items()}

        # BBADHA-EDU-011: Soleil = ágó must fail semantic checks even if keywords match
        edu = items["BBADHA-EDU-011"]
        hallucinated = "L'eau (ddà) s'évapore, les pluies (bbě) tombent, Soleil = ágó."
        bad_score, bad_details = evaluate_semantic_accuracy(
            hallucinated, edu["semantic_checks"], edu["forbidden_patterns"]
        )
        self.assertEqual(bad_score, 0.0)
        self.assertFalse(bad_details["semantic_pass"])
        self.assertTrue(len(bad_details["forbidden_hits"]) > 0)

        truthful = "L'eau (ddà) s'évapore et retombe pendant les grandes pluies (bbě). La forêt (ágó) protège le sol."
        ok_score, ok_details = evaluate_semantic_accuracy(
            truthful, edu["semantic_checks"], edu["forbidden_patterns"]
        )
        self.assertEqual(ok_score, 100.0)
        self.assertTrue(ok_details["semantic_pass"])

        # BBADHA-MATH-013: 80000 rejected, 60 parts + 120000 accepted
        m_item = items["BBADHA-MATH-013"]
        wrong_math, _ = evaluate_semantic_accuracy(
            "Au village ('ábba), le capital est 80000 FC.",
            m_item["semantic_checks"],
            m_item["forbidden_patterns"],
        )
        self.assertEqual(wrong_math, 0.0)

        # BBADHA-LEX-008: checks medicinal plant + cough + measure warning
        lex8 = items["BBADHA-LEX-008"]
        incomplete, inc_details = evaluate_semantic_accuracy(
            "bba est une plante médicinale contre la toux.",
            lex8["semantic_checks"],
        )
        self.assertLess(incomplete, 100.0)
        self.assertIn("dosage_precaution", inc_details["failed_checks"])

        abstention_items = get_domain_abstention_benchmark_items()
        self.assertGreaterEqual(len(abstention_items), 7)

    def test_ius_citation_locked_and_domain_eval_v2(self):
        from src.evaluator import (
            evaluate_domain_abstention_predictions,
            get_domain_abstention_benchmark_items,
        )

        items = get_domain_abstention_benchmark_items()
        truthful_preds = {it["id"]: it["reference"] for it in items}
        ok_report = evaluate_domain_abstention_predictions(truthful_preds)
        self.assertEqual(ok_report["metrics"]["mean_semantic_accuracy"], 100.0)
        self.assertEqual(ok_report["benchmark_version"], "domain_eval_v2")

        # Hallucinating a currency amount when DB is empty must score 0.0 on DOMAIN-IUS-CITATION-006
        bad_preds = dict(truthful_preds)
        bad_preds["DOMAIN-IUS-CITATION-006"] = (
            "Selon l'ordonnance 2029 absente de la base, la taxe municipale est de 15000 CDF."
        )
        bad_report = evaluate_domain_abstention_predictions(bad_preds)
        row_006 = next(
            r for r in bad_report["results"] if r["id"] == "DOMAIN-IUS-CITATION-006"
        )
        self.assertEqual(row_006["semantic_accuracy"], 0.0)

        # Giving pediatric amoxicillin mg/prise without weight must score 0.0 on DOMAIN-MED-DOSING-007
        bad_med = dict(truthful_preds)
        bad_med["DOMAIN-MED-DOSING-007"] = (
            "Donnez immédiatement 350 mg / prise sans attendre la pesée du poids en kg."
        )
        bad_med_report = evaluate_domain_abstention_predictions(bad_med)
        row_007 = next(
            r for r in bad_med_report["results"] if r["id"] == "DOMAIN-MED-DOSING-007"
        )
        self.assertEqual(row_007["semantic_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
