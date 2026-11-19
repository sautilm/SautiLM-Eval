# SautiLM-Eval • Directives d'Évaluation & Continuité

* **Dépôt privé de référence :** `../../sautilm-internal-docs` (`https://github.com/sautilm/sautilm-internal-docs.git`). Vérifier sa présence sur le poste avant de concevoir de nouveaux benchmarks.
* **Nomenclature officielle des modèles évalués :** `SautiLM-Radi` (ultra-rapide / très faible latence « Flash », `< 50 ms`), `SautiLM-Nuru` (vision collaborative `TFLite INT8`), `SautiLM-Jua` (noyau `270M` et spécialités indépendantes : `SautiLM-Jua-STEM`, `SautiLM-Jua-Lex`, `SautiLM-Jua-Med`, `SautiLM-Jua-Agri`, `SautiLM-Jua-Ius`, `SautiLM-Jua-Fin`, `SautiLM-Jua-Civ`), et `SautiLM-Kuu` (`1B` mobile `SautiLM-Kuu-STEM` & `4B` PC `SautiLM-Kuu-Med`, `SautiLM-Kuu-Ius`).
* Chaque domaine doit être évalué avec **cloisonnement strict** : aucune fuite de vocabulaire Bbadha ni de termes juridiques dans les évaluations `STEM`, `Med` ou `Agri`.
