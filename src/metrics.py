"""Core evaluation metrics for SautiLM-Eval (Bbadha Edition).

Implements standardized NLP and domain metrics tailored for tonal Central-Sudanic
and Bantu languages of Eastern DR Congo (starting with Bbadha / Kilendu):
- chrF++ (Character n-gram F-score + word n-grams, standard for low-resource morphology)
- BLEU-4 (Corpus and sentence-level n-gram precision with brevity penalty)
- Bbadha Tonal & Glottal Normalization Accuracy
- Exact Match (EM) for lexical headwords and deterministic arithmetic
"""

from collections import Counter
import math
import re
import unicodedata
from typing import Dict, List, Sequence, Tuple


def normalize_bbadha(text: str, strip_tones: bool = False) -> str:
    """Normalizes Bbadha orthography (glottal stops, apostrophes, and optional tonal marks)."""
    if not text:
        return ""
    cleaned = text.strip().replace("’", "'").replace("‘", "'").replace("ʼ", "'")
    if strip_tones:
        nfd = unicodedata.normalize("NFD", cleaned)
        cleaned = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
        cleaned = cleaned.replace("'", "")
    else:
        cleaned = unicodedata.normalize("NFC", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower()


def _extract_char_ngrams(text: str, n: int) -> Counter:
    chars = text.replace(" ", "")
    if len(chars) < n:
        return Counter()
    return Counter(chars[i : i + n] for i in range(len(chars) - n + 1))


def _extract_word_ngrams(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def compute_chrf_plus_plus(
    hypothesis: str,
    reference: str,
    char_order: int = 6,
    word_order: int = 2,
    beta: float = 2.0,
) -> float:
    """Computes chrF++ score (0.0 to 100.0) between hypothesis and reference."""
    hyp = hypothesis.strip().lower()
    ref = reference.strip().lower()
    if not hyp and not ref:
        return 100.0
    if not hyp or not ref:
        return 0.0

    beta_sq = beta**2
    f_scores: List[float] = []

    # Character n-grams (1 to char_order)
    for n in range(1, char_order + 1):
        hyp_ngrams = _extract_char_ngrams(hyp, n)
        ref_ngrams = _extract_char_ngrams(ref, n)
        if not hyp_ngrams and not ref_ngrams:
            continue
        overlap = sum((hyp_ngrams & ref_ngrams).values())
        hyp_total = sum(hyp_ngrams.values())
        ref_total = sum(ref_ngrams.values())
        if overlap == 0 or hyp_total == 0 or ref_total == 0:
            f_scores.append(0.0)
        else:
            prec = overlap / hyp_total
            rec = overlap / ref_total
            f_scores.append(((1 + beta_sq) * prec * rec) / (beta_sq * prec + rec))

    # Word n-grams (1 to word_order)
    hyp_words = re.findall(r"\w+", hyp)
    ref_words = re.findall(r"\w+", ref)
    for n in range(1, word_order + 1):
        hyp_ngrams = _extract_word_ngrams(hyp_words, n)
        ref_ngrams = _extract_word_ngrams(ref_words, n)
        if not hyp_ngrams and not ref_ngrams:
            continue
        overlap = sum((hyp_ngrams & ref_ngrams).values())
        hyp_total = sum(hyp_ngrams.values())
        ref_total = sum(ref_ngrams.values())
        if overlap == 0 or hyp_total == 0 or ref_total == 0:
            f_scores.append(0.0)
        else:
            prec = overlap / hyp_total
            rec = overlap / ref_total
            f_scores.append(((1 + beta_sq) * prec * rec) / (beta_sq * prec + rec))

    if not f_scores:
        return 0.0
    return round((sum(f_scores) / len(f_scores)) * 100.0, 2)


def compute_sentence_bleu(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """Computes smoothed sentence-level BLEU score (0.0 to 100.0)."""
    hyp_tokens = re.findall(r"\w+", hypothesis.lower())
    ref_tokens = re.findall(r"\w+", reference.lower())
    if not hyp_tokens or not ref_tokens:
        return 0.0

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        hyp_ngrams = _extract_word_ngrams(hyp_tokens, n)
        ref_ngrams = _extract_word_ngrams(ref_tokens, n)
        if not hyp_ngrams:
            precisions.append(0.0)
            continue
        overlap = sum((hyp_ngrams & ref_ngrams).values())
        total = sum(hyp_ngrams.values())
        # Add-1 smoothing for higher-order n-grams
        if n > 1:
            precisions.append((overlap + 1.0) / (total + 1.0))
        else:
            precisions.append(overlap / total if total > 0 else 0.0)

    if any(p == 0.0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / max_n
    bp = 1.0 if len(hyp_tokens) > len(ref_tokens) else math.exp(1.0 - len(ref_tokens) / len(hyp_tokens))
    return round(bp * math.exp(log_avg) * 100.0, 2)


def evaluate_target_keywords(
    response: str,
    expected_keywords: Sequence[str],
    allow_tonal_normalization: bool = True,
) -> Tuple[float, List[str]]:
    """Checks what percentage of expected Bbadha/domain keywords appear in the model response."""
    if not expected_keywords:
        return 100.0, []

    resp_exact = normalize_bbadha(response, strip_tones=False)
    resp_norm = normalize_bbadha(response, strip_tones=True)
    matched: List[str] = []

    for kw in expected_keywords:
        kw_exact = normalize_bbadha(kw, strip_tones=False)
        kw_norm = normalize_bbadha(kw, strip_tones=True)
        if kw_exact in resp_exact or (allow_tonal_normalization and kw_norm in resp_norm):
            matched.append(kw)

    score = round((len(matched) / len(expected_keywords)) * 100.0, 2)
    return score, matched
