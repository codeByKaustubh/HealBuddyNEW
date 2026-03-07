import re
from difflib import get_close_matches
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from spellchecker import SpellChecker


ALIAS_MAP: Dict[str, str] = {
    "cold": "Runny Nose",
    "stomach pain": "Abdominal Pain",
    "stomach ache": "Abdominal Pain",
    "belly pain": "Abdominal Pain",
    "abdomen pain": "Abdominal Pain",
    "shortness of breath": "Shortness Breath",
    "breathlessness": "Shortness Breath",
    "joint pain": "JointPain",
    "pain in body": "Body Ache",
    "body pain": "Body Ache",
    "whole body pain": "Body Ache",
    "body ache": "Body Ache",
}

_CONTEXT_PREFIXES: List[str] = [
    "i am suffering from",
    "i m suffering from",
    "im suffering from",
    "suffering from",
    "i have",
    "i am having",
    "i m having",
    "im having",
    "symptoms like",
    "like",
]

_NOISE_CHUNKS = {"etc", "etcetera", "and etc", "and etcetera"}
_SPELLCHECK_SKIP = {"etc", "etcetera"}
_SIGNATURE_STOPWORDS = {
    "i",
    "am",
    "m",
    "im",
    "have",
    "having",
    "from",
    "with",
    "in",
    "on",
    "at",
    "my",
    "the",
    "a",
    "an",
    "of",
    "is",
    "are",
    "feel",
    "feeling",
    "suffering",
}


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def get_target_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.strip().lower() == "disease":
            return col
    raise ValueError("No target column named 'disease' found (case-insensitive).")


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    target_col = get_target_column(df)
    return [c for c in df.columns if c != target_col]


def make_input_vector(feature_cols: List[str], selected: List[str]) -> np.ndarray:
    row = np.zeros(len(feature_cols), dtype=int)
    selected_set = set(selected)
    for i, feature in enumerate(feature_cols):
        if feature in selected_set:
            row[i] = 1
    return row


def _normalize_symptom(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _build_phrase_map(feature_cols: List[str]) -> Dict[str, str]:
    feature_map: Dict[str, str] = {_normalize_symptom(c): c for c in feature_cols}
    phrase_map: Dict[str, str] = {_normalize_phrase(c): c for c in feature_cols}

    for feature in feature_cols:
        canonical = feature
        phrase = _normalize_phrase(feature)
        tokens = phrase.split()

        if len(tokens) == 2:
            phrase_map[f"{tokens[1]} {tokens[0]}"] = canonical

        if tokens and tokens[0] in {"high", "low"} and len(tokens) >= 2:
            rest = " ".join(tokens[1:])
            phrase_map[f"{rest} {tokens[0]}"] = canonical

        if "pain" in tokens and len(tokens) >= 2:
            others = [t for t in tokens if t != "pain"]
            base = " ".join(others).strip()
            if base:
                phrase_map[f"pain in {base}"] = canonical
                phrase_map[f"{base} pain"] = canonical

        if "ache" in tokens and len(tokens) >= 2:
            others = [t for t in tokens if t != "ache"]
            base = " ".join(others).strip()
            if base:
                phrase_map[f"pain in {base}"] = canonical
                phrase_map[f"{base} pain"] = canonical
                phrase_map[f"{base} ache"] = canonical

    for alias, target in ALIAS_MAP.items():
        canonical = feature_map.get(_normalize_symptom(target))
        if canonical is not None:
            phrase_map[_normalize_phrase(alias)] = canonical

    return phrase_map


def _normalize_token(token: str) -> str:
    t = token.lower().strip()
    if t == "pains":
        return "pain"
    if t == "aches":
        return "ache"
    if t.endswith("ing") and len(t) > 5:
        t = t[:-3]
    return t


def _build_signature(text_norm: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", text_norm)
    norm_tokens = [_normalize_token(t) for t in tokens]
    filtered = [t for t in norm_tokens if t and t not in _SIGNATURE_STOPWORDS]
    if not filtered:
        return ""
    return " ".join(sorted(filtered))


def _build_signature_map(phrase_map: Dict[str, str]) -> Dict[str, str]:
    signature_map: Dict[str, str] = {}
    for phrase, canonical in phrase_map.items():
        sig = _build_signature(phrase)
        if sig and sig not in signature_map:
            signature_map[sig] = canonical
    return signature_map


def _strip_context_prefix(text_norm: str) -> str:
    cleaned = text_norm.strip()
    for prefix in _CONTEXT_PREFIXES:
        prefix_norm = _normalize_phrase(prefix)
        if cleaned == prefix_norm:
            return ""
        if cleaned.startswith(prefix_norm + " "):
            cleaned = cleaned[len(prefix_norm) + 1 :].strip()
            break
    return cleaned


def _build_spellchecker(feature_cols: List[str]) -> SpellChecker:
    checker = SpellChecker(distance=2)
    domain_words = set()

    for symptom in feature_cols:
        for token in re.findall(r"[a-z0-9]+", symptom.lower()):
            if token:
                domain_words.add(token)

    for alias, target in ALIAS_MAP.items():
        for token in re.findall(r"[a-z0-9]+", alias.lower()):
            if token:
                domain_words.add(token)
        for token in re.findall(r"[a-z0-9]+", target.lower()):
            if token:
                domain_words.add(token)

    checker.word_frequency.load_words(domain_words)
    return checker


def _correct_text_dictionary(
    raw_text: str, spellchecker: SpellChecker
) -> Tuple[str, Dict[str, str]]:
    corrections: Dict[str, str] = {}
    corrected_parts: List[str] = []
    cursor = 0

    for match in re.finditer(r"[A-Za-z]+", raw_text):
        start, end = match.span()
        token = match.group(0)
        lowered = token.lower()

        corrected_parts.append(raw_text[cursor:start])
        cursor = end

        if len(lowered) <= 2 or lowered in spellchecker:
            corrected_parts.append(token)
            continue
        if lowered in _SPELLCHECK_SKIP:
            corrected_parts.append(token)
            continue

        candidate = spellchecker.correction(lowered)
        if candidate and candidate != lowered:
            corrections[token] = candidate
            corrected_parts.append(candidate)
        else:
            corrected_parts.append(token)

    corrected_parts.append(raw_text[cursor:])
    return "".join(corrected_parts), corrections


def resolve_text_symptoms_with_spellcheck(
    raw_text: str, feature_cols: List[str]
) -> Tuple[List[str], List[str], Dict[str, str]]:
    if not raw_text.strip():
        return [], [], {}

    phrase_map = _build_phrase_map(feature_cols)
    signature_map = _build_signature_map(phrase_map)
    spellchecker = _build_spellchecker(feature_cols)
    corrected_text, corrections = _correct_text_dictionary(raw_text, spellchecker)

    resolved: List[str] = []
    text_norm = _normalize_phrase(corrected_text)
    occupied = [False] * len(text_norm)
    matches: List[Tuple[int, str]] = []

    phrases = sorted(phrase_map.keys(), key=len, reverse=True)
    signature_keys = list(signature_map.keys())
    for phrase in phrases:
        canonical = phrase_map[phrase]
        pattern = rf"\b{re.escape(phrase)}\b"
        for m in re.finditer(pattern, text_norm):
            if any(occupied[m.start() : m.end()]):
                continue
            for i in range(m.start(), m.end()):
                occupied[i] = True
            matches.append((m.start(), canonical))

    for _, canonical in sorted(matches, key=lambda x: x[0]):
        if canonical not in resolved:
            resolved.append(canonical)

    unmatched: List[str] = []
    chunks = [c.strip() for c in re.split(r"[,\n;]+|\band\b", corrected_text, flags=re.IGNORECASE) if c.strip()]
    for chunk in chunks:
        chunk_norm = _strip_context_prefix(_normalize_phrase(chunk))
        if not chunk_norm:
            continue
        if chunk_norm in _NOISE_CHUNKS:
            continue

        has_match = False
        chunk_sig = _build_signature(chunk_norm)
        if chunk_sig and chunk_sig in signature_map:
            canonical = signature_map[chunk_sig]
            if canonical not in resolved:
                resolved.append(canonical)
            has_match = True

        for phrase in phrases:
            if re.search(rf"\b{re.escape(phrase)}\b", chunk_norm):
                has_match = True
                break
        if not has_match:
            if chunk_sig and signature_keys:
                fuzzy_sig = get_close_matches(chunk_sig, signature_keys, n=1, cutoff=0.78)
                if fuzzy_sig:
                    canonical = signature_map[fuzzy_sig[0]]
                    if canonical not in resolved:
                        resolved.append(canonical)
                    has_match = True
        if not has_match:
            fuzzy = get_close_matches(chunk_norm, phrases, n=1, cutoff=0.72)
            if fuzzy:
                canonical = phrase_map[fuzzy[0]]
                if canonical not in resolved:
                    resolved.append(canonical)
                has_match = True
        if not has_match:
            unmatched.append(chunk)

    return resolved, unmatched, corrections


def resolve_text_symptoms(raw_text: str, feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    resolved, unmatched, _ = resolve_text_symptoms_with_spellcheck(raw_text, feature_cols)
    return resolved, unmatched


def suggest_closest_symptoms(unmatched: List[str], feature_cols: List[str]) -> Dict[str, str]:
    suggestions: Dict[str, str] = {}
    phrase_map = _build_phrase_map(feature_cols)
    signature_map = _build_signature_map(phrase_map)
    keys = list(phrase_map.keys())
    signature_keys = list(signature_map.keys())

    for item in unmatched:
        cleaned = _strip_context_prefix(_normalize_phrase(item))
        if not cleaned:
            continue
        if cleaned in _NOISE_CHUNKS:
            continue

        sig = _build_signature(cleaned)
        if sig in signature_map:
            suggestions[item] = signature_map[sig]
            continue
        if sig and signature_keys:
            sig_matches = get_close_matches(sig, signature_keys, n=1, cutoff=0.7)
            if sig_matches:
                suggestions[item] = signature_map[sig_matches[0]]
                continue

        matches = get_close_matches(cleaned, keys, n=1, cutoff=0.6)
        if matches:
            suggestions[item] = phrase_map[matches[0]]

    return suggestions
