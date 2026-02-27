import re
from difflib import get_close_matches
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ALIAS_MAP: Dict[str, str] = {
    "cold": "Runny Nose",
    "stomach pain": "Abdominal Pain",
    "stomach ache": "Abdominal Pain",
    "belly pain": "Abdominal Pain",
    "abdomen pain": "Abdominal Pain",
    "shortness of breath": "Shortness Breath",
    "breathlessness": "Shortness Breath",
    "joint pain": "JointPain",
    "weakness": "Fatigue",
    "feeling weak": "Fatigue",
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

    for alias, target in ALIAS_MAP.items():
        canonical = feature_map.get(_normalize_symptom(target))
        if canonical is not None:
            phrase_map[_normalize_phrase(alias)] = canonical

    return phrase_map


def resolve_text_symptoms(raw_text: str, feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    if not raw_text.strip():
        return [], []
    phrase_map = _build_phrase_map(feature_cols)

    resolved: List[str] = []
    text_norm = _normalize_phrase(raw_text)
    occupied = [False] * len(text_norm)
    matches: List[Tuple[int, str]] = []

    phrases = sorted(phrase_map.keys(), key=len, reverse=True)
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
    chunks = [c.strip() for c in re.split(r"[,\n;]+|\band\b", raw_text, flags=re.IGNORECASE) if c.strip()]
    for chunk in chunks:
        chunk_norm = _normalize_phrase(chunk)
        if not chunk_norm:
            continue
        has_match = False
        for phrase in phrases:
            if re.search(rf"\b{re.escape(phrase)}\b", chunk_norm):
                has_match = True
                break
        if not has_match:
            unmatched.append(chunk)

    return resolved, unmatched


def suggest_closest_symptoms(unmatched: List[str], feature_cols: List[str]) -> Dict[str, str]:
    suggestions: Dict[str, str] = {}
    phrase_map = _build_phrase_map(feature_cols)
    keys = list(phrase_map.keys())

    for item in unmatched:
        cleaned = _normalize_phrase(item)
        if not cleaned:
            continue
        matches = get_close_matches(cleaned, keys, n=1, cutoff=0.6)
        if matches:
            suggestions[item] = phrase_map[matches[0]]

    return suggestions
