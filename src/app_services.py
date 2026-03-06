from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import LabelEncoder

from src.config import RANDOM_STATE
from src.data import load_data as _load_data
from src.models import train_models as _train_models


DISEASE_CONTENT: Dict[str, Dict[str, str]] = {
    "Typhoid": {
        "description": "Typhoid fever is a bacterial infection caused by Salmonella Typhi.",
        "causes": "Usually spread through contaminated food or water and poor sanitation.",
        "symptoms": "Sustained fever, weakness, abdominal pain, headache, and loss of appetite.",
        "doctor": "Seek care promptly for persistent high fever, confusion, dehydration, or severe abdominal pain.",
        "prevention": "Safe drinking water, proper handwashing, cooked food hygiene, and vaccination where advised.",
    },
    "Flu": {
        "description": "Influenza is a viral respiratory infection that can spread rapidly.",
        "causes": "Influenza viruses transmitted by droplets or contaminated surfaces.",
        "symptoms": "Fever, cough, sore throat, body aches, fatigue, and chills.",
        "doctor": "See a doctor for breathing difficulty, chest pain, prolonged fever, or high-risk conditions.",
        "prevention": "Annual flu vaccine, hand hygiene, masking during outbreaks, and staying home when sick.",
    },
    "Asthma": {
        "description": "Asthma is a chronic condition causing airway inflammation and narrowing.",
        "causes": "Triggered by allergens, smoke, infection, cold air, or exercise in susceptible people.",
        "symptoms": "Wheezing, shortness of breath, chest tightness, and nighttime cough.",
        "doctor": "Urgent care is needed for severe breathlessness, bluish lips, or poor response to inhaler.",
        "prevention": "Avoid triggers, use preventer medication, and maintain an asthma action plan.",
    },
}


@st.cache_data
def load_data_cached(path: str) -> pd.DataFrame:
    return _load_data(path)


@st.cache_resource
def train_models_cached(df: pd.DataFrame):
    return _train_models(df, random_state=RANDOM_STATE)


def get_confidence_and_risk(probability: float) -> Tuple[str, str]:
    if probability >= 0.75:
        return "High confidence", "High likelihood"
    if probability >= 0.45:
        return "Moderate confidence", "Moderate likelihood"
    return "Low confidence", "Low likelihood"


def compute_similarity_probabilities(
    x_row: np.ndarray,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    X_arr = X_train.values.astype(int)
    x = x_row.astype(int).reshape(1, -1)

    intersections = (X_arr * x).sum(axis=1).astype(float)
    unions = ((X_arr + x) > 0).sum(axis=1).astype(float)
    sim = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)

    class_scores = np.zeros(n_classes, dtype=float)
    for cls in range(n_classes):
        cls_mask = y_train == cls
        if not np.any(cls_mask):
            continue
        cls_sim = sim[cls_mask]
        class_scores[cls] = 0.6 * np.max(cls_sim) + 0.4 * np.mean(cls_sim)

    score_sum = class_scores.sum()
    if score_sum <= 0:
        return np.zeros(n_classes, dtype=float)
    return class_scores / score_sum


def init_usage_log() -> None:
    if "prediction_logs" not in st.session_state:
        st.session_state["prediction_logs"] = []


def record_prediction(model_name: str, disease: str, probability: float, symptoms: List[str]) -> None:
    init_usage_log()
    signature = (
        model_name,
        disease,
        round(float(probability), 6),
        tuple(sorted(symptoms)),
    )
    if st.session_state.get("last_logged_signature") == signature:
        return
    st.session_state["prediction_logs"].append(
        {
            "Model": model_name,
            "Predicted Disease": disease,
            "Probability": float(probability),
            "Symptom Count": len(symptoms),
        }
    )
    st.session_state["last_logged_signature"] = signature


def get_disease_content(disease_name: str) -> Dict[str, str]:
    key = disease_name.strip().lower()
    for canonical_name, content in DISEASE_CONTENT.items():
        if canonical_name.lower() == key:
            return content
    return {
        "description": "This condition is present in the prediction dataset.",
        "causes": "Causes vary by condition and patient factors.",
        "symptoms": "Refer to recognized symptom patterns and medical consultation for full context.",
        "doctor": "Consult a healthcare professional if symptoms persist, worsen, or feel severe.",
        "prevention": "General prevention includes hygiene, vaccination where relevant, and early medical advice.",
    }


def dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    feature_cols = [c for c in df.columns if c.strip().lower() != "disease"]
    target_col = [c for c in df.columns if c.strip().lower() == "disease"][0]
    return {
        "num_rows": int(len(df)),
        "num_diseases": int(df[target_col].nunique()),
        "num_symptoms": int(len(feature_cols)),
        "disease_examples": sorted(df[target_col].astype(str).unique().tolist())[:12],
    }


def softmax_probabilities(model: Any, x_row: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x_row.reshape(1, -1))[0], dtype=float)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    total = raw.sum()
    if total <= 0:
        return np.full_like(raw, 1.0 / len(raw))
    return raw / total


def hybrid_probabilities(
    model: Any,
    x_row: np.ndarray,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_classes: int,
    model_weight: float = 0.75,
) -> np.ndarray:
    """
    Blend model probabilities with symptom-pattern similarity.
    This helps on tiny sparse datasets by leveraging nearest historical patterns.
    """
    model_probs = softmax_probabilities(model, x_row)
    sim_probs = compute_similarity_probabilities(x_row, X_train, y_train, n_classes)
    if np.allclose(sim_probs.sum(), 0.0):
        return model_probs

    blended = model_weight * model_probs + (1.0 - model_weight) * sim_probs
    blended = np.nan_to_num(blended, nan=0.0, posinf=0.0, neginf=0.0)
    blend_total = blended.sum()
    if blend_total <= 0:
        return model_probs
    return blended / blend_total


def consensus_probabilities(
    model_probs: np.ndarray,
    similarity_probs: np.ndarray,
    n_selected_symptoms: int,
    model_weight: float,
) -> np.ndarray:
    # With sparse symptom input, rely a bit more on similarity evidence.
    adjusted_model_weight = model_weight if n_selected_symptoms >= 3 else min(model_weight, 0.5)
    blended = adjusted_model_weight * model_probs + (1.0 - adjusted_model_weight) * similarity_probs
    blended = np.nan_to_num(blended, nan=0.0, posinf=0.0, neginf=0.0)
    total = blended.sum()
    if total <= 0:
        return model_probs
    return blended / total
