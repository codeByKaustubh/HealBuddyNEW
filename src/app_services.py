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
    return model.predict_proba(x_row.reshape(1, -1))[0]
