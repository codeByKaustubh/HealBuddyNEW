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
    "Allergic Rhinitis": {
        "description": "Allergic rhinitis is inflammation inside the nose caused by an allergic reaction.",
        "causes": "Often triggered by pollen, dust mites, pet dander, mold, or environmental irritants.",
        "symptoms": "Sneezing, runny or blocked nose, itchy eyes, watery eyes, and postnasal drip.",
        "doctor": "See a doctor if symptoms are frequent, disturb sleep, or are not controlled with routine care.",
        "prevention": "Limit allergen exposure, use masks in dusty areas, and follow prescribed allergy treatment.",
    },
    "Anemia": {
        "description": "Anemia occurs when the blood has too few healthy red blood cells or low hemoglobin.",
        "causes": "Common causes include iron deficiency, vitamin deficiency, chronic disease, or blood loss.",
        "symptoms": "Fatigue, weakness, dizziness, pale skin, shortness of breath, and headache.",
        "doctor": "Seek medical evaluation for persistent fatigue, breathlessness, fainting, or very pale skin.",
        "prevention": "Eat iron-rich foods, ensure adequate vitamin B12 and folate, and treat underlying causes.",
    },
    "Angina": {
        "description": "Angina is chest discomfort caused by reduced blood flow to heart muscle.",
        "causes": "Usually due to coronary artery narrowing from atherosclerosis and cardiovascular risk factors.",
        "symptoms": "Chest pressure or tightness, discomfort with exertion, and pain radiating to arm, jaw, or back.",
        "doctor": "Urgent assessment is needed for new, worsening, or prolonged chest pain.",
        "prevention": "Control blood pressure, diabetes, and cholesterol; stop smoking; stay active; take prescribed medicines.",
    },
    "Appendicitis": {
        "description": "Appendicitis is inflammation of the appendix and is often a surgical emergency.",
        "causes": "Most often caused by blockage of the appendix leading to infection and swelling.",
        "symptoms": "Pain starting near the navel then moving to the lower right abdomen, fever, nausea, and vomiting.",
        "doctor": "Go to emergency care immediately for severe abdominal pain with fever or vomiting.",
        "prevention": "No sure prevention; early diagnosis and treatment reduce complications like rupture.",
    },
    "Typhoid": {
        "description": "Typhoid fever is a bacterial infection caused by Salmonella Typhi.",
        "causes": "Usually spread through contaminated food or water and poor sanitation.",
        "symptoms": "Sustained fever, weakness, abdominal pain, headache, and loss of appetite.",
        "doctor": "Seek care promptly for persistent high fever, confusion, dehydration, or severe abdominal pain.",
        "prevention": "Safe drinking water, proper handwashing, cooked food hygiene, and vaccination where advised.",
    },
    "COVID19": {
        "description": "COVID-19 is a contagious respiratory illness caused by the SARS-CoV-2 virus.",
        "causes": "Spreads mainly through respiratory droplets and close contact with infected individuals.",
        "symptoms": "Fever, cough, sore throat, fatigue, breathing difficulty, and loss of smell or taste.",
        "doctor": "Seek urgent care for trouble breathing, persistent chest pain, confusion, or low oxygen levels.",
        "prevention": "Vaccination, good ventilation, hand hygiene, masking in high-risk settings, and staying home when unwell.",
    },
    "Common Cold": {
        "description": "The common cold is a mild viral infection of the nose and throat.",
        "causes": "Usually caused by rhinoviruses and spread through droplets, hands, and shared surfaces.",
        "symptoms": "Runny nose, sneezing, sore throat, mild cough, and low-grade fever.",
        "doctor": "Consult a doctor if symptoms persist beyond 10 days, worsen, or include high fever or breathing issues.",
        "prevention": "Frequent handwashing, avoiding close contact with sick people, and respiratory hygiene.",
    },
    "Conjunctivitis": {
        "description": "Conjunctivitis (pink eye) is inflammation of the eye's outer membrane.",
        "causes": "Can be viral, bacterial, allergic, or due to irritants such as smoke or chemicals.",
        "symptoms": "Red eyes, irritation, watery or sticky discharge, itching, and light sensitivity.",
        "doctor": "Seek care for severe pain, vision changes, intense light sensitivity, or symptoms in newborns.",
        "prevention": "Avoid touching eyes, wash hands often, avoid sharing towels, and follow contact lens hygiene.",
    },
    "Dehydration": {
        "description": "Dehydration occurs when the body loses more fluids than it takes in.",
        "causes": "Common causes include vomiting, diarrhea, fever, heat exposure, and inadequate fluid intake.",
        "symptoms": "Thirst, dry mouth, dizziness, dark urine, fatigue, and reduced urination.",
        "doctor": "Urgent care is needed for confusion, fainting, very low urine output, or inability to drink fluids.",
        "prevention": "Drink water regularly, use oral rehydration during illness, and increase fluids in hot weather.",
    },
    "Dengue": {
        "description": "Dengue is a mosquito-borne viral infection that can range from mild to severe.",
        "causes": "Transmitted by Aedes mosquitoes, especially in tropical and subtropical areas.",
        "symptoms": "High fever, severe headache, body pain, joint pain, rash, nausea, and fatigue.",
        "doctor": "Seek urgent care for bleeding, severe abdominal pain, persistent vomiting, or drowsiness.",
        "prevention": "Prevent mosquito bites, remove standing water, wear protective clothing, and use repellents.",
    },
    "Diabetes": {
        "description": "Diabetes is a chronic condition marked by high blood sugar levels.",
        "causes": "Due to reduced insulin production, insulin resistance, or both, influenced by genetics and lifestyle.",
        "symptoms": "Increased thirst, frequent urination, fatigue, blurred vision, and unexplained weight change.",
        "doctor": "Consult a doctor for persistent high sugar symptoms or signs of complications like foot wounds.",
        "prevention": "Maintain healthy weight, exercise regularly, follow a balanced diet, and monitor blood sugar as advised.",
    },
    "Flu": {
        "description": "Influenza is a viral respiratory infection that can spread rapidly.",
        "causes": "Influenza viruses transmitted by droplets or contaminated surfaces.",
        "symptoms": "Fever, cough, sore throat, body aches, fatigue, and chills.",
        "doctor": "See a doctor for breathing difficulty, chest pain, prolonged fever, or high-risk conditions.",
        "prevention": "Annual flu vaccine, hand hygiene, masking during outbreaks, and staying home when sick.",
    },
    "Food Poisoning": {
        "description": "Food poisoning is illness caused by contaminated food or drink.",
        "causes": "Usually due to bacteria, viruses, toxins, or poor food handling and storage.",
        "symptoms": "Nausea, vomiting, diarrhea, abdominal cramps, fever, and weakness.",
        "doctor": "Seek care for dehydration, blood in stool, high fever, severe pain, or symptoms lasting more than a few days.",
        "prevention": "Cook food thoroughly, avoid cross-contamination, refrigerate promptly, and drink safe water.",
    },
    "Gastritis": {
        "description": "Gastritis is inflammation of the stomach lining that may be acute or chronic.",
        "causes": "Causes include H. pylori infection, painkiller overuse, alcohol, stress, and autoimmune disease.",
        "symptoms": "Upper abdominal discomfort, nausea, bloating, indigestion, and occasional vomiting.",
        "doctor": "See a doctor for persistent abdominal pain, vomiting blood, black stools, or unexplained weight loss.",
        "prevention": "Avoid unnecessary painkiller use, limit alcohol, treat H. pylori when present, and eat regular balanced meals.",
    },
    "Heart Attack": {
        "description": "A heart attack occurs when blood flow to part of the heart is suddenly blocked.",
        "causes": "Most commonly due to rupture of a coronary plaque and formation of a blood clot.",
        "symptoms": "Chest pressure, pain spreading to arm or jaw, sweating, nausea, and shortness of breath.",
        "doctor": "Call emergency services immediately for suspected heart attack symptoms.",
        "prevention": "Control blood pressure, cholesterol, and diabetes; avoid tobacco; stay active; follow heart medicines.",
    },
    "Heat Stroke": {
        "description": "Heat stroke is a dangerous rise in body temperature due to prolonged heat exposure.",
        "causes": "Occurs in hot conditions, especially with dehydration, overexertion, or poor ventilation.",
        "symptoms": "High body temperature, confusion, headache, dizziness, nausea, and sometimes fainting.",
        "doctor": "This is a medical emergency; seek immediate care for confusion, collapse, or very high fever in heat.",
        "prevention": "Stay hydrated, avoid peak heat hours, wear light clothing, and rest in cool environments.",
    },
    "Hypertension": {
        "description": "Hypertension is persistently elevated blood pressure that can damage blood vessels and organs.",
        "causes": "Linked to genetics, high salt intake, obesity, stress, kidney disease, and inactivity.",
        "symptoms": "Often no symptoms; some people may have headache, dizziness, or blurred vision.",
        "doctor": "Get medical care for very high readings, chest pain, severe headache, or neurological symptoms.",
        "prevention": "Reduce salt, exercise regularly, limit alcohol, maintain healthy weight, and monitor blood pressure.",
    },
    "Hypothyroidism": {
        "description": "Hypothyroidism is an underactive thyroid gland producing too little thyroid hormone.",
        "causes": "Common causes include autoimmune thyroiditis, iodine deficiency, or thyroid treatment side effects.",
        "symptoms": "Fatigue, weight gain, cold intolerance, dry skin, constipation, and slowed thinking.",
        "doctor": "See a doctor for persistent fatigue, swelling, low mood, or progressive weight gain.",
        "prevention": "Routine thyroid monitoring in at-risk people and timely hormone replacement when diagnosed.",
    },
    "Irritable Bowel Syndrome": {
        "description": "Irritable bowel syndrome (IBS) is a functional gut disorder causing recurrent bowel symptoms.",
        "causes": "Related to altered gut sensitivity, bowel motility, stress, diet triggers, and gut-brain interaction.",
        "symptoms": "Abdominal pain, bloating, constipation, diarrhea, or alternating bowel habits.",
        "doctor": "Consult a doctor if there is weight loss, blood in stool, fever, anemia, or symptoms after age 50.",
        "prevention": "Identify trigger foods, manage stress, maintain fiber balance, and keep regular meal habits.",
    },
    "Kidney Stones": {
        "description": "Kidney stones are hard mineral deposits that form in the urinary tract.",
        "causes": "Often linked to low fluid intake, high salt diets, metabolic factors, or recurrent urinary infections.",
        "symptoms": "Severe side or back pain, painful urination, nausea, vomiting, and blood in urine.",
        "doctor": "Seek urgent care for severe pain, fever, vomiting, or inability to pass urine.",
        "prevention": "Increase water intake, reduce salt, adjust diet based on stone type, and follow medical advice.",
    },
    "Malaria": {
        "description": "Malaria is a mosquito-borne parasitic infection that can become severe quickly.",
        "causes": "Caused by Plasmodium parasites transmitted through bites of infected Anopheles mosquitoes.",
        "symptoms": "Fever with chills, sweating, headache, body ache, nausea, and fatigue.",
        "doctor": "Prompt medical testing and treatment are essential, especially after travel to endemic regions.",
        "prevention": "Use bed nets, repellents, protective clothing, and preventive medicines when traveling to risk areas.",
    },
    "Otitis Media": {
        "description": "Otitis media is infection or inflammation of the middle ear, common in children.",
        "causes": "Often follows upper respiratory infections due to fluid and germ buildup behind the eardrum.",
        "symptoms": "Ear pain, fever, hearing reduction, irritability, and sometimes ear discharge.",
        "doctor": "See a doctor for severe pain, prolonged fever, discharge, or recurrent ear infections.",
        "prevention": "Timely treatment of colds, smoke-free environment, vaccination, and proper feeding positions in infants.",
    },
    "Pneumonia": {
        "description": "Pneumonia is infection of the lungs causing inflammation of air sacs.",
        "causes": "Can be caused by bacteria, viruses, or fungi, especially in older adults or weakened immunity.",
        "symptoms": "Fever, cough with phlegm, chest pain, rapid breathing, and fatigue.",
        "doctor": "Urgent care is needed for breathlessness, low oxygen, chest pain, or confusion.",
        "prevention": "Vaccination, hand hygiene, smoking cessation, and early treatment of respiratory infections.",
    },
    "Sinusitis": {
        "description": "Sinusitis is inflammation of sinus cavities that may be acute or chronic.",
        "causes": "Usually follows viral infections; can also be due to allergies, nasal polyps, or bacterial infection.",
        "symptoms": "Facial pressure, blocked nose, thick nasal discharge, headache, and reduced smell.",
        "doctor": "Consult a doctor if symptoms last more than 10 days, recur often, or are severe.",
        "prevention": "Control allergies, avoid smoke exposure, keep nasal passages moist, and practice hand hygiene.",
    },
    "Tuberculosis": {
        "description": "Tuberculosis is a contagious bacterial infection that most often affects the lungs.",
        "causes": "Caused by Mycobacterium tuberculosis spread through airborne droplets from infected people.",
        "symptoms": "Persistent cough, fever, night sweats, weight loss, chest pain, and fatigue.",
        "doctor": "Seek medical testing for cough lasting over two weeks, blood in sputum, or unexplained weight loss.",
        "prevention": "Early diagnosis, completing full treatment, good ventilation, and public health screening measures.",
    },
    "Ulcerative Colitis": {
        "description": "Ulcerative colitis is chronic inflammation of the colon and rectum.",
        "causes": "Likely due to immune dysregulation with genetic and environmental influences.",
        "symptoms": "Abdominal pain, diarrhea often with blood, urgency, fatigue, and weight loss.",
        "doctor": "Seek specialist care for persistent bloody diarrhea, severe pain, dehydration, or fever.",
        "prevention": "No complete prevention; regular follow-up and adherence to treatment reduce flare-ups and complications.",
    },
    "Urinary Tract Infection": {
        "description": "A urinary tract infection (UTI) is a bacterial infection of the urinary system.",
        "causes": "Typically caused by bacteria entering the urethra, with higher risk in dehydration or urinary retention.",
        "symptoms": "Burning urination, frequent urination, urgency, lower abdominal discomfort, and cloudy urine.",
        "doctor": "See a doctor for fever, back pain, vomiting, blood in urine, or recurrent urinary symptoms.",
        "prevention": "Drink enough water, avoid delaying urination, maintain genital hygiene, and complete prescribed antibiotics.",
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
