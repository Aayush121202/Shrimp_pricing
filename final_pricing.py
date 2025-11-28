import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor  # needed so joblib can load the model
from huggingface_hub import hf_hub_download

# Hugging Face repo ids and filenames
DATASET_REPO_ID = "aayushpatel1212/shrimp_pricing_data"
MODEL_REPO_ID = "aayushpatel1212/shrimp_pricing_v1"

DATASET_FILENAME = "FINAL_COSTS_Vannamei.csv"
MODEL_FILENAME = "uplift_xgb.joblib"
FEATURES_FILENAME = "feature_columns.joblib"
CATEGORY_MAPS_FILENAME = "category_maps.joblib"

# ---- Form code to human readable mapping ----
FORM_CODE_LABELS = {
    0: "Head On Shell On",
    1: "Headless Shell On",
    2: "Easy Peel",
    3: "Peeled and Deveined Tail On",
    4: "Peeled and Deveined Tail Off",
}
LABEL_TO_FORM_CODE = {v: k for k, v in FORM_CODE_LABELS.items()}


# ------------- Load artifacts from Hugging Face -------------

@st.cache_resource
def load_artifacts():
    # dataset
    dataset_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=DATASET_FILENAME,
        repo_type="dataset",
    )
    df = pd.read_csv(dataset_path)

    # model and metadata
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model",
    )
    feat_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=FEATURES_FILENAME,
        repo_type="model",
    )
    catmap_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=CATEGORY_MAPS_FILENAME,
        repo_type="model",
    )

    model = joblib.load(model_path)
    feature_columns = joblib.load(feat_path)
    category_maps = joblib.load(catmap_path)

    treatment_categories = category_maps["treatment_categories"]
    form_code_categories = category_maps["form_code_categories"]

    # deterministic price grid
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["Size_Lower"] = pd.to_numeric(df["Size_Lower"], errors="coerce")
    df["Size_Upper"] = pd.to_numeric(df["Size_Upper"], errors="coerce")
    df["Form_Code"] = pd.to_numeric(df["Form_Code"], errors="coerce")
    df["Treatment"] = df["Treatment"].astype(str).str.strip()
    df["week_of_year"] = df["DATE"].dt.isocalendar().week.astype(int)

    needed_cols = [
        "Size_Lower",
        "Size_Upper",
        "Treatment",
        "Form_Code",
        "week_of_year",
        "FINAL_PRICE_4",
        "FINAL_PRICE_6",
        "FINAL_PRICE_8",
    ]
    df_ref = df.dropna(subset=needed_cols).copy()

    price_ref = (
        df_ref
        .groupby(
            ["Size_Lower", "Size_Upper", "Treatment", "Form_Code", "week_of_year"],
            as_index=False,
        )
        .agg({
            "FINAL_PRICE_4": "median",
            "FINAL_PRICE_6": "median",
            "FINAL_PRICE_8": "median",
        })
    )

    return price_ref, model, feature_columns, treatment_categories, form_code_categories


# load artifacts once here so they are available in the UI code
price_ref, ml_model, feature_columns, treatment_categories, form_code_categories = load_artifacts()


# ------------- Deterministic price helpers -------------

def parse_grade(grade_str: str):
    """
    Grade is of form 'a/b'.

    Your convention:
      a = Size_Upper (smaller number)
      b = Size_Lower (larger number)

    For 31/40:
      Size_Upper = 31
      Size_Lower = 40

    We return in dataset order: (Size_Lower, Size_Upper) = (40, 31).
    """
    grade_str = grade_str.strip()
    parts = grade_str.split("/")
    if len(parts) != 2:
        raise ValueError(f"Grade must be 'a/b' format, got: {grade_str}")

    a = float(parts[0])  # upper
    b = float(parts[1])  # lower

    size_upper = a
    size_lower = b

    return size_lower, size_upper  # dataset order


def get_deterministic_prices_from_inputs(
    price_ref: pd.DataFrame,
    grade: str,
    treatment: str,
    form_code: int,
):
    """
    Inputs:
      - grade: 'a/b'
      - treatment
      - form_code

    Internal:
      - uses today's date to get week_of_year
      - looks up deterministic FINAL_PRICE_4, 6, 8 from price_ref
    """
    # 1. Parse grade
    size_lower, size_upper = parse_grade(grade)

    # 2. Today and week
    today = pd.Timestamp.today()
    week = today.isocalendar().week

    # 3. Exact match
    mask_exact = (
        (price_ref["Size_Lower"] == size_lower) &
        (price_ref["Size_Upper"] == size_upper) &
        (price_ref["Treatment"] == treatment) &
        (price_ref["Form_Code"] == form_code) &
        (price_ref["week_of_year"] == week)
    )
    subset_exact = price_ref.loc[mask_exact]

    if not subset_exact.empty:
        row = subset_exact.iloc[0]
    else:
        # 4. Fallback - closest week for same spec
        mask_spec = (
            (price_ref["Size_Lower"] == size_lower) &
            (price_ref["Size_Upper"] == size_upper) &
            (price_ref["Treatment"] == treatment) &
            (price_ref["Form_Code"] == form_code)
        )
        subset_spec = price_ref.loc[mask_spec].copy()

        if subset_spec.empty:
            raise ValueError(
                f"No deterministic price found for spec: "
                f"grade={grade}, treatment={treatment}, form_code={form_code}"
            )

        subset_spec["week_diff"] = (subset_spec["week_of_year"] - week).abs()
        subset_spec = subset_spec.sort_values("week_diff")
        row = subset_spec.iloc[0]

    return {
        "FINAL_PRICE_4": float(row["FINAL_PRICE_4"]),
        "FINAL_PRICE_6": float(row["FINAL_PRICE_6"]),
        "FINAL_PRICE_8": float(row["FINAL_PRICE_8"]),
        "Size_Lower": float(size_lower),
        "Size_Upper": float(size_upper),
        "week_of_year_used": int(row["week_of_year"]),
        "today_week_of_year": int(week),
    }


# ------------- ML helpers -------------

def encode_treatment(treatment_str: str, treatment_categories):
    """
    Map treatment string to the same integer code used in training.
    """
    if treatment_str not in treatment_categories:
        raise ValueError(f"Unknown treatment: {treatment_str}")
    return treatment_categories.index(treatment_str)


def encode_form_code(form_code_original, form_code_categories):
    """
    Map original Form_Code (for example 0, 1, 2, 3, 4)
    to the integer code used in training.
    """
    for idx, val in enumerate(form_code_categories):
        if int(val) == int(form_code_original):
            return idx
    raise ValueError(f"Unknown form_code: {form_code_original}")


def make_feature_row(
    size_lower: float,
    size_upper: float,
    week: int,
    treatment: str,
    form_code: int,
    feature_columns,
    treatment_categories,
    form_code_categories,
) -> pd.DataFrame:
    """
    Build one feature row with encoded Treatment and Form_Code as in training.
    """
    treatment_code = encode_treatment(treatment, treatment_categories)
    form_code_code = encode_form_code(form_code, form_code_categories)

    data = {col: 0 for col in feature_columns}

    if "Size_Lower" in data:
        data["Size_Lower"] = size_lower
    if "Size_Upper" in data:
        data["Size_Upper"] = size_upper
    if "week_of_year" in data:
        data["week_of_year"] = week
    if "Treatment" in data:
        data["Treatment"] = treatment_code
    if "Form_Code" in data:
        data["Form_Code"] = form_code_code

    return pd.DataFrame([data])[feature_columns]


def predict_price_with_ml(
    price_ref: pd.DataFrame,
    model,
    feature_columns,
    treatment_categories,
    form_code_categories,
    grade: str,
    treatment: str,
    form_code: int,
    premium_pct: int,
):
    det = get_deterministic_prices_from_inputs(
        price_ref=price_ref,
        grade=grade,
        treatment=treatment,
        form_code=form_code,
    )

    premium_to_key = {
        4: "FINAL_PRICE_4",
        6: "FINAL_PRICE_6",
        8: "FINAL_PRICE_8",
    }
    if premium_pct not in premium_to_key:
        raise ValueError("Premium must be one of 4, 6, 8 percent")

    P_det = det[premium_to_key[premium_pct]]

    size_lower = det["Size_Lower"]
    size_upper = det["Size_Upper"]
    week = det["today_week_of_year"]

    X_row = make_feature_row(
        size_lower=size_lower,
        size_upper=size_upper,
        week=week,
        treatment=treatment,
        form_code=form_code,
        feature_columns=feature_columns,
        treatment_categories=treatment_categories,
        form_code_categories=form_code_categories,
    )

    uplift_pred = float(model.predict(X_row)[0])
    P_ml = P_det * uplift_pred

    return {
        "P_det": P_det,
        "P_ml": P_ml,
        "uplift_pred": uplift_pred,
        **det,
    }


# ------------- Streamlit UI -------------

st.set_page_config(page_title="Shrimp Pricing Model v1", page_icon="🦐")

st.title("Shrimp Pricing Model - Deterministic + ML uplift")

st.write(
    "Inputs: grade (a/b), treatment, form type, and premium. "
    "Backend uses today's date to pick week of year, "
    "fetches deterministic prices from Hugging Face dataset, "
    "then applies an XGBoost uplift model from Hugging Face."
)

# Inputs
default_grade = "31/40"
grade_input = st.text_input("Grade (a/b)", value=default_grade, help="Example: 31/40")

treatment_options = sorted(price_ref["Treatment"].dropna().unique().tolist())
treatment_input = st.selectbox("Treatment type", treatment_options)

# form type dropdown based on dataset form codes but shown as readable labels
available_form_codes = sorted(price_ref["Form_Code"].dropna().unique().tolist())
form_type_options = [
    FORM_CODE_LABELS[int(c)]
    for c in available_form_codes
    if int(c) in FORM_CODE_LABELS
]

form_type_input = st.selectbox("Form type", form_type_options)

# map back to numeric form_code for internal logic and ML
form_code_value = LABEL_TO_FORM_CODE[form_type_input]

premium_input = st.selectbox("Premium percent", [4, 6, 8], index=2)

# new adjustment input in cents
adjustment_cents = st.selectbox(
    "Client adjustment suggestion (cents per kg)",
    [-30, -20, -10, 10, 20, 30],
    index=3,  # default +10 cents, adjust if you like
)

if st.button("Calculate price"):
    try:
        res = predict_price_with_ml(
            price_ref=price_ref,
            model=ml_model,
            feature_columns=feature_columns,
            treatment_categories=treatment_categories,
            form_code_categories=form_code_categories,
            grade=grade_input,
            treatment=treatment_input,
            form_code=int(form_code_value),
            premium_pct=int(premium_input),
        )

        # compute adjusted price
        adjustment_usd = adjustment_cents / 100.0
        P_adjusted = res["P_ml"] + adjustment_usd

        st.subheader("Price per kg (USD)")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Deterministic price", f"{res['P_det']:.2f}")
        col2.metric("ML uplift factor", f"{res['uplift_pred']:.3f}")
        col3.metric("ML suggested price", f"{res['P_ml']:.2f}")
        col4.metric(
            f"Client adjusted price ({adjustment_cents:+d} cents)",
            f"{P_adjusted:.2f}",
        )

        st.caption(
            f"Size band (dataset): {int(res['Size_Lower'])}/{int(res['Size_Upper'])}, "
            f"current week: {res['today_week_of_year']}, "
            f"matched week in grid: {res['week_of_year_used']}."
        )

    except Exception as e:
        st.error(f"Error computing price: {e}")
