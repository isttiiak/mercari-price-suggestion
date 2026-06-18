import numpy as np
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

# Model directory path — relative to this file
MODEL_DIR = Path(__file__).parent / "model"


def load_artifacts() -> dict:
    """
    Load all saved encoders and vectorizers from Phase I.
    These must be the exact same objects used during training.
    """
    artifacts = {
        "tfidf_name" : joblib.load(MODEL_DIR / "tfidf_name.pkl"),
        "tfidf_desc" : joblib.load(MODEL_DIR / "tfidf_desc.pkl"),
        "le_cat1"    : joblib.load(MODEL_DIR / "le_cat1.pkl"),
        "le_cat2"    : joblib.load(MODEL_DIR / "le_cat2.pkl"),
        "le_cat3"    : joblib.load(MODEL_DIR / "le_cat3.pkl"),
        "le_brand"   : joblib.load(MODEL_DIR / "le_brand.pkl"),
        "model"      : joblib.load(MODEL_DIR / "lgbm_model.pkl"),
    }
    return artifacts


def clean_text(text: str) -> str:
    """
    Apply the same text cleaning as Phase I preprocessing:
    lowercase → strip [rm] tags → collapse whitespace.
    """
    text = str(text).lower()
    text = text.replace("[rm]", "")
    text = " ".join(text.split())
    return text


def split_category(category: str) -> tuple[str, str, str]:
    """
    Split category_name into 3 levels.
    Example: 'Women/Tops/T-Shirts' → ('Women', 'Tops', 'T-Shirts')
    Pads with 'missing' if fewer than 3 levels are provided.
    """
    parts = str(category).split("/", maxsplit=2)
    while len(parts) < 3:
        parts.append("missing")
    return parts[0], parts[1], parts[2]


def safe_label_encode(encoder, value: str) -> int:
    """
    Label encode a value safely.

    Tries the value as-is, then lowercase, then falls back to
    'missing' or 'no brand' (whichever exists in the encoder),
    and finally index 0 if none of the above are found.

    This prevents KeyError on unseen values at inference time.
    """
    classes = list(encoder.classes_)

    # Exact match
    if value in classes:
        return int(encoder.transform([value])[0])

    # Case-insensitive match
    lower = value.lower()
    if lower in classes:
        return int(encoder.transform([lower])[0])

    # Fallback to known sentinel values from Phase I
    for fallback in ("missing", "no brand", "No Brand"):
        if fallback in classes:
            return int(encoder.transform([fallback])[0])

    # Last resort — index 0
    return 0


def preprocess_input(
    name: str,
    category_name: str,
    brand_name: str,
    item_condition_id: int,
    shipping: int,
    item_description: str,
    artifacts: dict,
) -> csr_matrix:
    """
    Transform a single product listing into the feature matrix
    expected by the LightGBM model.

    Applies the exact same pipeline as Phase I feature engineering:
    text cleaning → length features → category split → TF-IDF
    → label encoding → hstack into a single sparse matrix.
    """
    # Step 1: Clean text
    name_clean = clean_text(name)
    desc_clean = clean_text(item_description)

    # Step 2: Length features (same as Phase I)
    name_len = len(name_clean.split())
    desc_len = len(desc_clean.split())

    # Step 3: Split category into 3 levels
    cat1, cat2, cat3 = split_category(category_name)

    # Step 4: is_branded binary flag
    is_branded = 0 if brand_name.strip().lower() in ("", "no brand", "missing") else 1

    # Step 5: TF-IDF transform using fitted vectorizers from Phase I
    X_name = artifacts["tfidf_name"].transform([name_clean])
    X_desc = artifacts["tfidf_desc"].transform([desc_clean])

    # Step 6: Label encode categories and brand
    cat1_enc  = safe_label_encode(artifacts["le_cat1"], cat1)
    cat2_enc  = safe_label_encode(artifacts["le_cat2"], cat2)
    cat3_enc  = safe_label_encode(artifacts["le_cat3"], cat3)
    brand_enc = safe_label_encode(artifacts["le_brand"], brand_name)

    # Step 7: Pack numeric features in the same column order as Phase I
    numeric = csr_matrix([[
        item_condition_id,
        shipping,
        name_len,
        desc_len,
        cat1_enc,
        cat2_enc,
        cat3_enc,
        brand_enc,
        is_branded,
    ]])

    # Step 8: Combine into final sparse matrix (same hstack order as Phase I)
    X = hstack([X_name, X_desc, numeric])

    return X