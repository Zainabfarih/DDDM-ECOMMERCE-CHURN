"""Central configuration for the ChurnGuard Streamlit app.

Every constant is traceable back to the project notebooks so the application
stays in lock-step with the trained model and the documented business framing.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths — anchored to the project root, never to the current working directory
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = PROJECT_ROOT / "streamlit_app"

DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
IMAGES_DIR    = PROJECT_ROOT / "images"
ASSETS_DIR    = APP_ROOT / "assets"

# Processed artefacts
RFM_PATH             = PROCESSED_DIR / "rfm_segmented.csv"
CLEAN_PATH           = PROCESSED_DIR / "clean_transactions.csv"
META_PATH            = PROCESSED_DIR / "meta.json"
AGG_MONTHLY_PATH     = PROCESSED_DIR / "agg_monthly.csv"
AGG_PRODUCTS_PATH    = PROCESSED_DIR / "agg_top_products.csv"
AGG_COUNTRY_PATH     = PROCESSED_DIR / "agg_country.csv"
AGG_SEASONALITY_PATH = PROCESSED_DIR / "agg_seasonality.csv"

# Model artefacts
MODEL_PATH         = MODELS_DIR / "churn_model_xgb.pkl"
FEATURES_PATH      = MODELS_DIR / "features.pkl"
SCALER_PATH        = MODELS_DIR / "scaler.pkl"
KMEANS_PATH        = MODELS_DIR / "kmeans.pkl"
KMEANS_SCALER_PATH = MODELS_DIR / "kmeans_scaler.pkl"

# --------------------------------------------------------------------------- #
# Business / model constants
# --------------------------------------------------------------------------- #
APP_TITLE    = "ChurnGuard"
APP_SUBTITLE = "Système d'aide à la décision anti-churn e-commerce"
AUTHORS      = "Zainab Farih & Assia Rguibi"

# Business / model constants
CHURN_THRESHOLD_DAYS       = 90        # churn = inactivity longer than this
MODEL_FEATURES             = ["LogFrequency", "LogMonetary", "LogAvgBasket", "Cluster"]
TARGET                     = "Churn"
CURRENCY                   = "£"        # transactions are in GBP
REPORTED_TEST_AUC          = 0.86
DEFAULT_DECISION_THRESHOLD = 0.50

# --------------------------------------------------------------------------- #
# Visual identity — must mirror the tokens in assets/styles.css
# Pastel palette from the notebook EDA — every color stays in this set.
# --------------------------------------------------------------------------- #
COLORS = {
    # Neutrals
    "ink":           "#2a3a4a",   # soft slate ink (not pure black)
    "ink_soft":      "#475669",
    "muted":         "#6b7888",
    "faint":         "#9ba8b5",
    "border":        "#e8e4dc",
    "surface":       "#ffffff",
    "surface_2":     "#f6f4ef",
    "bg":            "#fafaf7",

    # Pastels (notebook palette)
    "pastel_blue":   "#AEC6CF",
    "pastel_coral":  "#FFB7B2",
    "pastel_green":  "#C1E1C1",
    "pastel_peach":  "#FFDAC1",
    "pastel_mint":   "#B5EAD7",
    "pastel_pink":   "#FF9AA2",

    # Semantic — brand uses pastel blue, semantic uses the rest
    "accent":        "#AEC6CF",   # pastel blue — primary brand
    "accent_2":      "#7BA8B6",   # darker variant for hovers / strong text
    "accent_soft":   "#eaf0f3",   # very washed out — for soft backgrounds

    "positive":      "#C1E1C1",   # pastel green
    "positive_soft": "#eef7ee",
    "warning":       "#FFDAC1",   # pastel peach
    "warning_soft":  "#fcf3e8",
    "danger":        "#FFB7B2",   # pastel coral
    "danger_soft":   "#fceeed",

    # Aliases used elsewhere
    "primary":       "#AEC6CF",
    "success":       "#C1E1C1",
    "text":          "#2a3a4a",
    "card":          "#ffffff",
}

# Categorical palette for charts requiring 4+ distinct hues — strictly the six
# pastels from the notebook EDA, in the same order.
PASTEL = ["#AEC6CF", "#FFB7B2", "#C1E1C1", "#FFDAC1", "#B5EAD7", "#FF9AA2"]

CHURN_COLORS = {0: COLORS["positive"], 1: COLORS["danger"]}
CHURN_LABELS = {0: "Actif", 1: "Churn"}

# --------------------------------------------------------------------------- #
# K-Means segments (best_k = 2, oriented by recency)
#   Cluster 0 → low recency / high frequency  → active core
#   Cluster 1 → high recency / low frequency  → dormant / at risk
# --------------------------------------------------------------------------- #
SEGMENTS = {
    0: {
        "name":  "Cœur actif",
        "label": "Cœur actif",
        "code":  "A",
        "color": COLORS["pastel_green"],
        "desc":  "Clients fréquents et récents — faible taux de churn. À fidéliser.",
    },
    1: {
        "name":  "Clients dormants",
        "label": "Dormants",
        "code":  "D",
        "color": COLORS["pastel_coral"],
        "desc":  "Clients peu fréquents et inactivf — fort taux de churn. À reconquérir.",
    },
}


def get_segment_info(cluster_id: int) -> dict:
    """Return segment metadata for any cluster id, even if the silhouette
    selected K > 2 at runtime (defensive default avoids KeyError on chart
    color lookups and ``df["Cluster"].map(...)`` calls)."""
    cluster_id = int(cluster_id)
    if cluster_id in SEGMENTS:
        return SEGMENTS[cluster_id]
    fallback_color = PASTEL[cluster_id % len(PASTEL)]
    return {
        "name":  f"Segment {cluster_id}",
        "label": f"Segment {cluster_id}",
        "code":  str(cluster_id),
        "color": fallback_color,
        "desc":  "Segment additionnel issu du clustering.",
    }
