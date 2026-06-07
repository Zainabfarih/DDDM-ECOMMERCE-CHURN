"""Cached data access layer for the Streamlit app.

Loads the lightweight processed artefacts (RFM table + pre-computed aggregates +
meta). The heavy clean_transactions.csv is never loaded at runtime. If artefacts
are missing, `artifacts_status` reports it and `regenerate` rebuilds everything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import config as cfg  # noqa: E402

# Heavy services (pipeline, model_service) are imported lazily inside the
# functions that actually need them, so visiting a page that only reads
# light artefacts (RFM table, meta.json, aggregates) doesn't pay the cost of
# importing sklearn / joblib / xgboost.

REQUIRED_FILES = [
    cfg.RFM_PATH, cfg.META_PATH, cfg.AGG_MONTHLY_PATH,
    cfg.AGG_PRODUCTS_PATH, cfg.AGG_COUNTRY_PATH, cfg.AGG_SEASONALITY_PATH,
    cfg.KMEANS_PATH, cfg.KMEANS_SCALER_PATH,
]

def artifacts_status() -> dict:
    """Which artefacts exist and whether the app is ready to run."""
    status = {p.name: p.exists() for p in REQUIRED_FILES}
    status["_ready"] = all(status.values())
    return status

def regenerate(enrich_countries: bool = True, log=print,
               progress=None, force_full: bool = False) -> dict:
    """Rebuild every processed artefact from the raw data."""
    from services import pipeline as P  # heavy — lazy
    return P.build_artifacts(
        enrich_countries=enrich_countries,
        log=log,
        progress=progress,
        force_full=force_full,
    )

# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_rfm() -> pd.DataFrame:
    df = pd.read_csv(cfg.RFM_PATH)
    # Derived feature-engineering columns so every page shares them.
    df["AvgBasket"] = df["Monetary"] / df["Frequency"].replace(0, 1)
    df["LogFrequency"] = np.log1p(df["Frequency"])
    df["LogMonetary"] = np.log1p(df["Monetary"])
    df["LogAvgBasket"] = np.log1p(df["AvgBasket"])
    df["Statut"] = df["Churn"].map(cfg.CHURN_LABELS)
    # Use the get_segment_info helper so unexpected cluster ids (K > 2) get a
    # sensible name instead of NaN.
    df["Segment"] = df["Cluster"].apply(lambda c: cfg.get_segment_info(c)["name"])
    return df

@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    return json.loads(cfg.META_PATH.read_text())

@st.cache_data(show_spinner=False)
def load_monthly() -> pd.DataFrame:
    df = pd.read_csv(cfg.AGG_MONTHLY_PATH, parse_dates=["YearMonth"])
    return df

@st.cache_data(show_spinner=False)
def load_top_products() -> pd.DataFrame:
    return pd.read_csv(cfg.AGG_PRODUCTS_PATH)

@st.cache_data(show_spinner=False)
def load_country() -> pd.DataFrame:
    return pd.read_csv(cfg.AGG_COUNTRY_PATH)

@st.cache_data(show_spinner=False)
def load_seasonality() -> pd.DataFrame:
    return pd.read_csv(cfg.AGG_SEASONALITY_PATH, index_col=0)

@st.cache_resource(show_spinner=False)
def get_model_bundle():
    """Load (once) the model + KMeans artefacts used for inference."""
    from services import model_service as ms
    return ms.load_bundle()

def clear_caches():
    get_model_bundle.clear()
    score_full_base.clear()
    load_rfm.clear()
    load_meta.clear()
    load_monthly.clear()
    load_top_products.clear()
    load_country.clear()
    load_seasonality.clear()


@st.cache_data(show_spinner=False)
def score_full_base() -> pd.DataFrame:
    """RFM table augmented with the model's churn probability + risk level.

    Computed once per session and shared by every page that needs a scored
    customer base (Direction, Marketing, Operations). The cache invalidates
    automatically when ``clear_caches()`` runs after a regen.
    """
    bundle = get_model_bundle()
    df = load_rfm().copy()
    proba = bundle.model.predict_proba(df[cfg.MODEL_FEATURES].values)[:, 1]
    df["Churn_Probability"] = proba
    df["Risk_Level"] = pd.cut(
        proba, bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["Faible", "Modéré", "Élevé"],
    )
    return df
