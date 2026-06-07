"""Model service: load the shipped artefacts and run the prediction pipeline.

Key fidelity notes:
  * The final XGBoost classifier was trained on RAW (un-scaled) features, so we
    feed it raw values. ``scaler.pkl`` was only used for Logistic Regression and
    is therefore NOT applied at inference for XGBoost.
  * Feature order is fixed by ``features.pkl``:
    ['LogFrequency', 'LogMonetary', 'LogAvgBasket', 'Cluster'].
  * The ``Cluster`` feature is reproduced with the regenerated KMeans pipeline
    (kmeans.pkl + kmeans_scaler.pkl).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import config as cfg  # noqa: E402
from services import pipeline as P  # noqa: E402

@dataclass
class ModelBundle:
    model: object
    features: list
    kmeans: object
    kmeans_scaler: object

def load_bundle() -> ModelBundle:
    """Load every artefact required for inference. Raises FileNotFoundError with a
    clear message if an artefact is missing."""
    import joblib  # heavy — lazy

    missing = [p.name for p in (cfg.MODEL_PATH, cfg.FEATURES_PATH,
                                cfg.KMEANS_PATH, cfg.KMEANS_SCALER_PATH)
               if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Artefacts manquants : " + ", ".join(missing)
            + ". Lancez la préparation des données (page d'accueil)."
        )
    return ModelBundle(
        model=joblib.load(cfg.MODEL_PATH),
        features=list(joblib.load(cfg.FEATURES_PATH)),
        kmeans=joblib.load(cfg.KMEANS_PATH),
        kmeans_scaler=joblib.load(cfg.KMEANS_SCALER_PATH),
    )

# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(recency: float, frequency: float, monetary: float,
                      bundle: ModelBundle) -> dict:
    """Turn raw RFM values into the 4 model features, including the KMeans cluster."""
    frequency = max(float(frequency), 1.0)            # Frequency >= 1 (replace 0 -> 1)
    monetary = max(float(monetary), 0.0)
    avg_basket = monetary / frequency
    cluster = P.assign_cluster(recency, frequency, monetary,
                               bundle.kmeans, bundle.kmeans_scaler)
    return {
        "LogFrequency": float(np.log1p(frequency)),
        "LogMonetary": float(np.log1p(monetary)),
        "LogAvgBasket": float(np.log1p(avg_basket)),
        "Cluster": int(cluster),
        "AvgBasket": float(avg_basket),
    }

def predict_one(recency: float, frequency: float, monetary: float,
                bundle: ModelBundle, threshold: float = cfg.DEFAULT_DECISION_THRESHOLD) -> dict:
    """Predict churn probability for a single customer."""
    feats = engineer_features(recency, frequency, monetary, bundle)
    X = np.array([[feats[f] for f in bundle.features]], dtype=float)
    proba = float(bundle.model.predict_proba(X)[0, 1])
    return {
        **feats,
        "Recency": float(recency),
        "Frequency": float(frequency),
        "Monetary": float(monetary),
        "churn_proba": proba,
        "prediction": int(proba >= threshold),
        "segment": int(feats["Cluster"]),
    }

def predict_batch(df: pd.DataFrame, bundle: ModelBundle,
                  threshold: float = cfg.DEFAULT_DECISION_THRESHOLD) -> pd.DataFrame:
    """Vectorised batch prediction.

    ``df`` must contain columns Recency, Frequency, Monetary (case-insensitive).
    Rows with non-numeric or missing values in any of these columns are dropped
    from the prediction output (and reported back via the ``_dropped`` attribute
    on the result, so the UI can warn the user). Returns the input augmented
    with engineered features, churn probability and the binary decision.
    """
    cols = {c.lower(): c for c in df.columns}
    required = ["recency", "frequency", "monetary"]
    missing = [r for r in required if r not in cols]
    if missing:
        raise ValueError(
            "Colonnes manquantes : " + ", ".join(missing)
            + ". Le fichier doit contenir Recency, Frequency et Monetary."
        )

    work = df.copy()
    rec_raw  = pd.to_numeric(work[cols["recency"]],   errors="coerce")
    freq_raw = pd.to_numeric(work[cols["frequency"]], errors="coerce")
    mon_raw  = pd.to_numeric(work[cols["monetary"]],  errors="coerce")

    # Drop rows where any of the three required values failed to parse.
    valid = rec_raw.notna() & freq_raw.notna() & mon_raw.notna()
    n_dropped = int((~valid).sum())
    work = work.loc[valid].copy()
    rec  = rec_raw.loc[valid]
    freq = freq_raw.loc[valid].clip(lower=1)
    mon  = mon_raw.loc[valid].clip(lower=0)

    if len(work) == 0:
        raise ValueError(
            "Aucune ligne exploitable : Recency, Frequency et Monetary "
            "doivent être numériques et renseignés."
        )

    avg_basket = mon / freq
    log_x = np.log1p(
        np.column_stack([rec.values, freq.values, mon.values])
    )
    clusters = bundle.kmeans.predict(bundle.kmeans_scaler.transform(log_x))

    feat_df = pd.DataFrame({
        "LogFrequency": np.log1p(freq.values),
        "LogMonetary":  np.log1p(mon.values),
        "LogAvgBasket": np.log1p(avg_basket.values),
        "Cluster":      clusters.astype(int),
    })
    proba = bundle.model.predict_proba(feat_df[bundle.features].values)[:, 1]

    work["AvgBasket"]         = avg_basket.round(2).values
    work["Cluster"]           = clusters.astype(int)
    work["Churn_Probability"] = proba.round(4)
    work["Prediction"]        = (proba >= threshold).astype(int)
    work["Risk_Level"]        = pd.cut(
        proba, bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["Faible", "Modéré", "Élevé"],
    )
    work.attrs["dropped_rows"] = n_dropped
    return work
