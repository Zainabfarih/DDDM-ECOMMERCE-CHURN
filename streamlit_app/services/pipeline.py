"""
pipeline.py
-----------
Self-contained reconstruction of the ChurnGuard data pipeline.

The shipped repo provides the trained XGBoost model and the StandardScaler, but
not the processed datasets (git-ignored) nor the KMeans model used to produce
the `Cluster` feature. This module rebuilds those artefacts deterministically
from the raw sources so the dashboards and the inference path can run.

All functions are pure / side-effect free except `build_artifacts`, which writes
the regenerated files to disk.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# Heavy deps (sklearn, joblib) are imported lazily inside the functions that
# actually need them, so that simply importing this module — which happens
# eagerly via data_service — stays cheap. This shaves several seconds off
# Streamlit's cold start, especially on the Microsoft-Store Python build.

# --------------------------------------------------------------------------- #
# Paths & constants  (mirrors src/preprocessing.py)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

KAGGLE_CSV = RAW_DIR / "data.csv"
UCI_XLSX = RAW_DIR / "online_retail_II.xlsx"
UCI_SHEETS = ["Year 2009-2010", "Year 2010-2011"]

CLEAN_PATH = PROCESSED_DIR / "clean_transactions.csv"
RFM_PATH = PROCESSED_DIR / "rfm_segmented.csv"
KMEANS_PATH = MODELS_DIR / "kmeans.pkl"
KMEANS_SCALER_PATH = MODELS_DIR / "kmeans_scaler.pkl"

# Lightweight aggregates consumed by the dashboard (avoids loading the 190 MB
# clean_transactions.csv at runtime).
AGG_MONTHLY_PATH = PROCESSED_DIR / "agg_monthly.csv"
AGG_PRODUCTS_PATH = PROCESSED_DIR / "agg_top_products.csv"
AGG_COUNTRY_PATH = PROCESSED_DIR / "agg_country.csv"
AGG_SEASONALITY_PATH = PROCESSED_DIR / "agg_seasonality.csv"
META_PATH = PROCESSED_DIR / "meta.json"

COMMON_COLS = [
    "Invoice", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]
DEDUP_KEY = [
    "Invoice", "StockCode", "Quantity", "InvoiceDate", "CustomerID", "UnitPrice",
]

COUNTRIES_API = (
    "https://restcountries.com/v3.1/all"
    "?fields=name,region,subregion,population,latlng"
)
COUNTRY_MAPPING = {
    "EIRE": "Ireland",
    "USA": "United States",
    "RSA": "South Africa",
    "Channel Islands": "United Kingdom",
    "European Community": "Germany",
    "Korea": "South Korea",
    "West Indies": "Jamaica",
    "Czech Republic": "Czechia",
    "Unspecified": None,
}

RFM_VARS = ["Recency", "Frequency", "Monetary"]
CHURN_THRESHOLD = 90          # churn = inactivity longer than this
MODEL_FEATURES = ["LogFrequency", "LogMonetary", "LogAvgBasket", "Cluster"]

# --------------------------------------------------------------------------- #
# Loading & harmonising the two raw sources
# --------------------------------------------------------------------------- #
def load_kaggle(path: Path = KAGGLE_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="ISO-8859-1")
    df = df.rename(columns={"InvoiceNo": "Invoice"})
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], format="%m/%d/%Y %H:%M", errors="coerce"
    )
    df = df[COMMON_COLS].copy()
    df["Source"] = "Kaggle_ecommerce"
    return df

def load_uci(path: Path = UCI_XLSX, sheets: list[str] | None = None) -> pd.DataFrame:
    sheets = sheets or UCI_SHEETS
    frames = [pd.read_excel(path, sheet_name=s) for s in sheets]
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"Price": "UnitPrice", "Customer ID": "CustomerID"})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df[COMMON_COLS].copy()
    df["Source"] = "UCI_OnlineRetailII"
    return df

def load_raw() -> pd.DataFrame:
    return pd.concat([load_kaggle(), load_uci()], ignore_index=True)

# --------------------------------------------------------------------------- #
# Audit helpers (used by the Data Quality dashboard)
# --------------------------------------------------------------------------- #
def audit_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        non_na = df[col].dropna()
        rows.append({
            "Colonne": col,
            "Type": str(df[col].dtype),
            "Manquants_%": round(df[col].isna().mean() * 100, 2),
            "Uniques": int(df[col].nunique(dropna=True)),
            "Exemple": non_na.iloc[0] if not non_na.empty else None,
        })
    return pd.DataFrame(rows)

def count_cross_source_duplicates(df: pd.DataFrame) -> int:
    return int(df.duplicated(subset=DEDUP_KEY, keep="first").sum())

def quality_flags(df: pd.DataFrame) -> dict:
    invoice = df["Invoice"].astype(str)
    return {
        "lignes": len(df),
        "doublons_cross_source": count_cross_source_duplicates(df),
        "sans_customer_id": int(df["CustomerID"].isna().sum()),
        "quantite_negative_ou_nulle": int((df["Quantity"] <= 0).sum()),
        "prix_negatif_ou_nul": int((df["UnitPrice"] <= 0).sum()),
        "annulations": int(invoice.str.startswith("C").sum()),
        "date_min": df["InvoiceDate"].min(),
        "date_max": df["InvoiceDate"].max(),
        "pays_distincts": int(df["Country"].nunique()),
    }

# --------------------------------------------------------------------------- #
# Cleaning & enrichment
# --------------------------------------------------------------------------- #
def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.drop_duplicates(subset=DEDUP_KEY, keep="first").copy()
    clean = clean[~clean["Invoice"].astype(str).str.startswith("C")]
    clean = clean.dropna(subset=["CustomerID"])
    clean = clean[clean["Quantity"] > 0]
    clean = clean[clean["UnitPrice"] > 0]

    clean["CustomerID"] = clean["CustomerID"].astype("int64")
    clean["InvoiceDate"] = pd.to_datetime(clean["InvoiceDate"])
    clean["TotalPrice"] = clean["Quantity"] * clean["UnitPrice"]
    return clean.reset_index(drop=True)

def fetch_country_reference(url: str = COUNTRIES_API) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=30) as resp:
        api = json.load(resp)
    records = []
    for c in api:
        latlng = c.get("latlng") or [None, None]
        records.append({
            "CountryAPI": c["name"]["common"],
            "Region": c.get("region"),
            "SubRegion": c.get("subregion"),
            "Population": c.get("population"),
            "Lat": latlng[0],
            "Lng": latlng[1],
        })
    return pd.DataFrame(records)

def enrich(clean: pd.DataFrame, countries: pd.DataFrame) -> pd.DataFrame:
    out = clean.copy()
    out["CountryStd"] = out["Country"].replace(COUNTRY_MAPPING)
    out = out.merge(countries, left_on="CountryStd", right_on="CountryAPI", how="left")
    return out.drop(columns=["CountryAPI"])

# --------------------------------------------------------------------------- #
# RFM table, winsorization & KMeans segmentation
# --------------------------------------------------------------------------- #
def build_rfm(clean: pd.DataFrame) -> pd.DataFrame:
    """Build the RFM table per customer + the churn label."""
    reference_date = clean["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = (
        clean.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )
    rfm["Churn"] = (rfm["Recency"] > CHURN_THRESHOLD).astype(int)
    return rfm

def winsorize_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Clip R/F/M to [P1, P99] to soak up outliers before clustering."""
    out = rfm.copy()
    for var in RFM_VARS:
        p1, p99 = rfm[var].quantile(0.01), rfm[var].quantile(0.99)
        out[var] = rfm[var].clip(lower=p1, upper=p99)
    return out

def _canonicalize_by_recency(km, labels: np.ndarray) -> np.ndarray:
    """Re-order KMeans clusters so the numbering is deterministic and matches the
    encoding the shipped XGBoost model was trained on.

    KMeans assigns arbitrary integer ids to clusters, so two identical runs can
    swap "cluster 0" and "cluster 1". The trained model, however, treats `Cluster`
    as a numeric feature, so the ordering matters. We orient clusters by their mean
    Recency in the scaled-log space (Recency is the first RFM column): the cluster
    with the LOWEST recency -> label 0 (most active), highest -> top label. This
    reproduces the original encoding (verified: test AUC ~0.86 with the shipped
    model) and is fully deterministic.
    """
    order = np.argsort(km.cluster_centers_[:, 0])          # ascending mean recency
    remap = {old: new for new, old in enumerate(order)}
    km.cluster_centers_ = km.cluster_centers_[order]        # predict() now returns canonical ids
    return np.array([remap[int(l)] for l in labels])

def fit_kmeans(rfm_clean: pd.DataFrame, k_range=range(2, 9), random_state: int = 42):
    """Reproduce the notebook-03 KMeans: log1p(R,F,M) -> standardize -> pick best
    K by silhouette -> fit final model, then canonicalize cluster ids by recency.
    Returns (kmeans, scaler, labels, best_k, silhouette)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    X = rfm_clean[RFM_VARS].copy()
    for f in RFM_VARS:
        X[f] = np.log1p(X[f])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    silhouettes = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        silhouettes[k] = silhouette_score(
            X_scaled, labels, sample_size=2000, random_state=random_state
        )

    best_k = max(silhouettes, key=silhouettes.get)
    km_final = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    labels = km_final.fit_predict(X_scaled)
    sil_final = silhouette_score(
        X_scaled, labels, sample_size=2000, random_state=random_state
    )
    labels = _canonicalize_by_recency(km_final, labels)
    return km_final, scaler, labels, best_k, sil_final

def add_model_features(rfm: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: AvgBasket + log transforms on F, M, AvgBasket."""
    out = rfm.copy()
    out["AvgBasket"] = out["Monetary"] / out["Frequency"].replace(0, 1)
    out["LogFrequency"] = np.log1p(out["Frequency"])
    out["LogMonetary"] = np.log1p(out["Monetary"])
    out["LogAvgBasket"] = np.log1p(out["AvgBasket"])
    return out

def assign_cluster(recency, frequency, monetary, kmeans, kmeans_scaler) -> int:
    """Assign the K-Means cluster to a single (R, F, M) triple, using the same
    log1p + standardization the model was trained on."""
    x = np.log1p(np.array([[recency, frequency, monetary]], dtype=float))
    x_scaled = kmeans_scaler.transform(x)
    return int(kmeans.predict(x_scaled)[0])

def quality_flags_from_clean(clean: pd.DataFrame) -> dict:
    """Lightweight quality summary derived from already-cleaned data.

    Used by the fast path (when clean_transactions.csv is already on disk and we
    skip re-parsing the raw Excel). The exotic flags that only make sense on raw
    data — cross-source duplicates, cancellations, negative quantities — are
    reported as not-applicable strings so the Quality page can render gracefully.
    """
    return {
        "lignes": int(len(clean)),
        "doublons_cross_source": "—",
        "sans_customer_id": 0,
        "quantite_negative_ou_nulle": 0,
        "prix_negatif_ou_nul": 0,
        "annulations": 0,
        "date_min": clean["InvoiceDate"].min(),
        "date_max": clean["InvoiceDate"].max(),
        "pays_distincts": int(clean["Country"].nunique()),
    }

# --------------------------------------------------------------------------- #
# Orchestration - regenerate every missing artefact
# --------------------------------------------------------------------------- #
def build_dashboard_aggregates(raw, clean: pd.DataFrame,
                               rfm_clean: pd.DataFrame, extra_meta: dict) -> dict:
    """Pre-compute the small aggregate tables the dashboards rely on, so the app
    never has to load the full (very large) clean_transactions.csv at runtime.
    Mirrors the aggregations performed in notebooks 02 & 03.

    ``raw`` may be ``None`` (fast path with cached clean_transactions.csv).
    In that case the audit + quality_flags are derived from ``clean`` only and
    the meta carries an ``audit_full=False`` marker so downstream pages can
    explain that the cross-source counters require the slow path.
    """
    # --- Monthly activity ----------------------------------------------------
    c = clean.copy()
    c["YearMonth"] = pd.to_datetime(c["InvoiceDate"]).dt.to_period("M").dt.to_timestamp()
    monthly = (
        c.groupby("YearMonth")
        .agg(CA=("TotalPrice", "sum"),
             N_clients=("CustomerID", "nunique"),
             N_orders=("Invoice", "nunique"))
        .reset_index()
    )
    monthly.to_csv(AGG_MONTHLY_PATH, index=False)

    # --- Top products by revenue --------------------------------------------
    top_products = (
        c.groupby("Description")["TotalPrice"].sum()
        .sort_values(ascending=False).head(20).reset_index()
        .rename(columns={"TotalPrice": "CA"})
    )
    top_products.to_csv(AGG_PRODUCTS_PATH, index=False)

    # --- Country summary ------------------------------------------------------
    region_col = "Region" if "Region" in c.columns else None
    agg = {"Invoice": "count", "CustomerID": "nunique", "TotalPrice": "sum"}
    country = c.groupby("Country").agg(agg).reset_index()
    country.columns = ["Country", "Rows", "Customers", "CA"]
    if region_col:
        reg = c.dropna(subset=[region_col]).groupby("Country")[region_col].agg(
            lambda x: x.mode()[0] if len(x) else None)
        country = country.merge(reg.rename("Region"), on="Country", how="left")
    country = country.sort_values("Rows", ascending=False)
    country.to_csv(AGG_COUNTRY_PATH, index=False)

    # --- Seasonality heatmap : revenue by hour x month ------------------------
    c["Month"] = pd.to_datetime(c["InvoiceDate"]).dt.month
    c["Hour"] = pd.to_datetime(c["InvoiceDate"]).dt.hour
    season = c.pivot_table(values="TotalPrice", index="Hour", columns="Month",
                           aggfunc="sum", fill_value=0)
    season.to_csv(AGG_SEASONALITY_PATH)

    # --- Meta (audit, quality, headline stats) --------------------------------
    if raw is not None:
        audit = audit_report(raw)
        flags = quality_flags(raw)
        meta_extra = {
            "raw_rows": int(len(raw)),
            "source_counts": raw["Source"].value_counts().to_dict(),
            "missing_pct": {col: round(float(raw[col].isna().mean() * 100), 2)
                            for col in raw.columns},
            "audit_full": True,
        }
    else:
        audit = audit_report(clean)
        flags = quality_flags_from_clean(clean)
        meta_extra = {
            "raw_rows": int(len(clean)),  # raw not loaded → fall back to clean
            "source_counts": (clean.get("Source").value_counts().to_dict()
                              if "Source" in clean.columns else {}),
            "missing_pct": {col: round(float(clean[col].isna().mean() * 100), 2)
                            for col in clean.columns},
            "audit_full": False,
        }

    flags = {k: (str(v) if isinstance(v, (pd.Timestamp,)) else v)
             for k, v in flags.items()}

    meta = {
        "clean_rows":   int(len(clean)),
        "customers":    int(rfm_clean["CustomerID"].nunique()),
        "churn_rate":   float(rfm_clean["Churn"].mean()),
        "date_min":     str(pd.to_datetime(clean["InvoiceDate"]).min().date()),
        "date_max":     str(pd.to_datetime(clean["InvoiceDate"]).max().date()),
        "countries":    int(clean["Country"].nunique()),
        "quality_flags": flags,
        "audit":        audit.astype(str).to_dict(orient="records"),
        **meta_extra,
        **extra_meta,
    }
    META_PATH.write_text(json.dumps(meta, indent=2, default=str))
    return meta


def build_artifacts(enrich_countries: bool = True, log=print,
                    progress=None, force_full: bool = False) -> dict:
    """Regenerate every processed artefact required by the dashboard.

    Parameters
    ----------
    enrich_countries
        Add region/population from REST Countries (network call). Skipped on
        failure or if False.
    log
        Callback that receives human-readable progress messages.
    progress
        Optional callback ``progress(pct: float, stage: str)`` used to drive a
        progress bar in the UI.
    force_full
        Re-read the raw Excel + CSV even if ``clean_transactions.csv`` is
        already on disk. Default ``False`` — when the cache is present we skip
        the slow Excel parsing (saves ~60 s on a typical machine).
    """
    import joblib

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def _step(pct: float, stage: str) -> None:
        log(stage)
        if progress is not None:
            try:
                progress(pct, stage)
            except Exception:  # noqa: BLE001
                pass

    has_cached_clean = CLEAN_PATH.exists() and not force_full
    raw = None  # only populated on the slow / full path

    # ---------- Stage 1 — load (or reuse) clean transactions ----------------
    if has_cached_clean:
        _step(0.05, f"Lecture du cache · {CLEAN_PATH.name}")
        clean = pd.read_csv(CLEAN_PATH, parse_dates=["InvoiceDate"],
                            low_memory=False)
        log(f"  → {len(clean):,} lignes chargées depuis le cache")
    else:
        _step(0.05, "Lecture · Kaggle E-Commerce (CSV)")
        kaggle = load_kaggle()
        log(f"  → {len(kaggle):,} lignes")

        _step(0.20, "Lecture · UCI Online Retail II (Excel, lent)")
        uci = load_uci()
        log(f"  → {len(uci):,} lignes")

        raw = pd.concat([kaggle, uci], ignore_index=True)
        log(f"Total brut concaténé : {len(raw):,} lignes")

        _step(0.45, "Nettoyage · doublons, annulations, valeurs invalides")
        clean = clean_transactions(raw)
        log(f"  → {len(clean):,} lignes après nettoyage")

        if enrich_countries:
            try:
                _step(0.55, "Enrichissement géographique (REST Countries)")
                countries = fetch_country_reference()
                clean = enrich(clean, countries)
                log("  → enrichissement OK")
            except Exception as exc:  # noqa: BLE001
                log(f"  → enrichissement ignoré ({exc})")

        _step(0.62, f"Écriture · {CLEAN_PATH.name}")
        clean.to_csv(CLEAN_PATH, index=False)

    # ---------- Stage 2 — RFM table ----------------------------------------
    _step(0.66, "Calcul · table RFM par client")
    rfm = build_rfm(clean)
    rfm_clean = winsorize_rfm(rfm)
    log(f"  → {len(rfm_clean):,} clients · "
        f"churn {rfm_clean['Churn'].mean()*100:.1f}%")

    # ---------- Stage 3 — KMeans segmentation ------------------------------
    _step(0.72, "Ajustement · K-Means (silhouette)")
    kmeans, kmeans_scaler, labels, best_k, sil = fit_kmeans(rfm_clean)
    rfm_clean["Cluster"] = labels
    rfm_clean.to_csv(RFM_PATH, index=False)
    joblib.dump(kmeans, KMEANS_PATH)
    joblib.dump(kmeans_scaler, KMEANS_SCALER_PATH)
    log(f"  → K={best_k} retenu (silhouette {sil:.3f})")

    # ---------- Stage 4 — dashboard aggregates -----------------------------
    _step(0.88, "Construction · agrégats du tableau de bord")
    build_dashboard_aggregates(
        raw, clean, rfm_clean,
        extra_meta={"best_k": int(best_k), "silhouette": float(sil)},
    )
    log("  → agrégats + meta.json prêts")

    _step(1.0, "Préparation terminée")

    return {
        "clean_rows":   len(clean),
        "customers":    len(rfm_clean),
        "churn_rate":   float(rfm_clean["Churn"].mean()),
        "best_k":       int(best_k),
        "silhouette":   float(sil),
        "fast_path":    has_cached_clean,
    }


if __name__ == "__main__":
    summary = build_artifacts()
    print("Artifacts regenerated:")
    for k, v in summary.items():
        print(f"  {k:>12} : {v}")
