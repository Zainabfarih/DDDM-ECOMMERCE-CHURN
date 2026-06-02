"""
preprocessing.py
Collecte, audit, nettoyage et enrichissement des données — Phase 2.

Sources :
  - Kaggle E-Commerce Data      (data/raw/data.csv)
  - UCI Online Retail II        (data/raw/online_retail_II.xlsx, 2 feuilles)
  - REST Countries API          (référentiel pays pour l'enrichissement)
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Chemins & constantes
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


KAGGLE_CSV = "data/raw/data.csv"
UCI_XLSX = "data/raw/online_retail_II.xlsx"
UCI_SHEETS = ["Year 2009-2010", "Year 2010-2011"]

COMMON_COLS = [
    "Invoice", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]

# Colonnes qui identifient une transaction unique (détection des doublons)
DEDUP_KEY = [
    "Invoice", "StockCode", "Quantity", "InvoiceDate", "CustomerID", "UnitPrice",
]

COUNTRIES_API = (
    "https://restcountries.com/v3.1/all"
    "?fields=name,region,subregion,population,latlng"
)

# Correspondance des libellés pays du dataset vers les noms de l'API
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


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
def load_kaggle(path: str = KAGGLE_CSV) -> pd.DataFrame:
    """Charge la source Kaggle et l'aligne sur le schéma commun."""
    df = pd.read_csv(_resolve(path), encoding="ISO-8859-1")
    df = df.rename(columns={"InvoiceNo": "Invoice"})
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], format="%m/%d/%Y %H:%M", errors="coerce"
    )
    df = df[COMMON_COLS].copy()
    df["Source"] = "Kaggle_ecommerce"
    return df


def load_uci(path: str = UCI_XLSX, sheets: list[str] | None = None) -> pd.DataFrame:
    """Charge la source UCI Online Retail II et l'aligne sur le schéma commun."""
    sheets = sheets or UCI_SHEETS
    frames = []
    for sheet in sheets:
        frames.append(pd.read_excel(_resolve(path), sheet_name=sheet))
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"Price": "UnitPrice", "Customer ID": "CustomerID"})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df[COMMON_COLS].copy()
    df["Source"] = "UCI_OnlineRetailII"
    return df


def load_raw() -> pd.DataFrame:
    """Concatène les deux sources harmonisées avec une colonne Source."""
    kaggle = load_kaggle()
    uci = load_uci()
    return pd.concat([kaggle, uci], ignore_index=True)


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
def audit_report(df: pd.DataFrame) -> pd.DataFrame:
    """Type, taux de valeurs manquantes, cardinalité et exemple par colonne."""
    rows = []
    for col in df.columns:
        non_na = df[col].dropna()
        rows.append(
            {
                "Colonne": col,
                "Type": str(df[col].dtype),
                "Manquants_%": round(df[col].isna().mean() * 100, 2),
                "Uniques": int(df[col].nunique(dropna=True)),
                "Exemple": non_na.iloc[0] if not non_na.empty else None,
            }
        )
    return pd.DataFrame(rows)


def count_cross_source_duplicates(df: pd.DataFrame) -> int:
    """Transactions identiques présentes dans les deux sources."""
    return int(df.duplicated(subset=DEDUP_KEY, keep="first").sum())


def quality_flags(df: pd.DataFrame) -> dict:
    """Indicateurs de qualité : doublons, valeurs manquantes, aberrantes."""
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
# Nettoyage
# --------------------------------------------------------------------------- #
def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime doublons, annulations, lignes sans client et valeurs non
    positives, puis crée la variable TotalPrice."""
    clean = df.drop_duplicates(subset=DEDUP_KEY, keep="first").copy()
    clean = clean[~clean["Invoice"].astype(str).str.startswith("C")]
    clean = clean.dropna(subset=["CustomerID"])
    clean = clean[clean["Quantity"] > 0]
    clean = clean[clean["UnitPrice"] > 0]

    clean["CustomerID"] = clean["CustomerID"].astype("int64")
    clean["InvoiceDate"] = pd.to_datetime(clean["InvoiceDate"])
    clean["TotalPrice"] = clean["Quantity"] * clean["UnitPrice"]
    return clean.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Enrichissement
# --------------------------------------------------------------------------- #
def fetch_country_reference(url: str = COUNTRIES_API) -> pd.DataFrame:
    """Récupère région, sous-région, population et coordonnées par pays."""
    with urllib.request.urlopen(url, timeout=30) as resp:
        api = json.load(resp)
    records = []
    for c in api:
        latlng = c.get("latlng") or [None, None]
        records.append(
            {
                "CountryAPI": c["name"]["common"],
                "Region": c.get("region"),
                "SubRegion": c.get("subregion"),
                "Population": c.get("population"),
                "Lat": latlng[0],
                "Lng": latlng[1],
            }
        )
    return pd.DataFrame(records)


def enrich(clean: pd.DataFrame, countries: pd.DataFrame) -> pd.DataFrame:
    """Joint les transactions au référentiel pays."""
    out = clean.copy()
    out["CountryStd"] = out["Country"].replace(COUNTRY_MAPPING)
    out = out.merge(
        countries, left_on="CountryStd", right_on="CountryAPI", how="left"
    )
    return out.drop(columns=["CountryAPI"])


# --------------------------------------------------------------------------- #
# Pipeline complet
# --------------------------------------------------------------------------- #
def build_processed_dataset(
    out_path: str = "data/processed/clean_transactions.csv",
) -> pd.DataFrame:
    """Enchaîne chargement, nettoyage et enrichissement, puis sauvegarde le CSV."""
    raw = load_raw()
    clean = clean_transactions(raw)
    countries = fetch_country_reference()
    enriched = enrich(clean, countries)

    out = _resolve(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out, index=False)
    return enriched


if __name__ == "__main__":
    df = build_processed_dataset()
    print("Dataset final :", df.shape)
    print("Sauvegardé dans data/processed/clean_transactions.csv")
