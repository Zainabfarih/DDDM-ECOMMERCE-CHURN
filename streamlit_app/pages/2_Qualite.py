"""Page Qualité — audit complétude, cohérence, doublons, biais."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from utils import config as cfg  # noqa: E402
from components import ui, charts  # noqa: E402
from services import data_service as ds  # noqa: E402

ui.setup_page("Qualité")
ui.page_header(
    eyebrow="Audit qualité",
    title="Qualité des données",
    lead="Complétude, cohérence, doublons, biais géographique. "
         "Ce que les chiffres rapportés par la suite valent vraiment.",
)
ui.require_artifacts()

meta = ds.load_meta()
flags = meta.get("quality_flags", {})

# --- Quality flags ----------------------------------------------------------
ui.section("Indicateurs de qualité",
           "Mesurés sur les données brutes combinées.")

ui.metric_row([
    {"label": "Doublons cross-source",
     "value": f"{flags.get('doublons_cross_source', 0):,}".replace(",", " "),
     "tone": "neg"},
    {"label": "Sans CustomerID",
     "value": f"{flags.get('sans_customer_id', 0):,}".replace(",", " "),
     "tone": "warn"},
    {"label": "Annulations (préfixe C)",
     "value": f"{flags.get('annulations', 0):,}".replace(",", " "),
     "tone": "warn"},
    {"label": "Quantité ≤ 0",
     "value": f"{flags.get('quantite_negative_ou_nulle', 0):,}".replace(",", " "),
     "tone": "neg"},
])

# --- Missing values ---------------------------------------------------------
ui.section("Complétude par colonne",
           "Pourcentage de valeurs manquantes sur le schéma harmonisé.")

miss = meta.get("missing_pct", {})
if miss:
    miss_df = (pd.DataFrame({"Colonne": list(miss.keys()),
                             "Manquants_%": list(miss.values())})
               .sort_values("Manquants_%", ascending=False))
    st.plotly_chart(
        charts.bar(miss_df, "Colonne", "Manquants_%",
                   "Valeurs manquantes (%)"),
        use_container_width=True,
    )

# --- Geographic bias --------------------------------------------------------
ui.section(
    "Biais géographique",
    "Le marchand est britannique : la base est massivement orientée Royaume-Uni.",
)

country = ds.load_country()
total_rows = country["Rows"].sum()
uk_share = country.loc[country["Country"] == "United Kingdom", "Rows"].sum() / max(total_rows, 1)

b1, b2 = st.columns([1, 2])
with b1:
    ui.metric("Part du Royaume-Uni", f"{uk_share*100:.1f}%",
              delta="Toute lecture « marché international » est à interpréter avec prudence.",
              tone="neg")
with b2:
    st.plotly_chart(
        charts.bar(country.head(10), "Country", "Rows",
                   "Top 10 pays par volume", horizontal=True),
        use_container_width=True,
    )

# --- Cleaning funnel --------------------------------------------------------
ui.section("Effet du nettoyage",
           "Du brut concaténé au jeu exploitable.")

funnel = pd.DataFrame({
    "Étape":  ["Lignes brutes", "Après nettoyage"],
    "Lignes": [meta["raw_rows"], meta["clean_rows"]],
})
st.plotly_chart(
    charts.bar(funnel, "Étape", "Lignes",
               "Volumétrie avant / après nettoyage",
               color=cfg.COLORS["positive"]),
    use_container_width=True,
)
removed = meta["raw_rows"] - meta["clean_rows"]
st.caption(
    f"{removed:,} lignes retirées".replace(",", " ")
    + " (doublons cross-source, annulations, CustomerID manquants, "
    "quantités ou prix ≤ 0)."
)

# --- Data dictionary --------------------------------------------------------
ui.section("Dictionnaire & audit des variables")
audit = meta.get("audit", [])
if audit:
    st.dataframe(pd.DataFrame(audit), use_container_width=True, hide_index=True)

ui.footer()
