"""Page Données — sources combinées et table RFM par client."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from utils import config as cfg  # noqa: E402
from components import ui, charts  # noqa: E402
from services import data_service as ds  # noqa: E402

ui.setup_page("Données")
ui.page_header(
    eyebrow="Vue d'ensemble",
    title="Données",
    lead="Deux sources transactionnelles harmonisées, puis condensées en une "
         "table RFM client. Tout part de là.",
)
ui.require_artifacts()

meta = ds.load_meta()
rfm = ds.load_rfm()

# --- Headline ---------------------------------------------------------------
ui.metric_row([
    {"label": "Lignes brutes",        "value": f"{meta['raw_rows']/1e6:.2f} M",
     "tone": ""},
    {"label": "Après nettoyage",      "value": f"{meta['clean_rows']/1e6:.2f} M",
     "tone": "pos"},
    {"label": "Clients (RFM)",        "value": f"{meta['customers']:,}".replace(",", " "),
     "tone": "brand"},
    {"label": "Pays distincts",       "value": str(meta["countries"]),
     "tone": ""},
])

# --- Sources ----------------------------------------------------------------
ui.section("Sources de données",
           "Schéma harmonisé : Invoice, StockCode, Description, Quantity, "
           "InvoiceDate, UnitPrice, CustomerID, Country.")

c1, c2 = st.columns([1, 1.2])
with c1:
    src = meta.get("source_counts", {})
    if src:
        st.plotly_chart(
            charts.donut(list(src.keys()), list(src.values()),
                         "Répartition des lignes brutes",
                         colors=[cfg.COLORS["accent"], cfg.COLORS["positive"]]),
            use_container_width=True,
        )
with c2:
    st.markdown(
        """
        | Source | Format | Volume brut |
        |---|---|---|
        | Kaggle — *E-Commerce Data* | CSV | 541 909 |
        | UCI — *Online Retail II*   | Excel (2 feuilles) | 1 067 371 |

        Les deux jeux couvrent la même boutique en ligne britannique sur
        2009-2011. La plage 2010-2011 se chevauche : la déduplication
        cross-source est faite sur (Invoice, StockCode, Quantity, Date,
        CustomerID, UnitPrice).

        L'enrichissement géographique se fait via [REST Countries]
        (https://restcountries.com/) — région, sous-région, population.
        """
    )

# --- Monthly activity -------------------------------------------------------
ui.section("Activité commerciale dans le temps",
           "Pic d'activité en Q4 — saisonnalité typique du retail.")

monthly = ds.load_monthly()
t1, t2 = st.columns(2)
with t1:
    st.plotly_chart(
        charts.area_timeseries(monthly, "YearMonth", "CA",
                               f"Chiffre d'affaires mensuel ({cfg.CURRENCY})"),
        use_container_width=True,
    )
with t2:
    st.plotly_chart(
        charts.area_timeseries(monthly, "YearMonth", "N_clients",
                               "Clients actifs par mois",
                               color=cfg.COLORS["positive"]),
        use_container_width=True,
    )

# --- Top products & countries ----------------------------------------------
ui.section("Produits et marchés")
p1, p2 = st.columns(2)
with p1:
    top = ds.load_top_products().head(12)
    st.plotly_chart(
        charts.bar(top, "Description", "CA",
                   f"Top produits par CA ({cfg.CURRENCY})", horizontal=True),
        use_container_width=True,
    )
with p2:
    country = ds.load_country().head(12)
    st.plotly_chart(
        charts.bar(country, "Country", "Rows",
                   "Top pays par volume de transactions",
                   horizontal=True, color=cfg.COLORS["positive"]),
        use_container_width=True,
    )

# --- RFM table preview ------------------------------------------------------
ui.section("Table RFM par client",
           "Récence, fréquence, montant + label churn pour chaque client.")

st.dataframe(
    rfm[["CustomerID", "Recency", "Frequency", "Monetary", "AvgBasket",
         "Cluster", "Segment", "Statut"]].head(400),
    use_container_width=True,
    height=340,
    hide_index=True,
)
st.download_button(
    "Télécharger la table RFM (CSV)",
    rfm.to_csv(index=False).encode("utf-8"),
    "rfm_segmented.csv", "text/csv",
)

ui.footer()
