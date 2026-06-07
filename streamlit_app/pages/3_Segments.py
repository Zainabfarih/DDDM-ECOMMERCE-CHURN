"""Page Segments — distributions RFM, tests statistiques, K-Means."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from scipy import stats  # noqa: E402

from utils import config as cfg  # noqa: E402
from components import ui, charts  # noqa: E402
from services import data_service as ds  # noqa: E402

ui.setup_page("Segments")
ui.page_header(
    eyebrow="Analyse statistique",
    title="Segmentation & analyse statistique",
    lead="Distributions RFM, corrélations, tests de séparabilité, et clustering "
         "K-Means qui sert ensuite de feature au modèle.",
)
ui.require_artifacts()

rfm = ds.load_rfm()
RFM_VARS = ["Recency", "Frequency", "Monetary"]

# --- Distributions ----------------------------------------------------------
ui.section("Distributions RFM",
           "Forte asymétrie droite — la transformation log1p est appliquée en amont du modèle.")

c_top = st.columns([2, 3])
with c_top[0]:
    var = st.radio("Variable RFM", RFM_VARS, horizontal=True)
with c_top[1]:
    use_log = st.toggle("Échelle logarithmique (log1p)", value=False)

col = var
if use_log:
    log_col = f"Log{var}"
    if log_col not in rfm.columns:
        rfm = rfm.assign(**{log_col: np.log1p(rfm[var])})
    col = log_col

d1, d2 = st.columns(2)
with d1:
    st.plotly_chart(charts.histogram(rfm, col, f"Distribution — {col}"),
                    use_container_width=True)
with d2:
    st.plotly_chart(charts.box_by_churn(rfm, var, f"{var} par statut de churn"),
                    use_container_width=True)
st.caption(
    f"Skewness {var} (brut) = {rfm[var].skew():.2f} · "
    f"log1p = {np.log1p(rfm[var]).skew():.2f}"
)

# --- Correlations -----------------------------------------------------------
ui.section("Corrélations",
           "La récence est le signal numéro 1 — mais c'est mécanique : "
           "elle définit la cible.")

cc1, cc2 = st.columns([1.1, 1])
with cc1:
    st.plotly_chart(
        charts.correlation_heatmap(rfm, RFM_VARS + ["Churn"],
                                   "Heatmap de corrélation"),
        use_container_width=True,
    )
with cc2:
    corr_churn = rfm[RFM_VARS + ["Churn"]].corr()["Churn"].drop("Churn")
    corr_df = corr_churn.reset_index()
    corr_df.columns = ["Variable", "Corrélation"]
    st.dataframe(
        corr_df.style.format({"Corrélation": "{:.3f}"}),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "La récence est exclue des variables explicatives du modèle pour éviter "
        "une fuite de données (Churn = Recency > 90j)."
    )

# --- Statistical tests ------------------------------------------------------
ui.section("Tests Mann-Whitney U",
           "Séparabilité Actif / Churn sur chaque dimension RFM.")

actif = rfm[rfm.Churn == 0]
churned = rfm[rfm.Churn == 1]
rows = []
for v in RFM_VARS:
    u, p = stats.mannwhitneyu(actif[v], churned[v], alternative="two-sided")
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    rows.append({
        "Variable": v,
        "Médiane Actif": round(actif[v].median(), 2),
        "Médiane Churn": round(churned[v].median(), 2),
        "U": f"{u:,.0f}".replace(",", " "),
        "p-value": f"{p:.2e}",
        "Sig.": sig,
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.caption("Seuils : *** p<0.001 · ** p<0.01 · * p<0.05 · ns non significatif.")

# --- Clusters ---------------------------------------------------------------
ui.section("Segmentation K-Means",
           "K choisi par silhouette sur log(R, F, M) standardisées.")

profile = (rfm.groupby("Cluster")
           .agg(Recency=("Recency", "median"),
                Frequency=("Frequency", "median"),
                Monetary=("Monetary", "median"),
                Churn_rate=("Churn", "mean"),
                N=("CustomerID", "count"))
           .reset_index())
profile["Segment"] = profile["Cluster"].apply(
    lambda c: cfg.get_segment_info(c)["name"]
)

# Render one tile per actual cluster (handles K=2 today and K>2 later if the
# silhouette ever picks a different value).
seg_cols = st.columns(max(len(profile), 1))
for col_, (_, row) in zip(seg_cols, profile.iterrows()):
    cid = int(row["Cluster"])
    info = cfg.get_segment_info(cid)
    with col_:
        ui.metric(
            info["name"],
            f"{int(row['N']):,}".replace(",", " ") + " clients",
            delta=f"Churn {row['Churn_rate']*100:.0f}% · "
                  f"R={row['Recency']:.0f}j · F={row['Frequency']:.0f}",
            tone="pos" if cid == 0 else "neg",
        )

sc1, sc2 = st.columns(2)
sample = rfm.sample(min(2500, len(rfm)), random_state=42).copy()
sample["log_Recency"]   = np.log1p(sample["Recency"])
sample["log_Monetary"]  = np.log1p(sample["Monetary"])
sample["log_Frequency"] = np.log1p(sample["Frequency"])
with sc1:
    st.plotly_chart(
        charts.scatter_clusters(sample, "log_Recency", "log_Monetary",
                                "Récence vs Montant (log)"),
        use_container_width=True,
    )
with sc2:
    st.plotly_chart(charts.churn_by_segment(profile),
                    use_container_width=True)

# --- Seasonality ------------------------------------------------------------
ui.section("Saisonnalité",
           "Chiffre d'affaires par heure de la journée et par mois.")

season = ds.load_seasonality()
month_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
try:
    season.columns = [month_labels[int(c) - 1] for c in season.columns]
except Exception:  # noqa: BLE001
    pass
st.plotly_chart(
    charts.heatmap(season, "CA par heure × mois", xlabel="Mois", ylabel="Heure"),
    use_container_width=True,
)

ui.footer()
