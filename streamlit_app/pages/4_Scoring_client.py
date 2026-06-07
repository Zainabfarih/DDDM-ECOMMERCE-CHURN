"""Page Scoring client — score individuel à partir du comportement RFM."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from utils import config as cfg  # noqa: E402
from components import ui, charts  # noqa: E402
from services import data_service as ds, model_service as ms  # noqa: E402

ui.setup_page("Scoring client")
ui.page_header(
    eyebrow="Inférence",
    title="Scoring d'un client",
    lead="Estimez la probabilité de churn d'un client à partir de son comportement "
         "d'achat. La récence sert au calcul du segment mais ne pèse pas dans le score.",
)
ui.require_artifacts()

try:
    bundle = ds.get_model_bundle()
except Exception as e:  # noqa: BLE001
    st.error(f"Impossible de charger le modèle : {e}")
    st.stop()

rfm = ds.load_rfm()

# --- Form -------------------------------------------------------------------
with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        recency = st.number_input(
            "Récence (jours depuis le dernier achat)",
            min_value=0, max_value=1000, value=45, step=1,
            help="Sert au calcul du segment K-Means, pas au score lui-même.",
        )
    with c2:
        frequency = st.number_input(
            "Fréquence (nombre de commandes)",
            min_value=1, max_value=500, value=5, step=1,
        )
    with c3:
        monetary = st.number_input(
            f"Montant total dépensé ({cfg.CURRENCY})",
            min_value=0.0, max_value=500_000.0, value=1500.0, step=50.0,
        )
    threshold = st.slider(
        "Seuil de décision (probabilité de churn)",
        0.05, 0.95, cfg.DEFAULT_DECISION_THRESHOLD, 0.05,
    )
    submitted = st.form_submit_button("Calculer le score", type="primary",
                                       use_container_width=True)

# --- Quick presets ----------------------------------------------------------
with st.expander("Profils types"):
    st.markdown(
        "- **Fidèle** — Récence 10 j · Fréquence 20 · Montant 8 000 £  \n"
        "- **Moyen**  — Récence 45 j · Fréquence 5 · Montant 1 500 £  \n"
        "- **À risque** — Récence 200 j · Fréquence 1 · Montant 80 £"
    )

# --- Run scoring (results persisted in session_state) -----------------------
if submitted:
    placeholder = st.empty()
    with placeholder.container():
        ui.loading_card("Calcul du score…")
    res = ms.predict_one(recency, frequency, monetary, bundle, threshold)
    placeholder.empty()
    st.session_state["client_score"] = {
        "result":    res,
        "threshold": threshold,
        "recency":   recency,
        "frequency": frequency,
        "monetary":  monetary,
    }

# --- Render result if present ----------------------------------------------
state = st.session_state.get("client_score")
if state is None:
    ui.empty_state(
        "En attente d'un client",
        "Renseignez la récence, la fréquence et le montant ci-dessus, "
        "puis cliquez sur « Calculer le score ».",
        glyph="?",
    )
    ui.footer()
    st.stop()

res = state["result"]
threshold = state["threshold"]
recency = state["recency"]
frequency = state["frequency"]
monetary = state["monetary"]

seg = cfg.SEGMENTS[res["segment"]]
is_risk = res["prediction"] == 1

ui.result_panel(at_risk=is_risk, proba=res["churn_proba"], threshold=threshold)

g1, g2 = st.columns([1, 1])
with g1:
    st.plotly_chart(charts.gauge(res["churn_proba"]),
                    use_container_width=True)
with g2:
    ui.metric(
        f"Segment · {seg['name']}",
        seg["label"],
        delta=seg["desc"],
        tone="pos" if res["segment"] == 0 else "neg",
    )
    ui.metric(
        "Panier moyen",
        f"{cfg.CURRENCY} {res['AvgBasket']:,.2f}".replace(",", " "),
        tone="warn",
    )

# Engineered features feeding the model
ui.section("Variables transmises au modèle",
           "Les quatre features attendues par le modèle XGBoost.")

fcols = st.columns(4)
feat_view = [
    ("LogFrequency", res["LogFrequency"]),
    ("LogMonetary",  res["LogMonetary"]),
    ("LogAvgBasket", res["LogAvgBasket"]),
    ("Cluster",      res["Cluster"]),
]
for col_, (name, val) in zip(fcols, feat_view):
    with col_:
        display = str(int(val)) if name == "Cluster" else f"{val:.3f}"
        ui.metric(name, display, tone="brand")

# Peer comparison
ui.section("Positionnement vs la base clients",
           f"Comparaison à la cohorte « {seg['name']} ».")

seg_pop = rfm[rfm.Cluster == res["segment"]]
pc1, pc2, pc3 = st.columns(3)
with pc1:
    pct_freq = (seg_pop["Frequency"] < frequency).mean() * 100
    ui.metric(
        "Fréquence",
        f"P{pct_freq:.0f}",
        delta=f"médiane segment : {seg_pop['Frequency'].median():.0f}",
        tone="brand",
    )
with pc2:
    pct_mon = (seg_pop["Monetary"] < monetary).mean() * 100
    med_mon = f"{cfg.CURRENCY}{seg_pop['Monetary'].median():,.0f}".replace(",", " ")
    ui.metric(
        "Montant",
        f"P{pct_mon:.0f}",
        delta=f"médiane segment : {med_mon}",
        tone="warn",
    )
with pc3:
    ui.metric(
        "Churn du segment",
        f"{seg_pop['Churn'].mean()*100:.0f}%",
        delta=f"{len(seg_pop):,} clients".replace(",", " "),
        tone="neg" if res["segment"] == 1 else "pos",
    )

ui.footer()
