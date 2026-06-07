"""Vue Direction — tableau de bord exécutif.

Audience : équipe dirigeante. Lecture rapide de l'état du portefeuille client
et de l'enveloppe financière du dispositif anti-churn.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from utils import config as cfg  # noqa: E402
from components import ui, charts  # noqa: E402
from services import data_service as ds, model_service as ms  # noqa: E402

ui.setup_page("Vue Direction")
ui.page_header(
    eyebrow="Profil  ·  Direction",
    title="Tableau de bord exécutif",
    lead="Vue d'ensemble du portefeuille client : exposition au churn, "
         "segments à risque, et cadrage financier du dispositif de rétention.",
)
ui.require_artifacts()

meta = ds.load_meta()
rfm_scored = ds.score_full_base()

# --- Headline KPIs ----------------------------------------------------------
n_total = len(rfm_scored)
n_churn = int(rfm_scored["Churn"].sum())
churn_rate = n_churn / n_total
n_at_risk = int((rfm_scored["Churn_Probability"]
                 >= cfg.DEFAULT_DECISION_THRESHOLD).sum())

# Annualised CA exposure: Monetary covers the period so we annualise it
period_days = max(
    (pd.to_datetime(meta["date_max"]) - pd.to_datetime(meta["date_min"])).days,
    1,
)
years = period_days / 365.25
rfm_scored["MonetaryPerYear"] = rfm_scored["Monetary"] / years
clv_at_risk = float(
    (rfm_scored["MonetaryPerYear"] * rfm_scored["Churn_Probability"]).sum()
)
total_annual_ca = float(rfm_scored["MonetaryPerYear"].sum())
share_at_risk = clv_at_risk / max(total_annual_ca, 1)


def _short_money(v: float) -> str:
    if v >= 1e6:
        return f"{cfg.CURRENCY}{v/1e6:.2f} M"
    if v >= 1e3:
        return f"{cfg.CURRENCY}{v/1e3:.0f} K"
    return f"{cfg.CURRENCY}{v:.0f}"


ui.metric_row([
    {"label": "Clients analysés",
     "value": f"{n_total:,}".replace(",", " "),
     "tone": "brand"},
    {"label": "Taux de churn",
     "value": f"{churn_rate*100:.1f}%",
     "delta": f"{n_churn:,} clients > 90j".replace(",", " "),
     "tone": "neg"},
    {"label": "Revenu annuel à risque",
     "value": _short_money(clv_at_risk),
     "delta": f"{share_at_risk*100:.1f}% du CA annuel",
     "tone": "warn"},
    {"label": "Clients à risque",
     "value": f"{n_at_risk:,}".replace(",", " "),
     "delta": f"score modèle ≥ {cfg.DEFAULT_DECISION_THRESHOLD:.0%}",
     "tone": "neg"},
])

st.caption(
    f"Période couverte : {meta['date_min']} → {meta['date_max']} · "
    f"{meta['countries']} pays  ·  CA annuel base ≈ "
    f"{_short_money(total_annual_ca)}"
)

# --- Activité dans le temps ------------------------------------------------
ui.section("Activité commerciale",
           "Évolution mensuelle du chiffre d'affaires et des clients actifs.")

monthly = ds.load_monthly()
m1, m2 = st.columns(2)
with m1:
    st.plotly_chart(
        charts.area_timeseries(monthly, "YearMonth", "CA",
                                f"Chiffre d'affaires mensuel ({cfg.CURRENCY})"),
        use_container_width=True,
    )
with m2:
    st.plotly_chart(
        charts.area_timeseries(monthly, "YearMonth", "N_clients",
                                "Clients actifs par mois",
                                color=cfg.COLORS["positive"]),
        use_container_width=True,
    )

# --- Concentration du risque -----------------------------------------------
ui.section("Où se concentre le revenu à risque",
           "Le top 10–20 % des clients les plus à risque concentre une "
           "part disproportionnée du CA exposé.")

st.plotly_chart(
    charts.decile_risk_bar(rfm_scored,
                            prob_col="Churn_Probability",
                            monetary_col="MonetaryPerYear"),
    use_container_width=True,
)

# Single-line takeaway
top10_clv = float(
    (rfm_scored.nlargest(int(len(rfm_scored) * 0.10), "Churn_Probability")
     ["MonetaryPerYear"] * rfm_scored.nlargest(
         int(len(rfm_scored) * 0.10), "Churn_Probability"
     )["Churn_Probability"]).sum()
)
top10_share = top10_clv / max(clv_at_risk, 1)
st.caption(
    f"Le top 10 % des clients les plus à risque concentre "
    f"**{top10_share*100:.0f} %** du revenu annuel à risque "
    f"({_short_money(top10_clv)}). Cibler ce sous-ensemble suffit à "
    f"capturer la majorité de la valeur en jeu."
)

# --- État du portefeuille --------------------------------------------------
ui.section("État du portefeuille",
           "Distribution des clients par segment et exposition au risque.")

profile = (rfm_scored.groupby("Cluster")
           .agg(N=("CustomerID", "count"),
                Churn_rate=("Churn", "mean"),
                Recency=("Recency", "median"),
                Frequency=("Frequency", "median"),
                Monetary=("Monetary", "median"))
           .reset_index())
profile["Segment"] = profile["Cluster"].apply(
    lambda c: cfg.get_segment_info(c)["name"]
)

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(
        charts.donut(
            profile["Segment"].tolist(),
            profile["N"].tolist(),
            "Répartition par segment",
            colors=[cfg.get_segment_info(c)["color"]
                    for c in profile["Cluster"]],
        ),
        use_container_width=True,
    )
with c2:
    st.plotly_chart(charts.churn_by_segment(profile),
                    use_container_width=True)

# --- ROI simulator ----------------------------------------------------------
ui.section("Simulateur ROI",
           "Impact financier mensuel attendu, ajustable selon vos hypothèses.")

with st.container(border=True):
    cols = st.columns(2)
    with cols[0]:
        sim_clients = st.number_input(
            "Base mensuelle (clients actifs)",
            min_value=1_000, max_value=500_000, value=50_000, step=5_000,
            key="dir_sim_clients",
        )
        sim_churn = st.slider(
            "Taux de churn mensuel (%)", 1, 20, 5,
            key="dir_sim_churn") / 100
        sim_panier = st.number_input(
            "Panier moyen (DH)",
            min_value=50, max_value=5_000, value=500, step=50,
            key="dir_sim_panier",
        )
    with cols[1]:
        sim_precision = st.slider(
            "Précision du modèle (%)", 50, 95, 80,
            key="dir_sim_precision") / 100
        sim_conversion = st.slider(
            "Taux de conversion campagne (%)", 5, 40, 15,
            key="dir_sim_conversion") / 100
        sim_cost = st.number_input(
            "Coût mensuel (infra + ops, DH)",
            min_value=0, max_value=200_000, value=40_000, step=5_000,
            key="dir_sim_cost",
        )

# Computation
clients_perdus      = sim_clients * sim_churn
clients_cibles      = clients_perdus * sim_precision
clients_sauvegardes = clients_cibles * sim_conversion
ca_sauvegarde       = clients_sauvegardes * sim_panier
benefice_net        = ca_sauvegarde - sim_cost
roi                 = (benefice_net / sim_cost * 100) if sim_cost > 0 else 0

st.markdown("&nbsp;", unsafe_allow_html=True)
ui.metric_row([
    {"label": "Manque à gagner brut",
     "value": f"{clients_perdus*sim_panier:,.0f} DH".replace(",", " "),
     "tone": "neg"},
    {"label": "CA sauvegardé estimé",
     "value": f"{ca_sauvegarde:,.0f} DH".replace(",", " "),
     "delta": f"{int(clients_sauvegardes):,} clients retenus".replace(",", " "),
     "tone": "pos"},
    {"label": "Bénéfice net mensuel",
     "value": f"{benefice_net:,.0f} DH".replace(",", " "),
     "tone": "pos" if benefice_net > 0 else "neg"},
    {"label": "ROI prévisionnel",
     "value": f"{roi:.0f}%",
     "tone": "brand"},
])

# --- Recommandations -------------------------------------------------------
ui.section("Trois leviers d'action prioritaires")

recos = [
    ("Action immédiate",
     "Cibler les Dormants à score ≥ 0,65",
     "Email + SMS personnalisés, offre exclusive (-15 %).",
     "neg"),
    ("Court terme",
     "Programme de fidélité pour le Cœur actif",
     "Points + paliers, objectif +1 commande / trimestre.",
     "brand"),
    ("Moyen terme",
     "Optimiser l'expérience petits paniers",
     "Livraison gratuite à seuil + reco produit personnalisée.",
     "warn"),
]
for kicker, head, body, tone in recos:
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(
                f"<div style='font-size:.78rem;color:var(--cg-muted);"
                f"letter-spacing:.12em;font-weight:600;text-transform:uppercase;"
                f"margin-bottom:.3rem'>{kicker}</div>"
                f"<div style='font-size:1.08rem;color:var(--cg-ink);"
                f"font-weight:600;margin-bottom:.25rem'>{head}</div>"
                f"<div style='font-size:.92rem;color:var(--cg-muted);"
                f"font-style:italic'>{body}</div>",
                unsafe_allow_html=True,
            )
        with c2:
            tone_color = {
                "neg":   "var(--cg-danger)",
                "warn":  "var(--cg-warning)",
                "brand": "var(--cg-accent)",
                "pos":   "var(--cg-positive)",
            }[tone]
            st.markdown(
                f"<div style='height:100%;display:flex;align-items:center;"
                f"justify-content:flex-end'>"
                f"<div style='background:{tone_color};width:6px;height:48px;"
                f"border-radius:3px'></div></div>",
                unsafe_allow_html=True,
            )

# --- Limites méthodologiques -----------------------------------------------
ui.section("À garder à l'esprit")
with st.container(border=True):
    st.markdown(
        f"""
        - **Définition du churn** : inactivité de plus de **{cfg.CHURN_THRESHOLD_DAYS} jours**.
        - **Performance** : AUC ≈ {cfg.REPORTED_TEST_AUC:.2f} sur un hold-out 20 % stratifié.
        - **Biais** : base très majoritairement britannique. Les conclusions
          « marché international » sont à interpréter avec prudence.
        - **Validation production** : un A/B test confirmera l'effet causal
          des actions de rétention avant déploiement complet.
        """
    )

ui.footer()
