"""Vue Marketing — activation et liste opérationnelle.

Audience : équipes CRM et campagne. Récupérer les clients à risque,
comprendre leur segment, et déclencher des actions ciblées.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from utils import config as cfg  # noqa: E402
from components import ui, charts  # noqa: E402
from services import data_service as ds  # noqa: E402

ui.setup_page("Vue Marketing")
ui.page_header(
    eyebrow="Profil  ·  Marketing",
    title="Activation client",
    lead="Quels clients contacter en priorité, dans quel segment, "
         "et à quel moment de l'année.",
)
ui.require_artifacts()

rfm = ds.load_rfm()

# Score the entire base
try:
    rfm = ds.score_full_base()
except Exception as e:  # noqa: BLE001
    st.error(f"Impossible de scorer la base : {e}")
    st.stop()

# --- Headline KPIs ----------------------------------------------------------
n_high = int((rfm["Risk_Level"] == "Élevé").sum())
n_mod  = int((rfm["Risk_Level"] == "Modéré").sum())
n_low  = int((rfm["Risk_Level"] == "Faible").sum())

ui.metric_row([
    {"label": "Risque élevé",
     "value": f"{n_high:,}".replace(",", " "),
     "delta": "à contacter en priorité",
     "tone": "neg"},
    {"label": "Risque modéré",
     "value": f"{n_mod:,}".replace(",", " "),
     "delta": "à inclure dans les campagnes",
     "tone": "warn"},
    {"label": "Risque faible",
     "value": f"{n_low:,}".replace(",", " "),
     "delta": "à fidéliser",
     "tone": "pos"},
    {"label": "Probabilité moyenne",
     "value": f"{rfm['Churn_Probability'].mean()*100:.1f}%",
     "tone": "brand"},
])

# --- Cartographie des segments ---------------------------------------------
ui.section("Cartographie des clients",
           "Distribution dans l'espace log(Récence) × log(Montant).")

sample = rfm.sample(min(2500, len(rfm)), random_state=42).copy()
sample["log_Recency"]  = np.log1p(sample["Recency"])
sample["log_Monetary"] = np.log1p(sample["Monetary"])
st.plotly_chart(
    charts.scatter_clusters(sample, "log_Recency", "log_Monetary",
                             "Récence vs Montant (log)"),
    use_container_width=True,
)

# --- Top clients à risque --------------------------------------------------
ui.section("Top 50 clients à contacter",
           "Triés par probabilité de churn décroissante.")

top_n = st.slider("Nombre de clients à afficher", 10, 200, 50,
                  key="mkt_topn")

rfm_view = (rfm.copy()
            .sort_values("Churn_Probability", ascending=False)
            .head(top_n))

display_cols = ["CustomerID", "Recency", "Frequency", "Monetary",
                "AvgBasket", "Segment", "Churn_Probability", "Risk_Level"]

st.dataframe(
    rfm_view[display_cols].style.format({
        "Recency":    "{:.0f} j",
        "Frequency":  "{:.0f}",
        "Monetary":   f"{cfg.CURRENCY} {{:,.0f}}".replace(",", " "),
        "AvgBasket":  f"{cfg.CURRENCY} {{:,.2f}}".replace(",", " "),
        "Churn_Probability": "{:.1%}",
    }),
    use_container_width=True, height=420, hide_index=True,
)

# Download buttons
csv_high = rfm[rfm["Risk_Level"] == "Élevé"][display_cols].to_csv(index=False)
csv_top  = rfm_view[display_cols].to_csv(index=False)

dlc = st.columns(2)
with dlc[0]:
    st.download_button(
        f"Télécharger les {n_high} clients à risque élevé (CSV)",
        csv_high.encode("utf-8"),
        "clients_risque_eleve.csv", "text/csv",
        type="primary", use_container_width=True,
    )
with dlc[1]:
    st.download_button(
        f"Télécharger le top {top_n} affiché (CSV)",
        csv_top.encode("utf-8"),
        f"top_{top_n}_clients.csv", "text/csv",
        use_container_width=True,
    )

# --- Profil par segment ----------------------------------------------------
ui.section("Profil de chaque segment",
           "Cibler la bonne offre, au bon moment, au bon segment.")

profile = (rfm.groupby("Cluster")
           .agg(N=("CustomerID", "count"),
                Churn_rate=("Churn", "mean"),
                Recency=("Recency", "median"),
                Frequency=("Frequency", "median"),
                Monetary=("Monetary", "median"))
           .reset_index())
profile["Segment"] = profile["Cluster"].apply(
    lambda c: cfg.get_segment_info(c)["name"]
)

cols = st.columns(len(profile))
for col, (_, row) in zip(cols, profile.iterrows()):
    info = cfg.get_segment_info(int(row["Cluster"]))
    with col:
        with st.container(border=True):
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:.5rem;"
                f"margin-bottom:.6rem'>"
                f"<span style='width:10px;height:10px;border-radius:50%;"
                f"background:{info['color']}'></span>"
                f"<span style='font-weight:600;color:var(--cg-ink)'>"
                f"{info['name']}</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**{int(row['N']):,}**".replace(",", " ")
                + " clients · "
                + f"<span style='color:var(--cg-danger-strong)'>"
                f"{row['Churn_rate']*100:.0f}% churn</span>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Récence médiane : {row['Recency']:.0f} j · "
                f"Fréq. : {row['Frequency']:.0f}"
            )
            st.caption(info["desc"])

# --- Action playbook -------------------------------------------------------
ui.section("Action playbook",
           "Pour chaque segment : action recommandée, reach attendu, "
           "impact estimé. Ajustez les hypothèses ci-dessous.")

with st.container(border=True):
    cols = st.columns(3)
    with cols[0]:
        play_threshold = st.slider(
            "Seuil de score à contacter", 0.30, 0.90, 0.50, 0.05,
            key="mkt_play_threshold",
            help="Seul les clients avec un score ≥ seuil seront ciblés.",
        )
    with cols[1]:
        play_conv = st.slider(
            "Taux de conversion attendu (%)", 5, 40, 15,
            key="mkt_play_conv") / 100
    with cols[2]:
        play_avg_basket = st.number_input(
            f"Panier moyen retenu ({cfg.CURRENCY})",
            min_value=10, max_value=5_000, value=500, step=50,
            key="mkt_play_basket")

# Predefined action library — one per segment id
SEGMENT_ACTIONS: dict[int, dict] = {
    0: {
        "title":  "Programme de fidélité",
        "channel": "Email · push app · paliers de récompense",
        "tone":   "pos",
        "color":  "var(--cg-positive-strong)",
        "rationale": "Population peu à risque mais à grande valeur — "
                     "objectif fidéliser, pas reconquérir.",
    },
    1: {
        "title":  "Campagne reconquête multicanal",
        "channel": "Email + SMS · offre exclusive (-15 %) · suivi à J+7",
        "tone":   "neg",
        "color":  "var(--cg-danger-strong)",
        "rationale": "Forte concentration du risque — l'effort doit être "
                     "intensif et personnalisé.",
    },
}
default_action = {
    "title":  "Action standard",
    "channel": "Newsletter générale · à calibrer",
    "tone":   "warn",
    "color":  "var(--cg-warning-strong)",
    "rationale": "Segment additionnel — adapter l'effort au comportement.",
}

# Compute reach & expected impact per segment
st.markdown("&nbsp;", unsafe_allow_html=True)
for cid in sorted(rfm["Cluster"].unique()):
    seg_pop = rfm[rfm["Cluster"] == cid]
    info = cfg.get_segment_info(int(cid))
    action = SEGMENT_ACTIONS.get(int(cid), default_action)

    # Targets above threshold
    n_targets = int((seg_pop["Churn_Probability"] >= play_threshold).sum())
    n_saved = int(round(n_targets * play_conv))
    ca_saved = n_saved * play_avg_basket

    with st.container(border=True):
        c1, c2, c3 = st.columns([2.4, 1.2, 1.2])
        with c1:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:.5rem;"
                f"margin-bottom:.4rem'>"
                f"<span style='width:10px;height:10px;border-radius:50%;"
                f"background:{info['color']}'></span>"
                f"<span style='font-size:.78rem;color:var(--cg-muted);"
                f"letter-spacing:.1em;font-weight:600;text-transform:uppercase'>"
                f"{info['name']}</span></div>"
                f"<div style='font-size:1.05rem;font-weight:600;"
                f"color:var(--cg-ink);margin-bottom:.2rem'>"
                f"{action['title']}</div>"
                f"<div style='font-size:.85rem;color:var(--cg-muted);"
                f"font-style:italic;margin-bottom:.4rem'>"
                f"{action['channel']}</div>"
                f"<div style='font-size:.82rem;color:var(--cg-muted);"
                f"line-height:1.4'>{action['rationale']}</div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"<div style='font-size:.7rem;color:var(--cg-muted);"
                f"letter-spacing:.1em;font-weight:600;text-transform:uppercase;"
                f"margin-bottom:.2rem'>Reach (≥ {play_threshold:.0%})</div>"
                f"<div style='font-family:JetBrains Mono;font-size:1.5rem;"
                f"color:var(--cg-ink);font-weight:600'>"
                f"{n_targets:,}</div>".replace(",", " ")
                + f"<div style='font-size:.78rem;color:var(--cg-muted);"
                f"margin-top:.15rem'>clients à contacter</div>",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"<div style='font-size:.7rem;color:var(--cg-muted);"
                f"letter-spacing:.1em;font-weight:600;text-transform:uppercase;"
                f"margin-bottom:.2rem'>Impact attendu</div>"
                f"<div style='font-family:JetBrains Mono;font-size:1.5rem;"
                f"color:{action['color']};font-weight:600'>"
                f"{cfg.CURRENCY}{ca_saved/1000:.0f}K</div>"
                f"<div style='font-size:.78rem;color:var(--cg-muted);"
                f"margin-top:.15rem'>≈ {n_saved:,} clients retenus</div>"
                .replace(",", " "),
                unsafe_allow_html=True,
            )

# Total expected impact
total_targets = int((rfm["Churn_Probability"] >= play_threshold).sum())
total_saved   = int(round(total_targets * play_conv))
total_ca      = total_saved * play_avg_basket

st.markdown("&nbsp;", unsafe_allow_html=True)
st.markdown(
    f"<div style='background:var(--cg-accent-soft);border:1px solid "
    f"var(--cg-accent);border-radius:10px;padding:.85rem 1.1rem;"
    f"display:flex;justify-content:space-between;align-items:center'>"
    f"<div style='color:var(--cg-accent-2);font-weight:600;font-size:.95rem'>"
    f"Total attendu sur les hypothèses retenues</div>"
    f"<div style='font-family:JetBrains Mono;font-size:1.4rem;"
    f"color:var(--cg-accent-2);font-weight:700'>"
    f"{cfg.CURRENCY}{total_ca/1000:.0f}K · "
    f"{total_saved:,} clients sauvegardés"
    f"</div></div>".replace(",", " "),
    unsafe_allow_html=True,
)

# --- Saisonnalité ---------------------------------------------------------
ui.section("Top produits  ·  Saisonnalité",
           "Sur quoi capitaliser et à quel moment de l'année.")

p1, p2 = st.columns(2)
with p1:
    top = ds.load_top_products().head(12)
    st.plotly_chart(
        charts.bar(top, "Description", "CA",
                   f"Top produits par CA ({cfg.CURRENCY})", horizontal=True),
        use_container_width=True,
    )

season = ds.load_seasonality()
month_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
try:
    season.columns = [month_labels[int(c) - 1] for c in season.columns]
except Exception:  # noqa: BLE001
    pass

with p2:
    st.plotly_chart(
        charts.heatmap(season, "CA par heure × mois",
                        xlabel="Mois", ylabel="Heure"),
        use_container_width=True,
    )
st.caption(
    "Pic d'activité Q4 (Oct–Déc) — calibrer les campagnes de rétention "
    "avant la haute saison pour maximiser la conversion."
)

# --- CTAs vers le scoring ---------------------------------------------------
ui.section("Actions rapides")
cta = st.columns(2)
with cta[0]:
    with st.container(border=True):
        st.markdown(
            "<div style='font-weight:600;color:var(--cg-ink);"
            "margin-bottom:.4rem'>Scorer un client à la main</div>"
            "<div style='color:var(--cg-muted);font-size:.9rem;"
            "margin-bottom:.6rem'>"
            "Saisir un profil RFM et obtenir le score immédiatement."
            "</div>",
            unsafe_allow_html=True,
        )
        st.page_link("pages/4_Scoring_client.py",
                     label="Aller au scoring client →")
with cta[1]:
    with st.container(border=True):
        st.markdown(
            "<div style='font-weight:600;color:var(--cg-ink);"
            "margin-bottom:.4rem'>Scorer un fichier complet</div>"
            "<div style='color:var(--cg-muted);font-size:.9rem;"
            "margin-bottom:.6rem'>"
            "Importer un CSV et exporter les probabilités + niveaux de risque."
            "</div>",
            unsafe_allow_html=True,
        )
        st.page_link("pages/5_Scoring_par_lot.py",
                     label="Aller au scoring par lot →")

ui.footer()
