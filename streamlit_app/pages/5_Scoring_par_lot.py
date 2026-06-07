"""Page Scoring par lot — scoring d'un fichier CSV de clients."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from utils import config as cfg  # noqa: E402
from components import ui, charts  # noqa: E402
from services import data_service as ds, model_service as ms  # noqa: E402

ui.setup_page("Scoring par lot")
ui.page_header(
    eyebrow="Inférence",
    title="Scoring par lot",
    lead="Importez un fichier de clients (Recency, Frequency, Monetary) et "
         "récupérez les probabilités de churn et niveaux de risque associés.",
)
ui.require_artifacts()

try:
    bundle = ds.get_model_bundle()
except Exception as e:  # noqa: BLE001
    st.error(f"Impossible de charger le modèle : {e}")
    st.stop()

# --- Template + controls ----------------------------------------------------
template = pd.DataFrame({
    "CustomerID": [10001, 10002, 10003],
    "Recency":    [12, 210, 65],
    "Frequency":  [18, 1, 4],
    "Monetary":   [7600.0, 95.0, 1240.0],
})

c_top = st.columns([3, 2])
with c_top[0]:
    st.markdown(
        "**Format attendu** — un CSV avec a minima les colonnes "
        "`Recency`, `Frequency`, `Monetary`. `CustomerID` (et toute autre "
        "colonne) est conservée dans la sortie."
    )
    st.download_button(
        "Télécharger un modèle de fichier",
        template.to_csv(index=False).encode(),
        "template_clients.csv", "text/csv",
    )
with c_top[1]:
    threshold = st.slider("Seuil de décision",
                          0.05, 0.95, cfg.DEFAULT_DECISION_THRESHOLD, 0.05,
                          key="batch_threshold")

uploaded = st.file_uploader("Déposez votre fichier CSV", type=["csv"],
                             key="batch_upload")
use_demo = st.checkbox("Utiliser plutôt la base clients du projet (RFM)",
                       value=False, key="batch_use_demo")

# Determine input source
df_in: pd.DataFrame | None = None
if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
    except Exception as e:  # noqa: BLE001
        st.error(f"Lecture du fichier impossible : {e}")
elif use_demo:
    df_in = ds.load_rfm()[["CustomerID", "Recency", "Frequency", "Monetary"]].copy()

# --- Empty state ------------------------------------------------------------
if df_in is None:
    # Clear any previous scoring result if the input was removed
    st.session_state.pop("batch_scored", None)
    ui.empty_state(
        "Aucun fichier chargé",
        "Téléchargez le modèle ci-dessus pour visualiser le format attendu, "
        "ou cochez « base clients du projet » pour faire un test rapide.",
        glyph="↑",
    )
    ui.footer()
    st.stop()

ui.section("Aperçu du fichier")
st.dataframe(df_in.head(10), use_container_width=True, hide_index=True)

# --- Run scoring (writes into session_state so results survive reruns) ------
if st.button("Lancer le scoring", type="primary", use_container_width=True,
             key="batch_run"):
    placeholder = st.empty()
    with placeholder.container():
        ui.loading_card(
            f"Scoring de {len(df_in):,} clients en cours…".replace(",", " ")
        )
    try:
        scored = ms.predict_batch(df_in, bundle, threshold)
    except ValueError as e:
        placeholder.empty()
        st.error(str(e))
        st.stop()
    placeholder.empty()
    # Surface any rows the model couldn't score (non-numeric / blank R/F/M).
    dropped = int(scored.attrs.get("dropped_rows", 0))
    if dropped > 0:
        st.warning(
            f"{dropped:,} ligne(s) ignorée(s) : Recency, Frequency ou "
            f"Monetary non numériques.".replace(",", " "),
            icon="!",
        )
    # Persist across reruns (toggle, etc.). Keep alongside the threshold so we
    # can detect when the user changed it and invalidate the cached result.
    st.session_state["batch_scored"] = scored
    st.session_state["batch_scored_threshold"] = threshold
    st.session_state["batch_scored_size"] = len(df_in)

# --- Render results from session_state --------------------------------------
scored = st.session_state.get("batch_scored")
if scored is not None:
    # If the user moved the threshold slider after scoring, tell them the
    # result still reflects the old threshold and offer a one-click refresh.
    cached_threshold = st.session_state.get("batch_scored_threshold", threshold)
    if abs(cached_threshold - threshold) > 1e-9:
        st.info(
            f"Les résultats affichés utilisent le seuil **{cached_threshold:.0%}**. "
            f"Cliquez sur « Lancer le scoring » pour rafraîchir avec "
            f"**{threshold:.0%}**.",
            icon="↻",
        )

    n_risk = int(scored["Prediction"].sum())

    ui.section("Synthèse")
    ui.metric_row([
        {"label": "Clients scorés",
         "value": f"{len(scored):,}".replace(",", " "),
         "tone": "brand"},
        {"label": "Détectés à risque",
         "value": f"{n_risk:,}".replace(",", " "),
         "tone": "neg"},
        {"label": "Taux à risque",
         "value": f"{n_risk/len(scored)*100:.1f}%",
         "tone": "warn"},
        {"label": "Probabilité moyenne",
         "value": f"{scored['Churn_Probability'].mean()*100:.1f}%",
         "tone": "pos"},
    ])

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            charts.proba_distribution(
                scored["Churn_Probability"].values,
                threshold=cached_threshold,
            ),
            use_container_width=True,
        )
    with c2:
        risk_counts = (
            scored["Risk_Level"]
            .value_counts()
            .reindex(["Faible", "Modéré", "Élevé"])
            .fillna(0)
        )
        st.plotly_chart(
            charts.donut(
                risk_counts.index.tolist(),
                risk_counts.values.tolist(),
                "Répartition par niveau de risque",
                colors=[cfg.COLORS["positive"], cfg.COLORS["warning"],
                         cfg.COLORS["danger"]],
            ),
            use_container_width=True,
        )

    ui.section("Résultats détaillés")
    show_risk_only = st.toggle("Afficher uniquement les clients à risque",
                                value=False, key="batch_risk_only")
    view = scored[scored["Prediction"] == 1] if show_risk_only else scored
    st.caption(
        f"{len(view):,} ligne(s) affichée(s)".replace(",", " ")
        + (" · filtre actif : à risque uniquement" if show_risk_only else "")
    )
    st.dataframe(view, use_container_width=True, height=380, hide_index=True)

    st.download_button(
        "Télécharger les résultats (CSV)",
        scored.to_csv(index=False).encode("utf-8"),
        "churn_predictions.csv", "text/csv", type="primary",
    )

ui.footer()
