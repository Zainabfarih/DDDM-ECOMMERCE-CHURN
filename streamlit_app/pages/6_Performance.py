"""Page Performance — évaluation hold-out, ROC, importance, SHAP."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,  # noqa: E402
                              recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split  # noqa: E402

from utils import config as cfg  # noqa: E402
from components import ui, charts  # noqa: E402
from services import data_service as ds  # noqa: E402

ui.setup_page("Performance")
ui.page_header(
    eyebrow="Évaluation",
    title="Performance & interprétabilité",
    lead="Évaluation du modèle XGBoost sur un hold-out 20 % stratifié, "
         "avec les figures SHAP de référence.",
)
ui.require_artifacts()

try:
    bundle = ds.get_model_bundle()
except Exception as e:  # noqa: BLE001
    st.error(f"Impossible de charger le modèle : {e}")
    st.stop()

rfm = ds.load_rfm()


@st.cache_data(show_spinner=False)
def _evaluate_cached(rfm_signature: tuple, _bundle):
    """Hold-out evaluation. ``rfm_signature`` is just the (n_rows, churn_rate)
    tuple — when the user regenerates data and clear_caches() runs, the rfm
    cache invalidates which changes the signature and forces re-evaluation
    here too. ``_bundle`` is prefixed with ``_`` so Streamlit doesn't try to
    hash the model object."""
    X = rfm[cfg.MODEL_FEATURES]
    y = rfm[cfg.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    proba = _bundle.model.predict_proba(X_te.values)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "AUC":       roc_auc_score(y_te, proba),
        "F1":        f1_score(y_te, pred),
        "Precision": precision_score(y_te, pred),
        "Recall":    recall_score(y_te, pred),
    }
    fpr, tpr, _ = roc_curve(y_te, proba)
    cm = confusion_matrix(y_te, pred)
    return metrics, (fpr, tpr), cm, len(y_te)


def evaluate():
    """Pass a content signature so the cache invalidates when rfm changes."""
    sig = (len(rfm), float(rfm["Churn"].mean()))
    return _evaluate_cached(sig, bundle)


# Show a real loading state while the (cached) evaluation runs
eval_slot = st.empty()
with eval_slot.container():
    ui.loading_card("Évaluation du modèle sur le jeu de test…")

metrics, (fpr, tpr), cm, n_test = evaluate()
eval_slot.empty()

# --- Headline metrics -------------------------------------------------------
ui.metric_row([
    {"label": "AUC-ROC",   "value": f"{metrics['AUC']:.3f}",       "tone": "brand"},
    {"label": "F1-score",  "value": f"{metrics['F1']:.3f}",        "tone": "pos"},
    {"label": "Précision", "value": f"{metrics['Precision']:.3f}", "tone": "warn"},
    {"label": "Rappel",    "value": f"{metrics['Recall']:.3f}",    "tone": "neg"},
])
st.caption(
    f"Évalué sur {n_test:,} clients du jeu de test · ".replace(",", " ")
    + "XGBoost (n_estimators=150, max_depth=3, learning_rate=0.05)."
)

# --- ROC + confusion --------------------------------------------------------
ui.section("Courbe ROC & matrice de confusion")
r1, r2 = st.columns(2)
with r1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"XGBoost (AUC={metrics['AUC']:.3f})",
        line=dict(color=cfg.COLORS["accent"], width=2.5),
        fill="tozeroy", fillcolor="rgba(31,58,95,0.10)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Aléatoire",
        line=dict(color=cfg.COLORS["faint"], dash="dot", width=1.5),
        showlegend=False,
    ))
    fig.update_layout(
        title=dict(text="Courbe ROC", font=dict(size=14, color=cfg.COLORS["ink"]),
                    x=0, xanchor="left"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=cfg.COLORS["ink"], size=12),
        xaxis=dict(title="Taux de faux positifs",
                   gridcolor=cfg.COLORS["border"], range=[0, 1]),
        yaxis=dict(title="Taux de vrais positifs",
                   gridcolor=cfg.COLORS["border"], range=[0, 1]),
        legend=dict(x=0.5, y=0.08, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=8, r=8, t=44, b=8),
    )
    st.plotly_chart(fig, use_container_width=True)

with r2:
    labels = ["Actif", "Churn"]
    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=[f"Prédit {l}" for l in labels],
        y=[f"Réel {l}" for l in labels],
        colorscale=[[0, cfg.COLORS["surface_2"]],
                     [1, cfg.COLORS["accent"]]],
        text=cm, texttemplate="%{text}",
        textfont=dict(family="JetBrains Mono", color=cfg.COLORS["ink"], size=14),
        showscale=False,
    ))
    fig_cm.update_layout(
        title=dict(text="Matrice de confusion",
                    font=dict(size=14, color=cfg.COLORS["ink"]),
                    x=0, xanchor="left"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=cfg.COLORS["ink"], size=12),
        margin=dict(l=8, r=8, t=44, b=8),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# --- Feature importance -----------------------------------------------------
ui.section("Importance des variables",
           "Gain moyen du modèle XGBoost.")

importances = list(bundle.model.feature_importances_)
st.plotly_chart(
    charts.feature_importance(cfg.MODEL_FEATURES, importances,
                               "Importance des variables (XGBoost)"),
    use_container_width=True,
)
st.caption(
    "Le segment client (Cluster) et la fréquence d'achat ressortent comme "
    "principaux signaux de risque, en cohérence avec l'analyse SHAP."
)

# --- SHAP figures ----------------------------------------------------------
ui.section("Interprétabilité SHAP",
           "Importance globale, distribution des effets et explication locale.")

shap_imgs = [
    ("phase4_shap_global_bar.png", "Importance globale (SHAP)"),
    ("phase4_shap_beeswarm.png",   "Distribution des effets (beeswarm)"),
    ("phase4_shap_local.png",      "Explication locale (waterfall)"),
]
available = [(cfg.IMAGES_DIR / f, cap) for f, cap in shap_imgs
             if (cfg.IMAGES_DIR / f).exists()]
if available:
    icols = st.columns(len(available))
    for col_, (path, cap) in zip(icols, available):
        with col_:
            st.image(str(path), caption=cap, use_container_width=True)
else:
    ui.empty_state(
        "Figures SHAP introuvables",
        "Les images SHAP ne sont pas présentes dans le dossier `images/`. "
        "Régénérez-les pour les afficher ici.",
        glyph="!",
    )

cmp = cfg.IMAGES_DIR / "phase4_model_comparison.png"
if cmp.exists():
    ui.section("Comparaison des modèles",
               "Régression logistique vs Random Forest vs XGBoost.")
    st.image(str(cmp), use_container_width=True)

ui.footer()
