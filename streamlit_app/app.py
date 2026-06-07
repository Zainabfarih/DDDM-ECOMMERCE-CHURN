"""ChurnGuard — landing page.

Run from the project root with:
    streamlit run streamlit_app/app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from utils import config as cfg  # noqa: E402
from components import ui  # noqa: E402
from services import data_service as ds  # noqa: E402

ui.setup_page("Accueil")

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
ui.hero(
    eyebrow="E-commerce · Rétention client",
    title_html="Anticiper le départ d'un client,<br>avant qu'il ne parte <em>vraiment</em>.",
    lead=(
        "ChurnGuard transforme l'historique d'achat de votre base clients en "
        "scores de risque exploitables, avec une lecture honnête de leurs limites. "
        "De l'audit des données à la prédiction interprétable — bout en bout."
    ),
)

status = ds.artifacts_status()
ready = status["_ready"]

# --------------------------------------------------------------------------- #
# Headline KPIs (only when artefacts exist)
# --------------------------------------------------------------------------- #
if ready:
    meta = ds.load_meta()
    customers = f"{meta['customers']:,}".replace(",", " ")
    clean_rows = f"{meta['clean_rows']/1e6:.2f} M"
    churn_pct = f"{meta['churn_rate']*100:.1f}%"
    countries = str(meta["countries"])

    ui.kpi_bar([
        {"label": "Clients analysés",       "value": customers,
         "sub":   f"{meta['date_min']} → {meta['date_max']}"},
        {"label": "Transactions nettoyées", "value": clean_rows,
         "sub":   "Kaggle + UCI Online Retail II"},
        {"label": "Taux de churn (>90j)",   "value": churn_pct,
         "sub":   "Inactivité supérieure à 90 jours"},
        {"label": "Performance modèle",     "value": f"AUC {cfg.REPORTED_TEST_AUC:.2f}",
         "sub":   f"XGBoost · {countries} pays couverts"},
    ])

# --------------------------------------------------------------------------- #
# Preparation panel — inventory + run button
# --------------------------------------------------------------------------- #

# Human-friendly description of every artefact + which "category" it belongs to
ARTIFACT_INFO: dict[str, dict[str, str]] = {
    "rfm_segmented.csv":    {"what": "Table RFM par client + label Churn"},
    "meta.json":            {"what": "Audit, période couverte, biais géographique"},
    "agg_monthly.csv":      {"what": "Chiffre d'affaires + clients actifs par mois"},
    "agg_top_products.csv": {"what": "Top 20 produits par chiffre d'affaires"},
    "agg_country.csv":      {"what": "Volume + CA par pays"},
    "agg_seasonality.csv":  {"what": "Heatmap CA par heure × mois"},
    "kmeans.pkl":           {"what": "Segmentation K-Means pré-ajustée pour le scoring"},
    "kmeans_scaler.pkl":    {"what": "Scaler associé au K-Means (log1p + standardisation)"},
}

present = [n for n, ok in status.items() if ok and not n.startswith("_")]
missing = [n for n, ok in status.items() if not ok and not n.startswith("_")]

if ready:
    ui.section("Préparation des données",
               "Tout est en place. Vous pouvez régénérer si vos données sources changent.")
else:
    ui.section("Préparation des données",
               f"{len(missing)} artefact(s) à produire à partir de vos données brutes.")

# Status banner
if ready:
    st.markdown(
        "<div style='background:var(--cg-positive-soft);border:1px solid #bee0db;"
        "border-radius:10px;padding:.85rem 1.1rem;margin:.2rem 0 1rem 0;"
        "color:var(--cg-positive);font-size:.92rem;'>"
        "<b>Application prête.</b> Les huit artefacts dérivés sont disponibles, "
        "toutes les pages peuvent se charger immédiatement."
        "</div>",
        unsafe_allow_html=True,
    )
else:
    has_clean_cache = (cfg.CLEAN_PATH.exists())
    eta = "10–20 secondes" if has_clean_cache else "≈ 1 minute"
    extra = (
        " (votre fichier <code>clean_transactions.csv</code> est déjà présent, "
        "le pipeline va le réutiliser)"
        if has_clean_cache else
        " (la lecture du fichier Excel UCI prend la majorité du temps)"
    )
    st.markdown(
        f"<div style='background:var(--cg-warning-soft);border:1px solid #f1d8a8;"
        f"border-radius:10px;padding:.85rem 1.1rem;margin:.2rem 0 1rem 0;"
        f"color:var(--cg-warning);font-size:.92rem;'>"
        f"<b>Préparation requise.</b> Cliquez sur le bouton ci-dessous pour "
        f"générer les artefacts manquants à partir de <code>data/raw/</code>. "
        f"Temps estimé : <b>{eta}</b>{extra}."
        f"</div>",
        unsafe_allow_html=True,
    )

# Two-column inventory
inv_cols = st.columns(2)
with inv_cols[0]:
    st.markdown("**Déjà disponibles**")
    if present:
        for n in present:
            info = ARTIFACT_INFO.get(n, {})
            st.markdown(
                f"<div style='padding:.45rem 0;border-bottom:1px solid var(--cg-border);"
                f"font-size:.88rem;'>"
                f"<span style='color:var(--cg-positive);font-weight:600'>✓</span>&nbsp; "
                f"<code>{n}</code><br>"
                f"<span style='color:var(--cg-muted);font-size:.78rem;'>"
                f"{info.get('what','')}</span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("Aucun artefact pour l'instant.")
with inv_cols[1]:
    if missing:
        st.markdown("**Sera produit en cliquant**")
        for n in missing:
            info = ARTIFACT_INFO.get(n, {})
            st.markdown(
                f"<div style='padding:.45rem 0;border-bottom:1px solid var(--cg-border);"
                f"font-size:.88rem;'>"
                f"<span style='color:var(--cg-warning);font-weight:600'>○</span>&nbsp; "
                f"<code>{n}</code><br>"
                f"<span style='color:var(--cg-muted);font-size:.78rem;'>"
                f"{info.get('what','')}</span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("**Tout est en place.**")
        st.caption("Cliquez ci-dessous uniquement pour régénérer.")

st.markdown("&nbsp;", unsafe_allow_html=True)

# Options + run button
opt_cols = st.columns([3, 2])
with opt_cols[0]:
    enrich = st.checkbox(
        "Enrichissement géographique (REST Countries API)",
        value=True,
        help="Ajoute région, sous-région, population. "
             "Décochez si vous n'avez pas d'accès réseau.",
    )
    force_full = st.checkbox(
        "Audit complet (relire les fichiers bruts Kaggle + Excel)",
        value=False,
        help="Décoché : si clean_transactions.csv existe déjà, on saute la "
             "lecture des sources brutes — beaucoup plus rapide. "
             "Coché : tout est rejoué depuis zéro, nécessaire pour produire "
             "les compteurs « doublons cross-source » de la page Qualité.",
    )
with opt_cols[1]:
    run_label = "Régénérer les données" if ready else "Préparer les données"
    run = st.button(run_label, type="primary", use_container_width=True)

# Run it
if run:
    progress_bar = st.progress(0.0, text="Initialisation…")
    elapsed_box = st.empty()
    log_box = st.empty()
    logs: list[str] = []
    start = time.time()

    def _log(msg: str) -> None:
        logs.append(str(msg))
        log_box.code("\n".join(logs[-15:]), language="text")

    def _on_progress(pct: float, stage: str) -> None:
        progress_bar.progress(min(pct, 1.0),
                              text=f"{stage}  ·  {pct*100:.0f}%")
        elapsed_box.markdown(
            f"<div style='font-family:var(--cg-mono);font-size:.8rem;"
            f"color:var(--cg-muted);margin:.2rem 0 .6rem 0;'>"
            f"⏱ {time.time()-start:5.1f}s écoulées</div>",
            unsafe_allow_html=True,
        )

    try:
        with st.spinner("Pipeline en cours…"):
            summary = ds.regenerate(
                enrich_countries=enrich,
                force_full=force_full,
                log=_log,
                progress=_on_progress,
            )
            ds.clear_caches()
    except FileNotFoundError as e:
        progress_bar.empty()
        st.error(
            "Données brutes introuvables. Vérifiez que `data/raw/data.csv` "
            f"et `data/raw/online_retail_II.xlsx` sont en place.\n\n{e}"
        )
        st.stop()
    except Exception as e:  # noqa: BLE001
        progress_bar.empty()
        st.error(f"Erreur pendant la préparation : {e}")
        st.stop()

    duration = time.time() - start
    progress_bar.progress(1.0, text="Préparation terminée")
    elapsed_box.empty()

    path_used = "rapide (cache réutilisé)" if summary.get("fast_path") else "complet"
    st.success(
        f"✓ Préparation terminée en **{duration:.0f} s** ({path_used}). "
        f"{summary['customers']:,} clients · ".replace(",", " ")
        + f"churn {summary['churn_rate']*100:.1f}% · "
        f"K={summary['best_k']} (silhouette {summary['silhouette']:.3f})."
    )
    time.sleep(1.0)
    st.rerun()

# --------------------------------------------------------------------------- #
# Module navigation
# --------------------------------------------------------------------------- #
ui.section("Six modules, un fil conducteur",
           "Du jeu de données aux décisions de rétention.")

modules = [
    ("01", "Données",
     "Sources combinées, schéma, volumétrie, table RFM client."),
    ("02", "Qualité",
     "Audit complétude, doublons cross-source, biais géographique."),
    ("03", "Segments & EDA",
     "Distributions, corrélations, tests Mann-Whitney, K-Means."),
    ("04", "Scoring client",
     "Score un client à partir de son comportement (RFM)."),
    ("05", "Scoring par lot",
     "Importez un CSV, récupérez les probabilités et niveaux de risque."),
    ("06", "Performance",
     "Métriques hold-out, ROC, importance des variables, SHAP."),
]

row1 = st.columns(3)
for col, (n, t, d) in zip(row1, modules[:3]):
    with col:
        ui.feature_tile(n, t, d)

row2 = st.columns(3)
for col, (n, t, d) in zip(row2, modules[3:]):
    with col:
        ui.feature_tile(n, t, d)

st.markdown("&nbsp;", unsafe_allow_html=True)
st.info("Utilisez le **menu de gauche** pour naviguer entre les modules.",
        icon="↗")

# --------------------------------------------------------------------------- #
# Methodology footnote
# --------------------------------------------------------------------------- #
ui.section("Notes méthodologiques", "Pour rester honnête sur ce que fait le modèle.")
st.markdown(
    f"""
- **Définition du churn** — un client est considéré *churned* si son dernier
  achat date de plus de **{cfg.CHURN_THRESHOLD_DAYS} jours**.
- **Variables** — le modèle utilise la fréquence d'achat, le montant total,
  le panier moyen et le segment K-Means du client. La récence est volontairement
  **exclue** : elle définit la cible et serait une fuite de données.
- **Performance rapportée** — AUC ≈ {cfg.REPORTED_TEST_AUC:.2f} sur un hold-out
  20 % stratifié.
- **Limites connues** — base très majoritairement britannique, période courte
  (2009-2011), pas de signal navigation/marketing — les conclusions sont
  appropriées pour la démarche pédagogique du projet, à recalibrer pour un
  contexte production.
    """
)

ui.footer()
