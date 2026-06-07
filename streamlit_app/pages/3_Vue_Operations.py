"""Vue Opérations — santé du dispositif, qualité des données, modèle.

Audience : équipes data / IT. Tout le détail technique en un seul endroit,
organisé en quatre onglets pour rester lisible.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,  # noqa: E402
                              recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split  # noqa: E402

from utils import config as cfg  # noqa: E402
from components import ui, charts  # noqa: E402
from services import data_service as ds  # noqa: E402

ui.setup_page("Vue Opérations")
ui.page_header(
    eyebrow="Profil  ·  Opérations",
    title="Santé du dispositif",
    lead="Pipeline, qualité des données, exploration statistique et "
         "performance du modèle — tout au même endroit.",
)
ui.require_artifacts()

meta = ds.load_meta()
status = ds.artifacts_status()
flags = meta.get("quality_flags", {})
audit_full = meta.get("audit_full", True)
rfm = ds.load_rfm()

tabs = st.tabs([
    "Pipeline",
    "Qualité des données",
    "Distributions & tests",
    "Modèle",
])


# =========================================================================
# Tab 1 — Pipeline & artefacts
# =========================================================================
with tabs[0]:
    artefacts = [n for n in status if not n.startswith("_")]
    n_present = sum(1 for n in artefacts if status[n])
    n_total = len(artefacts)

    ui.metric_row([
        {"label": "Artefacts disponibles",
         "value": f"{n_present} / {n_total}",
         "delta": "tous prêts" if n_present == n_total else "préparation requise",
         "tone": "pos" if n_present == n_total else "warn"},
        {"label": "Période couverte",
         "value": f"{meta['date_min']}",
         "delta": f"→ {meta['date_max']}",
         "tone": "brand"},
        {"label": "Lignes nettoyées",
         "value": f"{meta['clean_rows']/1e6:.2f} M",
         "delta": f"depuis {meta['raw_rows']/1e6:.2f} M brutes",
         "tone": "brand"},
        {"label": "Type d'audit",
         "value": "complet" if audit_full else "rapide",
         "delta": ("relit les fichiers bruts" if audit_full
                   else "depuis le cache nettoyé"),
         "tone": "pos" if audit_full else "warn"},
    ])

    # Source breakdown
    ui.section("Sources combinées")
    src = meta.get("source_counts", {})
    s1, s2 = st.columns(2)
    with s1:
        if src:
            st.plotly_chart(
                charts.donut(list(src.keys()), list(src.values()),
                             "Répartition des lignes brutes",
                             colors=[cfg.COLORS["pastel_blue"],
                                     cfg.COLORS["pastel_green"]]),
                use_container_width=True,
            )
    with s2:
        st.markdown(
            """
            | Source | Format | Volume brut |
            |---|---|---|
            | Kaggle — *E-Commerce Data* | CSV | 541 909 |
            | UCI — *Online Retail II*   | Excel (2 feuilles) | 1 067 371 |

            Schéma harmonisé : `Invoice`, `StockCode`, `Description`,
            `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.
            Déduplication cross-source sur la période 2010-2011.
            """
        )

    # Artefact checklist
    ui.section("État des artefacts")
    with st.container(border=True):
        cols = st.columns(2)
        half = (len(artefacts) + 1) // 2
        for i, a in enumerate(artefacts):
            col = cols[0] if i < half else cols[1]
            ok = status[a]
            glyph = "✓" if ok else "✗"
            color = ("var(--cg-positive-strong)" if ok
                     else "var(--cg-warning-strong)")
            col.markdown(
                f"<div style='padding:.35rem 0;font-size:.9rem'>"
                f"<span style='color:{color};font-weight:600'>{glyph}</span>"
                f"&nbsp;&nbsp;<code style='font-size:.86rem'>{a}</code></div>",
                unsafe_allow_html=True,
            )

    # Volumetry funnel
    ui.section("Effet du nettoyage",
               "Du brut concaténé au jeu exploitable.")
    funnel = pd.DataFrame({
        "Étape":  ["Lignes brutes", "Après nettoyage"],
        "Lignes": [meta["raw_rows"], meta["clean_rows"]],
    })
    st.plotly_chart(
        charts.bar(funnel, "Étape", "Lignes",
                   "Volumétrie avant / après nettoyage",
                   color=cfg.COLORS["pastel_green"]),
        use_container_width=True,
    )
    removed = meta["raw_rows"] - meta["clean_rows"]
    st.caption(
        f"{removed:,} lignes retirées".replace(",", " ")
        + " (doublons, annulations, CustomerID manquants, "
        "quantités ou prix ≤ 0)."
    )


# =========================================================================
# Tab 2 — Qualité des données
# =========================================================================
with tabs[1]:
    ui.metric_row([
        {"label": "Doublons cross-source",
         "value": (f"{int(flags.get('doublons_cross_source', 0)):,}"
                   .replace(",", " ")
                   if isinstance(flags.get('doublons_cross_source'), (int, float))
                   else "n/a"),
         "tone": "neg"},
        {"label": "Sans CustomerID",
         "value": f"{int(flags.get('sans_customer_id', 0)):,}".replace(",", " "),
         "tone": "warn"},
        {"label": "Annulations",
         "value": (f"{int(flags.get('annulations', 0)):,}".replace(",", " ")
                   if isinstance(flags.get('annulations'), (int, float))
                   else "n/a"),
         "tone": "warn"},
        {"label": "Quantité ≤ 0",
         "value": (f"{int(flags.get('quantite_negative_ou_nulle', 0)):,}"
                   .replace(",", " ")
                   if isinstance(flags.get('quantite_negative_ou_nulle'),
                                 (int, float))
                   else "n/a"),
         "tone": "neg"},
    ])

    if not audit_full:
        st.info(
            "Audit en mode rapide — certains compteurs ne sont pas calculés. "
            "Pour les obtenir, relancez la préparation depuis l'accueil avec "
            "l'option « Audit complet ».",
            icon="↻",
        )

    # Missing values
    ui.section("Complétude par colonne")
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

    # Geographic bias
    ui.section("Biais géographique",
               "Le marchand est britannique : la base est massivement orientée UK.")
    country = ds.load_country()
    total_rows = country["Rows"].sum()
    uk_share = (country.loc[country["Country"] == "United Kingdom", "Rows"].sum()
                / max(total_rows, 1))
    b1, b2 = st.columns([1, 2])
    with b1:
        ui.metric("Part du Royaume-Uni",
                  f"{uk_share*100:.1f}%",
                  delta="lecture « marché international » à interpréter avec prudence",
                  tone="neg")
    with b2:
        st.plotly_chart(
            charts.bar(country.head(10), "Country", "Rows",
                       "Top 10 pays par volume", horizontal=True),
            use_container_width=True,
        )

    # Audit dictionary
    ui.section("Dictionnaire & audit des variables")
    audit = meta.get("audit", [])
    if audit:
        st.dataframe(pd.DataFrame(audit), use_container_width=True,
                     hide_index=True)


# =========================================================================
# Tab 3 — Distributions & tests statistiques
# =========================================================================
with tabs[2]:
    RFM_VARS = ["Recency", "Frequency", "Monetary"]

    ui.section("Distributions RFM",
               "Forte asymétrie droite — log1p en amont du modèle.")

    c_top = st.columns([2, 3])
    with c_top[0]:
        var = st.radio("Variable RFM", RFM_VARS, horizontal=True,
                       key="ops_var")
    with c_top[1]:
        use_log = st.toggle("Échelle logarithmique (log1p)", value=False,
                            key="ops_log")

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
        st.plotly_chart(charts.box_by_churn(rfm, var,
                                             f"{var} par statut de churn"),
                        use_container_width=True)
    st.caption(
        f"Skewness {var} (brut) = {rfm[var].skew():.2f} · "
        f"log1p = {np.log1p(rfm[var]).skew():.2f}"
    )

    # Correlations
    ui.section("Corrélations",
               "La récence est le signal numéro 1 — mais c'est mécanique.")
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
            "La récence définit la cible (Churn = Recency > 90j) ; elle est "
            "exclue des variables explicatives pour éviter une fuite."
        )

    # Mann-Whitney tests
    ui.section("Tests Mann-Whitney U",
               "Séparabilité Actif / Churn sur chaque dimension RFM.")
    actif = rfm[rfm.Churn == 0]
    churned = rfm[rfm.Churn == 1]
    rows = []
    for v in RFM_VARS:
        u, p = stats.mannwhitneyu(actif[v], churned[v],
                                   alternative="two-sided")
        sig = "***" if p < 0.001 else (
            "**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        rows.append({
            "Variable": v,
            "Médiane Actif": round(actif[v].median(), 2),
            "Médiane Churn": round(churned[v].median(), 2),
            "U": f"{u:,.0f}".replace(",", " "),
            "p-value": f"{p:.2e}",
            "Sig.": sig,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                 hide_index=True)
    st.caption("Seuils : *** p<0.001 · ** p<0.01 · * p<0.05 · ns non significatif.")


# =========================================================================
# Tab 4 — Modèle (performance + interprétabilité)
# =========================================================================
with tabs[3]:
    try:
        bundle = ds.get_model_bundle()
    except Exception as e:  # noqa: BLE001
        st.error(f"Impossible de charger le modèle : {e}")
        st.stop()

    @st.cache_data(show_spinner=False)
    def _evaluate(rfm_signature, _bundle):
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
        return (metrics, fpr, tpr, confusion_matrix(y_te, pred),
                len(y_te), y_te.values, proba)

    sig = (len(rfm), float(rfm["Churn"].mean()))
    eval_slot = st.empty()
    with eval_slot.container():
        ui.loading_card("Évaluation du modèle…")
    metrics, fpr, tpr, cm, n_test, y_test_arr, proba_test = _evaluate(sig, bundle)
    eval_slot.empty()

    ui.metric_row([
        {"label": "AUC-ROC",   "value": f"{metrics['AUC']:.3f}",       "tone": "brand"},
        {"label": "F1-score",  "value": f"{metrics['F1']:.3f}",        "tone": "pos"},
        {"label": "Précision", "value": f"{metrics['Precision']:.3f}", "tone": "warn"},
        {"label": "Rappel",    "value": f"{metrics['Recall']:.3f}",    "tone": "neg"},
    ])
    st.caption(f"Évalué sur {n_test:,} clients du jeu de test."
               .replace(",", " "))

    # ROC + confusion
    r1, r2 = st.columns(2)
    with r1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"AUC = {metrics['AUC']:.3f}",
            line=dict(color=cfg.COLORS["accent_2"], width=2.5),
            fill="tozeroy", fillcolor="rgba(174,198,207,0.25)",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color=cfg.COLORS["faint"], dash="dot", width=1.5),
            showlegend=False,
        ))
        fig.update_layout(
            title=dict(text="Courbe ROC",
                        font=dict(size=14, color=cfg.COLORS["ink"])),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color=cfg.COLORS["ink"], size=12),
            xaxis=dict(title="Faux positifs",
                       gridcolor=cfg.COLORS["border"], range=[0, 1]),
            yaxis=dict(title="Vrais positifs",
                       gridcolor=cfg.COLORS["border"], range=[0, 1]),
            margin=dict(l=8, r=8, t=44, b=8),
            legend=dict(x=0.5, y=0.08),
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
            textfont=dict(family="JetBrains Mono",
                          color=cfg.COLORS["ink"], size=14),
            showscale=False,
        ))
        fig_cm.update_layout(
            title=dict(text="Matrice de confusion",
                        font=dict(size=14, color=cfg.COLORS["ink"])),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color=cfg.COLORS["ink"], size=12),
            margin=dict(l=8, r=8, t=44, b=8),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Feature importance
    ui.section("Importance des variables",
               "Gain moyen du modèle XGBoost.")
    importances = list(bundle.model.feature_importances_)
    st.plotly_chart(
        charts.feature_importance(cfg.MODEL_FEATURES, importances,
                                   "Importance des variables"),
        use_container_width=True,
    )

    # Calibration & lift — proves the model is trustworthy as a probability
    ui.section("Calibration & courbe de gain",
               "Les scores ne sont pas que des classements : ils approximent "
               "la vraie probabilité de churn.")
    cl1, cl2 = st.columns(2)
    with cl1:
        st.plotly_chart(
            charts.calibration_plot(y_test_arr, proba_test),
            use_container_width=True,
        )
        st.caption(
            "Lecture : un point sur la diagonale signifie que les clients "
            "scorés à p ≈ 0,7 churnent effectivement à 70 %."
        )
    with cl2:
        st.plotly_chart(
            charts.lift_curve(y_test_arr, proba_test),
            use_container_width=True,
        )
        # Compute lift at top 20%
        order = np.argsort(-proba_test)
        y_sorted = y_test_arr[order]
        top20 = int(len(y_sorted) * 0.20)
        captured = float(y_sorted[:top20].sum() / max(y_sorted.sum(), 1))
        st.caption(
            f"En contactant les **20 % de la base** au score le plus élevé, "
            f"on capture **{captured*100:.0f} %** du churn — soit "
            f"**{captured/0.20:.1f}× mieux** que de tirer au hasard."
        )

    # SHAP figures from notebooks
    ui.section("Interprétabilité SHAP",
               "Importance globale, distribution des effets, explication locale.")
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
            "Les images ne sont pas dans `images/`.",
            glyph="!",
        )

ui.footer()
