"""Reusable UI primitives for the ChurnGuard Streamlit app.

The look & feel is owned by ``assets/styles.css``. Helpers here only emit the
HTML/CSS class names declared in that stylesheet, so the visual identity stays
consistent across every page.

IMPORTANT — every helper that emits HTML returns it as a single line, with no
leading indentation. Streamlit's markdown parser otherwise treats 4+ space
indentation as a code block and renders the literal markup.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import config as cfg  # noqa: E402


def _html(s: str) -> None:
    """Emit a raw HTML string. Strips inter-tag whitespace to prevent the
    markdown parser from interpreting indented blocks as code."""
    import re
    flat = re.sub(r">\s+<", "><", s.strip())
    st.markdown(flat, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Page bootstrap
# --------------------------------------------------------------------------- #
def setup_page(title: str, layout: str = "wide") -> None:
    """Set Streamlit page config, inject CSS, render brand + sidebar status."""
    st.set_page_config(
        page_title=f"{title} · {cfg.APP_TITLE}",
        page_icon="▪",
        layout=layout,
        initial_sidebar_state="expanded",
        menu_items={
            "About": (
                f"**{cfg.APP_TITLE}** — {cfg.APP_SUBTITLE}\n\n"
                f"Projet DDDM · {cfg.AUTHORS}"
            ),
        },
    )
    _inject_css()
    _sidebar_brand()
    _sidebar_status()


def _inject_css() -> None:
    css = _load_css(cfg.ASSETS_DIR / "styles.css")
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def _load_css(path: Path) -> str:
    """Read the CSS file once per Streamlit session (not per page load)."""
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _sidebar_brand() -> None:
    with st.sidebar:
        _html(
            f"<div class='cg-brand'>"
            f"<span class='mark'></span>"
            f"<span class='name'>Churn<em>Guard</em></span>"
            f"</div>"
            f"<div class='cg-tag-line'>{cfg.APP_SUBTITLE}</div>"
        )


def _sidebar_status() -> None:
    """Lightweight 'data ready / not ready' panel in the sidebar.

    Stashes the status dict on st.session_state so the page-level
    require_artifacts() guard can reuse it without a second filesystem hit.
    """
    from services import data_service as ds

    status = ds.artifacts_status()
    st.session_state["_cg_status"] = status   # consumed by require_artifacts()
    ready = status["_ready"]

    if ready:
        try:
            meta = ds.load_meta()
            customers = f"{meta['customers']:,}".replace(",", " ")
            churn = f"{meta['churn_rate']*100:.1f}%"
            period = f"{meta['date_min']} → {meta['date_max']}"
        except Exception:  # noqa: BLE001
            customers = churn = period = "—"
        with st.sidebar:
            _html(
                f"<div class='cg-side-meta'>"
                f"<div class='row'><span><span class='dot ok'></span>Données</span><b>prêtes</b></div>"
                f"<div class='row'><span>Clients</span><b>{customers}</b></div>"
                f"<div class='row'><span>Churn 90j</span><b>{churn}</b></div>"
                f"<div class='row' style='margin-top:.4rem;color:var(--cg-faint);'>"
                f"<span>Période</span>"
                f"<b style='font-family:var(--cg-mono);font-size:.72rem'>{period}</b>"
                f"</div>"
                f"</div>"
            )
    else:
        with st.sidebar:
            _html(
                "<div class='cg-side-meta'>"
                "<div class='row'><span><span class='dot warn'></span>Données</span>"
                "<b>à préparer</b></div>"
                "<div style='color:var(--cg-muted);font-size:.78rem;margin-top:.4rem;line-height:1.5'>"
                "Lancez la préparation depuis l'accueil pour activer les modules."
                "</div>"
                "</div>"
            )


# --------------------------------------------------------------------------- #
# Headers
# --------------------------------------------------------------------------- #
def hero(eyebrow: str, title_html: str, lead: str) -> None:
    """Landing-page hero. ``title_html`` may contain ``<em>…</em>`` for accent."""
    _html(
        f"<div class='cg-hero'>"
        f"<div class='eyebrow'>{eyebrow}</div>"
        f"<h1>{title_html}</h1>"
        f"<p class='lead'>{lead}</p>"
        f"</div>"
    )


def page_header(eyebrow: str, title: str, lead: str = "") -> None:
    """Compact per-page header used at the top of every secondary page."""
    lead_html = f"<p>{lead}</p>" if lead else ""
    _html(
        f"<div class='cg-page-head'>"
        f"<div class='eyebrow'>{eyebrow}</div>"
        f"<h1>{title}</h1>"
        f"{lead_html}"
        f"</div>"
    )


def section(title: str, hint: str = "") -> None:
    hint_html = f"<span class='hint'>{hint}</span>" if hint else ""
    _html(
        f"<div class='cg-section'>"
        f"<h3>{title}</h3>"
        f"{hint_html}"
        f"</div>"
    )


# --------------------------------------------------------------------------- #
# Metrics & cards
# --------------------------------------------------------------------------- #
def metric(label: str, value: str, delta: str = "", tone: str = "") -> None:
    """Single metric tile. ``tone`` ∈ {'', 'pos', 'neg', 'warn', 'brand'}."""
    cls = f"cg-metric {tone}".strip()
    delta_html = f"<div class='delta'>{delta}</div>" if delta else ""
    _html(
        f"<div class='{cls}'>"
        f"<span class='accent-bar'></span>"
        f"<div class='lbl'>{label}</div>"
        f"<div class='val'>{value}</div>"
        f"{delta_html}"
        f"</div>"
    )


def metric_row(items: list[dict]) -> None:
    """Layout helper: place N metric tiles on the same row."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            metric(**item)


def kpi_bar(items: list[dict]) -> None:
    """4-up borderless KPI strip used on the landing page only."""
    parts = ["<div class='cg-kpi-bar'>"]
    for it in items:
        sub = (f"<div class='sub'>{it['sub']}</div>"
               if it.get("sub") else "")
        parts.append(
            "<div class='cell'>"
            f"<div class='lbl'>{it['label']}</div>"
            f"<div class='val'>{it['value']}</div>"
            f"{sub}"
            "</div>"
        )
    parts.append("</div>")
    _html("".join(parts))


def feature_tile(num: str, title: str, desc: str) -> None:
    _html(
        f"<div class='cg-feature'>"
        f"<span class='num'>{num}</span>"
        f"<h4>{title}</h4>"
        f"<p>{desc}</p>"
        f"</div>"
    )


def pill(text: str, tone: str = "") -> str:
    """Return HTML for an inline pill (does not render directly)."""
    return f"<span class='cg-pill {tone}'><span class='dot'></span>{text}</span>"


# --------------------------------------------------------------------------- #
# Result panels (used on the prediction page)
# --------------------------------------------------------------------------- #
def result_panel(at_risk: bool, proba: float, threshold: float) -> None:
    if at_risk:
        _html(
            f"<div class='cg-result risk'>"
            f"<div class='kicker'>Recommandation</div>"
            f"<h2>Client à risque de churn</h2>"
            f"<p>Probabilité estimée : <strong>{proba*100:.1f}%</strong> "
            f"(seuil de décision {threshold:.0%}). "
            f"Une action de rétention ciblée est recommandée.</p>"
            f"</div>"
        )
    else:
        _html(
            f"<div class='cg-result safe'>"
            f"<div class='kicker'>Recommandation</div>"
            f"<h2>Client probablement actif</h2>"
            f"<p>Probabilité estimée : <strong>{proba*100:.1f}%</strong> "
            f"(seuil de décision {threshold:.0%}). Aucune action urgente.</p>"
            f"</div>"
        )


# --------------------------------------------------------------------------- #
# Empty / loading / error states
# --------------------------------------------------------------------------- #
def empty_state(title: str, body: str, glyph: str = "·") -> None:
    _html(
        f"<div class='cg-empty'>"
        f"<div class='ico'>{glyph}</div>"
        f"<h4>{title}</h4>"
        f"<p>{body}</p>"
        f"</div>"
    )


def loading_card(message: str = "Chargement…") -> None:
    """Animated loading card. Use as a placeholder while heavy work runs."""
    _html(
        f"<div class='cg-loading'>"
        f"<div class='cg-spinner'></div>"
        f"<div class='cg-loading-text'>{message}</div>"
        f"</div>"
    )


def skeleton_chart(height: int = 320) -> None:
    """Shimmering placeholder shaped like a chart, used while a figure renders."""
    _html(
        f"<div class='cg-skeleton-chart' style='height:{height}px'></div>"
    )


def not_ready_notice() -> None:
    """Shown when the processed artefacts haven't been generated yet."""
    empty_state(
        "Données non préparées",
        "Cette section a besoin des artefacts du pipeline pour s'afficher. "
        "Rendez-vous sur l'accueil et lancez la préparation des données.",
        glyph="!",
    )
    st.stop()


def require_artifacts() -> None:
    """Guard placed at the top of every data-driven page.

    Reads the status that was stashed by _sidebar_status during setup_page,
    so we don't pay for two filesystem checks per page.
    """
    status = st.session_state.get("_cg_status")
    if status is None:
        from services import data_service as ds
        status = ds.artifacts_status()
    if not status["_ready"]:
        not_ready_notice()


# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
def footer() -> None:
    _html(
        f"<div class='cg-footer'>"
        f"<div class='left'><b>{cfg.APP_TITLE}</b> — {cfg.APP_SUBTITLE}</div>"
        f"<div class='right'>Projet DDDM · {cfg.AUTHORS}</div>"
        f"</div>"
    )
