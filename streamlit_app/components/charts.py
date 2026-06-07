"""Plotly chart factory.

A thin wrapper around Plotly that gives every chart in the app a shared visual
language: subtle gridlines, the brand palette, no junk decorations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import config as cfg  # noqa: E402

# --------------------------------------------------------------------------- #
# Shared layout
# --------------------------------------------------------------------------- #
_BASE_LAYOUT = dict(
    template="simple_white",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=8, r=8, t=44, b=8),
    font=dict(family="Inter, system-ui, sans-serif",
              color=cfg.COLORS["ink"], size=12),
    title=dict(font=dict(size=14, color=cfg.COLORS["ink"]), x=0, xanchor="left"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=cfg.COLORS["muted"])),
    xaxis=dict(showgrid=True, gridcolor=cfg.COLORS["border"], gridwidth=1,
               zeroline=False, ticks="outside", ticklen=4,
               tickfont=dict(color=cfg.COLORS["muted"], size=11)),
    yaxis=dict(showgrid=True, gridcolor=cfg.COLORS["border"], gridwidth=1,
               zeroline=False, ticks="outside", ticklen=4,
               tickfont=dict(color=cfg.COLORS["muted"], size=11)),
)


def _style(fig: go.Figure, **overrides) -> go.Figure:
    layout = {**_BASE_LAYOUT, **overrides}
    fig.update_layout(**layout)
    return fig


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def gauge(proba: float, title: str = "Probabilité de churn") -> go.Figure:
    pct = proba * 100
    color = cfg.COLORS["danger"] if proba >= 0.5 else cfg.COLORS["positive"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 36, "color": cfg.COLORS["ink"],
                                         "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": cfg.COLORS["faint"]},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33], "color": cfg.COLORS["positive_soft"]},
                {"range": [33, 66], "color": cfg.COLORS["warning_soft"]},
                {"range": [66, 100], "color": cfg.COLORS["danger_soft"]},
            ],
            "threshold": {"line": {"color": cfg.COLORS["ink"], "width": 2},
                          "thickness": 0.75, "value": 50},
        },
        title={"text": title, "font": {"size": 13, "color": cfg.COLORS["muted"]}},
    ))
    fig.update_layout(**{**_BASE_LAYOUT, "height": 250,
                          "margin": dict(l=24, r=24, t=44, b=24)})
    return fig


def histogram(df: pd.DataFrame, col: str, title: str, log_x: bool = False,
              color_churn: bool = False) -> go.Figure:
    if color_churn and "Statut" in df.columns:
        fig = px.histogram(df, x=col, color="Statut", nbins=40, marginal="box",
                           color_discrete_map={"Actif": cfg.COLORS["positive"],
                                               "Churn": cfg.COLORS["danger"]},
                           opacity=0.85)
    else:
        fig = px.histogram(df, x=col, nbins=40,
                           color_discrete_sequence=[cfg.COLORS["accent"]])
    if log_x:
        fig.update_xaxes(type="log")
    return _style(fig, title=dict(text=title))


def box_by_churn(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    fig = px.box(df, x="Statut", y=col, color="Statut",
                 color_discrete_map={"Actif": cfg.COLORS["positive"],
                                     "Churn": cfg.COLORS["danger"]})
    return _style(fig, title=dict(text=title), showlegend=False)


def correlation_heatmap(df: pd.DataFrame, cols: list, title: str) -> go.Figure:
    corr = df[cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True,
                    color_continuous_scale=[
                        [0.0, cfg.COLORS["danger_soft"]],
                        [0.5, "#f5f4ef"],
                        [1.0, cfg.COLORS["accent_soft"]],
                    ],
                    zmin=-1, zmax=1, aspect="auto")
    fig.update_traces(textfont=dict(size=12, color=cfg.COLORS["ink"]))
    return _style(fig, title=dict(text=title), coloraxis_showscale=False)


def scatter_clusters(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    seg_colors = {info["name"]: info["color"]
                  for info in (cfg.get_segment_info(c) for c in df["Cluster"].unique())}
    fig = px.scatter(df, x=x, y=y, color="Segment",
                     color_discrete_map=seg_colors, opacity=0.55,
                     hover_data=["Recency", "Frequency", "Monetary"])
    fig.update_traces(marker=dict(size=5, line=dict(width=0)))
    return _style(fig, title=dict(text=title))


def bar(df: pd.DataFrame, x: str, y: str, title: str, *,
        color: str | None = None, horizontal: bool = False) -> go.Figure:
    if horizontal:
        fig = px.bar(df, x=y, y=x, orientation="h",
                     color_discrete_sequence=[color or cfg.COLORS["accent"]])
    else:
        fig = px.bar(df, x=x, y=y,
                     color_discrete_sequence=[color or cfg.COLORS["accent"]])
    fig.update_traces(marker_line_width=0)
    return _style(fig, title=dict(text=title), bargap=0.35)


def line(df: pd.DataFrame, x: str, y: str, title: str, *,
         color: str | None = None) -> go.Figure:
    fig = px.line(df, x=x, y=y,
                  color_discrete_sequence=[color or cfg.COLORS["accent"]])
    fig.update_traces(line=dict(width=2.5))
    return _style(fig, title=dict(text=title))


def area_timeseries(df: pd.DataFrame, x: str, y: str, title: str, *,
                    color: str | None = None) -> go.Figure:
    fig = px.area(df, x=x, y=y,
                  color_discrete_sequence=[color or cfg.COLORS["accent"]])
    fig.update_traces(line=dict(width=2),
                      fillcolor="rgba(31,58,95,0.10)")
    return _style(fig, title=dict(text=title))


def donut(labels, values, title: str, colors=None) -> go.Figure:
    palette = colors or [cfg.COLORS["accent"], cfg.COLORS["positive"],
                          cfg.COLORS["warning"], cfg.COLORS["danger"]]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.62,
                           marker=dict(colors=palette,
                                       line=dict(color=cfg.COLORS["surface"], width=2))))
    fig.update_traces(textinfo="percent+label",
                      textfont=dict(family="Inter", color=cfg.COLORS["ink"], size=12))
    return _style(fig, title=dict(text=title), showlegend=False)


def churn_by_segment(profile: pd.DataFrame) -> go.Figure:
    colors = [cfg.get_segment_info(c)["color"] for c in profile["Cluster"]]
    fig = go.Figure(go.Bar(
        x=profile["Segment"], y=profile["Churn_rate"] * 100,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v*100:.1f}%" for v in profile["Churn_rate"]],
        textposition="outside",
        textfont=dict(color=cfg.COLORS["ink"]),
    ))
    return _style(fig, title=dict(text="Taux de churn par segment"),
                   yaxis=dict(title="Taux de churn (%)",
                              gridcolor=cfg.COLORS["border"],
                              tickfont=dict(color=cfg.COLORS["muted"])),
                   showlegend=False, bargap=0.45)


def heatmap(matrix: pd.DataFrame, title: str, xlabel="", ylabel="") -> go.Figure:
    fig = px.imshow(matrix, aspect="auto",
                    color_continuous_scale=[
                        [0.0, cfg.COLORS["surface_2"]],
                        [0.5, "#cdd9e6"],
                        [1.0, cfg.COLORS["accent"]],
                    ])
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title=ylabel)
    return _style(fig, title=dict(text=title))


def feature_importance(features: list, importances: list, title: str) -> go.Figure:
    order = np.argsort(importances)
    fig = go.Figure(go.Bar(
        x=[importances[i] for i in order],
        y=[features[i] for i in order],
        orientation="h",
        marker=dict(color=cfg.COLORS["accent"], line=dict(width=0)),
        text=[f"{importances[i]:.3f}" for i in order],
        textposition="outside",
        textfont=dict(color=cfg.COLORS["muted"], family="JetBrains Mono", size=11),
    ))
    return _style(fig, title=dict(text=title),
                   xaxis=dict(title="Importance (gain)",
                              gridcolor=cfg.COLORS["border"]),
                   bargap=0.45, showlegend=False)


def proba_distribution(probas: np.ndarray, threshold: float = 0.5) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=probas, nbinsx=40,
        marker=dict(color=cfg.COLORS["accent"], line=dict(width=0)),
        opacity=0.9,
    ))
    fig.add_vline(x=threshold, line=dict(color=cfg.COLORS["danger"], width=2, dash="dot"),
                  annotation=dict(text=f"Seuil {threshold:.0%}",
                                   font=dict(color=cfg.COLORS["danger"], size=11),
                                   showarrow=False, yanchor="bottom"))
    return _style(fig, title=dict(text="Distribution des probabilités de churn"),
                   xaxis=dict(title="Probabilité"),
                   yaxis=dict(title="Clients"),
                   showlegend=False, bargap=0.05)
