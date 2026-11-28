"""Plotly charts in the SoundBridge visual language (graphite surface; series colours below are CVD-checked).

Colour roles (CVD separation was validated on the original #12111c surface; charts now sit on the
#1b1b1c graphite surface, which is close in luminance — re-check if the surface changes further):
* PAIR       - "your profile" violet vs "this track" aqua (2-series comparisons)
* CATEGORICAL- fixed order, never cycled; series beyond six fold into small multiples
* SEQ        - single-hue violet ramp for magnitude
* DIVERGING  - blue / neutral grey / orange for signed values (correlations)
* HIGHLIGHT  - one accent mark against muted marks for "this strategy vs the rest"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from .config import SHORT_LABELS

SURFACE = "#1b1b1c"
PAPER = "rgba(0,0,0,0)"
TEXT = "#ecebe8"
TEXT_2 = "#b4b2ac"
MUTED = "#878580"
GRID = "rgba(236,236,234,0.06)"
PROFILE = "#9085e9"   # violet
TRACK = "#199e70"     # aqua
ACCENT = "#9085e9"
MUTED_MARK = "#4a4663"
CATEGORICAL = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"]
SEQ = [[0.0, "#211d3d"], [0.35, "#4b3f9a"], [0.7, "#8a7cf0"], [1.0, "#e3ddff"]]
DIVERGING = [[0.0, "#3987e5"], [0.5, "#3a3848"], [1.0, "#d95926"]]
FONT = "'IBM Plex Sans', system-ui, sans-serif"


def _register() -> None:
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT, color=TEXT_2, size=12),
        title=dict(font=dict(family=FONT, size=15, color=TEXT), x=0.02, xanchor="left"),
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        colorway=CATEGORICAL,
        margin=dict(l=16, r=16, t=52, b=16),
        xaxis=dict(automargin=True, gridcolor=GRID, zeroline=False, linecolor="rgba(255,255,255,0.12)", tickcolor=MUTED,
                   title=dict(font=dict(color=TEXT_2))),
        yaxis=dict(automargin=True, gridcolor=GRID, zeroline=False, linecolor="rgba(255,255,255,0.12)", tickcolor=MUTED,
                   title=dict(font=dict(color=TEXT_2))),
        hoverlabel=dict(bgcolor="#232324", bordercolor="rgba(236,236,234,0.14)", font=dict(color=TEXT, family=FONT)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_2), orientation="h", y=1.08, x=0),
        bargap=0.25,
    )
    pio.templates["soundbridge"] = t
    pio.templates.default = "soundbridge"


_register()


def _finish(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(height=height, template="soundbridge")
    if title:
        fig.update_layout(title=title)
    return fig


# ---------------------------------------------------------------- Musical DNA
def radar(profile_values: dict[str, float], compare: dict[str, float] | None = None,
          compare_name: str = "This track", baseline: dict[str, float] | None = None,
          height: int = 400, profile_name: str = "Your Musical DNA") -> go.Figure:
    cats = [SHORT_LABELS.get(k, k) for k in profile_values]
    close = lambda v: list(v) + [list(v)[0]]  # noqa: E731
    fig = go.Figure()
    if baseline:
        fig.add_trace(go.Scatterpolar(r=close(baseline.values()), theta=close(cats), name="Catalogue average",
                                      line=dict(color=MUTED, width=1.5, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatterpolar(r=close(profile_values.values()), theta=close(cats), name=profile_name,
                                  fill="toself", fillcolor="rgba(144,133,233,0.22)",
                                  line=dict(color=PROFILE, width=2), marker=dict(size=8),
                                  hovertemplate="%{theta}: %{r:.2f}<extra></extra>"))
    if compare:
        fig.add_trace(go.Scatterpolar(r=close(compare.values()), theta=close(cats), name=compare_name,
                                      fill="toself", fillcolor="rgba(25,158,112,0.16)",
                                      line=dict(color=TRACK, width=2), marker=dict(size=8),
                                      hovertemplate="%{theta}: %{r:.2f}<extra></extra>"))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor=GRID, linecolor=GRID),
                   angularaxis=dict(gridcolor=GRID, linecolor="rgba(255,255,255,0.1)",
                                    tickfont=dict(color=TEXT, size=12))),
        showlegend=True, legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
        margin=dict(l=70, r=70, t=30, b=40),
    )
    return _finish(fig, height)


def feature_bars(summary: pd.DataFrame, height: int = 340) -> go.Figure:
    """Seed means (display scale) against catalogue means, one row per feature."""
    s = summary.iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=s["label"], x=s["display_mean"], orientation="h", name="Your seeds",
                         marker=dict(color=PROFILE, cornerradius=4),
                         customdata=np.c_[s["mean"], s["catalog_percentile"]],
                         hovertemplate="%{y}: %{customdata[0]:.2f} (higher than %{customdata[1]:.0f}% of catalogue)<extra></extra>"))
    fig.add_trace(go.Scatter(y=s["label"], x=s["display_catalog"], mode="markers", name="Catalogue average",
                             marker=dict(symbol="line-ns", size=18, line=dict(width=2, color=TEXT_2)),
                             hovertemplate="Catalogue average: %{x:.2f}<extra></extra>"))
    fig.update_layout(xaxis=dict(range=[0, 1], title="scaled 0–1 (tempo 50–200 BPM, loudness −30–0 dB)"),
                      legend=dict(y=1.12))
    return _finish(fig, height)


def comparison_bars(comp: pd.DataFrame, height: int = 340) -> go.Figure:
    """Signed difference (track − profile) in catalogue SDs per feature."""
    c = comp.iloc[::-1]
    colors = [TRACK if abs(z) <= 0.35 else ("#c98500" if abs(z) < 1 else "#d95926") for z in c["z_difference"]]
    fig = go.Figure(go.Bar(
        y=c["label"], x=c["z_difference"], orientation="h", marker=dict(color=colors, cornerradius=4),
        customdata=np.c_[c["your_profile"], c["this_track"], c["closeness"]],
        hovertemplate="%{y}<br>your profile %{customdata[0]:.2f} · track %{customdata[1]:.2f}<br>"
                      "%{x:+.2f} SD (%{customdata[2]})<extra></extra>"))
    fig.add_vrect(x0=-0.35, x1=0.35, fillcolor="rgba(25,158,112,0.08)", line_width=0)
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_layout(xaxis_title="difference from your profile (catalogue standard deviations)")
    return _finish(fig, height)


def attribution_bars(attr: pd.DataFrame, height: int = 320) -> go.Figure:
    a = attr.sort_values("attribution")
    fig = go.Figure(go.Bar(y=a["label"], x=a["attribution"], orientation="h",
                           marker=dict(color=[PROFILE if v > 0 else MUTED_MARK for v in a["attribution"]], cornerradius=4),
                           hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_layout(xaxis_title="contribution to encoder similarity (integrated gradients)")
    return _finish(fig, height)


# ---------------------------------------------------------------- distributions
def histogram(series: pd.Series, title: str, nbins: int = 40, color: str = ACCENT, height: int = 280,
              xtitle: str | None = None) -> go.Figure:
    fig = go.Figure(go.Histogram(x=series, nbinsx=nbins, marker=dict(color=color, line=dict(color=SURFACE, width=1)),
                                 hovertemplate="%{x}: %{y:,} tracks<extra></extra>"))
    fig.update_layout(xaxis_title=xtitle or title, yaxis_title="tracks", bargap=0.04)
    return _finish(fig, height, title)


def hbar(labels, values, title: str | None = None, height: int = 420, xtitle: str = "",
         highlight: set | None = None, fmt: str = ",.0f", hover_extra=None) -> go.Figure:
    labels, values = list(labels)[::-1], list(values)[::-1]
    colors = [ACCENT if (highlight is None or lbl in highlight) else MUTED_MARK for lbl in labels]
    fig = go.Figure(go.Bar(y=labels, x=values, orientation="h", marker=dict(color=colors, cornerradius=4),
                           text=[f"{v:{fmt}}" for v in values], textposition="outside",
                           textfont=dict(color=TEXT_2, size=11), cliponaxis=False,
                           hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>"))
    fig.update_layout(xaxis_title=xtitle, margin=dict(l=8, r=40, t=48, b=8))
    return _finish(fig, height, title)


def scatter_density(df: pd.DataFrame, x: str, y: str, title: str, height: int = 340) -> go.Figure:
    fig = go.Figure(go.Histogram2d(x=df[x], y=df[y], nbinsx=40, nbinsy=40, colorscale=SEQ,
                                   colorbar=dict(title="tracks", thickness=10, outlinewidth=0),
                                   hovertemplate=f"{x} %{{x}}<br>{y} %{{y}}<br>%{{z}} tracks<extra></extra>"))
    fig.update_layout(xaxis_title=SHORT_LABELS.get(x, x).lower(), yaxis_title=y)
    return _finish(fig, height, title)


def heatmap(m: pd.DataFrame, title: str, height: int = 460, diverging: bool = True, zmid: float = 0.0,
            fmt: str = ".2f") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=m.values, x=[SHORT_LABELS.get(c, c) for c in m.columns], y=[SHORT_LABELS.get(c, c) for c in m.index],
        colorscale=DIVERGING if diverging else SEQ, zmid=zmid if diverging else None,
        text=np.vectorize(lambda v: f"{v:{fmt}}")(m.values), texttemplate="%{text}",
        textfont=dict(size=10, color=TEXT), xgap=2, ygap=2,
        colorbar=dict(thickness=10, outlinewidth=0),
        hovertemplate="%{y} × %{x}: %{z:" + fmt + "}<extra></extra>"))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _finish(fig, height, title)


def box_by_group(df: pd.DataFrame, group: str, value: str, title: str, order=None, height: int = 380) -> go.Figure:
    fig = go.Figure()
    for g in order or sorted(df[group].unique()):
        fig.add_trace(go.Box(y=df.loc[df[group] == g, value], name=str(g), marker_color=ACCENT,
                             line=dict(color=ACCENT, width=1.5), fillcolor="rgba(144,133,233,0.18)",
                             boxpoints=False))
    fig.update_layout(showlegend=False, yaxis_title=value)
    return _finish(fig, height, title)


# ---------------------------------------------------------------- maps & lab
def sound_map(coords: np.ndarray, labels: pd.Series, highlight_idx=None, seed_idx=None, title: str = "",
              height: int = 480, sample: int = 6000) -> go.Figure:
    rng = np.random.default_rng(0)
    base = rng.choice(len(coords), size=min(sample, len(coords)), replace=False)
    fig = go.Figure(go.Scattergl(x=coords[base, 0], y=coords[base, 1], mode="markers", name="Catalogue (sample)",
                                 marker=dict(size=3, color="rgba(185,181,207,0.18)"),
                                 text=labels.iloc[base], hovertemplate="%{text}<extra></extra>"))
    if highlight_idx is not None and len(highlight_idx):
        h = np.asarray(highlight_idx)
        fig.add_trace(go.Scattergl(x=coords[h, 0], y=coords[h, 1], mode="markers", name="Recommendations",
                                   marker=dict(size=10, color=TRACK, line=dict(color=SURFACE, width=2)),
                                   text=labels.iloc[h], hovertemplate="%{text}<extra></extra>"))
    if seed_idx is not None and len(seed_idx):
        s = np.asarray(seed_idx)
        fig.add_trace(go.Scattergl(x=coords[s, 0], y=coords[s, 1], mode="markers", name="Your seeds",
                                   marker=dict(size=14, symbol="star", color=PROFILE, line=dict(color=SURFACE, width=2)),
                                   text=labels.iloc[s], hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(xaxis=dict(showticklabels=False, title="embedding PC 1"),
                      yaxis=dict(showticklabels=False, title="embedding PC 2"))
    return _finish(fig, height, title)


def strategy_metric_bars(summary: pd.DataFrame, metric: str, labels: dict, highlight: str | None = None,
                         title: str | None = None, ci: pd.DataFrame | None = None, height: int = 300,
                         fmt: str = ".2f") -> go.Figure:
    names = [labels.get(s.split("@")[0], s) if "@" not in s else short_strategy(s) for s in summary.index]
    vals = summary[metric].to_numpy()
    colors = [ACCENT if (highlight and s.startswith(highlight)) else MUTED_MARK for s in summary.index]
    err = None
    if ci is not None and metric in ci.columns:
        e = ci.reindex(summary.index)[metric].to_numpy()
        if np.isfinite(e).any():
            err = dict(type="data", array=np.nan_to_num(e), color=TEXT_2, thickness=1.2, width=4)
    fig = go.Figure(go.Bar(y=names[::-1], x=vals[::-1], orientation="h",
                           marker=dict(color=colors[::-1], cornerradius=4),
                           error_x=dict(err, array=err["array"][::-1]) if err else None,
                           text=[f"{v:{fmt}}" for v in vals[::-1]], textposition="outside",
                           textfont=dict(color=TEXT_2, size=11), cliponaxis=False,
                           hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>"))
    fig.update_layout(margin=dict(l=8, r=48, t=40, b=8))
    return _finish(fig, height, title)


SHORT_STRATEGY = {"content_raw": "Raw features", "content": "Encoder", "popularity_ranked": "Popularity-ranked",
                  "genre_aware": "Genre-aware", "popularity_aware": "Discovery", "diversity_mmr": "MMR", "hybrid": "Hybrid"}


def short_strategy(s: str) -> str:
    base, _, suffix = s.partition("@")
    name = SHORT_STRATEGY.get(base, base)
    return f"{name} {suffix.replace('lambda', 'λ').replace('novelty', 'nov')}" if suffix else name


def tradeoff_scatter(summary: pd.DataFrame, x: str, y: str, labels: dict, xtitle: str, ytitle: str,
                     title: str = "", height: int = 420) -> go.Figure:
    names = [short_strategy(s) for s in summary.index]
    positions = ["top center", "bottom center", "middle right", "middle left"]
    order = np.argsort(summary[x].to_numpy())
    textpos = [positions[int(np.where(order == i)[0][0]) % len(positions)] for i in range(len(names))]
    fig = go.Figure(go.Scatter(
        x=summary[x], y=summary[y], mode="markers+text", text=names, textposition=textpos, cliponaxis=False,
        textfont=dict(color=TEXT_2, size=11),
        marker=dict(size=14, color=[ACCENT if s.startswith("hybrid") else "#3987e5" for s in summary.index],
                    line=dict(color=SURFACE, width=2)),
        hovertemplate="%{text}<br>" + xtitle + ": %{x:.3f}<br>" + ytitle + ": %{y:.3f}<extra></extra>"))
    pad = lambda v: (v.max() - v.min()) * 0.18 + 1e-3  # noqa: E731
    fig.update_layout(xaxis=dict(title=xtitle, range=[summary[x].min() - pad(summary[x]), summary[x].max() + pad(summary[x])]),
                      yaxis=dict(title=ytitle, range=[summary[y].min() - pad(summary[y]), summary[y].max() + pad(summary[y])]))
    return _finish(fig, height, title)


def popularity_violin(df: pd.DataFrame, group: str, labels: dict, title: str = "", height: int = 380) -> go.Figure:
    fig = go.Figure()
    for s in df[group].unique():
        name = short_strategy(s)
        fig.add_trace(go.Violin(y=df.loc[df[group] == s, "popularity"], name=name, box_visible=True,
                                meanline_visible=True, points=False,
                                line=dict(color=ACCENT if s.startswith("hybrid") else "#3987e5", width=1.5),
                                fillcolor="rgba(144,133,233,0.20)" if s.startswith("hybrid") else "rgba(57,135,229,0.14)"))
    fig.update_layout(showlegend=False, yaxis_title="popularity (0–100)")
    return _finish(fig, height, title)


def gauge_ring(value: float, label: str, color: str = ACCENT) -> str:
    """Small inline SVG ring used inside track cards (value 0-1)."""
    v = float(np.clip(value, 0, 1))
    r, c = 15, 2 * np.pi * 15
    return (
        f'<svg width="40" height="40" viewBox="0 0 40 40" aria-label="{label} {v:.0%}">'
        f'<circle cx="20" cy="20" r="{r}" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="4"/>'
        f'<circle cx="20" cy="20" r="{r}" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" '
        f'stroke-dasharray="{c * v:.1f} {c:.1f}" transform="rotate(-90 20 20)"/>'
        f'<text x="20" y="24" text-anchor="middle" font-size="11" font-weight="600" fill="{TEXT}">{v * 100:.0f}</text>'
        "</svg>"
    )
