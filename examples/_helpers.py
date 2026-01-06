"""Utility helpers shared across example notebooks."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative

PALETTE = qualitative.Set2


def seed_everything(seed: int = 7) -> None:
    """Set NumPy's random seed (useful for reproducible notebooks)."""
    np.random.seed(seed)


def generate_piecewise_gaussian(
    lengths: Sequence[int],
    means: Sequence[float],
    sigma: float = 0.3,
    seed: int | None = 7,
) -> Tuple[pd.DataFrame, List[int]]:
    """Create a toy time series with abrupt mean shifts."""
    if seed is not None:
        np.random.seed(seed)
    if len(lengths) != len(means):
        raise ValueError("lengths and means must have the same size")

    segments = []
    change_indices: List[int] = []
    t = 0
    for length, mean in zip(lengths, means):
        segment = mean + sigma * np.random.randn(length)
        segments.append(
            pd.DataFrame(
                {
                    "t": np.arange(t, t + length),
                    "value": segment,
                    "true_mean": mean,
                }
            )
        )
        t += length
        change_indices.append(t)

    df = pd.concat(segments, ignore_index=True)
    # The final cumulative length is not a changepoint; drop it
    if change_indices:
        change_indices = change_indices[:-1]
    return df, change_indices


def plot_series_with_cp(
    df: pd.DataFrame,
    value_col: str = "value",
    cp_prob_col: str | None = None,
    changepoints: Iterable[int] | None = None,
    title: str = "",
    height: int = 420,
):
    """Plot the time series and (optionally) changepoint probabilities."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["t"],
            y=df[value_col],
            mode="lines",
            name="Observation",
            line=dict(color=PALETTE[0], width=2),
        )
    )

    if cp_prob_col is not None and cp_prob_col in df:
        fig.add_trace(
            go.Scatter(
                x=df["t"],
                y=df[cp_prob_col],
                mode="lines",
                name="P(changepoint)",
                line=dict(color=PALETTE[1], width=2, dash="dot"),
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis2=dict(
                title="Changepoint probability",
                overlaying="y",
                side="right",
                range=[0, 1],
            )
        )

    if changepoints:
        for cp in changepoints:
            fig.add_vline(
                x=cp,
                line=dict(color="#888", width=1.2, dash="dash"),
                opacity=0.5,
            )

    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        xaxis_title="Time step",
        yaxis_title=value_col,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.02),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
