"""Small reusable helpers for the Titanic EDA notebook.

Keeping them here keeps the notebook focused on analysis and interpretation.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Consistent colours / labels for the target variable
C_DIED, C_SURV = "#d1495b", "#2a9d8f"
LABELS = {0: "Did not survive", 1: "Survived"}
PALETTE = {"Did not survive": C_DIED, "Survived": C_SURV}


def save_fig(fig, name, folder):
    """Save a figure (tight, 150 dpi) into `folder` and return its path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    return path


def rate_bar(ax, data, col, order=None, title=None, target="Survived", color=C_SURV, xlabel=None):
    """Bar chart of the mean of `target` per group, annotated with % and group size."""
    g = data.groupby(col, observed=True)[target].agg(["mean", "count"])
    if order is not None:
        g = g.loc[order]
    ax.bar(g.index.astype(str), g["mean"], color=color, edgecolor="white", width=0.7)
    ax.axhline(data[target].mean(), ls="--", c="grey", lw=1)
    for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
        ax.text(i, m + 0.02, f"{m:.0%}\nn={n}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Survival rate")
    ax.set_xlabel(xlabel if xlabel is not None else col)
    ax.set_title(title or f"Survival rate by {col}")
    return g


def cramers_v(x, y):
    """Cramér's V (0-1) effect size for association between two categorical variables."""
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


def rank_biserial(a, b):
    """Rank-biserial correlation from Mann-Whitney U (-1..1). Sign: positive if a tends to be larger."""
    u = stats.mannwhitneyu(a, b).statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def cohens_d(a, b):
    """Cohen's d with pooled standard deviation."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / pooled)


def effect_label(v):
    """Rule-of-thumb labels for |effect size| (Cohen-style thresholds)."""
    v = abs(v)
    if v < 0.1:
        return "negligible"
    if v < 0.3:
        return "small"
    if v < 0.5:
        return "medium"
    return "large"


def iqr_fences(s, k=1.5):
    """Tukey fences (Q1 - k*IQR, Q3 + k*IQR)."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr
