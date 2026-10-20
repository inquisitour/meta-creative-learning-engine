"""Quasi-binomial logistic regression with effect (sum-to-zero) coding, in plain numpy.

Why hand-rolled: it keeps the dependency list to numpy/pandas and makes every
step explainable. Why this model: raw per-level averages are confounded when
creatives are produced unevenly (e.g. social-proof hooks happen to be shipped
mostly as video). A multivariable model separates the effect of each attribute.

Each ad is one grouped-binomial observation: y successes out of m trials.
  stage 1 (CTR): y = clicks,     m = impressions
  stage 2 (CVR): y = purchases,  m = clicks
Overdispersion (ads differ beyond binomial noise) is handled with the
quasi-binomial dispersion factor, which widens the confidence intervals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

Z95 = 1.96  # two-sided 95% interval (used for winner/loser calls)


@dataclass
class LevelEffect:
    attribute: str
    level: str
    coef: float   # log-odds deviation from the attribute's unweighted mean level
    se: float


@dataclass
class FitResult:
    effects: list[LevelEffect]
    dispersion: float
    n_obs: int


def _design(df: pd.DataFrame, attributes: list[str]):
    """Effect coding: for each attribute, L-1 columns; the last level is -1 on all."""
    cols = [np.ones(len(df))]
    names = ["intercept"]
    level_map: dict[str, list[str]] = {}
    for attr in attributes:
        levels = sorted(df[attr].unique())
        level_map[attr] = levels
        for lv in levels[:-1]:
            col = (df[attr] == lv).astype(float).to_numpy()
            col = col - (df[attr] == levels[-1]).astype(float).to_numpy()
            cols.append(col)
            names.append(f"{attr}={lv}")
    return np.column_stack(cols), names, level_map


def fit_quasibinomial(
    df: pd.DataFrame, attributes: list[str], y_col: str, m_col: str, iters: int = 50
) -> FitResult:
    d = df[df[m_col] > 0].reset_index(drop=True)
    X, names, level_map = _design(d, attributes)
    y = d[y_col].to_numpy(float)
    m = d[m_col].to_numpy(float)
    p = X.shape[1]

    # IRLS for the binomial GLM (logit link)
    beta = np.zeros(p)
    beta[0] = np.log((y.sum() + 0.5) / (m.sum() - y.sum() + 0.5))
    ridge = 1e-8 * np.eye(p)
    for _ in range(iters):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        mu = np.clip(mu, 1e-9, 1 - 1e-9)
        w = m * mu * (1 - mu)
        z = eta + (y / m - mu) / (mu * (1 - mu))
        XtWX = X.T @ (w[:, None] * X) + ridge
        new = np.linalg.solve(XtWX, X.T @ (w * z))
        if np.max(np.abs(new - beta)) < 1e-9:
            beta = new
            break
        beta = new

    eta = X @ beta
    mu = np.clip(1 / (1 + np.exp(-eta)), 1e-9, 1 - 1e-9)
    w = m * mu * (1 - mu)
    cov_unscaled = np.linalg.inv(X.T @ (w[:, None] * X) + ridge)
    pearson = np.sum((y - m * mu) ** 2 / (m * mu * (1 - mu)))
    dof = max(len(d) - p, 1)
    phi = max(1.0, pearson / dof)  # never *narrow* the intervals
    cov = cov_unscaled * phi

    index = {n: i for i, n in enumerate(names)}
    effects: list[LevelEffect] = []
    for attr, levels in level_map.items():
        for lv in levels:
            c = np.zeros(p)
            if lv != levels[-1]:
                c[index[f"{attr}={lv}"]] = 1.0
            else:  # last level = minus the sum of the others (sum-to-zero)
                for other in levels[:-1]:
                    c[index[f"{attr}={other}"]] = -1.0
            coef = float(c @ beta)
            se = float(np.sqrt(c @ cov @ c))
            effects.append(LevelEffect(attr, lv, coef, se))
    return FitResult(effects=effects, dispersion=float(phi), n_obs=len(d))
