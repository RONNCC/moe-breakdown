"""Concentration / localizability metrics computed from an attribution vector phi.

See study_design_C1_C4.md Sec 2.5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ConcentrationMetrics:
    entropy: float             # normalized entropy H (lower = more concentrated)
    gini: float                # Gini coefficient of |phi| (higher = more concentrated)
    top_5_fraction: float      # share of total |phi| mass held by top-5 players
    top_10pct_fraction: float  # share of total |phi| mass held by top-10% of players
    n_players: int
    n_nonzero: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "entropy": self.entropy,
            "gini": self.gini,
            "top_5_fraction": self.top_5_fraction,
            "top_10pct_fraction": self.top_10pct_fraction,
            "n_players": self.n_players,
            "n_nonzero": self.n_nonzero,
        }


def _normalized_entropy(p: np.ndarray) -> float:
    """Shannon entropy of the normalized |phi| distribution, divided by log(n)
    so the result is in [0, 1] regardless of player-set size (needed to
    compare MoE models with different expert counts, and MoE vs dense with
    very different player-set sizes).
    """
    n = len(p)
    if n <= 1:
        return 0.0
    mass = np.abs(p)
    total = mass.sum()
    if total <= 0:
        return 0.0
    phat = mass / total
    nz = phat[phat > 0]
    h = -np.sum(nz * np.log(nz))
    return float(h / np.log(n))


def _gini(p: np.ndarray) -> float:
    mass = np.sort(np.abs(p))
    n = len(mass)
    if n == 0 or mass.sum() == 0:
        return 0.0
    cum = np.cumsum(mass)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def _top_fraction(p: np.ndarray, top_n: int) -> float:
    mass = np.abs(p)
    total = mass.sum()
    if total <= 0:
        return 0.0
    top = np.sort(mass)[::-1][:top_n]
    return float(top.sum() / total)


def compute_concentration_metrics(phi: np.ndarray) -> ConcentrationMetrics:
    n = len(phi)
    top10pct_n = max(1, int(round(0.10 * n)))
    return ConcentrationMetrics(
        entropy=_normalized_entropy(phi),
        gini=_gini(phi),
        top_5_fraction=_top_fraction(phi, 5),
        top_10pct_fraction=_top_fraction(phi, top10pct_n),
        n_players=n,
        n_nonzero=int(np.count_nonzero(phi)),
    )


def localizability_ratio(h_dense: float, h_moe: float) -> float:
    """LR = H_dense / H_moe. LR > 1 means MoE is more localizable (RQ2 / C4)."""
    if h_moe <= 0:
        return float("inf")
    return h_dense / h_moe
