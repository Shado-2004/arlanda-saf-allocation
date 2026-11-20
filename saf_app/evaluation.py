"""Held-out evaluation on independently seeded synthetic weather draws.

Sampling method (published in the UI and CSV):
  1. Draw the hidden outcome s ~ p with the held-out seed.
  2. The visible signal for flight i is g_i(s); the frozen policy maps that
     signal to a route. The policy function receives only the signal.
  3. Fresh within-state variation: contrail EF of every (draw, flight, route)
     is multiplied by a mean-one lognormal factor with spread sigma, common to
     all methods. Fuel life-cycle terms are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import Problem
from .strategies import StrategyResult


def policy_table(P: Problem, res: StrategyResult) -> dict:
    """Freeze a strategy as {(flight, signal) -> route}. Raises if routes differ within a signal."""
    n, B, R, S = P.shape
    tab = {}
    for i in range(n):
        for s in range(S):
            key = (i, int(P.groups[i, s]))
            r = int(res.route[i, s])
            if key in tab and tab[key] != r:
                raise AssertionError(f"Route differs within pooled signal for {P.flights[i]}")
            tab[key] = r
    return tab


@dataclass
class HeldOut:
    scores: pd.DataFrame       # draws × strategies
    hidden: np.ndarray
    seed: int
    sigma: float
    draws: int
    method: str


def sample_heldout(P: Problem, results: dict[str, StrategyResult], seed: int,
                   draws: int = 2000, sigma: float = 0.35) -> HeldOut:
    rng = np.random.default_rng(seed)
    n, B, R, S = P.shape
    hidden = rng.choice(S, size=draws, p=P.p)
    mult = rng.lognormal(mean=-0.5 * sigma ** 2, sigma=sigma, size=(draws, n, R)) if sigma > 0 \
        else np.ones((draws, n, R))
    out = {}
    for name, res in results.items():
        pol = policy_table(P, res)
        x = res.x
        tot = np.zeros(draws)
        for i in range(n):
            b = int(x[i])
            sig = P.groups[i, hidden]                         # what dispatch sees
            r = np.array([pol[(i, int(g))] for g in sig])      # route from signal only
            tot += (P.fuel[i, b, r, hidden] + P.contrail[i, b, r, hidden] * mult[np.arange(draws), i, r]
                    + P.extra[i, b, r, hidden])
        out[name] = tot
    method = (f"Hidden outcome drawn from the planning probabilities; signal = forecast group visible "
              f"at each route cutoff; contrail EF × lognormal(mean 1, σ={sigma}) per draw/flight/route, "
              f"common to all methods; seed {seed}; {draws} draws.")
    return HeldOut(pd.DataFrame(out), hidden, seed, sigma, draws, method)


def paired_summary(ho: HeldOut, focus: str = "Joint stochastic") -> pd.DataFrame:
    rows = []
    f = ho.scores[focus]
    for c in ho.scores.columns:
        if c == focus:
            continue
        d = ho.scores[c] - f            # positive = focus better
        rows.append({"comparator": c, "mean_paired_saving": float(d.mean()),
                     "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
                     "frac_focus_worse": float((d < -1e-9).mean()),
                     "frac_tie": float((d.abs() <= 1e-9).mean()), "n_draws": len(d)})
    return pd.DataFrame(rows)


def strongest_comparator(results: dict[str, StrategyResult], exclude=("Joint stochastic",),
                         reference_name: str = "Ordinary fuel, adaptive routes") -> str:
    cand = {k: v.expected for k, v in results.items() if k not in exclude and k != reference_name}
    return min(cand, key=cand.get)
