"""The two unitless teaching examples (total climate-score points).

They are fed to the same solver as the Stockholm case but kept separate from
the physical fuel calculations and from the held-out evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import Problem, info_groups

GRADES = ["Ordinary", "SAF"]
ROUTES = ["Original", "Alternative"]

EXAMPLE_A_TABLE = pd.DataFrame(
    [("A", "Original", 12, 7), ("A", "Alternative", 5, 4),
     ("B", "Original", 9, 5), ("B", "Alternative", 8, 8)],
    columns=["Flight", "Route", "Ordinary", "SAF"])

# rows: weather, prob, then (ordinary, SAF) for A orig, A alt, B orig, B alt
EXAMPLE_B_TABLE = pd.DataFrame(
    [("Dry", 0.60, "10 / 5", "8 / 6", "7 / 6", "9 / 7"),
     ("Humid", 0.40, "7 / 6", "8 / 7", "18 / 7", "14 / 5")],
    columns=["Weather", "Probability", "A original", "A alternative", "B original", "B alternative"])


def _problem(scores: np.ndarray, scenarios, p, mode="revealed", name="") -> Problem:
    n = scores.shape[0]
    q = np.zeros((n, 2))
    q[:, 1] = 1.0  # one unit of SAF per aircraft
    groups = np.tile(info_groups(mode, scenarios), (n, 1))
    return Problem(
        flights=["A", "B"], grades=GRADES, routes=ROUTES, scenarios=list(scenarios),
        p=np.array(p, dtype=float), fuel=np.zeros_like(scores), contrail=scores,
        groups=groups, q_stock=q, dedicated=np.array([False, True]), opening_stock_kg=1.0,
        units="score points", meta={"name": name, "kind": "teaching"})


def example_a() -> Problem:
    # C[i,b,r,s]
    C = np.zeros((2, 2, 2, 1))
    for _, row in EXAMPLE_A_TABLE.iterrows():
        i = 0 if row.Flight == "A" else 1
        r = ROUTES.index(row.Route)
        C[i, 0, r, 0] = row.Ordinary
        C[i, 1, r, 0] = row.SAF
    return _problem(C, ["Single outcome"], [1.0], "revealed", "Example A")


def example_b(mode: str = "revealed") -> Problem:
    C = np.zeros((2, 2, 2, 2))
    cols = [(0, 0, "A original"), (0, 1, "A alternative"), (1, 0, "B original"), (1, 1, "B alternative")]
    for s, row in EXAMPLE_B_TABLE.iterrows():
        for i, r, col in cols:
            o, sf = (float(v) for v in row[col].split("/"))
            C[i, 0, r, s] = o
            C[i, 1, r, s] = sf
    P = _problem(C, list(EXAMPLE_B_TABLE.Weather), list(EXAMPLE_B_TABLE.Probability), mode,
                 "Example B")
    P.representative = 0  # dry
    return P
