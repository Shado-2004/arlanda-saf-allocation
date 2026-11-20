"""Per-flight explanation of the fleet decision."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import Problem
from .solver import SolveError, solve
from .strategies import StrategyResult


def best_policy(P: Problem, i: int, b: int, w: float = 1.0):
    """Expected score for flight i at grade b with its best feasible route policy.

    Returns (value or None, {group: route}, per-scenario reported scores).
    """
    n, B, R, S = P.shape
    C_opt, C = P.score(w), P.score(1.0)
    if not P.grade_allowed[i, b]:
        return None, {}, None
    pol, per = {}, np.zeros(S)
    for g in np.unique(P.groups[i]):
        ss = np.where(P.groups[i] == g)[0]
        best = None
        for r in range(R):
            if not P.feasible[i, b, r, ss].all():
                continue
            v = float((P.p[ss] * C_opt[i, b, r, ss]).sum())
            if best is None or v < best[0] - 1e-12:
                best = (v, r)
        if best is None:
            return None, {}, None
        pol[int(g)] = best[1]
        per[ss] = C[i, b, best[1], ss]
    return float(P.p @ per), pol, per


def flight_table(P: Problem, i: int, w: float = 1.0) -> pd.DataFrame:
    rows = []
    for b, g in enumerate(P.grades):
        v, pol, per = best_policy(P, i, b, w)
        rows.append({"grade": g, "expected_score": v,
                     "route_by_signal": ", ".join(f"signal {k}: {P.routes[r]}" for k, r in pol.items()) or "—",
                     "pure_SAF_kg": float(P.q_stock[i, b]),
                     "truck_loads": float(P.truck_loads[i, b]),
                     "premium_eur": float(P.premium_eur[i, b]),
                     "eligible": bool(P.grade_allowed[i, b]),
                     "reason": P.grade_reason.get((i, b), "")})
    return pd.DataFrame(rows)


def saving_per_tonne(P: Problem, i: int, b_from: int = 0, b_to: int = 1, w: float = 1.0) -> dict:
    v0, _, _ = best_policy(P, i, b_from, w)
    v1, _, _ = best_policy(P, i, b_to, w)
    den_t = (P.q_stock[i, b_to] - P.q_stock[i, b_from]) / 1000.0
    num = None if (v0 is None or v1 is None) else v0 - v1
    return {"from": P.grades[b_from], "to": P.grades[b_to], "numerator_t": num,
            "denominator_t_SAF": den_t,
            "ratio": None if (num is None or abs(den_t) < 1e-12) else num / den_t,
            "zero_denominator": abs(den_t) < 1e-12}


def explain_decision(P: Problem, res: StrategyResult, i: int, w: float = 1.0,
                     cvar_weight: float = 0.0) -> str:
    """One sentence on why flight i got (or did not get) its grade in the fleet plan."""
    name = P.flights[i]
    b = int(res.x[i])
    alt = 1 if b == 0 else 0
    s = saving_per_tonne(P, i, 0, 1, w)
    if b == 0 and not P.grade_allowed[i, 1]:
        return f"{name} keeps ordinary fuel: the SAF option is unavailable ({P.grade_reason.get((i, 1))})."
    if s["numerator_t"] is None:
        return f"{name}: no feasible route policy for one of the grades, so the comparison is not defined."
    try:
        forced = solve(P, contrail_weight=w, forced={i: alt}, cvar_weight=cvar_weight)
        delta = forced.expected_score - res.expected
    except SolveError:
        forced, delta = None, None
    binding = [k for k, v in res.binding.items() if v.get("binding")]
    bind_txt = f" Binding limits: {', '.join(binding)}." if binding else ""
    ratio = s["ratio"]
    rtxt = f"{ratio:.2f} t CO₂e per t of pure SAF" if ratio is not None else "no SAF difference"
    if b == 1:
        if forced is None:
            return f"{name} receives SAF ({rtxt}); every plan without it is infeasible.{bind_txt}"
        return (f"{name} receives SAF ({rtxt}); forcing it back to ordinary fuel and re-solving the fleet "
                f"worsens the expected score by {delta:.3f} {P.units}.{bind_txt}")
    if forced is None:
        return f"{name} keeps ordinary fuel; giving it SAF cannot be made feasible with the shared limits.{bind_txt}"
    return (f"{name} keeps ordinary fuel ({rtxt} if it had SAF); forcing SAF onto it and re-solving the "
            f"fleet changes the expected score by {delta:+.3f} {P.units}, because the stock is better "
            f"used elsewhere.{bind_txt}")
