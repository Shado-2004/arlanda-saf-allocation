"""The five competing fuel-commitment strategies plus the ordinary-fuel reference.

Every strategy commits fuel first and then receives the *same* stage-two
route replanning, information groups, stock, slots, cash cap, feasibility
limits and prices. Grade index 0 is the ordinary/reference fuel.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import Problem
from .solver import SolveError, _route_policy, evaluate_commitment, resource_check, solve

STRATEGY_NAMES = [
    "Fixed priority",
    "Original-route targeting",
    "Route-first targeting",
    "Joint deterministic",
    "Joint stochastic",
]
REFERENCE_NAME = "Ordinary fuel, adaptive routes"


@dataclass
class StrategyResult:
    name: str
    x: np.ndarray
    route: np.ndarray
    per_flight: np.ndarray      # (n,S) reported score
    scenario_scores: np.ndarray  # (S,)
    expected: float
    considered: str
    notes: list = field(default_factory=list)
    binding: dict = field(default_factory=dict)

    def components(self, P: Problem):
        n, B, R, S = P.shape
        idx = (np.arange(n)[:, None], self.x[:, None], self.route, np.arange(S)[None, :])
        fuel = (P.fuel[idx] * P.p[None, :]).sum()
        con = (P.contrail[idx] * P.p[None, :]).sum()
        ext = (P.extra[idx] * P.p[None, :]).sum()
        return float(fuel), float(con), float(ext)

    def saf_stock_kg(self, P: Problem) -> float:
        return float(P.q_stock[np.arange(len(self.x)), self.x].sum())

    def saf_burn_kg(self, P: Problem) -> float:
        if P.saf_burn_kg is None:
            return float("nan")
        n, B, R, S = P.shape
        idx = (np.arange(n)[:, None], self.x[:, None], self.route, np.arange(S)[None, :])
        return float((P.saf_burn_kg[idx] * P.p[None, :]).sum())

    def cash_eur(self, P: Problem) -> float:
        if P.cash_eur is None:
            return float("nan")
        return float(P.cash_eur[np.arange(len(self.x)), self.x].sum())

    def premium_eur(self, P: Problem) -> float:
        return float(P.premium_eur[np.arange(len(self.x)), self.x].sum())

    def comparison_eur(self, P: Problem) -> float:
        if P.comparison_eur is None:
            return float("nan")
        n, B, R, S = P.shape
        idx = (np.arange(n)[:, None], self.x[:, None], self.route, np.arange(S)[None, :])
        return float((P.comparison_eur[idx] * P.p[None, :]).sum())


def _finish(P, name, x, considered, w, notes=None, binding=None):
    route, per, scen, exp = evaluate_commitment(P, x, w)
    return StrategyResult(name, x, route, per, scen, exp, considered, notes or [], binding or {})


def _fallback_value(P, i, b, C_opt):
    """Best no-update route value when a fixed route is infeasible for (i,b)."""
    n, B, R, S = P.shape
    vals = [float(P.p @ C_opt[i, b, r]) for r in range(R) if P.feasible[i, b, r].all()]
    return min(vals) if vals else None


def fixed_priority(P: Problem, w: float = 1.0) -> StrategyResult:
    n, B, R, S = P.shape
    x = np.zeros(n, dtype=int)
    b_pref = P.preferred_grade
    skipped = []
    for i in P.priority_order:
        trial = x.copy()
        trial[i] = b_pref
        if not P.grade_allowed[i, b_pref]:
            skipped.append(f"{P.flights[i]}: {P.grade_reason.get((i, b_pref), 'grade ineligible')}")
            continue
        if resource_check(P, trial):
            skipped.append(f"{P.flights[i]}: {resource_check(P, trial)[0]}")
            continue
        try:
            _route_policy(P, trial, P.score(w))
        except SolveError as e:
            skipped.append(str(e))
            continue
        x = trial
    order = ", ".join(P.flights[i] for i in P.priority_order)
    return _finish(P, "Fixed priority", x,
                   f"Published order ({order}) filled with {P.grades[b_pref]} until a common "
                   "limit binds. A simple proxy for a seasonal policy; no weather or route information.",
                   w, skipped)


def original_route_targeting(P: Problem, w: float = 1.0) -> StrategyResult:
    n, B, R, S = P.shape
    C = P.score(w)
    V = np.zeros((n, B))
    notes = []
    for i in range(n):
        for b in range(B):
            if P.feasible[i, b, 0].all():
                V[i, b] = float(P.p @ C[i, b, 0])
            else:
                fb = _fallback_value(P, i, b, C)
                V[i, b] = fb if fb is not None else 1e9
                notes.append(f"{P.flights[i]} / {P.grades[b]}: planned route not feasible in every "
                             "outcome; valued at best fixed alternative.")
    sol = solve(P, contrail_weight=w, x_values=V)
    return _finish(P, "Original-route targeting", sol.x,
                   "Expected savings on the planned profiles only, through the same stock- and "
                   "budget-constrained allocation solver.", w, notes, sol.binding)


def route_first_targeting(P: Problem, w: float = 1.0) -> StrategyResult:
    n, B, R, S = P.shape
    C = P.score(w)
    route0 = _route_policy(P, np.zeros(n, dtype=int), C)
    V = np.zeros((n, B))
    notes = []
    for i in range(n):
        for b in range(B):
            ok = all(P.feasible[i, b, route0[i, s], s] for s in range(S))
            if ok:
                V[i, b] = float(sum(P.p[s] * C[i, b, route0[i, s], s] for s in range(S)))
            else:
                fb = _fallback_value(P, i, b, C)
                V[i, b] = fb if fb is not None else 1e9
                notes.append(f"{P.flights[i]} / {P.grades[b]}: ordinary-fuel route infeasible; "
                             "valued at best fixed alternative.")
    sol = solve(P, contrail_weight=w, x_values=V)
    return _finish(P, "Route-first targeting", sol.x,
                   "Adaptive routes planned with ordinary fuel first; SAF then allocated by benefit "
                   "on those fixed routes. Replanning is allowed after fuel commitment.", w, notes,
                   sol.binding)


def joint_deterministic(P: Problem, w: float = 1.0) -> StrategyResult:
    rep = P.representative
    sol = solve(P, contrail_weight=w, scenario_subset=[rep])
    return _finish(P, "Joint deterministic", sol.x,
                   f"Fuel and routes chosen together for one representative scenario "
                   f"('{P.scenarios[rep]}'); fuel decision frozen, then evaluated with the same "
                   "forecast-based route policy.", w, [], sol.binding)


def joint_stochastic(P: Problem, w: float = 1.0, cvar_weight: float = 0.0) -> StrategyResult:
    sol = solve(P, contrail_weight=w, cvar_weight=cvar_weight)
    extra = f" Risk weight on worst-5% CVaR = {cvar_weight:g}." if cvar_weight else ""
    return _finish(P, "Joint stochastic", sol.x,
                   "Fuel chosen while accounting for every scenario's probability and the route "
                   "choices available under each later forecast signal." + extra, w, [], sol.binding)


def ordinary_reference(P: Problem, w: float = 1.0) -> StrategyResult:
    n = len(P.flights)
    return _finish(P, REFERENCE_NAME, np.zeros(n, dtype=int),
                   "No SAF; routes adapt to the forecast signal.", w)


def run_all(P: Problem, w: float = 1.0, cvar_weight: float = 0.0) -> dict[str, StrategyResult]:
    out = {
        "Fixed priority": fixed_priority(P, w),
        "Original-route targeting": original_route_targeting(P, w),
        "Route-first targeting": route_first_targeting(P, w),
        "Joint deterministic": joint_deterministic(P, w),
        "Joint stochastic": joint_stochastic(P, w, cvar_weight),
        REFERENCE_NAME: ordinary_reference(P, w),
    }
    for r in out.values():
        viol = resource_check(P, r.x)
        if viol:
            raise SolveError(f"{r.name} violates shared limits: {viol}")
    return out
