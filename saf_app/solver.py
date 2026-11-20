"""Two-stage mixed-integer allocation solver (SciPy HiGHS via scipy.optimize.milp).

minimise   sum_s p[s] * sum_{i,b,r} C[i,b,r,s] * y[i,b,r,g_i(s)]
subject to sum_b x[i,b] = 1                                   each flight
           sum_r y[i,b,r,g] = x[i,b]                          each flight, grade, signal
           y[i,b,r,g_i(s)] <= a[i,b,r,s]                      each supported outcome
           cumulative q_stock * x by service time <= opening stock + receipts
           truck loads overlapping a service window <= window capacity
           number of dedicated deliveries <= max_special
           sum premium * x <= cash cap                        (optional)
           x[i,b] = 0 for ineligible grades

Optional CVaR term (Rockafellar–Uryasev): + lambda * (eta + 1/alpha * sum_s p_s u_s),
u_s >= Z_s - eta, u_s >= 0, which handles fractional probability at the alpha boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .model import Problem


class SolveError(RuntimeError):
    pass


@dataclass
class Solution:
    status: int
    message: str
    x: np.ndarray                # (n,) chosen grade index per flight
    route: np.ndarray            # (n,S) chosen route index per flight & outcome
    objective_opt: float         # optimiser objective (weight applied, no tie terms)
    expected_score: float        # physically reported (weight 1) expected score
    scenario_scores: np.ndarray  # (S,) fleet score per outcome (weight 1)
    binding: dict = field(default_factory=dict)
    cvar: float | None = None


def _tie_scale(C: np.ndarray) -> float:
    m = np.nanmean(np.abs(C)) if C.size else 1.0
    return max(m, 1e-9) * 1e-7


def _route_policy(P: Problem, x: np.ndarray, C_opt: np.ndarray):
    """Best feasible route per signal group for a fixed fuel commitment.

    Tie rule: lowest route index (planned profile first).
    Returns route[i,s] or raises SolveError if no feasible recourse exists.
    """
    n, B, R, S = P.shape
    route = np.zeros((n, S), dtype=int)
    for i in range(n):
        b = int(x[i])
        for g in np.unique(P.groups[i]):
            ss = np.where(P.groups[i] == g)[0]
            best, best_r = None, None
            for r in range(R):
                if not P.feasible[i, b, r, ss].all():
                    continue
                val = float((P.p[ss] * C_opt[i, b, r, ss]).sum())
                if best is None or val < best - 1e-12:
                    best, best_r = val, r
            if best_r is None:
                raise SolveError(f"No feasible route for {P.flights[i]} with {P.grades[b]} "
                                 f"in signal group {g}.")
            route[i, ss] = best_r
    return route


def evaluate_commitment(P: Problem, x: np.ndarray, contrail_weight: float = 1.0):
    """Exact expected score of a fixed first-stage plan with adaptive recourse.

    Route policy is chosen with the optimiser weight; the reported score uses weight 1.
    """
    C_opt = P.score(contrail_weight)
    C = P.score(1.0)
    route = _route_policy(P, x, C_opt)
    n, B, R, S = P.shape
    per = np.zeros((n, S))
    for i in range(n):
        per[i] = C[i, int(x[i]), route[i], np.arange(S)]
    scen = per.sum(axis=0)
    return route, per, scen, float(P.p @ scen)


def resource_check(P: Problem, x: np.ndarray) -> list[str]:
    """Return a list of violated first-stage limits for commitment x (empty = ok)."""
    n, B, R, S = P.shape
    out = []
    xb = np.zeros((n, B))
    xb[np.arange(n), x.astype(int)] = 1
    if np.any(xb * ~P.grade_allowed):
        for i in range(n):
            if not P.grade_allowed[i, int(x[i])]:
                out.append(f"{P.flights[i]}: grade {P.grades[int(x[i])]} not eligible "
                           f"({P.grade_reason.get((i, int(x[i])), 'ineligible')})")
    used = (P.q_stock * xb).sum(axis=1)
    for k, t in enumerate(sorted(set(P.service_start_min.tolist()))):
        cap = P.opening_stock_kg + sum(kg for (tt, kg) in P.receipts if tt <= t)
        tot = used[P.service_start_min <= t].sum()
        if tot > cap + 1e-6:
            out.append(f"Cumulative SAF stock exceeded by t={t:.0f} min ({tot:.0f} > {cap:.0f} kg)")
            break
    loads = (P.truck_loads * xb * P.dedicated[None, :]).sum(axis=1)
    for w in P.slot_windows:
        ov = (P.service_start_min < w.end_min) & (P.service_end_min > w.start_min)
        if loads[ov].sum() > w.capacity + 1e-9:
            out.append(f"Truck slots exceeded in window {w.label or (w.start_min, w.end_min)}")
    if P.max_special is not None:
        k = int((xb * P.dedicated[None, :]).sum())
        if k > P.max_special:
            out.append(f"Dedicated refuellings {k} > maximum {P.max_special}")
    if P.cash_cap_eur is not None:
        spend = float((P.premium_eur * xb).sum())
        if spend > P.cash_cap_eur + 1e-6:
            out.append(f"Cash premium €{spend:,.0f} > cap €{P.cash_cap_eur:,.0f}")
    return out


def solve(P: Problem, *, contrail_weight: float = 1.0, scenario_subset: list[int] | None = None,
          x_values: np.ndarray | None = None, forced: dict | None = None,
          cvar_weight: float = 0.0, cvar_alpha: float = 0.05,
          time_limit: float = 60.0) -> Solution:
    """Solve the fuel-and-route MILP.

    scenario_subset : optimise against only these outcomes (renormalised), used by
                      the joint-deterministic method; groups become 'revealed'.
    x_values        : (n,B) objective on x only (targeting rules). Routes are still
                      required to be feasible for every supported outcome.
    forced          : {flight_index: grade_index} fixes x for explanation re-solves.
    """
    n, B, R, S_all = P.shape
    scen = list(range(S_all)) if scenario_subset is None else list(scenario_subset)
    p = P.p[scen] / P.p[scen].sum()
    groups = P.groups[:, scen].copy()
    if scenario_subset is not None:
        groups = np.tile(np.arange(len(scen)), (n, 1))
    S = len(scen)
    C_opt = P.score(contrail_weight)[:, :, :, scen]
    feas = P.feasible[:, :, :, scen]
    tie = _tie_scale(C_opt)

    # ---- variable indexing
    xi = {(i, b): i * B + b for i in range(n) for b in range(B)}
    nv = n * B
    yi = {}
    gl = [np.unique(groups[i]) for i in range(n)]
    for i in range(n):
        for b in range(B):
            for r in range(R):
                for g in gl[i]:
                    yi[(i, b, r, int(g))] = nv
                    nv += 1
    use_cvar = cvar_weight > 0
    if use_cvar:
        eta_idx = nv
        u_idx = {s: nv + 1 + s for s in range(S)}
        nv += 1 + S

    c = np.zeros(nv)
    lb, ub = np.zeros(nv), np.ones(nv)
    integrality = np.ones(nv)
    for (i, b), k in xi.items():
        if not P.grade_allowed[i, b]:
            ub[k] = 0
        if forced is not None and i in forced:
            lb[k] = ub[k] = 1.0 if b == forced[i] else 0.0
        if x_values is not None:
            c[k] += x_values[i, b]
        # tie rule: prefer less SAF, then earlier flights keep ordinary fuel
        c[k] += tie * (P.q_stock[i, b] / (P.q_stock.max() + 1e-9)) * (1 + i / (10 * n))
    for (i, b, r, g), k in yi.items():
        ss = np.where(groups[i] == g)[0]
        if not feas[i, b, r, ss].all():
            ub[k] = 0
        if x_values is None:
            c[k] += float((p[ss] * C_opt[i, b, r, ss]).sum())
        c[k] += tie * 0.1 * r  # tie rule: planned profile first
    if use_cvar:
        lb[eta_idx], ub[eta_idx] = -np.inf, np.inf
        integrality[eta_idx] = 0
        c[eta_idx] += cvar_weight
        for s, k in u_idx.items():
            ub[k] = np.inf
            integrality[k] = 0
            c[k] += cvar_weight * p[s] / cvar_alpha

    rows, lo, hi = [], [], []
    A = lil_matrix((0, nv))

    def add(coefs: dict, l, h):
        rows.append(coefs)
        lo.append(l)
        hi.append(h)

    for i in range(n):
        add({xi[(i, b)]: 1.0 for b in range(B)}, 1, 1)
        for b in range(B):
            for g in gl[i]:
                d = {yi[(i, b, r, int(g))]: 1.0 for r in range(R)}
                d[xi[(i, b)]] = -1.0
                add(d, 0, 0)
    # cumulative stock
    stock_rows = []
    for t in sorted(set(P.service_start_min.tolist())):
        cap = P.opening_stock_kg + sum(kg for (tt, kg) in P.receipts if tt <= t)
        if not np.isfinite(cap):
            continue
        d = {}
        for i in np.where(P.service_start_min <= t)[0]:
            for b in range(B):
                if P.q_stock[i, b] > 0:
                    d[xi[(i, b)]] = float(P.q_stock[i, b])
        if d:
            stock_rows.append((len(rows), f"SAF stock available by t={t:.0f} min", cap))
            add(d, -np.inf, cap)
    slot_rows = []
    for w in P.slot_windows:
        ov = (P.service_start_min < w.end_min) & (P.service_end_min > w.start_min)
        d = {}
        for i in np.where(ov)[0]:
            for b in range(B):
                if P.dedicated[b] and P.truck_loads[i, b] > 0:
                    d[xi[(i, b)]] = float(P.truck_loads[i, b])
        if d:
            slot_rows.append((len(rows), f"Truck slots {w.label}", w.capacity))
            add(d, -np.inf, w.capacity)
    special_row = None
    if P.max_special is not None:
        d = {xi[(i, b)]: 1.0 for i in range(n) for b in range(B) if P.dedicated[b]}
        if d:
            special_row = (len(rows), "Maximum dedicated refuellings", P.max_special)
            add(d, -np.inf, P.max_special)
    cash_row = None
    if P.cash_cap_eur is not None:
        d = {xi[(i, b)]: float(P.premium_eur[i, b]) for i in range(n) for b in range(B)
             if P.premium_eur[i, b] != 0}
        if d:
            cash_row = (len(rows), "Cash cap on SAF premium", P.cash_cap_eur)
            add(d, -np.inf, P.cash_cap_eur)
    if use_cvar:
        for s in range(S):
            d = {u_idx[s]: 1.0, eta_idx: 1.0}
            for i in range(n):
                g = int(groups[i, s])
                for b in range(B):
                    for r in range(R):
                        d[yi[(i, b, r, g)]] = d.get(yi[(i, b, r, g)], 0) - float(C_opt[i, b, r, s])
            add(d, 0, np.inf)

    A = lil_matrix((len(rows), nv))
    for k, d in enumerate(rows):
        for j, v in d.items():
            A[k, j] = v
    cons = LinearConstraint(A.tocsr(), np.array(lo), np.array(hi)) if rows else None
    res = milp(c, constraints=cons, integrality=integrality, bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-9})
    if res.status != 0 or res.x is None:
        raise SolveError(f"Solver status {res.status}: {res.message}. "
                         "No plan is displayed because the problem is infeasible or unsolved.")
    z = res.x
    x = np.array([int(np.argmax([z[xi[(i, b)]] for b in range(B)])) for i in range(n)])

    # Recover routes on the full outcome set with the true information structure.
    route, per, scen_scores, exp = evaluate_commitment(P, x, contrail_weight)
    C_opt_full = P.score(contrail_weight)
    obj_opt = float(sum(P.p[s] * C_opt_full[i, x[i], route[i, s], s]
                        for i in range(n) for s in range(S_all)))
    if scenario_subset is not None:
        # objective that the deterministic planner believed in
        obj_opt = float(sum(p[k] * C_opt_full[i, x[i], route[i, s], s]
                            for k, s in enumerate(scen) for i in range(n)))
    binding = {}
    act = A.tocsr() @ z
    for lst in (stock_rows, slot_rows):
        for (k, label, cap) in lst:
            binding[label] = {"used": float(act[k]), "limit": float(cap),
                              "binding": bool(act[k] >= cap - 1e-6)}
    for item in (special_row, cash_row):
        if item:
            k, label, cap = item
            binding[label] = {"used": float(act[k]), "limit": float(cap),
                              "binding": bool(act[k] >= cap - 1e-6)}
    cvar_val = None
    if use_cvar:
        cvar_val = cvar(scen_scores, P.p, cvar_alpha)
    return Solution(res.status, res.message, x, route, obj_opt, exp, scen_scores, binding, cvar_val)


def cvar(values: np.ndarray, p: np.ndarray, alpha: float = 0.05) -> float:
    """Probability-weighted mean of the worst alpha tail (higher = worse).

    Handles a fractional probability at the alpha boundary exactly.
    """
    order = np.argsort(-values)
    rem, acc = alpha, 0.0
    for k in order:
        take = min(p[k], rem)
        acc += take * values[k]
        rem -= take
        if rem <= 1e-15:
            break
    return acc / alpha


def enumerate_best(P: Problem, contrail_weight: float = 1.0):
    """Exhaustive enumeration over first-stage plans (small problems only)."""
    import itertools
    n, B, R, S = P.shape
    best = None
    for combo in itertools.product(range(B), repeat=n):
        x = np.array(combo)
        if resource_check(P, x):
            continue
        try:
            route, per, scen, exp = evaluate_commitment(P, x, contrail_weight)
        except SolveError:
            continue
        val = float(P.p @ P.score(contrail_weight)[np.arange(n)[:, None], x[:, None], route,
                                                   np.arange(S)[None, :]].sum(axis=0))
        if best is None or val < best[0] - 1e-12:
            best = (val, x, exp)
    return best
