"""One complete optimisation run: build, solve, evaluate, tabulate, export.

The UI only ever displays numbers that come out of `run_case`.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluation import HeldOut, paired_summary, sample_heldout, strongest_comparator
from .model import Problem
from .stockholm import Case, build_problem
from .strategies import REFERENCE_NAME, STRATEGY_NAMES, StrategyResult, run_all


def case_fingerprint(case: Case) -> str:
    payload = {
        "flights": case.flights.to_json(), "scen": case.scenarios.to_json(),
        "opts": case.options.to_json(), "supply": case.supply, "settings": case.settings,
        "fuel": case.fuel.__dict__, "kind": case.kind,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


@dataclass
class Run:
    case_name: str
    kind: str
    label: str
    fingerprint: str
    P: Problem
    detail: pd.DataFrame
    results: dict
    heldout: HeldOut
    paired: pd.DataFrame
    strongest: str
    summary: pd.DataFrame
    scenario_table: pd.DataFrame
    allocation: pd.DataFrame


def strategy_summary(P: Problem, results: dict[str, StrategyResult], benchmark: str,
                     ho: HeldOut) -> pd.DataFrame:
    rows = []
    ref = results[REFERENCE_NAME]
    bench = results.get(benchmark, ref)
    for name in STRATEGY_NAMES + [REFERENCE_NAME]:
        r = results[name]
        f, c, e = r.components(P)
        rows.append({
            "strategy": name,
            "departures_included": len(P.flights),
            "fuel_lifecycle_tCO2e": f, "contrail_tCO2e": c, "extra_tCO2e": e,
            "expected_score_tCO2e": r.expected,
            "heldout_mean_tCO2e": float(ho.scores[name].mean()),
            "pure_SAF_from_stock_kg": r.saf_stock_kg(P),
            "pure_SAF_burned_kg": r.saf_burn_kg(P),
            "SAF_flights": int((r.x > 0).sum()),
            "cash_paid_eur": r.cash_eur(P),
            "comparison_cost_eur": r.comparison_eur(P),
            "extra_cost_vs_ordinary_eur": r.comparison_eur(P) - ref.comparison_eur(P),
            "saf_premium_cash_eur": r.premium_eur(P),
            "diff_vs_benchmark_tCO2e": r.expected - bench.expected,
        })
    df = pd.DataFrame(rows)
    saved = df.expected_score_tCO2e.iloc[-1] - df.expected_score_tCO2e
    df["climate_saving_vs_ordinary_tCO2e"] = saved
    df["eur_per_t_saved"] = [(x / s) if s > 1e-9 else np.nan
                             for x, s in zip(df.extra_cost_vs_ordinary_eur, saved)]
    return df


def allocation_table(P: Problem, res: StrategyResult, detail: pd.DataFrame, case: Case) -> pd.DataFrame:
    n, B, R, S = P.shape
    fl = case.flights.reset_index(drop=True)
    vis = P.meta.get("visibility")
    rows = []
    for i in range(n):
        b = int(res.x[i])
        d = detail[(detail.study_id == P.flights[i]) & (detail.grade == P.grades[b])]
        d0 = d[d.route == "reference"].iloc[0]
        routes = {}
        for s in range(S):
            routes[P.scenarios[s]] = P.routes[res.route[i, s]]
        exp_score = float(P.p @ res.per_flight[i])
        rows.append({
            "study_id": P.flights[i], "flight": fl.loc[i, "flight_number"], "to": fl.loc[i, "destination"],
            "STD_UTC": pd.Timestamp(fl.loc[i, "std_utc"]).strftime("%H:%M"),
            "fuel": "SAF blend" if b > 0 else "Reference Jet A-1",
            "grade": P.grades[b],
            "blend_mass_pct": 100 * d0.saf_mass_frac_delivered,
            "blend_vol_pct": 100 * d0.saf_vol_frac_delivered,
            "uplift_kg": d0.uplift_kg,
            "pure_SAF_stock_kg": float(P.q_stock[i, b]),
            "truck_loads": int(P.truck_loads[i, b]),
            **{f"route | {k}": v for k, v in routes.items()},
            "signal_groups": "/".join(str(g) for g in P.groups[i]),
            "update_visible": bool(vis.iloc[i].has_update) if vis is not None else True,
            "expected_score_tCO2e": exp_score,
        })
    return pd.DataFrame(rows)


def run_case(case: Case) -> Run:
    P, detail = build_problem(case)
    st = case.settings
    results = run_all(P, st["contrail_weight"], st.get("cvar_weight", 0.0))
    ho = sample_heldout(P, results, int(st["heldout_seed"]), int(st["heldout_draws"]),
                        float(st["heldout_sigma"]))
    paired = paired_summary(ho)
    strongest = strongest_comparator(results)
    summ = strategy_summary(P, results, st.get("benchmark", "Original-route targeting"), ho)
    sc_rows = []
    for name, r in results.items():
        for s, sn in enumerate(P.scenarios):
            sc_rows.append({"strategy": name, "scenario": sn, "probability": P.p[s],
                            "fleet_score_tCO2e": r.scenario_scores[s]})
    alloc = allocation_table(P, results["Joint stochastic"], detail, case)
    return Run(case.name, case.kind, case.run_label, case_fingerprint(case), P, detail, results, ho,
               paired, strongest, summ, pd.DataFrame(sc_rows), alloc)


def export_csv(run: Run, case: Case) -> str:
    """CSV with units, strategy, seeds and assumptions; matches the dashboard table."""
    buf = io.StringIO()
    st = case.settings
    hdr = [
        f"# run_label,{run.label}",
        f"# data_kind,{'SYNTHETIC - illustrative, not historical' if run.kind == 'synthetic' else 'sourced cohort with modelled decisions'}",
        f"# fingerprint,{run.fingerprint}",
        f"# departures_included,{len(run.P.flights)}",
        f"# climate_basis,t CO2e QAEI 100-year; k100={case.fuel.k_contrail:.6e} t/J",
        f"# planning_evaluation,exact expectation over scenario table (planning seed {st['planning_seed']}; no sampling)",
        f"# heldout_seed,{run.heldout.seed}",
        f"# heldout_method,{run.heldout.method}",
        f"# info_mode,{st['info_mode']}",
        f"# saf_contrail_response_scale,{st['response_scale']}",
        f"# opening_stock_kg,{case.supply['opening_stock_kg']}",
        f"# prices,EASA 2025 EU reference: Jet {case.fuel.price_J_eur_t} EUR/t; biofuel {case.fuel.price_S_eur_t} EUR/t",
        f"# CI,Jet {case.fuel.CI_J} g/MJ; SAF {case.fuel.CI_S} g/MJ (UCO-HEFA sensitivity value)",
    ]
    buf.write("\n".join(hdr) + "\n")
    run.summary.to_csv(buf, index=False, float_format="%.6g")
    return buf.getvalue()
