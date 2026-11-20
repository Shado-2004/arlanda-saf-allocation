"""Preset registry and helpers that keep editable tables consistent."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .stockholm import (DEFAULT_SCENARIOS, ROUTES, Case, build_sourced_options, load_tracks,
                        sourced_case, synthetic_case, to_min)

PRESETS = {
    "Sourced cohort (12 ARN departures, 17 Nov 2026)": "sourced",
    "Illustrative Stockholm simulation (12 synthetic rows)": "synthetic",
    "Small teaching fixture (4 synthetic rows)": "small",
}


def make_preset(key: str) -> Case:
    kind = PRESETS.get(key, key)
    if kind == "sourced":
        return sourced_case()
    c = synthetic_case()
    if kind == "small":
        keep = ["SYN-01", "SYN-03", "SYN-08", "SYN-12"]
        c.flights = c.flights[c.flights.study_id.isin(keep)].reset_index(drop=True)
        c.manifest = c.manifest[c.manifest.study_id.isin(keep)].reset_index(drop=True)
        c.options = c.options[c.options.study_id.isin(keep)].reset_index(drop=True)
        c.supply["opening_stock_kg"] = 2600.0
        c.supply["receipt_kg"] = 0.0
        c.name = "Small teaching fixture"
    return c


FLIGHT_DEFAULTS = {
    "modelled_aircraft": "Airbus A320-251N", "modelled_engine": "2 × CFM LEAP-1A26",
    "track_distance_km": 600.0, "peak_alt_ft": 37000.0, "cruise_km_available": 200.0,
    "trip_fuel_kg": 2190.0, "taxi_kg": 200.0, "reserve_kg": 2000.0, "uplift_kg": 4500.0,
    "M0_kg": 0.0, "w0": 0.0, "tank_capacity_kg": 18700.0, "lower_allowed": True,
    "higher_allowed": True, "lower_penalty_pct": 1.8, "higher_penalty_pct": 0.3,
    "level_changes": 2, "mass_for_level_change_kg": 65000.0, "eligible_for_saf": True,
    "energy_source": "assumption (row added in app)", "destination": "CPH", "flight_number": "",
}


def sanitize_flights(df: pd.DataFrame, case: Case) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    for k, v in FLIGHT_DEFAULTS.items():
        if k not in df:
            df[k] = v
        df[k] = df[k].where(df[k].notna(), v) if df[k].dtype != bool else df[k].fillna(v)
    used = set()
    for i in range(len(df)):
        sid = df.at[i, "study_id"]
        if not isinstance(sid, str) or not sid.strip() or sid in used:
            k = 1
            while f"SYN-{k:02d}" in used or f"SYN-{k:02d}" in set(df.study_id.dropna()):
                k += 1
            df.at[i, "study_id"] = f"SYN-{k:02d}"
        used.add(df.at[i, "study_id"])
        std = df.at[i, "std_utc"] if "std_utc" in df else None
        if not isinstance(std, str) or not std:
            df.at[i, "std_utc"] = "2026-11-17T12:00:00Z"
        if "route_cutoff_min" not in df or pd.isna(df.at[i, "route_cutoff_min"]):
            df.at[i, "route_cutoff_min"] = to_min(df.at[i, "std_utc"]) - case.settings["route_cutoff_min_before_std"]
    for b in ("lower_allowed", "higher_allowed", "eligible_for_saf"):
        df[b] = df[b].astype(bool)
    return df


def ensure_options(case: Case, rebuild_surrogate: bool = False) -> None:
    """Make sure every (flight, route, scenario) has an EF row; rebuild sourced screen if asked."""
    if case.kind == "sourced" and rebuild_surrogate:
        _, fc = load_tracks()
        new = build_sourced_options(case.flights, case.scenarios, fc, case.settings)
        old = case.options
        keep = old[old.ef_source.isin(["team estimate", "edited in app"])]
        if len(keep):
            new = new.set_index(["study_id", "route", "scenario"])
            k = keep.set_index(["study_id", "route", "scenario"])
            common = new.index.intersection(k.index)
            new.loc[common, ["EF_ordinary_J", "ef_source"]] = k.loc[common, ["EF_ordinary_J", "ef_source"]]
            new = new.reset_index()
        case.options = new
        return
    idx = pd.MultiIndex.from_product([case.flights.study_id, ROUTES, case.scenarios.scenario],
                                     names=["study_id", "route", "scenario"])
    o = case.options.drop_duplicates(["study_id", "route", "scenario"]).set_index(["study_id", "route", "scenario"])
    o = o.reindex(idx)
    o["EF_ordinary_J"] = o["EF_ordinary_J"].fillna(0.0)
    o["ef_source"] = o["ef_source"].fillna("assumed zero (no estimate entered)")
    case.options = o.reset_index()


def ef_wide(case: Case) -> pd.DataFrame:
    w = case.options.pivot_table(index=["study_id", "route"], columns="scenario", values="EF_ordinary_J",
                                 aggfunc="first").reindex(columns=list(case.scenarios.scenario))
    return (w / 1e12).reset_index()


def apply_ef_wide(case: Case, wide: pd.DataFrame) -> None:
    long = wide.melt(id_vars=["study_id", "route"], var_name="scenario", value_name="EF_TJ")
    long["EF_ordinary_J"] = long.EF_TJ.astype(float).fillna(0.0) * 1e12
    o = case.options.set_index(["study_id", "route", "scenario"])
    for _, r in long.iterrows():
        key = (r.study_id, r.route, r.scenario)
        if key in o.index and not np.isclose(o.at[key, "EF_ordinary_J"], r.EF_ordinary_J, rtol=1e-9, atol=1e3):
            o.at[key, "EF_ordinary_J"] = r.EF_ordinary_J
            o.at[key, "ef_source"] = "edited in app"
    case.options = o.reset_index()


def clone(case: Case) -> Case:
    return copy.deepcopy(case)
