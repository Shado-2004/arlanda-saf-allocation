"""Stockholm Arlanda case: data loading, presets and Problem construction.

Two presets:
  * "Sourced cohort" – the 12 documented ARN departures in data/arlanda. Rows,
    times and tracks are sourced; energy, uplift, supply and EF are modelled
    or assumed and labelled as such.
  * "Illustrative Stockholm simulation" – 12 synthetic rows (SYN-xx) with a
    hand-built EF table, kept out of historical counts and exports.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import physics as ph
from .evidence import annotate_manifest
from .model import InputError, Problem, SlotWindow, info_groups

DATA = Path(__file__).resolve().parent.parent / "data" / "arlanda"
T0 = pd.Timestamp("2026-11-17T00:00:00Z")
ROUTES = ["reference", "lower", "higher"]
ROUTE_LABEL = {"reference": "Planned profile", "lower": "2,000 ft lower", "higher": "2,000 ft higher"}
A320NEO_CEILING_FT = 39_800


def to_min(ts) -> float:
    if ts is None or (isinstance(ts, float) and math.isnan(ts)) or ts == "":
        return float("nan")
    return (pd.Timestamp(ts).tz_convert("UTC") - T0).total_seconds() / 60.0


def from_min(m: float) -> str:
    return (T0 + pd.Timedelta(minutes=float(m))).strftime("%d %b %H:%M UTC")


def isa_pressure_hpa(alt_ft: float) -> float:
    h = alt_ft * 0.3048
    if h <= 11000:
        return 1013.25 * (1 - 2.25577e-5 * h) ** 5.25588
    return 226.32 * math.exp(-(h - 11000) / 6341.62)


# ---------------------------------------------------------------------------
@dataclass
class Case:
    name: str
    kind: str                     # "sourced" | "synthetic"
    manifest: pd.DataFrame        # annotated historical (or synthetic) rows
    flights: pd.DataFrame         # editable modelling inputs, one row per departure
    scenarios: pd.DataFrame       # scenario, probability, rhi_multiplier, partial_pool
    options: pd.DataFrame         # long table: study_id, route, scenario, EF_ordinary_J, ...
    supply: dict
    settings: dict
    fuel: ph.FuelParams = field(default_factory=ph.FuelParams)
    notes: list = field(default_factory=list)

    @property
    def run_label(self) -> str:
        if self.kind == "sourced":
            n_team = int((self.options.get("ef_source", pd.Series(dtype=str)) == "team estimate").sum())
            tag = "team EF estimates" if n_team else "teaching-surrogate EF"
            return f"Sourced cohort · modelled decisions ({tag}; energy, uplift and supply assumed)"
        return "Illustrative Stockholm simulation · synthetic rows, not historical"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SUPPLY = {
    "opening_stock_kg": 4400.0,          # ~2 trial-size truckloads of neat SAF (assumption)
    "receipt_time_utc": "2026-11-17T12:00:00Z",
    "receipt_kg": 2200.0,
    "truck_capacity_L": 8000.0,          # Quante et al. trial truck (observation, not a 2026 fact)
    "trucks_per_window": 2,
    "window_minutes": 60,
    "service_minutes": 35,
    "service_offset_min": 60,            # special refuelling starts this long before STD
    "booking_lead_h": 6.0,               # supplier must be booked this long before service
    "max_special": 6,
    "fee_eur": 250.0,
    "cash_cap_enabled": False,
    "cash_cap_eur": 15000.0,
}

DEFAULT_SETTINGS = {
    "fuel_deadline_utc": "2026-11-16T22:00:00Z",   # after the 16 Nov 18Z cycle was published
    "planning_cycle": "2026-11-16T18:00:00Z",
    "route_cutoff_min_before_std": 90,
    "info_mode": "partial_update",
    "response_scale": 1.0,
    "rel_change_at_ref": -0.15,
    "contrail_weight": 1.0,
    "detour_energy_scale": 1.0,
    "cvar_weight": 0.0,
    "ef_lw_J_per_km": 1.0e11,
    "ef_sw_J_per_km": 6.0e11,
    "heldout_seed": 20261117,
    "heldout_draws": 2000,
    "heldout_sigma": 0.35,
    "planning_seed": 0,
    "benchmark": "Original-route targeting",
    "blend_sensitivity": False,
    "sens_fraction": 0.40,
    "sens_basis": "volume",
    "contrail_horizon": "100-year",
}

DEFAULT_SCENARIOS = pd.DataFrame({
    "scenario": ["dry", "moderate ISS", "high ISS"],
    "probability": [0.40, 0.35, 0.25],
    "rhi_multiplier": [0.90, 1.00, 1.25],
    "pooled_in_partial": [False, True, True],
    "description": ["Forecast humidity 10% too high", "Forecast as issued",
                    "Forecast misses thin ice-supersaturated layers (+25% RHi)"],
})


# ---------------------------------------------------------------------------
# Sourced cohort
# ---------------------------------------------------------------------------
def load_manifest(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA / "flight_manifest.csv"
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)


_CACHE: dict = {}


def load_tracks():
    if "tracks" not in _CACHE:
        summ = pd.read_csv(DATA / "tracks" / "summary.csv")
        fc = pd.read_csv(DATA / "tracks" / "forecast_at_observed_waypoints.csv")
        _CACHE["tracks"] = (summ, fc)
    return _CACHE["tracks"]


def contrail_screen_cached(study_id, cycle, route, rhi_mult, lw, sw):
    key = ("screen", study_id, cycle, route, round(float(rhi_mult), 6), float(lw), float(sw))
    if key not in _CACHE:
        _, fc = load_tracks()
        _CACHE[key] = contrail_screen(fc, study_id, cycle, route, rhi_mult, lw, sw)
    return _CACHE[key]


def _default_flight_inputs(man: pd.DataFrame, summ: pd.DataFrame) -> pd.DataFrame:
    rows = []
    s = summ.set_index("study_id")
    for _, m in man.iterrows():
        sid = m.study_id
        dist = float(s.loc[sid, "sampled_airborne_distance_km"]) if sid in s.index else float(m.great_circle_km)
        peak = float(s.loc[sid, "max_reported_altitude_ft"]) if sid in s.index else 37000.0
        trip = 600.0 + 2.65 * dist     # kg ordinary fuel, linear A20N proxy (assumption)
        reserve = 2000.0
        taxi = 200.0
        contingency = 0.05 * trip
        uplift = math.ceil((trip + contingency + reserve + taxi) / 10.0) * 10.0
        cruise_km = dist - 400.0
        alt_ok = cruise_km >= 150.0
        hi_ok = alt_ok and peak + 2000 <= A320NEO_CEILING_FT
        std = to_min(m.scheduled_departure_utc)
        rows.append({
            "study_id": sid, "flight_number": m.flight_number, "destination": m.destination_iata,
            "std_utc": m.scheduled_departure_utc,
            "modelled_aircraft": "Airbus A320-251N", "modelled_engine": "2 × CFM LEAP-1A26",
            "track_distance_km": round(dist, 1), "peak_alt_ft": peak,
            "cruise_km_available": round(cruise_km, 1),
            "trip_fuel_kg": round(trip, 0), "taxi_kg": taxi, "reserve_kg": reserve,
            "uplift_kg": uplift, "M0_kg": 0.0, "w0": 0.0, "tank_capacity_kg": 18_700.0,
            "lower_allowed": alt_ok, "higher_allowed": hi_ok,
            "lower_penalty_pct": 1.8, "higher_penalty_pct": 0.3,
            "level_changes": 2, "mass_for_level_change_kg": 65_000.0,
            "route_cutoff_min": std - DEFAULT_SETTINGS["route_cutoff_min_before_std"],
            "eligible_for_saf": True,
            "energy_source": "assumption: linear A20N burn proxy × 43 MJ/kg",
        })
    return pd.DataFrame(rows)


def level_reason(f: pd.Series, route: str) -> str:
    if route == "reference":
        return ""
    if f.cruise_km_available < 150:
        return f"short leg: {f.cruise_km_available:.0f} km of cruise after climb/descent (< 150 km)"
    if route == "higher" and f.peak_alt_ft + 2000 > A320NEO_CEILING_FT:
        return f"{f.peak_alt_ft + 2000:,.0f} ft exceeds A320neo ceiling {A320NEO_CEILING_FT:,} ft"
    return "disabled by user"


def contrail_screen(fc: pd.DataFrame, study_id: str, cycle: str, route: str,
                    rhi_mult: float, lw: float, sw: float):
    """Persistent-contrail screen along the observed track at one level.

    Persistent if RHi × multiplier ≥ 100% and T below the Schmidt–Appleman
    threshold. Returns (contrail_km, signed EF in J, detail DataFrame).
    The EF per km is a *teaching surrogate*: longwave warming minus a shortwave
    term proportional to sin(solar elevation).
    """
    d = fc[(fc.study_id == study_id) & (fc.source_forecast_init_utc == cycle)].copy()
    if d.empty:
        d = fc[fc.study_id == study_id].copy()
        if d.empty:
            return 0.0, 0.0, d
        first = sorted(d.source_forecast_init_utc.unique())[0]
        d = d[d.source_forecast_init_utc == first].copy()
    d = d.sort_values("waypoint_utc").reset_index(drop=True)
    off = {"reference": 0, "lower": -2000, "higher": 2000}[route]
    tcol = {"reference": "temperature_K", "lower": "temperature_K_lower_2000ft",
            "higher": "temperature_K_higher_2000ft"}[route]
    rcol = {"reference": "relative_humidity_over_ice_pct", "lower": "relative_humidity_over_ice_pct_lower_2000ft",
            "higher": "relative_humidity_over_ice_pct_higher_2000ft"}[route]
    lat, lon = d.latitude_deg.to_numpy(), d.longitude_deg.to_numpy()
    seg = np.zeros(len(d))
    for k in range(len(d) - 1):
        seg[k] = ph.haversine_km(lat[k], lon[k], lat[k + 1], lon[k + 1])
    if len(d) > 1:
        seg[-1] = np.median(seg[:-1])
    alt = d.assumed_pressure_altitude_ft.to_numpy() + off
    T = d[tcol].to_numpy(dtype=float)
    rhi = d[rcol].to_numpy(dtype=float) * rhi_mult
    tsac = np.array([ph.sac_threshold_K(isa_pressure_hpa(a)) for a in alt])
    persistent = np.nan_to_num(rhi) >= 100.0
    persistent &= np.nan_to_num(T, nan=999) < tsac
    when = pd.to_datetime(d.waypoint_utc, utc=True)
    elev = np.array([ph.solar_elevation_deg(la, lo, w.to_pydatetime()) for la, lo, w in zip(lat, lon, when)])
    ef_km = lw - sw * np.clip(np.sin(np.radians(elev)), 0, None)
    ef = float((seg * ef_km * persistent).sum())
    d = d.assign(segment_km=seg, rhi_scaled=rhi, T_sac_K=tsac, persistent=persistent,
                 solar_elev_deg=elev, ef_J=seg * ef_km * persistent, level_ft=alt)
    return float((seg * persistent).sum()), ef, d


def build_sourced_options(flights: pd.DataFrame, scenarios: pd.DataFrame, fc: pd.DataFrame,
                          settings: dict) -> pd.DataFrame:
    rows = []
    for _, f in flights.iterrows():
        for route in ROUTES:
            for _, sc in scenarios.iterrows():
                km, ef, _ = contrail_screen_cached(f.study_id, settings["planning_cycle"], route,
                                            float(sc.rhi_multiplier), settings["ef_lw_J_per_km"],
                                            settings["ef_sw_J_per_km"])
                rows.append({"study_id": f.study_id, "route": route, "scenario": sc.scenario,
                             "contrail_km": round(km, 1), "EF_ordinary_J": ef,
                             "ef_source": "teaching surrogate (GFS screen)",
                             "forecast_issue_utc": settings["planning_cycle"]})
    return pd.DataFrame(rows)


def apply_team_estimates(options: pd.DataFrame, flights: pd.DataFrame, team: pd.DataFrame) -> tuple:
    """Merge a filled option_scenario_inputs CSV. Rows with values override the surrogate.

    Only ordinary-fuel rows set EF_ordinary_J; route_energy_MJ overrides energy.
    Returns (options, flights, n_rows_used).
    """
    opts = options.copy()
    fl = flights.copy()
    used = 0
    t = team.dropna(subset=["EF_contrail_j"], how="all") if "EF_contrail_j" in team else team.iloc[0:0]
    for _, r in t.iterrows():
        if str(r.get("fuel_option", "")).startswith("reference") and pd.notna(r.get("EF_contrail_j")):
            m = (opts.study_id == r.study_id) & (opts.route == r.route_option) & (opts.scenario == r.weather_scenario_id)
            if m.any():
                opts.loc[m, "EF_ordinary_J"] = float(r.EF_contrail_j)
                opts.loc[m, "ef_source"] = "team estimate"
                opts.loc[m, "forecast_issue_utc"] = r.get("forecast_issue_utc", "")
                used += 1
    return opts, fl, used


def sourced_case() -> Case:
    man_raw = load_manifest()
    if man_raw.empty:
        c = synthetic_case()
        c.notes.append("12 proposed departures · manifest pending — showing the illustrative simulation.")
        return c
    man, notes = annotate_manifest(man_raw)
    summ, fc = load_tracks()
    flights = _default_flight_inputs(man, summ)
    scen = DEFAULT_SCENARIOS.copy()
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    options = build_sourced_options(flights, scen, fc, settings)
    return Case("Sourced cohort", "sourced", man, flights, scen, options,
                copy.deepcopy(DEFAULT_SUPPLY), settings, ph.FuelParams(), notes)


# ---------------------------------------------------------------------------
# Synthetic preset
# ---------------------------------------------------------------------------
# EF_ordinary in units of 3 TJ for (reference, lower, higher) × (dry, moderate, high).
# None = level unavailable.
_SYN = [
    # id, dest, dist_km, std, EF table, note
    ("SYN-01", "CPH", 600, "05:00", [[0, 2, 9], [0, 0, 1], [0, 1, 4]], "warming at night; cheap lower detour"),
    ("SYN-02", "OSL", 420, "05:40", [[0, 2, 10], None, None], "short leg: planned profile only, humid-sensitive"),
    ("SYN-03", "LHR", 1650, "06:10", [[0, 18, 40], [0, 2, 6], [0, 8, 22]], "cheap detour, large avoided EF"),
    ("SYN-04", "CDG", 1600, "07:40", [[1, 10, 24], [0, 9, 22], [0, 2, 5]], "expensive higher detour helps"),
    ("SYN-05", "BER", 880, "07:45", [[0, 4, 12], [0, 3, 11], None], "modest benefit, at ceiling"),
    ("SYN-06", "OSL", 430, "08:05", [[0, 0, 0], None, None], "never forms persistent contrails"),
    ("SYN-07", "CPH", 590, "10:50", [[0, 3, 6], [0, 2, 5], [0, 0, 1]], "expensive detour, small EF"),
    ("SYN-08", "LHR", 1640, "11:10", [[0, -6, -14], [0, -1, -3], [0, -4, -9]], "midday cooling case"),
    ("SYN-09", "CPH", 595, "15:05", [[0, 6, 15], [0, 1, 3], [0, 5, 13]], "dusk warming"),
    ("SYN-10", "LHR", 1570, "15:20", [[0, 14, 30], [0, 12, 28], [0, 1, 4]], "high-humidity sensitive"),
    ("SYN-11", "CDG", 1630, "16:55", [[0, 9, 26], [0, 3, 8], [0, 8, 24]], "night warming"),
    ("SYN-12", "OSL", 430, "19:45", [[0, 3, 12], None, None], "short leg: planned profile only, night warming"),
]


def synthetic_case() -> Case:
    rows, opts, man = [], [], []
    scen = DEFAULT_SCENARIOS.copy()
    scen["rhi_multiplier"] = np.nan  # EF table is entered directly
    for sid, dest, dist, std, table, note in _SYN:
        std_utc = f"2026-11-17T{std}:00Z"
        trip = 600.0 + 2.65 * dist
        uplift = math.ceil((trip * 1.05 + 2200) / 10) * 10
        lower_ok = table[1] is not None
        higher_ok = table[2] is not None
        rows.append({
            "study_id": sid, "flight_number": "", "destination": dest, "std_utc": std_utc,
            "modelled_aircraft": "Airbus A320-251N", "modelled_engine": "2 × CFM LEAP-1A26",
            "track_distance_km": float(dist), "peak_alt_ft": 37000.0 if higher_ok else 39000.0,
            "cruise_km_available": float(dist - 400),
            "trip_fuel_kg": round(trip), "taxi_kg": 200.0, "reserve_kg": 2000.0,
            "uplift_kg": uplift, "M0_kg": 0.0, "w0": 0.0, "tank_capacity_kg": 18_700.0,
            "lower_allowed": lower_ok, "higher_allowed": higher_ok,
            "lower_penalty_pct": 0.6 if sid == "SYN-03" else 1.8,
            "higher_penalty_pct": 4.0 if sid in ("SYN-04", "SYN-07") else 0.3,
            "level_changes": 2, "mass_for_level_change_kg": 65_000.0,
            "route_cutoff_min": to_min(std_utc) - 90, "eligible_for_saf": True,
            "energy_source": "illustrative assumption",
        })
        man.append({"study_id": sid, "flight_number": "", "origin_iata": "ARN", "destination_iata": dest,
                    "scheduled_departure_utc": std_utc, "is_synthetic": True,
                    "flight_status": "synthetic", "aircraft_status": "synthetic",
                    "engine_status": "synthetic", "operator_status": "synthetic",
                    "conflict": "", "documented": False, "audit_note": note})
        for r, route in enumerate(ROUTES):
            for s, sc in enumerate(scen.scenario):
                ef = 3.0 * table[r][s] * 1e12 if table[r] is not None else 0.0
                opts.append({"study_id": sid, "route": route, "scenario": sc,
                             "contrail_km": np.nan, "EF_ordinary_J": ef,
                             "ef_source": "synthetic teaching table", "forecast_issue_utc": ""})
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    settings["rel_change_at_ref"] = -0.30  # illustrative, synthetic preset only
    return Case("Illustrative Stockholm simulation", "synthetic", pd.DataFrame(man), pd.DataFrame(rows),
                scen, pd.DataFrame(opts), copy.deepcopy(DEFAULT_SUPPLY), settings, ph.FuelParams(),
                ["Synthetic rows use internal IDs only and are excluded from historical counts and exports."])


# ---------------------------------------------------------------------------
# Forecast information per flight
# ---------------------------------------------------------------------------
def update_visibility(flights: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Which GFS cycle each flight's dispatcher can see at its route cutoff.

    A flight whose cutoff precedes publication of any cycle newer than the
    fuel-deadline cycle sees *no update* regardless of the global mode.
    """
    try:
        wi = pd.read_csv(DATA / "weather_index.csv")
        wi = wi[wi.role == "archived forecast"]
        cyc = wi.groupby("forecast_init_utc").provider_index_modified_utc.max().reset_index()
        cyc["avail_min"] = cyc.provider_index_modified_utc.map(to_min)
    except Exception:
        cyc = pd.DataFrame({"forecast_init_utc": [], "avail_min": []})
    plan_min = to_min(settings["planning_cycle"])
    out = []
    for _, f in flights.iterrows():
        cut = float(f.route_cutoff_min)
        seen = cyc[cyc.avail_min <= cut]
        latest = seen.forecast_init_utc.max() if len(seen) else ""
        newer = bool(latest) and to_min(latest) > plan_min
        out.append({"study_id": f.study_id, "route_cutoff": from_min(cut),
                    "latest_cycle_visible": latest or "none", "has_update": newer})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Build the optimisation Problem
# ---------------------------------------------------------------------------
def grade_list(case: Case) -> list[ph.Grade]:
    g = [ph.REFERENCE, ph.HEFA_35]
    s = case.settings
    if s.get("blend_sensitivity"):
        g.append(ph.Grade(f"Sensitivity_{s['sens_fraction']:.0%}_{s['sens_basis']}", float(s["sens_fraction"]),
                          s["sens_basis"], True, "Blend sensitivity (not the main case)"))
    return g


def route_energy_MJ(f: pd.Series, route: str, fp: ph.FuelParams, scale: float = 1.0) -> float:
    base = float(f.trip_fuel_kg) * fp.H_J
    if route == "reference":
        return base
    pct = float(f.lower_penalty_pct if route == "lower" else f.higher_penalty_pct) / 100.0
    cruise_frac = max(0.0, min(1.0, float(f.cruise_km_available) / max(float(f.track_distance_km), 1.0)))
    level = float(f.level_changes) * float(f.mass_for_level_change_kg) * 9.81 * 609.6 / 0.35 / 1e6
    return base + scale * (base * pct * cruise_frac + level)


def build_problem(case: Case) -> tuple[Problem, pd.DataFrame]:
    fl = case.flights.reset_index(drop=True)
    sc = case.scenarios.reset_index(drop=True)
    st, sup, fp = case.settings, case.supply, case.fuel
    grades = grade_list(case)
    n, B, R, S = len(fl), len(grades), len(ROUTES), len(sc)
    if n == 0:
        raise InputError("No departures in the case.")
    p = sc.probability.to_numpy(dtype=float)
    if np.any(~np.isfinite(p)):
        raise InputError("Every scenario needs a probability.")
    fuel = np.zeros((n, B, R, S))
    con = np.zeros((n, B, R, S))
    feas = np.ones((n, B, R, S), dtype=bool)
    reasons, greasons = {}, {}
    allowed = np.ones((n, B), dtype=bool)
    q = np.zeros((n, B))
    prem = np.zeros((n, B))
    cash = np.zeros((n, B))
    comp = np.zeros((n, B, R, S))
    sburn = np.zeros((n, B, R, S))
    loads = np.zeros((n, B))
    ded = np.array([g.dedicated_delivery for g in grades])
    detail = []
    opt = case.options.set_index(["study_id", "route", "scenario"])
    fuel_deadline = to_min(st["fuel_deadline_utc"])
    svc_start = np.array([to_min(t) - sup["service_offset_min"] for t in fl.std_utc])
    svc_end = svc_start + sup["service_minutes"]

    for i, f in fl.iterrows():
        if f.uplift_kg < 0 or f.M0_kg < 0 or f.reserve_kg < 0:
            raise InputError(f"{f.study_id}: fuel quantities must be non-negative.")
        for b, g in enumerate(grades):
            w_u = g.mass_fraction(fp)
            if g.dedicated_delivery:
                book_by = svc_start[i] - 60 * sup["booking_lead_h"]
                if not g.spec_compliant(fp):
                    allowed[i, b] = False
                    greasons[(i, b)] = (f"{g.volume_fraction(fp):.1%} by volume exceeds the HEFA-SPK "
                                        f"Annex A2 limit of {fp.hefa_a2_max_vol:.0%}")
                elif not bool(f.eligible_for_saf):
                    allowed[i, b] = False
                    greasons[(i, b)] = "marked ineligible for the dedicated delivery"
                elif fuel_deadline > book_by + 1e-9:
                    allowed[i, b] = False
                    greasons[(i, b)] = (f"supplier booking cutoff {from_min(book_by)} is before the "
                                        f"fuel deadline {from_min(fuel_deadline)}")
            vol_L = f.uplift_kg / ph.blend_density(w_u, fp.rho_S, fp.rho_J) * 1000
            loads[i, b] = math.ceil(vol_L / sup["truck_capacity_L"] - 1e-9) if g.dedicated_delivery else 0
            if g.dedicated_delivery and loads[i, b] > sup["trucks_per_window"]:
                allowed[i, b] = False
                greasons[(i, b)] = (f"needs {int(loads[i, b])} truck loads ({vol_L:,.0f} L) but only "
                                    f"{sup['trucks_per_window']} trucks per window")
            for r, route in enumerate(ROUTES):
                E = route_energy_MJ(f, route, fp, st["detour_energy_scale"])
                br = ph.burn(E, f.uplift_kg, w_u, fp, f.M0_kg, f.w0, g.dedicated_delivery)
                cr = ph.costs(br, w_u, fp, sup["fee_eur"] if g.dedicated_delivery else 0.0)
                q[i, b] = br.q_stock_kg
                prem[i, b] = cr.premium_eur if g.dedicated_delivery else 0.0
                cash[i, b] = cr.cash_eur
                level_ok = route == "reference" or bool(f[f"{route}_allowed"])
                need = br.m_burn_kg + f.taxi_kg + f.reserve_kg
                fuel_ok = need <= f.M0_kg + f.uplift_kg + 1e-6
                tank_ok = f.M0_kg + f.uplift_kg <= f.tank_capacity_kg + 1e-6
                for s, scen_name in enumerate(sc.scenario):
                    key = (f.study_id, route, scen_name)
                    ef0 = float(opt.loc[key, "EF_ordinary_J"]) if key in opt.index else 0.0
                    ef = ph.blended_EF(ef0, br.w_onboard, st["response_scale"], st["rel_change_at_ref"])
                    fuel[i, b, r, s] = br.L_fuel_t
                    con[i, b, r, s] = ph.contrail_score_t(ef, fp)
                    comp[i, b, r, s] = cr.comparison_eur
                    sburn[i, b, r, s] = br.m_S_kg
                    if not level_ok:
                        feas[i, b, r, s] = False
                        reasons[(i, b, r, s)] = level_reason(f, route)
                    elif not fuel_ok:
                        feas[i, b, r, s] = False
                        reasons[(i, b, r, s)] = (f"burn {br.m_burn_kg:,.0f} + taxi + reserve "
                                                 f"{need:,.0f} kg exceeds fuel onboard {f.M0_kg + f.uplift_kg:,.0f} kg")
                    elif not tank_ok:
                        feas[i, b, r, s] = False
                        reasons[(i, b, r, s)] = "fuel onboard exceeds declared tank capacity"
                    detail.append({
                        "study_id": f.study_id, "grade": g.name, "route": route, "scenario": scen_name,
                        "uplift_kg": f.uplift_kg, "saf_mass_frac_delivered": w_u,
                        "saf_vol_frac_delivered": g.volume_fraction(fp), "q_SAF_kg": br.q_SAF_kg,
                        "q_stock_kg": br.q_stock_kg, "w_onboard": br.w_onboard,
                        "route_energy_MJ": E, "H_mix_MJ_kg": br.H_mix, "m_burn_kg": br.m_burn_kg,
                        "m_SAF_burn_kg": br.m_S_kg, "m_jet_burn_kg": br.m_J_kg,
                        "remaining_kg": br.remaining_kg, "EF_ordinary_J": ef0, "EF_J": ef,
                        "L_fuel_t": br.L_fuel_t, "L_contrail_t": con[i, b, r, s],
                        "C_t": br.L_fuel_t + con[i, b, r, s], "cash_eur": cr.cash_eur,
                        "comparison_eur": cr.comparison_eur, "premium_eur": prem[i, b],
                        "truck_loads": loads[i, b], "feasible": bool(feas[i, b, r, s] and allowed[i, b]),
                        "reason": reasons.get((i, b, r, s), greasons.get((i, b), "")),
                    })

    base_groups = info_groups(st["info_mode"], list(sc.scenario),
                              list(sc.scenario[sc.pooled_in_partial.astype(bool)]))
    vis = update_visibility(fl, st)
    groups = np.tile(base_groups, (n, 1))
    for i in range(n):
        if not bool(vis.iloc[i].has_update) and case.kind == "sourced":
            groups[i] = 0  # no newer cycle published before this flight's route cutoff
    day_start, day_end = math.floor(svc_start.min() / 60) * 60, math.ceil(svc_end.max() / 60) * 60
    wmin = sup["window_minutes"]
    windows = [SlotWindow(t, t + wmin, int(sup["trucks_per_window"]), f"{from_min(t)[7:12]}–{from_min(t + wmin)[7:12]}")
               for t in np.arange(day_start, day_end, wmin)]
    receipts = []
    if sup.get("receipt_kg", 0) and sup.get("receipt_time_utc"):
        receipts = [(to_min(sup["receipt_time_utc"]), float(sup["receipt_kg"]))]
    order = list(np.argsort(svc_start, kind="stable"))
    rep_name = st.get("representative")
    rep = list(sc.scenario).index(rep_name) if rep_name in list(sc.scenario) else int(np.argmax(p))
    P = Problem(
        flights=list(fl.study_id), grades=[g.name for g in grades], routes=ROUTES,
        scenarios=list(sc.scenario), p=p, fuel=fuel, contrail=con, feasible=feas,
        infeasible_reason=reasons, grade_allowed=allowed, grade_reason=greasons, groups=groups,
        q_stock=q, dedicated=ded, premium_eur=prem, service_start_min=svc_start,
        service_end_min=svc_end, truck_loads=loads, opening_stock_kg=float(sup["opening_stock_kg"]),
        receipts=receipts, slot_windows=windows, max_special=int(sup["max_special"]),
        cash_cap_eur=float(sup["cash_cap_eur"]) if sup.get("cash_cap_enabled") else None,
        priority_order=order, preferred_grade=1, units="t CO2e", representative=rep,
        cash_eur=cash, comparison_eur=comp, saf_burn_kg=sburn,
        meta={"case": case.name, "kind": case.kind, "visibility": vis},
    )
    return P, pd.DataFrame(detail)
