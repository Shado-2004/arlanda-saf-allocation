"""Climate-Optimal SAF Allocation Under Uncertainty | Stockholm Arlanda

Run:  python -m streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from saf_app import physics as ph
from saf_app.evidence import SyntheticExportBlocked, historical_export, roster_summary
from saf_app.examples import EXAMPLE_A_TABLE, EXAMPLE_B_TABLE, example_a, example_b
from saf_app.explain import explain_decision, flight_table, saving_per_tonne
from saf_app.model import InputError
from saf_app.presets import (PRESETS, apply_ef_wide, clone, ef_wide, ensure_options, make_preset,
                             sanitize_flights)
from saf_app.runner import case_fingerprint, export_csv, run_case
from saf_app.solver import SolveError
from saf_app.stockholm import (DATA, ROUTE_LABEL, ROUTES, apply_team_estimates, contrail_screen_cached,
                               from_min, load_tracks, route_energy_MJ, to_min)
from saf_app.strategies import REFERENCE_NAME, STRATEGY_NAMES

NAVY, TEAL, GREY, LIGHT = "#0b2545", "#12867a", "#8a99ad", "#e6f2f0"
QUESTION = ("Which Stockholm departures should receive the available 35%-by-mass SAF blend, how much "
            "pure SAF would that use, and when would an altitude change help more?")

st.set_page_config(page_title="SAF allocation · Stockholm Arlanda", layout="wide")
st.markdown(f"""
<style>
  .stApp {{ background:#ffffff; color:{NAVY}; }}
  h1,h2,h3,h4,p,li,label,span,div {{ color:{NAVY}; }}
  .runlabel {{ display:inline-block; padding:4px 10px; border-left:4px solid {TEAL};
              background:{LIGHT}; font-weight:600; margin:4px 0 8px 0; }}
  .synth {{ border-left-color:#b5651d; background:#fbf1e6; }}
  .stale {{ padding:6px 10px; border-left:4px solid #b00020; background:#fdecee; font-weight:600; }}
</style>""", unsafe_allow_html=True)


def plot_layout(fig, title="", height=380, **kw):
    fig.update_layout(template="plotly_white", title=title, height=height,
                      font=dict(color=NAVY, size=13), margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.2), **kw)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch")


def fmt_t(x, d=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
ss = st.session_state
DEFAULT_PRESET = list(PRESETS)[0]


def reset(preset: str):
    for k in [k for k in ss.keys() if k.startswith("w_")]:
        del ss[k]
    ss.preset = preset
    ss.case = make_preset(preset)
    ss.base_flights = ss.case.flights.copy()
    ss.base_scen = ss.case.scenarios.copy()
    ss.base_opts = ss.case.options.copy()
    ss.opts_sig = None
    ss.error = None
    try:
        ss.run = run_case(ss.case)
    except (InputError, SolveError) as e:
        ss.run, ss.error = None, str(e)


if "case" not in ss:
    reset(DEFAULT_PRESET)

case = ss.case

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Case and run")
    preset = st.selectbox("Preset", list(PRESETS), index=list(PRESETS).index(ss.preset), key="w_preset")
    if preset != ss.preset:
        reset(preset)
        st.rerun()
    c1, c2 = st.columns(2)
    run_clicked = c1.button("Run optimisation", type="primary", width="stretch")
    if c2.button("Reset to preset", width="stretch"):
        reset(ss.preset)
        st.rerun()
    auto = st.toggle("Recompute automatically on change", value=True, key="w_auto",
                     help="Off: edits are flagged until you press Run optimisation.")

    S, U = case.settings, case.supply
    st.header("Sensitivity")
    S["response_scale"] = st.slider("SAF effect on contrails (response scale)", 0.0, 2.0,
                                    float(S["response_scale"]), 0.1, key="w_resp",
                                    help="0 = zero-cruise-response sensitivity for the LEAP engine: SAF keeps "
                                         "only its fuel life-cycle effect.")
    modes = {"No update (one group)": "no_update",
             "Partial update (dry distinct; moderate/high alike)": "partial_update",
             "Scenario revealed (perfect stage-two information)": "revealed"}
    inv = {v: k for k, v in modes.items()}
    S["info_mode"] = modes[st.radio("What dispatch can distinguish at the route cutoff", list(modes),
                                    index=list(modes).index(inv[S["info_mode"]]), key="w_info")]
    U["opening_stock_kg"] = st.number_input("Opening dedicated stock (kg pure HEFA-SPK)", 0.0, 50000.0,
                                            float(U["opening_stock_kg"]), 100.0, key="w_stock")
    U["receipt_kg"] = st.number_input("Mid-day receipt (kg pure SAF, at receipt time)", 0.0, 50000.0,
                                      float(U["receipt_kg"]), 100.0, key="w_receipt")
    U["trucks_per_window"] = int(st.number_input("Truck slots per service window", 0, 10,
                                                 int(U["trucks_per_window"]), 1, key="w_trucks"))
    U["max_special"] = int(st.number_input("Maximum special refuellings per day", 0, 30,
                                           int(U["max_special"]), 1, key="w_maxsp"))
    U["cash_cap_enabled"] = st.toggle("Cap extra cash spent on the blend", value=bool(U["cash_cap_enabled"]),
                                      key="w_capon")
    U["cash_cap_eur"] = st.number_input("Cash cap on SAF premium + delivery fees (€)", 0.0, 1e6,
                                        float(U["cash_cap_eur"]), 500.0, key="w_cap",
                                        disabled=not U["cash_cap_enabled"],
                                        help="Enable the toggle above to apply this cap.")
    S["detour_energy_scale"] = st.slider("Extra energy for altitude changes (× default)", 0.0, 3.0,
                                         float(S["detour_energy_scale"]), 0.1, key="w_detour")
    with st.expander("Timing and supplier"):
        U["booking_lead_h"] = st.number_input("Supplier booking lead time (h before service)", 0.0, 48.0,
                                              float(U["booking_lead_h"]), 0.5, key="w_lead")
        fd = st.text_input("Common fuel-allocation deadline (UTC, ISO)", S["fuel_deadline_utc"], key="w_fd")
        try:
            pd.Timestamp(fd)
            S["fuel_deadline_utc"] = fd
        except Exception:
            st.error("Not a valid timestamp; keeping the previous deadline.")
        U["fee_eur"] = st.number_input("Delivery fee per special uplift (€)", 0.0, 10000.0,
                                       float(U["fee_eur"]), 50.0, key="w_fee")
        U["truck_capacity_L"] = st.number_input("Finished blend per truck load (L)", 1000.0, 40000.0,
                                                float(U["truck_capacity_L"]), 500.0, key="w_truckL")
    with st.expander("Fuel properties and prices"):
        f = case.fuel
        f.CI_S = st.number_input("SAF life-cycle intensity (g CO₂e/MJ)", 0.0, 120.0, float(f.CI_S), 0.1,
                                 key="w_cis", help="13.9 = ICAO CORSIA used-cooking-oil HEFA default (sensitivity).")
        f.CI_J = st.number_input("Jet A-1 life-cycle intensity (g CO₂e/MJ)", 0.0, 150.0, float(f.CI_J), 0.1, key="w_cij")
        f.price_J_eur_t = st.number_input("Jet A-1 price (€/t)", 0.0, 5000.0, float(f.price_J_eur_t), 10.0, key="w_pj")
        f.price_S_eur_t = st.number_input("HEFA price (€/t)", 0.0, 10000.0, float(f.price_S_eur_t), 10.0, key="w_ps")
        f.rho_S = st.number_input("HEFA-SPK density (kg/m³)", 700.0, 850.0, float(f.rho_S), 1.0, key="w_rhos")
        f.rho_J = st.number_input("Jet A-1 density (kg/m³)", 750.0, 850.0, float(f.rho_J), 1.0, key="w_rhoj")
    with st.expander("Blend sensitivity (separate from the 35%-by-mass case)"):
        S["blend_sensitivity"] = st.toggle("Add a third grade", value=bool(S["blend_sensitivity"]), key="w_bs")
        S["sens_fraction"] = st.slider("Sensitivity SAF fraction", 0.0, 0.6, float(S["sens_fraction"]), 0.01,
                                       key="w_bsf", disabled=not S["blend_sensitivity"])
        S["sens_basis"] = st.radio("Basis", ["volume", "mass"], index=["volume", "mass"].index(S["sens_basis"]),
                                   key="w_bsb", horizontal=True, disabled=not S["blend_sensitivity"])
        st.caption("Grades above 50% by volume are screened out (HEFA-SPK Annex A2). The optimum blend "
                   "remains an empirical question.")
    with st.expander("Advanced"):
        S["contrail_weight"] = st.slider("Optimiser preference weight on contrails (dimensionless)", 0.0, 2.0,
                                         float(S["contrail_weight"]), 0.1, key="w_cw",
                                         help="Changes what the optimiser values. Reported scores stay at weight 1.")
        S["cvar_weight"] = st.slider("Risk aversion: weight on worst-5% CVaR", 0.0, 2.0,
                                     float(S["cvar_weight"]), 0.1, key="w_cvar",
                                     help="Weight 0 reproduces the main result.")
        S["contrail_horizon"] = st.selectbox("Climate horizon", ["100-year", "20-year (unavailable)"],
                                             index=0, key="w_hor")
        if S["contrail_horizon"].startswith("20"):
            st.warning("20-year view unavailable: no documented 20-year contrail factor with compatible "
                       "fuel intensities is bundled. Results stay on the QAEI 100-year basis.")
            S["contrail_horizon"] = "100-year"
        S["heldout_seed"] = int(st.number_input("Held-out seed", 0, 2**31 - 1, int(S["heldout_seed"]), 1, key="w_seed"))
        S["heldout_draws"] = int(st.number_input("Held-out draws", 100, 20000, int(S["heldout_draws"]), 100, key="w_draws"))
        S["heldout_sigma"] = st.slider("Within-state EF spread σ (lognormal)", 0.0, 1.0,
                                       float(S["heldout_sigma"]), 0.05, key="w_sigma")
        if case.kind == "sourced":
            S["ef_lw_J_per_km"] = st.number_input("Surrogate longwave EF (J per contrail km)", 0.0, 1e13,
                                                  float(S["ef_lw_J_per_km"]), 1e10, format="%.2e", key="w_lw")
            S["ef_sw_J_per_km"] = st.number_input("Surrogate shortwave EF per sin(sun elevation) (J/km)", 0.0, 1e13,
                                                  float(S["ef_sw_J_per_km"]), 1e10, format="%.2e", key="w_sw")

# ---------------------------------------------------------------------------
# Layout skeleton (inputs in tab 2 are read before the run is computed)
# ---------------------------------------------------------------------------
header = st.container()
tab1, tab2, tab3, tab4 = st.tabs(["Decision board", "Flights and forecasts", "Compare strategies",
                                  "How the decision works"])
with tab2:
    inputs_box = st.container()
    outputs_box = st.container()

with inputs_box:
    if case.kind == "sourced":
        st.markdown("##### The 12 case records (historical, read-only)")
        summary = roster_summary(case.manifest)
        man = case.manifest
        st.dataframe(man[["study_id", "flight_number", "destination_iata", "scheduled_departure_local",
                          "scheduled_departure_utc", "reported_departure_utc", "marketing_carrier",
                          "operating_carrier", "registration", "observed_aircraft_variant", "modelled_aircraft",
                          "modelled_engine", "flight_status", "aircraft_status", "engine_status",
                          "operator_status", "flight_source_url"]],
                     hide_index=True, width="stretch",
                     column_config={"flight_source_url": st.column_config.LinkColumn("Evidence")})
        c = st.columns(4)
        c[0].metric("Rows documented (flown + airframe verified)", f"{summary['documented']} / 12")
        c[1].metric("Destinations", " · ".join(summary["destinations"]))
        c[2].metric("Verified-aircraft subset", f"{summary['airframe_verified']} of 12",
                    help="Airframe variant matches the modelled A320-251N with an aircraft-specific source.")
        c[3].metric("Engine subvariant verified", f"{summary['engine_verified']} of 12")
        st.caption("Operating carrier is blank in the source for rows where the tracker shows only SAS-group "
                   "attribution; it is left blank here rather than guessed. Codeshares are de-duplicated by "
                   "aircraft and departure before counting.")
        if summary["conflicts"]:
            st.markdown("**Conflicting evidence**")
            for cf in summary["conflicts"]:
                st.markdown(f"- {cf['study_id']} {cf['flight_number']}: {cf['conflict']}")
        for n_ in case.notes:
            st.caption(n_)
        try:
            hist = historical_export(case.manifest)
            st.download_button("Download historical manifest with evidence status (CSV)",
                               hist.to_csv(index=False), "arn_manifest_evidence.csv", "text/csv")
        except SyntheticExportBlocked as e:
            st.warning(str(e))
    else:
        summary = None
        st.markdown("##### Synthetic rows (illustrative, not historical)")
        st.dataframe(case.manifest, hide_index=True, width="stretch")
        try:
            historical_export(case.manifest)
        except SyntheticExportBlocked as e:
            st.caption(f"Historical export disabled: {e}")

    st.markdown("##### Modelling inputs per departure (editable; historical records are not changed)")
    st.caption("Sourced: study ID, times, track distance and peak altitude. Assumed: trip fuel (linear A20N "
               "proxy), reserve, uplift, carry-in M0 = 0, level-change penalties, tank capacity.")
    locked = ["study_id", "flight_number", "destination", "std_utc", "track_distance_km", "peak_alt_ft"] \
        if case.kind == "sourced" else []
    ed = st.data_editor(ss.base_flights, key="w_flights", hide_index=True, width="stretch",
                        num_rows="dynamic" if case.kind != "sourced" else "fixed", disabled=locked,
                        column_config={
                            "route_cutoff_min": st.column_config.NumberColumn("Route cutoff (min after 17 Nov 00Z)"),
                            "uplift_kg": st.column_config.NumberColumn("Uplift (kg)"),
                            "M0_kg": st.column_config.NumberColumn("Carry-in M0 (kg)"),
                            "w0": st.column_config.NumberColumn("Carry-in SAF mass fraction"),
                        })
    case.flights = sanitize_flights(ed, case)
    if case.kind != "sourced":
        keep = list(case.flights.study_id)
        man = ss.case.manifest
        extra_rows = [{"study_id": sid, "is_synthetic": True, "flight_status": "synthetic",
                       "audit_note": "added in app"} for sid in keep if sid not in set(man.study_id)]
        case.manifest = pd.concat([man[man.study_id.isin(keep)], pd.DataFrame(extra_rows)], ignore_index=True)
        st.caption(f"{len(case.flights)} departures in this synthetic run (add or delete rows above).")

    st.markdown("##### Weather scenarios and what dispatch can distinguish")
    st.caption("Each scenario is one coherent weather field. Sourced cohort: the GFS forecast at each waypoint "
               "with relative humidity over ice scaled by the multiplier (an illustrative forecast-error "
               "assumption). Probabilities must sum to 1.")
    sc_ed = st.data_editor(ss.base_scen, key="w_scen", hide_index=True, num_rows="dynamic", width="stretch",
                           column_config={"probability": st.column_config.NumberColumn(format="%.3f"),
                                          "pooled_in_partial": st.column_config.CheckboxColumn(
                                              "Pooled under partial update")})
    sc_ed = sc_ed.dropna(subset=["scenario"]).drop_duplicates("scenario").reset_index(drop=True)
    sc_ed["probability"] = pd.to_numeric(sc_ed["probability"], errors="coerce").fillna(0.0)
    sc_ed["pooled_in_partial"] = sc_ed["pooled_in_partial"].fillna(False).astype(bool)
    if case.kind == "sourced":
        sc_ed["rhi_multiplier"] = pd.to_numeric(sc_ed["rhi_multiplier"], errors="coerce").fillna(1.0)
    case.scenarios = sc_ed
    tot = case.scenarios.probability.sum()
    (st.success if abs(tot - 1) < 1e-6 else st.error)(f"Probabilities sum to {tot:.4f}")
    scen_names = list(case.scenarios.scenario)
    if scen_names:
        default_rep = int(np.argmax(case.scenarios.probability.to_numpy()))
        rep = st.selectbox("Representative scenario for the joint deterministic method", scen_names,
                           index=default_rep, key="w_rep", help="Defaults to the most probable scenario.")
        S["representative"] = rep

    # EF table: base options rebuilt only when what they depend on changes
    sig = (case.scenarios[["scenario"] + (["rhi_multiplier"] if case.kind == "sourced" else [])].to_json(),
           S["ef_lw_J_per_km"], S["ef_sw_J_per_km"], tuple(case.flights.study_id))
    if ss.opts_sig is not None and ss.opts_sig != sig:
        tmp = clone(case)
        tmp.options = ss.base_opts
        ensure_options(tmp, rebuild_surrogate=True)
        ensure_options(tmp)
        ss.base_opts = tmp.options
    ss.opts_sig = sig
    case.options = ss.base_opts
    ensure_options(case)
    st.markdown("##### Signed contrail energy forcing with ordinary fuel (TJ per flight, route, scenario)")
    wide = ef_wide(case)
    ef_ed = st.data_editor(wide, key=f"w_ef_{abs(hash(sig)) % 10**8}", hide_index=True, width="stretch",
                           disabled=["study_id", "route"])
    apply_ef_wide(case, ef_ed)
    src_counts = case.options.ef_source.value_counts()
    st.caption("Sources: " + " · ".join(f"{k}: {v} option rows" for k, v in src_counts.items()) +
               ". Positive warms, negative cools. The surrogate is a teaching estimate, not CoCiP.")
    up = st.file_uploader("Import team EF/energy estimates (filled option_scenario_inputs CSV)", type="csv",
                          key="w_upload")
    if up is not None and st.button("Apply imported estimates"):
        team = pd.read_csv(up)
        ss.base_opts, _, used = apply_team_estimates(ss.base_opts, case.flights, team)
        st.success(f"{used} option rows now use team estimates.")
        st.rerun()

with tab3:
    bench = st.selectbox("Benchmark for paired differences", STRATEGY_NAMES + [REFERENCE_NAME],
                         index=(STRATEGY_NAMES + [REFERENCE_NAME]).index(S.get("benchmark", "Original-route targeting")),
                         key="w_bench")
    S["benchmark"] = bench

fp = case_fingerprint(case)
if run_clicked or (auto and (ss.run is None or ss.run.fingerprint != fp)):
    try:
        ss.run = run_case(case)
        ss.error = None
    except (InputError, SolveError, ValueError) as e:
        ss.error = str(e)
run = ss.run
stale = run is None or run.fingerprint != fp

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
with header:
    st.title("Climate-Optimal SAF Allocation Under Uncertainty | Stockholm Arlanda")
    st.markdown(f"#### {QUESTION}")
    if case.kind == "sourced":
        st.markdown(f"**{summary['headline']}** · {summary['flown']} flown in dated trackers · "
                    f"{summary['airframe_verified']}/12 airframes verified · engines modelled (subvariant unverified)")
    else:
        st.markdown("**12 proposed departures · manifest pending** for this preset — the rows below are synthetic."
                    if case.name.startswith("Illustrative") else "**Small teaching fixture** · synthetic rows")
    if run is not None:
        cls = "runlabel synth" if run.kind != "sourced" else "runlabel"
        st.markdown(f'<span class="{cls}">Run label: {run.label}</span>', unsafe_allow_html=True)
    if ss.error:
        st.error(f"Inputs not solved: {ss.error}")
    if stale and run is not None:
        st.markdown('<div class="stale">Inputs have changed: the results below belong to the previous run. '
                    'Press “Run optimisation”.</div>', unsafe_allow_html=True)

if run is None:
    st.stop()

P, R = run.P, run.results
js = R["Joint stochastic"]
st.caption("Counterfactual coursework results. Real dispatch needs approved flight levels, aircraft "
           "performance checks and air traffic clearance; the model never certifies a fuel batch.")

# ===========================================================================
# TAB 1 — Decision board
# ===========================================================================
with tab1:
    alloc = run.allocation
    saf_used = js.saf_stock_kg(P)
    avail = case.supply["opening_stock_kg"] + case.supply.get("receipt_kg", 0)
    ref = R[REFERENCE_NAME]
    extra = js.comparison_eur(P) - ref.comparison_eur(P)
    m = st.columns(5)
    m[0].metric("Flights receiving SAF", f"{int((js.x > 0).sum())} of {len(P.flights)}")
    m[1].metric("Pure SAF drawn from stock", f"{saf_used:,.0f} kg", f"{avail - saf_used:,.0f} kg left",
                delta_color="off")
    m[2].metric("Pure SAF actually burned (expected)", f"{js.saf_burn_kg(P):,.0f} kg",
                help="Less than the uplift: reserve fuel stays onboard.")
    m[3].metric("Expected climate score", f"{js.expected:,.2f} t CO₂e",
                f"{js.expected - ref.expected:+,.2f} t vs ordinary fuel", delta_color="inverse")
    m[4].metric("Extra cost vs ordinary fuel", f"€{extra:,.0f}",
                help="Comparison cost: burned fuel valued by component plus delivery fees.")
    st.markdown("##### Fleet plan · joint stochastic method")
    show_cols = ["study_id", "flight", "to", "STD_UTC", "fuel", "blend_mass_pct", "blend_vol_pct",
                 "uplift_kg", "pure_SAF_stock_kg"] + [c for c in alloc.columns if c.startswith("route |")] + \
                ["update_visible", "expected_score_tCO2e"]
    st.dataframe(alloc[show_cols], hide_index=True, width="stretch", column_config={
        "blend_mass_pct": st.column_config.NumberColumn("SAF % by mass", format="%.1f"),
        "blend_vol_pct": st.column_config.NumberColumn("SAF % by volume", format="%.2f"),
        "uplift_kg": st.column_config.NumberColumn("Uplift (kg finished fuel)", format="%.0f"),
        "pure_SAF_stock_kg": st.column_config.NumberColumn("Pure SAF from stock (kg)", format="%.0f"),
        "expected_score_tCO2e": st.column_config.NumberColumn("Expected score (t CO₂e)", format="%.3f"),
        "update_visible": st.column_config.CheckboxColumn("Forecast update visible"),
        "STD_UTC": "STD (UTC)",
        **{c: st.column_config.TextColumn(f"Route if {c.split('| ')[1]}") for c in alloc.columns if c.startswith("route |")},
    })
    st.caption("Routes are decided at each flight's route cutoff using only the forecast signal it can see; "
               "outcomes it cannot tell apart share a route.")

    st.markdown("##### Explain one flight")
    labels = [f"{a.study_id} {a.flight} → {a.to} ({a.STD_UTC} UTC)" for a in alloc.itertuples()]
    sel = st.selectbox("Flight", labels, key="w_sel")
    i = labels.index(sel)
    ft = flight_table(P, i, S["contrail_weight"])
    sp = saving_per_tonne(P, i, 0, 1, S["contrail_weight"])
    cA, cB = st.columns([3, 2])
    with cA:
        ft_show = ft.rename(columns={"grade": "Grade", "expected_score": "Expected score (t CO₂e)",
                                     "route_by_signal": "Best route policy", "pure_SAF_kg": "Pure SAF (kg)",
                                     "truck_loads": "Truck loads", "premium_eur": "SAF premium + fee (€)",
                                     "eligible": "Eligible", "reason": "Why unavailable"})
        ft_show.insert(0, "Option", ["Best ordinary-fuel alternative", "Best SAF alternative"] +
                       ["Blend sensitivity"] * (len(ft_show) - 2))
        st.dataframe(ft_show, hide_index=True, width="stretch",
                     column_config={"Expected score (t CO₂e)": st.column_config.NumberColumn(format="%.3f")})
        num = sp["numerator_t"]
        if sp["zero_denominator"]:
            st.markdown("Saving per tonne of pure SAF: **zero denominator** (no extra SAF uplifted).")
        elif num is None:
            st.markdown("Saving per tonne of pure SAF: undefined (one grade has no feasible route policy).")
        else:
            st.markdown(f"Saving per tonne of pure SAF ({sp['from']} → {sp['to']}): "
                        f"**{num:.3f} t CO₂e ÷ {sp['denominator_t_SAF']:.3f} t pure SAF = "
                        f"{sp['ratio']:.2f} t CO₂e per t SAF**. This ranks individual opportunities; the fleet "
                        "choice also depends on uplift size and slots.")
        with st.spinner("Re-solving the fleet with this flight's other option forced…"):
            st.info(explain_decision(P, js, i, S["contrail_weight"], S["cvar_weight"]))
    with cB:
        fl = case.flights.iloc[i]
        dist = float(fl.track_distance_km)
        cruise_alt = float(fl.peak_alt_ft)
        climb = min(200.0, dist * 0.45)
        xs = [0, climb, dist - climb, dist]
        fig = go.Figure()
        chosen = set(P.routes[r] for r in js.route[i])
        for route, dz, dash in (("reference", 0, "solid"), ("higher", 2000, "dot"), ("lower", -2000, "dash")):
            avail_r = route == "reference" or bool(fl[f"{route}_allowed"])
            a = cruise_alt + dz
            ys = [0, a, a, 0]
            if route != "reference":
                xs_r = [climb, climb + 40, dist - climb - 40, dist - climb]
                ys = [cruise_alt, a, a, cruise_alt]
            else:
                xs_r = xs
            name = ROUTE_LABEL[route] + ("" if avail_r else " — unavailable")
            if route in chosen:
                name += " ✓ chosen in ≥1 signal"
            fig.add_trace(go.Scatter(x=xs_r, y=ys, mode="lines", name=name,
                                     line=dict(color=TEAL if route in chosen else (GREY if avail_r else "#d0d6de"),
                                               width=4 if route in chosen else 2, dash=dash)))
            fig.add_annotation(x=dist * {"reference": 0.5, "higher": 0.3, "lower": 0.7}[route], y=a,
                               text=ROUTE_LABEL[route] + ("" if avail_r else " (n/a)"),
                               showarrow=False, yshift=12 if dz >= 0 else -12, font=dict(size=11))
        plot_layout(fig, f"Schematic profiles · {alloc.iloc[i].flight or alloc.iloc[i].study_id}", 330,
                    xaxis_title="Distance flown (km)", yaxis_title="Pressure altitude (ft)")
        show(fig)
        st.caption("Schematic, not to scale. Only feasible levels can be chosen; reasons below.")
        reasons = {P.routes[k[2]]: v for k, v in P.infeasible_reason.items() if k[0] == i}
        for r_, why in reasons.items():
            st.caption(f"• {ROUTE_LABEL[r_]}: {why}")

    with st.expander("Show the calculation for this flight"):
        d = run.detail[(run.detail.study_id == P.flights[i])]
        st.markdown("Per option: `b = (w/ρS)/(w/ρS+(1−w)/ρJ)`, `q_SAF = U·w`, "
                    "`w_onboard = (M0·w0 + U·w)/(M0+U)`, `H_mix = (1−w_on)H_J + w_on·H_S`, "
                    "`m_burn = E/H_mix`, `L_fuel = (m_J·H_J·CI_J + m_S·H_S·CI_S)/10⁶`, "
                    "`L_contrail = k100·EF`, `C = L_fuel + L_contrail + L_extra`.")
        st.dataframe(d.drop(columns=["study_id"]), hide_index=True, width="stretch")

# ===========================================================================
# TAB 2 — Flights and forecasts
# ===========================================================================
with outputs_box:
    vis = P.meta.get("visibility")
    if vis is not None and case.kind == "sourced":
        st.markdown("**Forecast visible to each dispatcher at its route cutoff (GFS publication times)**")
        st.dataframe(vis, hide_index=True, width="stretch")
        st.caption("A flight whose cutoff precedes publication of any cycle newer than the fuel-deadline "
                   "cycle sees no update, whatever the global mode.")
    st.markdown("##### Supply at ARN (all editable assumptions)")
    st.json({k: v for k, v in case.supply.items()}, expanded=False)
    st.caption("Quante et al. report a dedicated ~8,000 L truck and remote-stand fuelling in the 2023 trial. "
               "That motivates these scenarios; it does not establish trucks, slots or uplifts on 17 Nov 2026. "
               "Truck throughput and deadline results apply only to the assumptions entered.")

    if case.kind == "sourced":
        st.markdown("##### Weather along the observed track")
        sid = st.selectbox("Flight", list(case.flights.study_id), key="w_wx")
        _, fc = load_tracks()
        d = fc[fc.study_id == sid]
        fig = go.Figure()
        for k, (cyc, g) in enumerate(d.groupby("source_forecast_init_utc")):
            fig.add_trace(go.Scatter(x=pd.to_datetime(g.waypoint_utc), y=g.relative_humidity_over_ice_pct,
                                     mode="lines+markers", name=f"GFS {cyc[5:16].replace('T', ' ')}Z",
                                     marker=dict(size=4)))
        fig.add_hline(y=100, line_dash="dash", line_color=NAVY, annotation_text="ice saturation (100%)")
        plot_layout(fig, f"{sid}: relative humidity over ice at the flown level, by forecast cycle", 340,
                    xaxis_title="Waypoint time (UTC)", yaxis_title="RHi (%)")
        show(fig)
        km, ef, det = contrail_screen_cached(sid, S["planning_cycle"], "reference", 1.0,
                                             S["ef_lw_J_per_km"], S["ef_sw_J_per_km"])
        st.caption(f"At the planning cycle with no humidity scaling: {km:.0f} km meets the persistence screen "
                   f"(RHi ≥ 100% and T below the Schmidt–Appleman threshold), surrogate EF {ef / 1e12:.2f} TJ. "
                   "Contrails form when the plume mixes to liquid saturation (cold enough air); they persist only "
                   "in ice-supersaturated air. Their warming depends on sunlight (shortwave cooling by day), "
                   "the surface and clouds below, and how the contrail-cirrus evolves.")

# ===========================================================================
# TAB 3 — Compare strategies
# ===========================================================================
with tab3:
    summ = run.summary
    compact = st.toggle("Compact pitch view", value=False, key="w_compact")
    names = list(summ.strategy)
    short = [n.replace(" targeting", "").replace("Ordinary fuel, adaptive routes", "Ordinary fuel") for n in names]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=short, y=summ.fuel_lifecycle_tCO2e, name="Fuel life-cycle",
                         marker_color=NAVY, text=[f"{v:.1f}" for v in summ.fuel_lifecycle_tCO2e],
                         textposition="inside", insidetextanchor="middle"))
    fig.add_trace(go.Bar(x=short, y=summ.contrail_tCO2e, name="Signed contrail",
                         marker_color=TEAL, marker_pattern_shape="/"))
    fig.add_trace(go.Scatter(x=short, y=summ.expected_score_tCO2e, mode="markers+text", name="Total",
                             marker=dict(symbol="diamond", size=11, color="#b5651d"),
                             text=[f"total {v:.2f}" for v in summ.expected_score_tCO2e],
                             textposition="top center"))
    plot_layout(fig, f"Expected climate score, {len(P.flights)} departures (lower is better)",
                420 if not compact else 330, barmode="relative",
                yaxis_title="t CO₂e (QAEI 100-year)", xaxis_title="Strategy",
                yaxis_range=[0, float(summ.expected_score_tCO2e.max()) * 1.12])
    fc1, fc2 = st.columns([3, 2])
    with fc1:
        show(fig)
    with fc2:
        fcon = go.Figure(go.Bar(x=short, y=summ.contrail_tCO2e, marker_color=TEAL, marker_pattern_shape="/",
                                text=[f"{v:+.3f}" for v in summ.contrail_tCO2e], textposition="outside"))
        plot_layout(fcon, "Signed contrail contribution (+ warms, − cools)", 420 if not compact else 330,
                    yaxis_title="t CO₂e", xaxis_title="Strategy")
        show(fcon)
    bm = summ.set_index("strategy").loc[bench, "expected_score_tCO2e"]
    if not compact:
        c1, c2 = st.columns(2)
        with c1:
            f2 = go.Figure(go.Bar(x=names, y=summ.expected_score_tCO2e - bm, marker_color=[
                TEAL if v < -1e-9 else (GREY if abs(v) <= 1e-9 else "#b00020") for v in summ.expected_score_tCO2e - bm],
                text=[f"{v:+.3f}" for v in summ.expected_score_tCO2e - bm], textposition="outside"))
            plot_layout(f2, f"Difference from {bench} (negative = better)", 340,
                        yaxis_title="Δ expected score (t CO₂e)", xaxis_title="Strategy")
            show(f2)
        with c2:
            f3 = go.Figure()
            f3.add_trace(go.Bar(x=names, y=summ.pure_SAF_from_stock_kg, name="Pure SAF from stock (kg)",
                                marker_color=NAVY, text=[f"{v:,.0f}" for v in summ.pure_SAF_from_stock_kg]))
            f3.add_trace(go.Bar(x=names, y=summ.pure_SAF_burned_kg, name="Pure SAF burned (kg)",
                                marker_color=TEAL, text=[f"{v:,.0f}" for v in summ.pure_SAF_burned_kg]))
            plot_layout(f3, "SAF used", 340, barmode="group", yaxis_title="kg pure SAF", xaxis_title="Strategy")
            show(f3)
    tbl = summ.copy()
    st.dataframe(tbl, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%.3f") for c in tbl.columns if c.endswith("tCO2e")} | {
        c: st.column_config.NumberColumn(format="€%.0f") for c in tbl.columns if c.endswith("eur")} | {
        "eur_per_t_saved": st.column_config.NumberColumn("€ per t CO₂e saved (only if saving > 0)", format="€%.0f"),
        "pure_SAF_from_stock_kg": st.column_config.NumberColumn(format="%.0f"),
        "pure_SAF_burned_kg": st.column_config.NumberColumn(format="%.0f")})
    st.caption("Cash paid = uplift × €/t ÷ 1000 + fees. Comparison cost values opening and closing fuel by "
               "component at the same prices (so leftover SAF is not counted as spent) plus fees once.")
    if not compact:
        st.markdown("##### Scenario breakdown")
        sct = run.scenario_table
        f4 = go.Figure()
        for k, sn in enumerate(P.scenarios):
            d = sct[sct.scenario == sn]
            f4.add_trace(go.Bar(x=d.strategy, y=d.fleet_score_tCO2e, name=f"{sn} (p={P.p[k]:.2f})",
                                text=[f"{v:.1f}" for v in d.fleet_score_tCO2e]))
        plot_layout(f4, "Fleet score if each scenario occurs", 360, barmode="group",
                    yaxis_title="t CO₂e", xaxis_title="Strategy")
        show(f4)
        st.markdown("##### Held-out synthetic weather (independent seed)")
        pr = run.paired.copy()
        st.dataframe(pr, hide_index=True, width="stretch", column_config={
            "mean_paired_saving": st.column_config.NumberColumn("Mean paired saving of joint stochastic (t CO₂e)", format="%.3f"),
            "se": st.column_config.NumberColumn("Std. error", format="%.3f"),
            "frac_focus_worse": st.column_config.NumberColumn("Share of draws joint does worse", format="%.3f"),
            "frac_tie": st.column_config.NumberColumn("Share tied", format="%.3f"),
            "n_draws": "Draws"})
        strong = run.strongest
        sv = float(pr.set_index("comparator").loc[strong, "mean_paired_saving"])
        d_saf = js.saf_stock_kg(P) - R[strong].saf_stock_kg(P)
        st.markdown(f"Strongest feasible comparator (same stock and routing access): **{strong}**. Added value "
                    f"of joint stochastic planning on held-out weather: **{sv:+.3f} t CO₂e** "
                    f"({len(P.flights)} departures). SAF difference: {d_saf:+,.0f} kg pure SAF.")
        if abs(sv) < 1e-6:
            st.caption("A tie on held-out weather is a valid result: here the stock size, not the weather, "
                       "decides who gets the fuel.")
        st.caption(run.heldout.method + " Planning results are exact expectations (no sampling).")
    try:
        csv = export_csv(run, case)
        st.download_button("Download comparison CSV (units, strategy, seed, assumptions)", csv,
                           f"saf_comparison_{run.kind}_{run.fingerprint}.csv", "text/csv")
    except Exception as e:  # pragma: no cover
        st.error(str(e))

# ===========================================================================
# TAB 4 — How the decision works
# ===========================================================================
with tab4:
    st.markdown("##### Decision timeline")
    fl = case.flights
    fig = go.Figure()
    fdl = to_min(S["fuel_deadline_utc"])
    fig.add_vline(x=fdl / 60, line_color=NAVY, line_dash="dash")
    fig.add_annotation(x=fdl / 60, y=1.02, yref="paper", text="Fuel commitment (all flights)", showarrow=False)
    for k, f_ in enumerate(fl.itertuples()):
        std = to_min(f_.std_utc) / 60
        book = (to_min(f_.std_utc) - U["service_offset_min"] - 60 * U["booking_lead_h"]) / 60
        fig.add_trace(go.Scatter(x=[book, f_.route_cutoff_min / 60, std], y=[k] * 3, mode="markers+lines",
                                 marker=dict(symbol=["square", "circle", "triangle-right"], size=10,
                                             color=[GREY, TEAL, NAVY]),
                                 line=dict(color="#d0d6de"), showlegend=False,
                                 hovertext=["supplier booking cutoff", "route cutoff", "departure"]))
    fig.update_yaxes(tickvals=list(range(len(fl))), ticktext=[f"{a} {b}" for a, b in zip(fl.study_id, fl.flight_number)])
    plot_layout(fig, "■ booking cutoff · ● route cutoff · ▶ departure (hours after 17 Nov 00:00 UTC)", 420,
                xaxis_title="Hours after 17 Nov 2026 00:00 UTC (negative = 16 Nov)", yaxis_title="Departure")
    show(fig)
    st.markdown("""
1. **Fuel deadline (stage 1)** – one grade per flight is committed for all possible weather outcomes, before the
   first supplier booking closes. The pure SAF component draws on the segregated ARN stock.
2. **Route cutoff (stage 2)** – each dispatcher sees only its forecast signal and picks a feasible level.
   Outcomes in the same signal group must share a route.
3. **Flight** – the hidden weather happens; the score counts fuel life-cycle CO₂e plus signed contrail forcing.
""")

    st.markdown("##### The two worked examples (solved live by the same code)")
    from saf_app.strategies import run_all
    ea, eb, eb0 = run_all(example_a()), run_all(example_b()), run_all(example_b("no_update"))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Example A — why choose fuel and route together?** (score points, one outcome)")
        st.dataframe(EXAMPLE_A_TABLE, hide_index=True)
        o, j = ea["Original-route targeting"].expected, ea["Joint stochastic"].expected
        st.markdown(f"Original-route targeting gives SAF to **{'AB'[int(np.argmax(ea['Original-route targeting'].x))]}** "
                    f"and scores **{o:g}** after rerouting. Joint planning gives SAF to "
                    f"**{'AB'[int(np.argmax(ea['Joint stochastic'].x))]}** and scores **{j:g}** "
                    f"({(o - j) / o:.1%} lower). Ordinary fuel with adaptive routes: **{ea[REFERENCE_NAME].expected:g}**. "
                    f"Joint deterministic = {ea['Joint deterministic'].expected:g}.")
    with c2:
        st.markdown("**Example B — does uncertainty change who gets SAF?** (ordinary / SAF)")
        st.dataframe(EXAMPLE_B_TABLE, hide_index=True)
        d, s_ = eb["Joint deterministic"], eb["Joint stochastic"]
        st.markdown(f"Dry-only planner → SAF to **{'AB'[int(np.argmax(d.x))]}**: {d.scenario_scores[0]:g} dry, "
                    f"{d.scenario_scores[1]:g} humid, expected **{d.expected:.1f}**. Stochastic planner → SAF to "
                    f"**{'AB'[int(np.argmax(s_.x))]}**: {s_.scenario_scores[0]:g} and {s_.scenario_scores[1]:g}, "
                    f"expected **{s_.expected:.1f}**. With no update the joint optimum is "
                    f"**{eb0['Joint stochastic'].expected:.1f}**.")

    with st.expander("Show the calculation: optimisation model"):
        st.code("""minimise   Σ_s p[s] Σ_{i,b,r} C[i,b,r,s] · y[i,b,r,g_i(s)]
subject to Σ_b x[i,b] = 1                                    each flight
           Σ_r y[i,b,r,g] = x[i,b]                           each flight, grade, signal group
           y[i,b,r,g_i(s)] ≤ a[i,b,r,s]                       route feasible in every pooled outcome
           Σ_{i: service starts ≤ t} q_stock[i,b]·x[i,b] ≤ opening stock + receipts by t
           Σ_{i overlapping window w} loads[i,b]·x[i,b] ≤ truck slots in w
           Σ_i Σ_{b dedicated} x[i,b] ≤ maximum special refuellings
           Σ premium_eur[i,b]·x[i,b] ≤ cash cap           (if enabled)
           x[i,b] = 0 for ineligible grades (booking cutoff missed, Annex A2, truck volume)
Optional:  + λ·CVaR_5%(fleet score)   via η + (1/0.05) Σ_s p_s u_s,  u_s ≥ Z_s − η, u_s ≥ 0
Tie rule:  tiny penalties prefer less SAF, the planned profile, and earlier flights keeping ordinary fuel.
Solver:    scipy.optimize.milp (HiGHS); status is checked before any plan is shown.""", language="text")
    with st.expander("Show the calculation: fuel and climate equations with live unit checks"):
        fp_ = ph.FuelParams()
        b35 = ph.mass_to_volume_fraction(0.35, fp_.rho_S, fp_.rho_J)
        be = ph.breakeven_EF_J(100, fp_, H=43.2, CI=89)
        st.code(f"""b         = (w_u/ρS)/(w_u/ρS + (1−w_u)/ρJ)        35% mass → {b35:.4%} volume (760/800 kg/m³)
q_SAF     = U·w_u                                 1,000 kg × 0.35 = {1000 * 0.35:.0f} kg pure SAF
w_onboard = (M0·w0 + U·w_u)/(M0 + U)              1,000 kg at 35% onto 1,000 kg Jet A-1 → {ph.burn(1, 1000, .35, fp_, 1000, 0).w_onboard:.3f}
H_mix     = (1−w_on)·H_J + w_on·H_S                 MJ/kg
m_burn    = route_energy / H_mix                  kg
L_fuel    = (m_J·H_J·CI_J + m_S·H_S·CI_S)/10⁶      t CO₂e
k100      = 0.42/(5.1e14 · 31,536,000 · 8.8e-14 · 1000) = {ph.K100:.4e} t CO₂e/J = {ph.K100 * 1e12:.3f} t per TJ
L_contrail= k100 · EF (signed: + warms, − cools)
Break-even: 100 kg fossil detour at 43.2 MJ/kg, 89 g/MJ = {100 * 43.2 * 89 / 1e6:.4f} t → must avoid {be / 1e12:.2f} TJ
EF_blend  = EF_ordinary + response_scale · δ,  δ = {S['rel_change_at_ref']:+.2f} · (w_onboard/0.35) · EF_ordinary""",
                language="text")
        st.caption("δ is an illustrative teaching coefficient. It is not the CFM56-5B4 ~11% simulation result and "
                   "not the 42% taxi nvPM statistic; a particle-number change alone cannot set a warming change.")

    st.markdown("##### 60-second demo script")
    top = alloc[alloc.fuel == "SAF blend"]
    first = top.iloc[0] if len(top) else alloc.iloc[0]
    st.markdown(f"""
1. *(10 s)* “Twelve SAS-group A320neo departures from Arlanda on 17 November 2026. We have
   {avail:,.0f} kg of pure HEFA for a 35%-by-mass blend — enough for only a few aircraft.”
2. *(15 s)* Decision board: “The joint plan fuels **{', '.join(f'{a.flight or a.study_id}→{a.to}' for a in top.itertuples()) or 'no flights'}**,
   drawing {saf_used:,.0f} kg from stock, for {js.expected - ref.expected:+.2f} t CO₂e against ordinary fuel.”
3. *(15 s)* Pick **{first.flight or first.study_id}** → read the sentence: why it gets fuel, the saving per tonne,
   and the levels it may fly under each forecast signal.
4. *(15 s)* Sidebar → switch *What dispatch can distinguish* to **No update** and then **Scenario revealed**.
   Say whether the list of SAF flights changes (often it does not — stock size dominates; that is a finding).
5. *(5 s)* Compare tab: “Same stock, same slots, five methods; held-out weather shows the paired difference.”
""")

    st.markdown("##### Evidence from Arlanda")
    st.markdown("""
- **Row-level flight evidence** (published, dated trackers): the 12 manifest rows, registrations and ADS-B tracks
  (adsb.lol contributors, ODbL). SK531's two trackers report different times; SK1415 replaced SK1417 after
  conflicting tail evidence; one 59,200 ft point was removed from SK1407.
- **2023 demonstration evidence** — Quante et al. (2025): 84 demonstration legs Stockholm–Copenhagen, both
  directions, 17 Jan–2 Feb 2023; segregated nominal 35%-by-mass HEFA uplifted at ARN; dedicated ~8,000 L truck;
  LEAP-1A26 aircraft. The oft-quoted ~11% contrail EF reduction came from a **CFM56-5B4** simulation — context only.
- **Ground engine evidence** — Schmitt et al. (2026): Copenhagen taxi plumes, A320-251N with LEAP-1A26; for a
  34% HEFA blend the median nvPM number EI fell 42% (95% interval 17–62%); 85 blend-fuelled and 30 reference
  services (a different denominator from Quante's 84 legs). Not used as a cruise EF coefficient.
- **Private or disputed inputs**: ARN stock, trucks, slots, booking deadlines, uplifts, payloads and batch
  intensity are unknown for 17 Nov 2026 and are editable assumptions here.
- **Counterfactual results**: every allocation and saving in this app.
""")
    st.caption("Data-pack note: `one_trial_size_truck_equivalent_neat_saf` is listed as 2,199,483.2 “kg”; "
               f"8,000 L × {ph.blend_density(.35, 760, 800):.2f} kg/m³ × 0.35 = "
               f"{8 * ph.blend_density(.35, 760, 800) * .35:,.1f} kg. The app uses the recalculated value.")

    st.markdown("##### European outlook to 2036 (separate scale)")
    pol = pd.read_csv(DATA / "refueleu_policy_shares.csv")
    show_years = pol[pol.year.isin([2026, 2030, 2035, 2036])]
    f5 = go.Figure(go.Bar(x=pol.year.astype(str), y=pol.total_SAF_minimum_pct, marker_color=[
        TEAL if y in (2026, 2030, 2035, 2036) else "#c9d3df" for y in pol.year],
        text=[f"{v:g}%" for v in pol.total_SAF_minimum_pct], textposition="outside"))
    plot_layout(f5, "ReFuelEU Annex I minimum SAF supply share, covered EU airports", 320,
                xaxis_title="Year", yaxis_title="Minimum SAF share of fuel supplied (%)")
    show(f5)
    st.dataframe(show_years[["year", "total_SAF_minimum_pct", "metric", "geography"]], hide_index=True)
    st.caption("Scenario context: Boeing Cascade ReFuelEU scenario (traffic within or departing the EU, fuel shares "
               "consistent with the policy). Metric: minimum SAF supply share (%); geography: covered EU airport "
               "supply; years 2025–2036; extracted with the data pack (Nov 2026). Cascade's simulated emissions "
               "series is not bundled (interactive terms acceptance), so no emissions curve is drawn. EU-level "
               "shares cannot set these 12 aircraft's tank composition, price or ARN stock, and a one-day sample "
               "does not support a European annual savings estimate.")

    st.markdown("##### Sources and labels")
    src = pd.DataFrame([
        ("12-flight manifest, tracks, GFS forecasts", "published (row-level evidence)", "Which flights flew, where, and the weather forecast"),
        ("Quante et al. 2025, doi:10.1016/j.jatrs.2024.100049", "published", "Delivery logistics: segregated truck, ~8,000 L"),
        ("Schmitt et al. 2026, PMC12969261", "published", "Ground nvPM evidence only"),
        ("Teoh et al. 2022; Woeldgen et al. 2025", "published", "Targeted SAF allocation context"),
        ("Voigt et al. 2021; Märkl et al. 2024", "published", "Exhaust and contrail ice measurements"),
        ("Engberg et al. 2025; pycontrails", "published", "Signed EF forecasting method"),
        ("ICAO Annex A2 50% vol; CORSIA UCO-HEFA 13.9 g/MJ", "published", "Blend limit; SAF intensity sensitivity"),
        ("Boeing Cascade 89 g/MJ; ReFuelEU scenario", "published", "Conventional intensity; European outlook"),
        ("EASA 2025 €640/t and €1,925/t (EU)", "published", "Illustrative price basis, 2025, EU"),
        ("QAEI 100-year factor k100", "course convention", "Contrail EF → t CO₂e"),
        ("Densities 800/760, LHV 43/44, burn proxy, stock, trucks, fees, RHi multipliers, EF/km", "illustrative assumption", "Editable"),
        ("Allocations, routes, scores, costs, savings", "calculated result", "Counterfactual coursework outputs"),
        ("Imperial–Rolls-Royce QRITOS", "published (announcement)", "Existing work on targeting SAF; scope as reported"),
    ], columns=["Input or finding", "Label", "Used for"])
    st.dataframe(src, hide_index=True, width="stretch")
    st.markdown("##### Limits")
    st.markdown("""
- Contrail EF is a teaching surrogate from a GFS humidity/temperature screen along the observed tracks, not CoCiP;
  thin ice-supersaturated layers are smoothed by the 0.25° grid. Route-specific CoCiP estimates can be imported.
- SAF's effect on the contrail formation threshold is not modelled; a SAF-fuelled flight can still leave a
  persistent contrail. Reducing a cooling contrail can worsen the score.
- Equal-efficiency approximation for route energy; flights are independent (no rotations; carry-in M0 = 0 by default).
- Scenarios are illustrative forecast-error states, not a calibrated ensemble.
""")
