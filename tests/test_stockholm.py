import io

import numpy as np
import pandas as pd
import pytest

from saf_app.evidence import (SyntheticExportBlocked, annotate_manifest, deduplicate_codeshares,
                              evidence_status, historical_export, roster_summary)
from saf_app.presets import make_preset
from saf_app.runner import export_csv, run_case
from saf_app.stockholm import build_problem, load_manifest, sourced_case, synthetic_case


@pytest.fixture(scope="module")
def sourced_run():
    c = sourced_case()
    return c, run_case(c)


def test_twelve_roster_slots_and_destinations():
    man, _ = annotate_manifest(load_manifest())
    s = roster_summary(man)
    assert s["rows"] == 12 and s["complete"]
    assert set(s["destinations"]) == {"CPH", "OSL", "BER", "LHR", "CDG"}
    assert s["headline"].startswith("12 departures")


def test_manifest_pending_headline():
    s = roster_summary(pd.DataFrame(columns=["is_synthetic", "documented", "flight_status",
                                             "aircraft_status", "engine_status", "operator_status",
                                             "conflict", "destination_iata", "study_id", "flight_number"]))
    assert s["headline"] == "12 proposed departures · manifest pending"


def test_incomplete_records_headline():
    man, _ = annotate_manifest(load_manifest().head(7))
    assert roster_summary(man)["headline"] == "7 of 12 departures documented · manifest incomplete"


def test_duplicate_codeshare_removed():
    man = load_manifest()
    dup = man.iloc[[0]].copy()
    dup["flight_number"] = "XX9999"
    dup["marketing_carrier"] = "Partner"
    dup["operating_carrier"] = "SAS"
    man = man.copy()
    man.loc[0, "operating_carrier"] = "SAS"
    out, notes = deduplicate_codeshares(pd.concat([man, dup], ignore_index=True))
    assert len(out) == 12
    assert "XX9999" in notes[0]
    assert "SK1415" in out.flight_number.tolist()


def test_scheduled_versus_flown_and_disputed_aircraft():
    row = load_manifest().iloc[0].copy()
    assert evidence_status(row)["flight_status"].startswith("flown")
    sched = row.copy()
    sched["observed_status"] = ""
    sched["reported_departure_utc"] = np.nan
    assert evidence_status(sched)["flight_status"] == "scheduled only (timetable)"
    disp = row.copy()
    disp["observed_aircraft_variant"] = "Airbus A320-214"
    assert evidence_status(disp)["aircraft_status"].startswith("disputed")
    assert evidence_status(row)["engine_status"] == "subvariant unverified"


def test_conflict_listed():
    man, _ = annotate_manifest(load_manifest())
    s = roster_summary(man)
    assert any(c["flight_number"] == "SK531" for c in s["conflicts"])


def test_synthetic_export_blocked():
    c = synthetic_case()
    with pytest.raises(SyntheticExportBlocked):
        historical_export(c.manifest)
    r = run_case(c)
    assert "SYNTHETIC" in export_csv(r, c)


def test_synthetic_preset_has_12_rows_5_destinations():
    c = synthetic_case()
    assert len(c.flights) == 12 and set(c.flights.destination) == {"CPH", "OSL", "BER", "LHR", "CDG"}
    assert c.flights.study_id.str.startswith("SYN-").all()
    ef = c.options.EF_ordinary_J
    assert (ef > 0).any() and (ef == 0).any() and (ef < 0).any()


def test_booking_cutoff_and_window_eligibility():
    c = sourced_case()
    c.supply["booking_lead_h"] = 8.0       # ARN-01 service 04:00Z -> book by 20:00Z on 16th < deadline 22:00Z
    P, _ = build_problem(c)
    assert not P.grade_allowed[0, 1]
    assert "booking cutoff" in P.grade_reason[(0, 1)]
    c.supply["booking_lead_h"] = 6.0
    c.supply["trucks_per_window"] = 1      # 2-load LHR uplifts no longer fit one window
    P, _ = build_problem(c)
    lhr = [i for i, d in enumerate(c.flights.destination) if d == "LHR"]
    assert all(not P.grade_allowed[i, 1] for i in lhr)
    c.supply["trucks_per_window"] = 0
    r = run_case(c)
    assert all(res.x.sum() == 0 for res in r.results.values())


def test_reserve_and_fuel_feasibility():
    c = sourced_case()
    c.flights.loc[0, "uplift_kg"] = 1000.0     # not enough for trip + reserve
    P, detail = build_problem(c)
    assert not P.feasible[0].any()
    ok = detail[detail.feasible].merge(c.flights[["study_id", "reserve_kg", "taxi_kg"]], on="study_id")
    assert len(ok) > 0
    assert (ok.remaining_kg >= 0).all()
    assert (ok.remaining_kg >= ok.reserve_kg + ok.taxi_kg - 1e-6).all()   # reserve compliance


def test_short_leg_only_planned_profile():
    c = sourced_case()
    P, _ = build_problem(c)
    osl = [i for i, d in enumerate(c.flights.destination) if d == "OSL"]
    for i in osl:
        assert P.feasible[i, :, 0].all() and not P.feasible[i, :, 1:].any()


def test_taxi_statistic_and_cascade_year_do_not_change_ef(tmp_path):
    c1 = sourced_case()
    P1, _ = build_problem(c1)
    c2 = sourced_case()
    c2.settings["ground_nvpm_number_median_reduction"] = 99   # unrelated statistic
    c2.settings["cascade_year"] = 2036
    P2, _ = build_problem(c2)
    assert np.array_equal(P1.contrail, P2.contrail)


def test_zero_response_keeps_fuel_benefit():
    c = sourced_case()
    c.settings["response_scale"] = 0.0
    P, _ = build_problem(c)
    assert np.allclose(P.contrail[:, 1], P.contrail[:, 0])
    assert (P.fuel[:, 1] < P.fuel[:, 0]).all()


def test_all_strategies_same_inventory(sourced_run):
    c, r = sourced_run
    cap = c.supply["opening_stock_kg"] + c.supply["receipt_kg"]
    for res in r.results.values():
        assert res.saf_stock_kg(r.P) <= cap + 1e-6
    assert (r.summary.departures_included == 12).all()


def test_csv_matches_dashboard(sourced_run):
    c, r = sourced_run
    csv = export_csv(r, c)
    body = "\n".join(l for l in csv.splitlines() if not l.startswith("#"))
    df = pd.read_csv(io.StringIO(body))
    assert list(df.strategy) == list(r.summary.strategy)
    assert np.allclose(df.expected_score_tCO2e, r.summary.expected_score_tCO2e, rtol=1e-5)
    assert f"heldout_seed,{c.settings['heldout_seed']}" in csv
    assert "t CO2e" in csv


def test_fresh_start_documented_default():
    c = make_preset("Sourced cohort (12 ARN departures, 17 Nov 2026)")
    r = run_case(c)
    js = r.results["Joint stochastic"]
    assert c.settings["info_mode"] == "partial_update"
    assert c.supply["opening_stock_kg"] == 4400 and c.supply["receipt_kg"] == 2200
    assert r.label.startswith("Sourced cohort · modelled decisions")
    # documented default result (README): three SAF flights, 6,580 kg pure SAF
    assert int((js.x > 0).sum()) == 3
    assert js.saf_stock_kg(r.P) == pytest.approx(6580, abs=5)
    assert r.P.groups[0].tolist() == [0, 0, 0]   # ARN-01 cutoff precedes the 00Z cycle
