"""Manifest handling: evidence status, codeshare de-duplication, conflicts.

The historical manifest is never edited by modelling controls. Synthetic rows
carry `is_synthetic=True` and are blocked from historical counts and exports.
"""
from __future__ import annotations

import pandas as pd

SAS_GROUP_OPERATORS = {"SAS", "SAS CONNECT", "SAS LINK", "SCANDINAVIAN AIRLINES"}
ROSTER_SIZE = 12
STUDY_DESTINATIONS = ["CPH", "OSL", "BER", "LHR", "CDG"]

MANIFEST_COLUMNS = [
    "study_id", "flight_number", "origin_iata", "destination_iata",
    "scheduled_departure_local", "scheduled_departure_utc",
    "reported_departure_local", "reported_departure_utc",
    "marketing_carrier", "operating_carrier", "registration",
    "observed_aircraft_variant", "modelled_aircraft", "modelled_engine",
]


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == ""


def deduplicate_codeshares(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One physical departure = same origin, destination, scheduled UTC and registration.

    The operating carrier's own flight number is kept when identifiable;
    otherwise the first row is kept and the others are listed as codeshares.
    """
    notes = []
    key_cols = ["origin_iata", "destination_iata", "scheduled_departure_utc"]
    df = df.copy()
    df["_reg"] = df.get("registration", pd.Series([""] * len(df))).fillna("").astype(str)
    keep_idx = []
    for _, grp in df.groupby(key_cols + ["_reg"], sort=False, dropna=False):
        if len(grp) == 1 or grp["_reg"].iloc[0] == "":
            keep_idx.extend(grp.index.tolist())
            continue
        op = grp[grp["marketing_carrier"].fillna("").str.upper()
                 == grp["operating_carrier"].fillna("").str.upper()]
        keep = op.index[0] if len(op) else grp.index[0]
        keep_idx.append(keep)
        dropped = [f for f in grp.loc[grp.index != keep, "flight_number"]]
        notes.append(f"{grp.loc[keep, 'flight_number']} kept; codeshare(s) {', '.join(map(str, dropped))} "
                     "removed (same aircraft and departure).")
    out = df.loc[sorted(keep_idx)].drop(columns="_reg")
    return out, notes


def evidence_status(row: pd.Series) -> dict:
    """Classify one manifest row. Returns status fields shown in charts and exports."""
    flown_text = str(row.get("observed_status", "")).lower()
    has_flight_src = not _blank(row.get("flight_source_url"))
    has_reported = not _blank(row.get("reported_departure_utc"))
    if "flown" in flown_text and has_flight_src and has_reported:
        flight = "flown (dated operating evidence)"
    elif not _blank(row.get("scheduled_departure_utc")):
        flight = "scheduled only (timetable)"
    else:
        flight = "unresolved"
    reg = row.get("registration")
    variant = str(row.get("observed_aircraft_variant", "") or "")
    modelled = str(row.get("modelled_aircraft", "") or "")
    if _blank(reg):
        aircraft = "unresolved"
    elif variant and modelled and variant.strip() == modelled.strip() and not _blank(row.get("aircraft_type_source_url")):
        aircraft = "airframe verified"
    elif variant and modelled and variant.strip() != modelled.strip():
        aircraft = "disputed (observed differs from modelled)"
    else:
        aircraft = "registration only"
    eng_text = str(row.get("engine_evidence", "") or "").lower()
    engine = "subvariant unverified" if ("not confirmed" in eng_text or "unverified" in eng_text) \
        else ("verified" if eng_text else "unresolved")
    op = str(row.get("operating_carrier", "") or "").strip()
    grp = str(row.get("group_attribution", "") or "")
    if op.upper() in SAS_GROUP_OPERATORS:
        operator = f"SAS-group operated ({op})"
    elif op:
        operator = f"operated by {op} (not SAS group)"
    elif "sas group" in grp.lower():
        operator = "SAS-group attribution from tracker; operating carrier blank"
    else:
        operator = "unresolved"
    note = str(row.get("audit_note", "") or "")
    conflict = ""
    low = note.lower()
    if any(w in low for w in ("discrepan", "conflict", "different", "differ")):
        conflict = note
    complete = flight.startswith("flown") and aircraft == "airframe verified"
    return {"flight_status": flight, "aircraft_status": aircraft, "engine_status": engine,
            "operator_status": operator, "conflict": conflict,
            "documented": complete}


def annotate_manifest(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dedup, notes = deduplicate_codeshares(df)
    st = dedup.apply(evidence_status, axis=1, result_type="expand")
    out = pd.concat([dedup.reset_index(drop=True), st.reset_index(drop=True)], axis=1)
    out["is_synthetic"] = False
    return out, notes


def roster_summary(annot: pd.DataFrame) -> dict:
    hist = annot[~annot.get("is_synthetic", False)]
    n = len(hist)
    return {
        "rows": n,
        "roster_slots": ROSTER_SIZE,
        "complete": n >= ROSTER_SIZE,
        "documented": int(hist["documented"].sum()) if n else 0,
        "flown": int(hist["flight_status"].str.startswith("flown").sum()) if n else 0,
        "airframe_verified": int((hist["aircraft_status"] == "airframe verified").sum()) if n else 0,
        "engine_verified": int((hist["engine_status"] == "verified").sum()) if n else 0,
        "sas_operated_explicit": int(hist["operator_status"].str.startswith("SAS-group operated").sum()) if n else 0,
        "conflicts": hist.loc[hist["conflict"] != "", ["study_id", "flight_number", "conflict"]].to_dict("records") if n else [],
        "destinations": sorted(hist["destination_iata"].unique().tolist()) if n else [],
        "headline": (f"{n} departures · manifest loaded" if n >= ROSTER_SIZE
                     else (f"{ROSTER_SIZE} proposed departures · manifest pending" if n == 0
                           else f"{n} of {ROSTER_SIZE} departures documented · manifest incomplete")),
    }


class SyntheticExportBlocked(ValueError):
    pass


def historical_export(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows suitable for a *historical* export; refuses synthetic rows."""
    if "is_synthetic" in df and df["is_synthetic"].any():
        raise SyntheticExportBlocked(
            "Synthetic (illustrative) rows cannot be exported as historical records.")
    return df
