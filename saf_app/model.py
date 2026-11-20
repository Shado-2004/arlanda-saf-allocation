"""Problem definition shared by the solver, the strategies and the UI.

A `Problem` holds precomputed option scores C[i,b,r,s] split into a fuel
life-cycle component, a signed contrail component and an `extra` component,
plus every resource limit. Both the unitless teaching examples (score points)
and the physical Stockholm case are expressed in this one structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class InputError(ValueError):
    """Raised for invalid user inputs, with a message suitable for the UI."""


@dataclass
class SlotWindow:
    start_min: float
    end_min: float
    capacity: int
    label: str = ""


@dataclass
class Problem:
    flights: list[str]
    grades: list[str]
    routes: list[str]
    scenarios: list[str]
    p: np.ndarray                      # (S,)
    fuel: np.ndarray                   # (n,B,R,S) score units (t CO2e or points)
    contrail: np.ndarray               # (n,B,R,S) signed
    extra: np.ndarray | None = None    # (n,B,R,S) L_extra, defaults to 0
    feasible: np.ndarray | None = None  # (n,B,R,S) bool  a[i,b,r,s]
    infeasible_reason: dict = field(default_factory=dict)   # (i,b,r,s) -> text
    grade_allowed: np.ndarray | None = None   # (n,B) bool
    grade_reason: dict = field(default_factory=dict)         # (i,b) -> text
    groups: np.ndarray | None = None   # (n,S) int signal group visible to flight i
    q_stock: np.ndarray | None = None  # (n,B) kg pure SAF drawn from dedicated stock
    dedicated: np.ndarray | None = None  # (B,) bool
    premium_eur: np.ndarray | None = None  # (n,B)
    service_start_min: np.ndarray | None = None  # (n,)
    service_end_min: np.ndarray | None = None    # (n,)
    truck_loads: np.ndarray | None = None        # (n,B) slots used
    opening_stock_kg: float = float("inf")
    receipts: list = field(default_factory=list)  # [(time_min, kg)]
    slot_windows: list = field(default_factory=list)
    max_special: int | None = None
    cash_cap_eur: float | None = None
    representative: int | None = None   # scenario index for joint deterministic
    priority_order: list | None = None  # flight indices for fixed priority
    preferred_grade: int | None = None  # grade index preferred by fixed priority
    units: str = "t CO2e"
    # reporting-only arrays (optional)
    cash_eur: np.ndarray | None = None        # (n,B)
    comparison_eur: np.ndarray | None = None  # (n,B,R,S)
    saf_burn_kg: np.ndarray | None = None     # (n,B,R,S)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def __post_init__(self):
        n, B, R, S = self.shape
        self.p = np.asarray(self.p, dtype=float)
        self.fuel = np.asarray(self.fuel, dtype=float)
        self.contrail = np.asarray(self.contrail, dtype=float)
        if self.extra is None:
            self.extra = np.zeros((n, B, R, S))
        if self.feasible is None:
            self.feasible = np.ones((n, B, R, S), dtype=bool)
        if self.grade_allowed is None:
            self.grade_allowed = np.ones((n, B), dtype=bool)
        if self.groups is None:  # revealed by default
            self.groups = np.tile(np.arange(S), (n, 1))
        if self.q_stock is None:
            self.q_stock = np.zeros((n, B))
        if self.dedicated is None:
            self.dedicated = np.array([b > 0 for b in range(B)])
        if self.premium_eur is None:
            self.premium_eur = np.zeros((n, B))
        if self.service_start_min is None:
            self.service_start_min = np.arange(n, dtype=float)
        if self.service_end_min is None:
            self.service_end_min = self.service_start_min + 1
        if self.truck_loads is None:
            self.truck_loads = np.where(self.dedicated[None, :], 1, 0) * np.ones((n, B))
        if self.representative is None:
            self.representative = int(np.argmax(self.p))  # first most probable
        if self.priority_order is None:
            self.priority_order = list(range(n))
        if self.preferred_grade is None:
            self.preferred_grade = B - 1
        self.validate()

    @property
    def shape(self):
        return (len(self.flights), len(self.grades), len(self.routes), len(self.scenarios))

    def score(self, contrail_weight: float = 1.0) -> np.ndarray:
        """C[i,b,r,s]; physically reported score uses weight 1."""
        return self.fuel + contrail_weight * self.contrail + self.extra

    def validate(self):
        n, B, R, S = self.shape
        if S == 0 or n == 0:
            raise InputError("At least one flight and one weather scenario are required.")
        if np.any(~np.isfinite(self.p)) or np.any(self.p < 0):
            raise InputError("Scenario probabilities must be non-negative numbers.")
        if abs(self.p.sum() - 1.0) > 1e-6:
            raise InputError(f"Scenario probabilities must sum to 1 (they sum to {self.p.sum():.4f}).")
        for name, arr in (("fuel", self.fuel), ("contrail", self.contrail), ("extra", self.extra)):
            if arr.shape != (n, B, R, S):
                raise InputError(f"{name} table has shape {arr.shape}, expected {(n, B, R, S)}")
        if self.groups.shape != (n, S):
            raise InputError("Signal groups must give one group per flight and scenario.")

    def copy_with(self, **kw) -> "Problem":
        import copy
        new = copy.copy(self)
        for k, v in kw.items():
            setattr(new, k, v)
        new.validate()
        return new


# ---------------------------------------------------------------------------
# Information structures
# ---------------------------------------------------------------------------
def info_groups(mode: str, scenarios: list[str], partial_pool: list[str] | None = None) -> np.ndarray:
    """Return a (S,) group id per scenario for a forecast-information mode.

    no_update        -> one group
    partial_update   -> scenarios in `partial_pool` look alike; others distinct
    revealed         -> each scenario is its own group
    """
    S = len(scenarios)
    if mode == "no_update":
        return np.zeros(S, dtype=int)
    if mode == "revealed":
        return np.arange(S, dtype=int)
    if mode == "partial_update":
        pool = set(partial_pool or scenarios[1:])
        ids, out, next_id = {}, np.zeros(S, dtype=int), 0
        for s, name in enumerate(scenarios):
            key = "__pool__" if name in pool else name
            if key not in ids:
                ids[key] = next_id
                next_id += 1
            out[s] = ids[key]
        return out
    raise InputError(f"unknown information mode {mode!r}")
