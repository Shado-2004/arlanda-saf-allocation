"""Physical fuel, climate-score and cost calculations.

Pure functions with explicit units. Nothing here knows about Streamlit or the
solver. Every formula matches the brief's "Show the calculation" text.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants (course convention / illustrative assumptions, all editable via
# FuelParams so nothing is hard-wired into the model).
# ---------------------------------------------------------------------------
SECONDS_PER_YEAR = 31_536_000
EARTH_AREA_M2 = 5.1e14
QAEI_EFFICACY = 0.42
QAEI_AGWP_FACTOR = 8.8e-14  # W m^-2 yr per kg CO2 basis used by QAEI Tutorial 3


def k100_t_per_joule() -> float:
    """QAEI 100-year conversion, t CO2e per joule of contrail energy forcing.

    k100 = 0.42 / (5.1e14 * 31_536_000 * 8.8e-14 * 1000)
    """
    return QAEI_EFFICACY / (EARTH_AREA_M2 * SECONDS_PER_YEAR * QAEI_AGWP_FACTOR * 1000)


K100 = k100_t_per_joule()  # ~2.967e-13 t CO2e / J  (~0.297 t per TJ)


@dataclass
class FuelParams:
    rho_J: float = 800.0        # kg/m3 Jet A-1 (illustrative assumption)
    rho_S: float = 760.0        # kg/m3 HEFA-SPK (illustrative assumption)
    H_J: float = 43.0           # MJ/kg (illustrative assumption)
    H_S: float = 44.0           # MJ/kg (illustrative assumption)
    CI_J: float = 89.0          # g CO2e/MJ (Boeing Cascade conventional default)
    CI_S: float = 13.9          # g CO2e/MJ (ICAO CORSIA UCO-HEFA default, sensitivity)
    price_J_eur_t: float = 640.0    # EASA 2025 EU reference
    price_S_eur_t: float = 1925.0   # EASA 2025 EU aviation biofuel reference
    hefa_a2_max_vol: float = 0.50   # ICAO HEFA-SPK Annex A2 limit, volume basis
    k_contrail: float = field(default_factory=k100_t_per_joule)


# ---------------------------------------------------------------------------
# Blend basis conversions
# ---------------------------------------------------------------------------
def mass_to_volume_fraction(w_u: float, rho_S: float, rho_J: float) -> float:
    """b = (w/rho_S) / (w/rho_S + (1-w)/rho_J)."""
    if not 0.0 <= w_u <= 1.0:
        raise ValueError("mass fraction must lie in [0, 1]")
    if w_u == 0.0:
        return 0.0
    return (w_u / rho_S) / (w_u / rho_S + (1.0 - w_u) / rho_J)


def volume_to_mass_fraction(b: float, rho_S: float, rho_J: float) -> float:
    """w = b*rho_S / (b*rho_S + (1-b)*rho_J)."""
    if not 0.0 <= b <= 1.0:
        raise ValueError("volume fraction must lie in [0, 1]")
    if b == 0.0:
        return 0.0
    return b * rho_S / (b * rho_S + (1.0 - b) * rho_J)


def blend_density(w_u: float, rho_S: float, rho_J: float) -> float:
    """Density of the finished blend, kg/m3 (ideal mixing by volume)."""
    return 1.0 / (w_u / rho_S + (1.0 - w_u) / rho_J)


@dataclass(frozen=True)
class Grade:
    """A named fuel grade. `fraction` is interpreted on `basis`."""
    name: str
    fraction: float
    basis: str = "mass"          # "mass" or "volume"
    dedicated_delivery: bool = False
    label: str = ""

    def mass_fraction(self, fp: FuelParams) -> float:
        if self.basis == "mass":
            return float(self.fraction)
        if self.basis == "volume":
            return volume_to_mass_fraction(self.fraction, fp.rho_S, fp.rho_J)
        raise ValueError(f"unknown basis {self.basis!r}")

    def volume_fraction(self, fp: FuelParams) -> float:
        if self.basis == "volume":
            return float(self.fraction)
        return mass_to_volume_fraction(self.fraction, fp.rho_S, fp.rho_J)

    def spec_compliant(self, fp: FuelParams) -> bool:
        """Annex A2 HEFA-SPK limit applied on a volume basis.

        This only screens the nominal blend ratio; it never certifies a batch.
        """
        return self.volume_fraction(fp) <= fp.hefa_a2_max_vol + 1e-12


REFERENCE = Grade("reference_Jet_A1", 0.0, "mass", False, "Declared reference Jet A-1 (0% SAF assumed)")
HEFA_35 = Grade("HEFA_35wt", 0.35, "mass", True, "HEFA-SPK/Jet A-1, 35% SAF by mass")


# ---------------------------------------------------------------------------
# Tank mixing and burn
# ---------------------------------------------------------------------------
@dataclass
class BurnResult:
    U_kg: float
    q_SAF_kg: float          # pure SAF in the new delivery
    q_stock_kg: float        # pure SAF charged to the segregated stock
    w_onboard: float         # mixed onboard SAF mass fraction
    H_mix: float             # MJ/kg
    m_burn_kg: float
    m_S_kg: float
    m_J_kg: float
    remaining_kg: float
    L_fuel_t: float          # life-cycle t CO2e


def burn(route_energy_MJ: float, U_kg: float, w_u: float, fp: FuelParams,
         M0_kg: float = 0.0, w0: float = 0.0, dedicated: bool = True) -> BurnResult:
    """Well-mixed tank: deliver U kg at w_u onto M0 kg at w0 and burn route energy."""
    if U_kg < 0 or M0_kg < 0:
        raise ValueError("fuel masses must be non-negative")
    q_SAF = U_kg * w_u
    q_stock = q_SAF if dedicated else 0.0
    total = M0_kg + U_kg
    w_on = (M0_kg * w0 + U_kg * w_u) / total if total > 0 else 0.0
    H_mix = (1.0 - w_on) * fp.H_J + w_on * fp.H_S
    m_burn = route_energy_MJ / H_mix
    m_S = w_on * m_burn
    m_J = (1.0 - w_on) * m_burn
    L_fuel = (m_J * fp.H_J * fp.CI_J + m_S * fp.H_S * fp.CI_S) / 1_000_000
    return BurnResult(U_kg, q_SAF, q_stock, w_on, H_mix, m_burn, m_S, m_J,
                      total - m_burn, L_fuel)


def contrail_score_t(EF_J: float, fp: FuelParams) -> float:
    """L_contrail = k100 * EF (signed). Positive EF warms, negative cools."""
    return fp.k_contrail * EF_J


def blended_EF(EF_ordinary: float, w_onboard: float, response_scale: float,
               rel_change_at_ref: float = -0.15, w_ref: float = 0.35) -> float:
    """Teaching surrogate: EF = EF_ordinary + response_scale * delta_EF_for_blend.

    delta_EF_for_blend = rel_change_at_ref * (w_onboard / w_ref) * EF_ordinary.
    Zero for ordinary fuel; at response_scale = 0 SAF has no contrail effect.
    Because it scales EF_ordinary, a *cooling* contrail is weakened as well,
    which can worsen the score. This coefficient is NOT derived from the
    CFM56 11% simulation or the 42% taxi nvPM statistic.
    """
    if w_onboard <= 0:
        return EF_ordinary
    delta = rel_change_at_ref * (w_onboard / w_ref) * EF_ordinary
    return EF_ordinary + response_scale * delta


def breakeven_EF_J(extra_fuel_kg: float, fp: FuelParams, H: float | None = None,
                   CI: float | None = None) -> float:
    """EF (J) a detour must avoid to pay back its extra fossil fuel."""
    H = fp.H_J if H is None else H
    CI = fp.CI_J if CI is None else CI
    extra_t = extra_fuel_kg * H * CI / 1_000_000
    return extra_t / fp.k_contrail


# ---------------------------------------------------------------------------
# Economics
# ---------------------------------------------------------------------------
@dataclass
class CostResult:
    cash_eur: float          # purchase cash for the uplift + fee
    comparison_eur: float    # burned fuel valued by component + fee (inventory-neutral)
    premium_eur: float       # SAF price premium on the pure SAF uplift + fee


def costs(br: BurnResult, w_u: float, fp: FuelParams, fee_eur: float = 0.0) -> CostResult:
    U = br.U_kg
    cash = (U * (1 - w_u) * fp.price_J_eur_t + U * w_u * fp.price_S_eur_t) / 1000 + fee_eur
    comparison = (br.m_J_kg * fp.price_J_eur_t + br.m_S_kg * fp.price_S_eur_t) / 1000 + fee_eur
    premium = U * w_u * (fp.price_S_eur_t - fp.price_J_eur_t) / 1000 + fee_eur
    return CostResult(cash, comparison, premium)


def eur_per_t_saved(extra_eur: float, saving_t: float) -> float | None:
    """Abatement cost, reported only for strictly positive climate savings."""
    if saving_t is None or saving_t <= 1e-12:
        return None
    return extra_eur / saving_t


# ---------------------------------------------------------------------------
# Contrail formation screen (Schmidt–Appleman, Schumann 1996 approximation)
# ---------------------------------------------------------------------------
def sac_threshold_K(p_hPa: float, EI_H2O: float = 1.23, Q_J_kg: float = 43.2e6,
                    eta: float = 0.38) -> float:
    """Approximate threshold temperature (K) for contrail formation at liquid saturation."""
    cp, eps = 1004.0, 0.622
    G = EI_H2O * cp * (p_hPa * 100.0) / (eps * Q_J_kg * (1.0 - eta))  # Pa/K
    x = math.log(G - 0.053)
    return 273.15 - 46.46 + 9.43 * x + 0.72 * x * x


def solar_elevation_deg(lat: float, lon: float, when_utc) -> float:
    """Approximate solar elevation (NOAA low-precision formula)."""
    doy = when_utc.timetuple().tm_yday
    hour = when_utc.hour + when_utc.minute / 60 + when_utc.second / 3600
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                    - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    tst = hour * 60 + eqt + 4 * lon
    ha = math.radians(tst / 4 - 180)
    la = math.radians(lat)
    cz = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    return math.degrees(math.asin(max(-1.0, min(1.0, cz))))


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))
