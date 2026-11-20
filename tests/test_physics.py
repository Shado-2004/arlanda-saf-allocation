import pytest

from saf_app import physics as ph

FP = ph.FuelParams()


def test_35_percent_mass_is_350_kg_per_tonne():
    assert ph.burn(1.0, 1000, 0.35, FP).q_SAF_kg == pytest.approx(350.0)


def test_volume_fraction_and_round_trip():
    b = ph.mass_to_volume_fraction(0.35, 760, 800)
    assert b == pytest.approx(0.3618, abs=1e-4)
    assert ph.volume_to_mass_fraction(b, 760, 800) == pytest.approx(0.35, abs=1e-12)
    g = ph.Grade("v", b, "volume", True)
    assert g.mass_fraction(FP) == pytest.approx(0.35)


def test_onboard_dilution_with_conventional_carry_in():
    br = ph.burn(43_000, 1000, 0.35, FP, M0_kg=1000, w0=0.0)
    assert br.w_onboard == pytest.approx(0.175)
    assert br.q_stock_kg == pytest.approx(350)          # stock charge unchanged
    assert br.m_S_kg < br.q_SAF_kg


def test_zero_saf_grade():
    br = ph.burn(43_000, 2000, 0.0, FP)
    assert br.q_SAF_kg == 0 and br.w_onboard == 0 and br.m_S_kg == 0
    assert br.L_fuel_t == pytest.approx(1000 * 43 * 89 / 1e6)


def test_k100_and_breakeven():
    assert ph.K100 * 1e12 == pytest.approx(0.297, abs=5e-4)
    assert ph.breakeven_EF_J(100, FP, H=43.2, CI=89) / 1e12 == pytest.approx(1.30, abs=0.005)


def test_negative_ef_cools_and_saf_can_worsen_cooling():
    assert ph.contrail_score_t(-1e12, FP) < 0
    ef_cool = -1e13
    ef_blend = ph.blended_EF(ef_cool, 0.35, 1.0, -0.15)
    assert ef_blend > ef_cool       # cooling weakened -> worse score


def test_zero_contrail_response_separate_from_zero_total_benefit():
    # Zero contrail response: EF unchanged, but fuel life-cycle still benefits.
    assert ph.blended_EF(5e12, 0.35, 0.0) == 5e12
    a = ph.burn(43_000, 1000, 0.0, FP).L_fuel_t
    b = ph.burn(43_000, 1000, 0.35, FP).L_fuel_t
    assert b < a
    # Zero total benefit: equal intensities and zero response.
    fp = ph.FuelParams(CI_S=89.0, H_S=43.0)
    assert ph.burn(43_000, 1000, 0.35, fp).L_fuel_t == pytest.approx(ph.burn(43_000, 1000, 0.0, fp).L_fuel_t)


def test_annex_a2_volume_limit():
    assert ph.HEFA_35.spec_compliant(FP)
    assert not ph.Grade("x", 0.55, "volume", True).spec_compliant(FP)
    assert not ph.Grade("y", 0.49, "mass", True).spec_compliant(FP)   # 50.3% by volume


def test_leftover_non_negative_and_costs():
    br = ph.burn(4000 * 43, 6000, 0.35, FP)
    assert br.remaining_kg >= 0
    c = ph.costs(br, 0.35, FP, fee_eur=250)
    assert c.cash_eur == pytest.approx((6000 * 0.65 * 640 + 6000 * 0.35 * 1925) / 1000 + 250)
    assert c.premium_eur == pytest.approx(6000 * 0.35 * (1925 - 640) / 1000 + 250)
    assert c.comparison_eur < c.cash_eur
    assert ph.eur_per_t_saved(100, 0) is None
    assert ph.eur_per_t_saved(100, -1) is None


def test_invalid_fractions_rejected():
    with pytest.raises(ValueError):
        ph.mass_to_volume_fraction(1.2, 760, 800)
    with pytest.raises(ValueError):
        ph.burn(1, -5, 0.35, FP)
