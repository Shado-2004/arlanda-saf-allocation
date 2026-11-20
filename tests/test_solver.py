import numpy as np
import pytest

from saf_app.evaluation import policy_table, sample_heldout
from saf_app.examples import example_a, example_b
from saf_app.model import InputError, Problem, SlotWindow, info_groups
from saf_app.solver import SolveError, cvar, enumerate_best, resource_check, solve
from saf_app.strategies import STRATEGY_NAMES, run_all


def random_problem(seed, n=4, S=3, mode="partial_update"):
    rng = np.random.default_rng(seed)
    fuel = rng.uniform(5, 10, size=(n, 2, 3, S))
    fuel[:, 1] -= rng.uniform(0.5, 2, size=(n, 1, 1))
    con = rng.normal(0, 3, size=(n, 2, 3, S))
    feas = rng.random((n, 2, 3, S)) > 0.2
    feas[:, :, 0, :] = True
    q = np.zeros((n, 2))
    q[:, 1] = rng.uniform(1, 3, n)
    p = rng.dirichlet(np.ones(S))
    scen = [f"s{k}" for k in range(S)]
    return Problem([f"F{i}" for i in range(n)], ["ord", "saf"], ["o", "lo", "hi"], scen, p, fuel, con,
                   feasible=feas, q_stock=q, opening_stock_kg=float(q[:, 1].sum() * 0.5),
                   groups=np.tile(info_groups(mode, scen), (n, 1)))


@pytest.mark.parametrize("seed", range(8))
def test_milp_equals_enumeration(seed):
    P = random_problem(seed)
    assert solve(P).expected_score == pytest.approx(enumerate_best(P)[0], abs=1e-8)


def test_zero_stock_gives_no_saf():
    P = example_a().copy_with(opening_stock_kg=0.0)
    for r in run_all(P).values():
        assert r.x.sum() == 0


def test_zero_slots_gives_no_saf():
    P = example_a().copy_with(slot_windows=[SlotWindow(-1e9, 1e9, 0, "all day")])
    assert solve(P).x.sum() == 0


def test_binding_purchase_budget():
    P = example_a().copy_with(premium_eur=np.array([[0, 100.0], [0, 100.0]]), cash_cap_eur=50.0)
    sol = solve(P)
    assert sol.x.sum() == 0
    P2 = P.copy_with(cash_cap_eur=100.0)
    sol2 = solve(P2)
    assert sol2.x.tolist() == [0, 1]
    assert sol2.binding["Cash cap on SAF premium"]["binding"]


def test_ineligible_flight():
    P = example_a()
    allowed = np.ones((2, 2), bool)
    allowed[1, 1] = False
    P2 = P.copy_with(grade_allowed=allowed)
    assert solve(P2).x.tolist() == [1, 0]


def test_invalid_probabilities():
    P = example_b()
    with pytest.raises(InputError):
        P.copy_with(p=np.array([0.7, 0.4]))
    with pytest.raises(InputError):
        P.copy_with(p=np.array([1.2, -0.2]))


def test_infeasible_routes_and_no_recourse():
    P = example_a()
    feas = np.ones(P.feasible.shape, bool)
    feas[1, 1, 0, 0] = False          # B with SAF cannot fly original
    sol = solve(P.copy_with(feasible=feas))
    assert sol.expected_score == pytest.approx(12.0)   # B-SAF must take alt (8) -> SAF to A better
    feas[1, 1, :, :] = False          # B with SAF has no feasible route at all
    sol = solve(P.copy_with(feasible=feas))
    assert sol.x[1] == 0
    feas[:, 0, :, :] = False
    feas[:, 1, :, :] = False
    with pytest.raises(SolveError):
        solve(P.copy_with(feasible=feas))


def test_negative_ef_options_are_handled():
    P = example_a()
    con = P.contrail.copy() - 20       # all cooling
    sol = solve(P.copy_with(contrail=con))
    assert sol.expected_score == pytest.approx(10 - 40)


def test_equal_weather_states_match_merged():
    P = example_b()
    C = np.concatenate([P.contrail, P.contrail[:, :, :, :1]], axis=3)
    P2 = Problem(P.flights, P.grades, P.routes, ["dry_a", "humid", "dry_b"], np.array([0.3, 0.4, 0.3]),
                 np.zeros_like(C), C, q_stock=P.q_stock, opening_stock_kg=1.0,
                 groups=np.tile([0, 1, 0], (2, 1)))
    assert solve(P2).expected_score == pytest.approx(13.2)


def test_ties_are_reproducible():
    P = example_a()
    con = np.full(P.contrail.shape, 5.0)
    P2 = P.copy_with(contrail=con)
    a, b = solve(P2), solve(P2)
    assert a.x.tolist() == b.x.tolist() == [0, 0]       # tie rule prefers less SAF
    assert (a.route == 0).all()                          # and the planned profile


def test_pooled_outcomes_share_route():
    P = random_problem(3, mode="partial_update")
    for r in run_all(P).values():
        policy_table(P, r)  # raises if routes differ within a pooled signal
        for i in range(len(P.flights)):
            for g in np.unique(P.groups[i]):
                assert len(set(r.route[i, P.groups[i] == g])) == 1


def test_all_strategies_respect_same_limits():
    P = random_problem(5).copy_with(max_special=1)
    res = run_all(P)
    for name in STRATEGY_NAMES:
        assert not resource_check(P, res[name].x)
        assert (res[name].x > 0).sum() <= 1


def test_cvar_weight_zero_reproduces_and_fractional_tail():
    P = random_problem(7)
    assert solve(P, cvar_weight=0.0).expected_score == pytest.approx(solve(P).expected_score)
    v = np.array([10.0, 5.0, 1.0])
    p = np.array([0.03, 0.5, 0.47])
    assert cvar(v, p, 0.05) == pytest.approx((0.03 * 10 + 0.02 * 5) / 0.05)
    s = solve(P, cvar_weight=1.0)
    assert s.cvar is not None


def test_heldout_policy_sees_only_signal():
    P = random_problem(2, mode="partial_update")
    res = run_all(P)
    ho1 = sample_heldout(P, res, seed=11, draws=500, sigma=0.3)
    ho2 = sample_heldout(P, res, seed=11, draws=500, sigma=0.3)
    assert ho1.scores.equals(ho2.scores)
    # with sigma 0 the held-out mean converges to the exact expectation
    ho = sample_heldout(P, res, seed=3, draws=40000, sigma=0.0)
    for name, r in res.items():
        assert ho.scores[name].mean() == pytest.approx(r.expected, rel=0.02)


def test_contrail_weight_changes_optimiser_not_reporting():
    P = example_a()
    s1 = solve(P, contrail_weight=1.0)
    s0 = solve(P, contrail_weight=0.0)   # optimiser ignores contrails (all of score here)
    assert s1.expected_score == pytest.approx(10)
    assert s0.expected_score >= s1.expected_score
