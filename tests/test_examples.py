import numpy as np
import pytest

from saf_app.examples import example_a, example_b
from saf_app.solver import enumerate_best, solve
from saf_app.strategies import REFERENCE_NAME, run_all


def test_example_a_numbers():
    r = run_all(example_a())
    assert r["Original-route targeting"].x.tolist() == [1, 0]          # favours A (5 vs 4)
    assert r["Original-route targeting"].expected == pytest.approx(12)  # 4 + 8
    assert r["Joint stochastic"].x.tolist() == [0, 1]                   # SAF to B
    assert r["Joint stochastic"].expected == pytest.approx(10)          # 5 + 5
    assert (12 - 10) / 12 == pytest.approx(0.1667, abs=1e-4)
    assert r[REFERENCE_NAME].expected == pytest.approx(13)
    assert r["Joint deterministic"].expected == pytest.approx(r["Joint stochastic"].expected)


def test_example_b_numbers():
    r = run_all(example_b("revealed"))
    d, s = r["Joint deterministic"], r["Joint stochastic"]
    assert d.x.tolist() == [1, 0]
    assert d.scenario_scores.tolist() == pytest.approx([12, 20])
    assert d.expected == pytest.approx(15.2)
    assert s.x.tolist() == [0, 1]
    assert s.scenario_scores.tolist() == pytest.approx([14, 12])
    assert s.expected == pytest.approx(13.2)


def test_example_b_no_update():
    assert run_all(example_b("no_update"))["Joint stochastic"].expected == pytest.approx(14.2)


@pytest.mark.parametrize("P", [example_a(), example_b("revealed"), example_b("no_update")])
def test_solver_matches_enumeration(P):
    best = enumerate_best(P)
    sol = solve(P)
    assert sol.expected_score == pytest.approx(best[0])


def test_examples_are_score_points_not_physical():
    assert example_a().units == "score points"
    assert np.all(example_a().fuel == 0)
