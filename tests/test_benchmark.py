"""Benchmark problems: constraints derived from the true expressions must match hand derivations."""
import numpy as np

from co2sr.benchmark.problems import PROBLEMS, by_name, derive_constraints, function, make_data, uniform_boxes


def test_constraints_match_hand_derivation():
    assert derive_constraints(by_name("I.39.22")) == [[0, 1.0], [1, 1.0], [2, -1.0]]      # n*T/V
    assert derive_constraints(by_name("I.6.20a")) == [[0, -1.0]]                         # Gaussian tail, theta >= 1
    assert derive_constraints(by_name("I.18.4")) == [[2, 1.0], [3, 1.0]]                 # masses: sign changes


def test_every_problem_is_finite_and_constrained():
    for p in PROBLEMS:
        Xtr, ytr, Xte, yte, ex = make_data(p, 50, 200, 0.05, 0.7, 0)
        assert np.all(np.isfinite(ytr)) and np.all(np.isfinite(yte)) and ex.any() and (~ex).any()
        assert len(derive_constraints(p)) >= 1
        assert np.all(np.isfinite(function(p)(np.asarray([p.hi], float))))


def test_uniform_boxes_cover_the_domain():
    LO, HI = uniform_boxes([0, 0, 0], [1, 2, 3], 64)
    assert len(LO) == 64 and np.isclose(np.prod(HI - LO, axis=1).sum(), 6.0)
