"""Tests for the corrections of October 2026: statistics against deterministic
baselines, and the complexity measure. Run with:  python -m pytest -q tests"""
import numpy as np

from co2sr.gp.tree import const, op, var
from co2sr.metrics import simplify_gp
from co2sr.stats import compare

def test_sign_test_against_deterministic_baseline():
    x = np.linspace(29.0, 30.0, 30)
    p, name = compare(x, np.array([31.0]))
    assert name == "sign_test" and p < 1e-6
    p, name = compare(np.array([31.0]), x)
    assert name == "sign_test" and p < 1e-6
    p, name = compare(x, x + 0.01)
    assert name == "mann_whitney"


def test_simplified_size_never_exceeds_tree():
    names = ["engine_size", "cylinders"]
    t = op("add", op("aq", var(0), var(1)), op("aq", op("mul", const(2.0), var(0)), var(1)))
    s = simplify_gp(t.to_dict(), names)
    assert s["n_nodes_simplified"] <= t.size()
    assert "aq(" in s["equation_simplified"]
