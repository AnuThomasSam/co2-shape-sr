"""Tests for the branch-and-bound verifier (gp/verify.py) and the DCSR-V engine mode."""
import numpy as np

from co2sr.gp.engine import GPRegressor
from co2sr.gp.scaling import Scaling
from co2sr.gp.tree import const, op, var
from co2sr.gp.verify import signed_slopes, verify

from test_gp_engine import SPEC, random_trees

ONE = Scaling(np.array([0.0]), np.array([1.0]), 0.0, 1.0)


def _roots(n_ctx=3, rng=None):
    rng = rng or np.random.default_rng(0)
    LO = np.column_stack([rng.uniform(0.6, 2.0, n_ctx), rng.uniform(1, 3, n_ctx), rng.uniform(0, 1, n_ctx)])
    HI = LO.copy()
    HI[:, 0] = LO[:, 0] + rng.uniform(1.0, 4.0, n_ctx)       # only engine size varies inside a context
    return LO, HI


def test_dependency_problem_resolved_by_bisection():
    # g = x*x - 3x on [2, 5]: g' = 2x - 3 >= 1. Interval form of the derivative is
    # [2,5]+[2,5]-3 = [1,7]: fine. Use g = x*(6 - x) + x*x on [1, 4]: g' = 6 exactly, but the
    # interval derivative of x*(6-x) is (6-x) + x*(-1) = [2,5] + [-4,-1] = [-2, 4]; adding (x*x)' = [2,8]
    # gives [0, 12] ... still certified. A clear case: g = x*(4 - x) + 2*x*x on [0.5, 3.5], g' = 4 + 2x > 0;
    # interval: (4 - x) - x + 4x = [0.5,3.5]->[0.5,3.5] + [-3.5,-0.5] + [2,14] = [-1, 17] -> not proved.
    t = op("add", op("mul", var(0), op("sub", const(4.0), var(0))), op("mul", const(2.0), op("mul", var(0), var(0))))
    LO, HI = np.array([[0.5]]), np.array([[3.5]])
    g = np.zeros(1, int)
    plain = verify(t, ONE, LO, HI, g, max_depth=0, probe=False)
    bb = verify(t, ONE, LO, HI, g, max_depth=12, max_boxes=5000)
    assert plain.status == "undecided"
    assert bb.status == "certified" and bb.n_bisections > 0


def test_counterexample_is_a_true_violation():
    t = op("mul", op("sub", var(0), const(2.0)), op("sub", var(0), const(2.0)))   # (x-2)^2, decreasing on x<2
    LO, HI = np.array([[0.6, 0, 0]]), np.array([[5.0, 0, 0]])
    v = verify(t, ONE, LO, HI, np.zeros(1, int))
    assert v.status == "refuted"
    s = signed_slopes(t, ONE, v.cx_X, v.cx_g)
    assert np.all(s < 0) and np.all(v.cx_X[:, 0] < 2.0)


def test_soundness_on_random_trees():
    """A certified verdict must never be contradicted by dense sampling; B&B never loses an IA proof."""
    rng = np.random.default_rng(3)
    n_cert = n_gain = 0
    for t in random_trees(600, function_set="extended", seed=5):
        LO, HI = _roots(3, rng)
        g = np.zeros(len(LO), int)
        plain = verify(t, ONE, LO, HI, g, max_depth=0, probe=False)
        bb = verify(t, ONE, LO, HI, g, max_depth=12, max_boxes=20000)
        if plain.status == "certified":
            assert bb.status == "certified"
        if bb.status == "certified":
            n_cert += 1
            n_gain += plain.status != "certified"
            u = rng.random((400, 1))
            X = np.repeat(LO, 400, axis=0)
            X[:, 0] = (LO[:, 0][:, None] + u.T * (HI[:, 0] - LO[:, 0])[:, None]).ravel()
            s = signed_slopes(t, ONE, X, np.zeros(len(X), int))
            assert np.all(s >= -1e-7), t.to_string()
        if bb.status == "refuted":
            assert np.all(signed_slopes(t, ONE, bb.cx_X, bb.cx_g) < 0)
    assert n_cert > 20 and n_gain >= 1


def test_negative_scaling_slope_flips_the_constraint():
    t = var(0)
    neg = Scaling(np.array([0.0]), np.array([-1.0]), 0.0, -1.0)
    assert verify(t, ONE, np.array([[1.0]]), np.array([[2.0]]), np.zeros(1, int)).status == "certified"
    assert verify(t, neg, np.array([[1.0]]), np.array([[2.0]]), np.zeros(1, int)).status == "refuted"


def test_dcsr_v_engine_returns_certified_model():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(1, 6, 300), rng.integers(3, 9, 300).astype(float), rng.uniform(0, 1, 300)])
    y = 100 + 20 * X[:, 0] + 5 * X[:, 1] + rng.normal(0, 5, 300)
    LO, HI = _roots(6, rng)
    spec = {**SPEC, "function_set": "basic", "constraint": "verified", "require_engine_size": True,
            "verify_max_boxes": 2000, "verify_max_depth": 8}
    gp = GPRegressor(spec, 1).fit(X, y, np.zeros(300, int), 1, LO=LO, HI=HI, g_box=np.zeros(len(LO), int))
    assert gp.info["final_verdict"] == "certified"
    assert gp.verify_final(LO, HI, np.zeros(len(LO), int))["bb_verdict"] == "certified"
    assert gp.info["verify_bb_calls"] > 0
