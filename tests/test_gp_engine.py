"""Correctness tests for the GP engine. Run with:  python -m pytest -q tests"""
import numpy as np
import pytest
import sympy as sp

from co2sr.gp import primitives as P
from co2sr.gp.engine import GPRegressor
from co2sr.gp.scaling import fit_scaling
from co2sr.gp.tree import Node, const, op, var

SPEC = dict(population_size=60, generations=5, tournament_size=5, init_depth=[2, 5], max_depth=8,
            max_size=40, const_range=[-3.0, 3.0], p_var_terminal=0.6, p_crossover=0.7,
            p_subtree_mutation=0.15, p_point_mutation=0.1, p_constant_mutation=0.05, hall_of_fame=20,
            function_set="extended", linear_scaling=True, fuel_mode="scaling", min_group_rows=5,
            constraint="none", lam=0.0, feasibility_tol=1e-9)


def random_trees(n, function_set="extended", seed=0):
    gp = GPRegressor({**SPEC, "function_set": function_set}, seed)
    return [gp._random_tree(3, int(gp.rng.integers(2, 6)), full=bool(i % 2)) for i in range(n)]


def test_derivative_matches_sympy_and_finite_differences():
    rng = np.random.default_rng(1)
    names = ["x0", "x1", "x2"]
    syms = sp.symbols(names)
    checked = 0
    for t in random_trees(1000):
        X = rng.uniform(0.5, 3.0, size=(5, 3))
        v, d = t.evaluate_dual(X, 0)
        if not (np.all(np.isfinite(v)) and np.all(np.isfinite(d))):
            continue
        expr = t.to_sympy(names)
        f = sp.lambdify(syms, expr, "numpy")
        df = sp.lambdify(syms, sp.diff(expr, syms[0]), "numpy")
        v_ref = np.broadcast_to(np.asarray(f(*X.T), float), (5,))
        d_ref = np.broadcast_to(np.asarray(df(*X.T), float), (5,))
        if not (np.all(np.isfinite(v_ref)) and np.all(np.isfinite(d_ref))):
            continue
        np.testing.assert_allclose(v, v_ref, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(d, d_ref, rtol=1e-6, atol=1e-6)
        h = 1e-6
        Xp, Xm = X.copy(), X.copy()
        Xp[:, 0] += h
        Xm[:, 0] -= h
        fd = (t.evaluate(Xp) - t.evaluate(Xm)) / (2 * h)
        scale = np.maximum(1.0, np.abs(d))
        if np.all(np.isfinite(fd)) and np.max(np.abs(d)) < 1e6:
            assert np.all(np.abs(fd - d) / scale < 1e-3)
        checked += 1
    assert checked > 300


def test_interval_bounds_contain_sampled_values():
    rng = np.random.default_rng(2)
    for t in random_trees(500, seed=3):
        lo = rng.uniform(0.5, 3.0, size=(4, 3))
        hi = lo + rng.uniform(0.0, 0.5, size=(4, 3))
        vlo, vhi, dlo, dhi = t.evaluate_interval(lo, hi, 0)
        for b in range(4):
            pts = lo[b] + rng.random((200, 3)) * (hi[b] - lo[b])
            v, d = t.evaluate_dual(pts, 0)
            ok = np.isfinite(v) & np.isfinite(d)
            if not ok.any():
                continue
            tol = 1e-7 * (1 + np.abs(v[ok]))
            assert np.all(v[ok] >= vlo[b] - tol) and np.all(v[ok] <= vhi[b] + tol)
            tol_d = 1e-7 * (1 + np.abs(d[ok]))
            assert np.all(d[ok] >= dlo[b] - tol_d) and np.all(d[ok] <= dhi[b] + tol_d)


def test_group_scaling_matches_lstsq():
    rng = np.random.default_rng(3)
    g = rng.normal(size=300)
    groups = rng.integers(0, 3, 300)
    y = np.array([1.0, -2.0, 0.5])[groups] + np.array([3.0, 1.0, -4.0])[groups] * g + rng.normal(0, 0.1, 300)
    sc = fit_scaling(g, y, groups, 3, min_rows=5)
    for k in range(3):
        m = groups == k
        A = np.column_stack([np.ones(m.sum()), g[m]])
        coef = np.linalg.lstsq(A, y[m], rcond=None)[0]
        np.testing.assert_allclose([sc.a[k], sc.b[k]], coef, rtol=1e-8, atol=1e-8)


def test_constraint_uses_sign_of_scaling_slope():
    # g = -x0 is decreasing, but with b < 0 the scaled prediction is increasing.
    t = op("sub", const(0.0), var(0))
    X = np.column_stack([np.linspace(1, 5, 50)])
    y = 10 + 3 * X[:, 0]
    sc = fit_scaling(t.evaluate(X), y, np.zeros(50, int), 1, 5)
    from co2sr.gp.constraints import check_points
    c = check_points(t, sc, X, np.zeros(50, int), 1.0, 1e-9)
    assert sc.b[0] < 0 and c.feasible


def test_original_dcsr_equation_is_flagged():
    """Regression test for the error found in the code review: the reported DCSR
    equation, written in primitive variables, decreases in engine size in places."""
    from co2sr.gp.constraints import check_points
    d, c = var(0), var(1)
    # (((log(exp(exp(c))) * c) - c^2) - (c^2 - log(((d^2 - 2.5189) - (2.5189 + c)) * (exp(d) * exp(c^2)))))
    # The clipped exp of the original code is not reproducible with strict operators, so the
    # engine-size part, which carries the violation, is tested: d + log(|d^2 - 5.0378 - c|).
    inner = op("sub", op("mul", d, d), op("add", const(5.0378), c))
    t = op("add", d, op("log", op("mul", inner, inner)))   # log(u^2) = 2 log|u|
    X = np.array([[2.5, 4.0], [3.0, 6.0], [3.3, 6.0], [2.0, 4.0]])
    sc = fit_scaling(t.evaluate(X), np.array([200., 260., 265., 190.]), np.zeros(4, int), 1, 1, enabled=False)
    res = check_points(t, sc, X, np.zeros(4, int), 1.0, 1e-9)
    assert not res.feasible


@pytest.mark.parametrize("mode", ["none", "train", "sample", "feasibility", "interval"])
def test_engine_runs_and_constrained_modes_return_feasible_models(mode):
    rng = np.random.default_rng(4)
    n = 400
    X = np.column_stack([rng.uniform(1, 6, n), rng.choice([4, 6, 8], n).astype(float)])
    groups = rng.integers(0, 2, n)
    y = 80 + 30 * X[:, 0] ** 0.8 + 4 * X[:, 1] + 20 * groups + rng.normal(0, 5, n)
    Xq = np.column_stack([rng.uniform(0.6, 8.5, 300), rng.choice([3, 4, 6, 8, 12], 300).astype(float)])
    gq = rng.integers(0, 2, 300)
    cyl = np.array([3, 4, 6, 8, 12], float)
    LO = np.column_stack([np.tile(np.linspace(0.6, 8.0, 8), 5), np.repeat(cyl, 8)])
    HI = LO.copy()
    HI[:, 0] += (8.5 - 0.6) / 8
    LO = np.vstack([LO, LO]); HI = np.vstack([HI, HI]); gb = np.repeat([0, 1], len(LO) // 2)
    spec = {**SPEC, "function_set": "basic", "constraint": mode, "lam": 100.0,
            "population_size": 80, "generations": 8}
    m = GPRegressor(spec, seed=1).fit(X, y, groups, 2, X_val=X[:50], y_val=y[:50], g_val=groups[:50],
                                      X_q=Xq, g_q=gq, LO=LO, HI=HI, g_box=gb,
                                      X_full=X, y_full=y, g_full=groups)
    pred = m.predict(X, groups)
    assert np.all(np.isfinite(pred))
    assert np.corrcoef(pred, y)[0, 1] > 0.5
    if mode in ("feasibility", "interval") and m.info["feasible_on_constraint_set"]:
        assert np.all(m.slope(Xq, gq) >= -1e-9)
    if mode == "interval" and m.info["feasible_on_constraint_set"]:
        assert m.certify(LO, HI, gb)["certified"]
    d = m.to_dict()
    m2 = GPRegressor.from_dict(d)
    np.testing.assert_allclose(m2.predict(X, groups), pred)


def test_sign_consistent_scaling_never_reverses_a_group():
    rng = np.random.default_rng(5)
    g = rng.normal(size=400)
    groups = rng.integers(0, 3, 400)
    slopes = np.array([3.0, 2.0, -1.5])          # group 2 would reverse the shape
    y = 10 + slopes[groups] * g + rng.normal(0, 0.1, 400)
    free = fit_scaling(g, y, groups, 3, 5, sign_consistent=False)
    cons = fit_scaling(g, y, groups, 3, 5, sign_consistent=True)
    assert free.b[2] < 0
    assert np.all(np.sign(cons.b) == np.sign(cons.b0))
    m = groups == 2                                # intercept refitted for the fixed slope
    np.testing.assert_allclose(cons.a[2], y[m].mean() - cons.b0 * g[m].mean(), rtol=1e-10)


def test_require_engine_size_ranks_equations_without_it_last():
    from co2sr.gp.engine import Record
    gp = GPRegressor({**SPEC, "require_engine_size": True}, 0)
    with_e = Record(tree=var(0), key="a", nmse=0.9, valid=True, extra={"has_engine": True, "size": 1})
    without = Record(tree=var(1), key="b", nmse=0.1, valid=True, extra={"has_engine": False, "size": 1})
    assert gp._sel_key(with_e) < gp._sel_key(without)


def test_final_model_uses_engine_size_when_required():
    rng = np.random.default_rng(6)
    n = 300
    X = np.column_stack([rng.uniform(1, 6, n), rng.choice([4, 6, 8], n).astype(float)])
    y = 100 + 5 * X[:, 0] + 25 * X[:, 1] + rng.normal(0, 3, n)    # cylinders dominate
    spec = {**SPEC, "function_set": "basic", "require_engine_size": True, "population_size": 60,
            "generations": 6, "parsimony": 0.0002, "selection_tolerance": 0.01}
    m = GPRegressor(spec, 2).fit(X, y, np.zeros(n, int), 1, X_val=X[:60], y_val=y[:60], g_val=np.zeros(60, int))
    assert 0 in m.tree.variables()
