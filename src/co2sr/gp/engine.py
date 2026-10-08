"""Genetic programming engine shared by Standard GP and all DCSR variants.

Constraint modes (spec["constraint"]):
  none         Standard GP. No constraint during evolution.
  train        DCSR-T. Penalty lam * V on training points (original formulation,
               with the derivative corrected to the total derivative).
  sample       DCSR-S. Penalty lam * V on training points plus a fixed sample Q_D
               of the declared domain.
  feasibility  DCSR-F. Deb's feasibility rules with V on training points plus Q_D.
  interval     DCSR-IA. Deb's feasibility rules with interval-arithmetic bounds
               of the slope over a fixed partition of the declared domain.
  verified     DCSR-V. Deb's feasibility rules with a three-valued verdict from
               budgeted branch-and-bound verification (gp/verify.py) over root
               boxes covering the declared domain, a cheap pre-filter on the
               training points and an archive of counterexamples, and
               counterexample feedback: points refuting one model are added to
               the archive and screen every later candidate.

A penalty schedule (spec["lam_schedule"] = "geometric") raises lam from
lam_start to lam during the run (a dynamic penalty in the spirit of Haider et
al., EUROCAST 2024; it is not a reimplementation of their method).

Fitness uses the normalised MSE (MSE / var(y)) after group-wise linear scaling,
so the penalty weight lam is in units of "one variance per std(y)/litre".
The final model is chosen from a hall of fame by inner-validation error,
preferring candidates that satisfy the constraint.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import primitives as P
from .constraints import BIG, check_interval, check_points
from .verify import signed_slopes, verify
from .scaling import Scaling, fit_scaling
from .tree import Node

INVALID = (3, np.inf)


@dataclass
class Record:
    tree: Node
    key: str
    nmse: float = np.inf
    violation: float = 0.0
    feasible: bool = True
    valid: bool = False
    scaling: Scaling | None = None
    sel: tuple = INVALID
    val_nmse: float = np.nan
    extra: dict = field(default_factory=dict)


class GPRegressor:
    constraints = ((0, 1.0),)      # class default, so models saved before this attribute existed still load
    def __init__(self, spec: dict, seed: int):
        self.spec = spec
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.functions = P.FUNCTION_SETS[spec["function_set"]]
        self.mode = spec["constraint"]
        self.lam = float(spec.get("lam", 0.0))
        self.tol = float(spec.get("feasibility_tol", 1e-9))
        self.constraints = tuple((int(j), float(sg)) for j, sg in spec.get("constraints", [[0, 1]]))
        self.lam_final = self.lam
        self.vstats = {"prefilter_refuted": 0, "bb_certified": 0, "bb_refuted": 0, "bb_undecided": 0,
                       "bb_boxes": 0, "bb_calls": 0, "cx_added": 0}
        self.history: list[dict] = []
        self.n_evaluations = 0
        self.tree: Node | None = None
        self.scaling: Scaling | None = None
        self.info: dict = {}

    # ================================================================ random trees
    def _terminal(self, n_vars: int) -> Node:
        if self.rng.random() < self.spec["p_var_terminal"]:
            return Node("var", value=int(self.rng.integers(0, n_vars)))
        lo, hi = self.spec["const_range"]
        return Node("const", value=float(self.rng.uniform(lo, hi)))

    def _random_tree(self, n_vars: int, depth: int, full: bool) -> Node:
        if depth <= 1 or (not full and self.rng.random() < 0.3):
            return self._terminal(n_vars)
        name = self.functions[int(self.rng.integers(0, len(self.functions)))]
        return Node(name, [self._random_tree(n_vars, depth - 1, full) for _ in range(P.ARITY[name])])

    def _init_population(self, n_vars: int) -> list[Node]:
        dmin, dmax = self.spec["init_depth"]
        pop, keys, tries = [], set(), 0
        while len(pop) < self.spec["population_size"] and tries < 50 * self.spec["population_size"]:
            tries += 1
            depth = int(self.rng.integers(dmin, dmax + 1))
            t = self._random_tree(n_vars, depth, full=bool(self.rng.random() < 0.5))
            k = t.key()
            if k in keys or t.size() > self.spec["max_size"]:
                continue
            keys.add(k)
            pop.append(t)
        return pop

    # ================================================================ variation
    def _pick_node(self, tree: Node):
        nodes = tree.walk()
        internal = [n for n in nodes if not n[0].is_terminal()]
        if internal and self.rng.random() < 0.9:
            return internal[int(self.rng.integers(0, len(internal)))]
        return nodes[int(self.rng.integers(0, len(nodes)))]

    @staticmethod
    def _replace(root: Node, target, new: Node) -> Node:
        node, parent, idx = target
        if parent is None:
            return new
        parent.children[idx] = new
        return root

    def _crossover(self, p1: Node, p2: Node) -> Node:
        child = p1.copy()
        target = self._pick_node(child)
        donor = self._pick_node(p2)[0].copy()
        return self._replace(child, target, donor)

    def _subtree_mutation(self, p: Node, n_vars: int) -> Node:
        child = p.copy()
        target = self._pick_node(child)
        new = self._random_tree(n_vars, int(self.rng.integers(1, 4)), full=False)
        return self._replace(child, target, new)

    def _point_mutation(self, p: Node, n_vars: int) -> Node:
        child = p.copy()
        nodes = child.walk()
        node = nodes[int(self.rng.integers(0, len(nodes)))][0]
        if node.op == "const":
            node.value = float(node.value + self.rng.normal(0.0, 0.1 * abs(node.value) + 0.1))
        elif node.op == "var":
            node.value = int(self.rng.integers(0, n_vars))
        else:
            same = [f for f in self.functions if P.ARITY[f] == P.ARITY[node.op] and f != node.op]
            if same:
                node.op = same[int(self.rng.integers(0, len(same)))]
        return child

    def _constant_mutation(self, p: Node) -> Node:
        child = p.copy()
        for node, _, _ in child.walk():
            if node.op == "const":
                node.value = float(node.value * (1.0 + self.rng.normal(0.0, 0.1)))
        return child

    def _ok(self, t: Node) -> bool:
        return t.size() <= self.spec["max_size"] and t.depth() <= self.spec["max_depth"]

    # ================================================================ evaluation
    def _evaluate(self, tree: Node) -> Record:
        self.n_evaluations += 1
        rec = Record(tree=tree, key=tree.key())
        need_dv = self.mode in ("train", "sample", "feasibility")
        if need_dv:
            v, dv = tree.evaluate_dual(self.X, 0)
        else:
            v, dv = tree.evaluate(self.X), None
        if not np.all(np.isfinite(v)) or (dv is not None and not np.all(np.isfinite(dv))):
            return rec
        sc = self._scale(v, self.y, self.g)
        pred = sc.apply(v, self.g)
        if not np.all(np.isfinite(pred)):
            return rec
        rec.nmse = float(np.mean((self.y - pred) ** 2) / self.y_var)
        rec.scaling = sc
        rec.valid = True
        rec.extra["size"] = tree.size()
        rec.extra["has_engine"] = 0 in tree.variables()
        C = self.constraints
        if self.mode == "train":
            c = check_points(tree, sc, self.X, self.g, self.y_std, self.tol, dv=dv, constraints=C)
            rec.violation, rec.feasible = c.violation, c.feasible
        elif self.mode in ("sample", "feasibility"):
            ct = check_points(tree, sc, self.X, self.g, self.y_std, self.tol, dv=dv, constraints=C)
            cq = check_points(tree, sc, self.Xq, self.gq, self.y_std, self.tol, constraints=C)
            n_t, n_q = len(self.y), len(self.gq)
            rec.violation = (ct.violation * n_t + cq.violation * n_q) / (n_t + n_q)
            rec.feasible = ct.feasible and cq.feasible
        elif self.mode == "interval":
            ci = check_interval(tree, sc, self.LO, self.HI, self.gb, self.y_std, self.tol, constraints=C)
            rec.violation, rec.feasible = ci.violation, ci.feasible
        elif self.mode == "verified":
            self._evaluate_verified(rec, sc)
        rec.sel = self._sel_key(rec)
        return rec

    # ---------------------------------------------------------------- DCSR-V
    def _screen_points(self):
        """Training points (optional) plus the counterexample archive."""
        parts_X, parts_g = [], []
        if self.spec.get("prefilter_training", True):
            parts_X.append(self.X); parts_g.append(self.g)
        if self.cx_X is not None and len(self.cx_X):
            parts_X.append(self.cx_X); parts_g.append(self.cx_g)
        if not parts_X:
            return None, None
        return np.concatenate(parts_X), np.concatenate(parts_g)

    def _evaluate_verified(self, rec: Record, sc: Scaling) -> None:
        Xs, gs = self._screen_points()
        if Xs is not None:
            sl = signed_slopes(rec.tree, sc, Xs, gs, self.constraints)
            if np.any(sl < -self.tol):
                self.vstats["prefilter_refuted"] += 1
                neg = np.where(np.isfinite(sl), np.maximum(0.0, -sl), BIG)
                rec.violation, rec.feasible = float(neg.mean() / self.y_std), False
                rec.extra["verdict"] = "refuted"
                return
        v = self._verify(rec.tree, sc)
        rec.extra["verdict"] = v.status
        rec.extra["undecided_fraction"] = v.undecided_fraction
        if v.status == "certified":
            rec.violation, rec.feasible = 0.0, True
        elif v.status == "refuted":
            ws = v.worst_slope if np.isfinite(v.worst_slope) else -BIG * self.y_std
            rec.violation, rec.feasible = float(-ws / self.y_std), False
            if self.spec.get("counterexample_feedback", True) and v.cx_X is not None:
                self._add_counterexamples(v.cx_X, v.cx_g)
        else:  # undecided
            policy = self.spec.get("undecided_policy", "between")
            rec.violation = float(v.undecided_fraction)
            rec.feasible = policy == "feasible"

    def _verify(self, tree: Node, sc: Scaling):
        v = verify(tree, sc, self.LO, self.HI, self.gb, self.tol,
                   int(self.spec.get("verify_max_boxes", 4000)), int(self.spec.get("verify_max_depth", 10)),
                   self.constraints, probe=bool(self.spec.get("verify_probe", True)))
        self.vstats["bb_calls"] += 1
        self.vstats["bb_boxes"] += v.n_boxes
        self.vstats["bb_" + v.status] += 1
        return v

    def _add_counterexamples(self, X, g):
        cap = int(self.spec.get("counterexample_archive", 1000))
        X = np.asarray(X, float); g = np.asarray(g, int)
        fin = np.all(np.isfinite(X), axis=1)
        X, g = X[fin], g[fin]
        if not len(X):
            return
        if self.cx_X is None or not len(self.cx_X):
            self.cx_X, self.cx_g = X, g
        else:
            self.cx_X = np.concatenate([self.cx_X, X])[-cap:]
            self.cx_g = np.concatenate([self.cx_g, g])[-cap:]
        self.vstats["cx_added"] += len(X)

    def _scale(self, v, y, g):
        return fit_scaling(v, y, g, self.n_groups, self.spec["min_group_rows"], self.spec["linear_scaling"],
                           self.spec.get("sign_consistent_scaling", False))

    def _sel_key(self, rec: Record) -> tuple:
        """Smaller is better. Level 0 = acceptable, 1 = infeasible (Deb), 2 = lacks engine size, 3 = invalid."""
        if not rec.valid:
            return INVALID
        if self.spec.get("require_engine_size", False) and not rec.extra.get("has_engine", True):
            # An equation without engine size satisfies the monotonicity constraint trivially and
            # cannot answer the research question; it ranks below every equation that uses it.
            return (2, rec.nmse)
        score = rec.nmse + float(self.spec.get("parsimony", 0.0)) * rec.extra.get("size", 0)
        if self.mode in ("none",):
            return (0, score)
        if self.mode in ("train", "sample"):
            return (0, score + self.lam * rec.violation)
        # Deb's feasibility rules
        if rec.feasible:
            return (0, score)
        if self.mode == "verified" and rec.extra.get("verdict") == "undecided" \
                and self.spec.get("undecided_policy", "between") == "between":
            # neither proved nor refuted: above every refuted model, below every certified one
            return (0.5, score)
        return (1, rec.violation)

    def _tournament(self, recs: list[Record]) -> Record:
        k = self.spec["tournament_size"]
        idx = self.rng.integers(0, len(recs), k)
        best = recs[idx[0]]
        for i in idx[1:]:
            if recs[i].sel < best.sel:
                best = recs[i]
        return best

    # ================================================================ fit
    def fit(self, X, y, groups, n_groups, X_val=None, y_val=None, g_val=None,
            X_q=None, g_q=None, LO=None, HI=None, g_box=None, X_full=None, y_full=None, g_full=None):
        t0 = time.perf_counter()
        self.X, self.y = np.asarray(X, float), np.asarray(y, float)
        self.g = np.asarray(groups, int)
        self.n_groups = int(n_groups)
        self.y_var = float(np.var(self.y)) or 1.0
        self.y_std = float(np.sqrt(self.y_var))
        self.Xq, self.gq = X_q, g_q
        self.LO, self.HI, self.gb = LO, HI, g_box
        if self.mode in ("sample", "feasibility") and X_q is None:
            raise ValueError(f"constraint '{self.mode}' needs a domain sample")
        if self.mode in ("interval", "verified") and LO is None:
            raise ValueError(f"constraint '{self.mode}' needs domain boxes")
        self.cx_X, self.cx_g = None, None
        if self.mode != "none" and self.spec.get("lam_schedule", "constant") == "geometric":
            self.lam = float(self.spec.get("lam_start", 1e-3))
        n_vars = self.X.shape[1]

        pop = [self._evaluate(t) for t in self._init_population(n_vars)]
        hof: dict[str, Record] = {}
        H = self.spec["hall_of_fame"]
        probs = np.array([self.spec["p_crossover"], self.spec["p_subtree_mutation"],
                          self.spec["p_point_mutation"], self.spec["p_constant_mutation"]])
        p_repro = max(0.0, 1.0 - probs.sum())
        probs = np.append(probs, p_repro) / (probs.sum() + p_repro)

        for gen in range(self.spec["generations"] + 1):
            if self.spec.get("lam_schedule", "constant") == "geometric" and self.mode in ("train", "sample"):
                T = max(1, self.spec["generations"])
                l0 = float(self.spec.get("lam_start", 1e-3))
                self.lam = l0 * (self.lam_final / l0) ** (gen / T)
                for r in pop:
                    r.sel = self._sel_key(r)
                for r in hof.values():
                    r.sel = self._sel_key(r)
            pop.sort(key=lambda r: r.sel)
            for r in pop[:H]:
                if r.valid and (r.key not in hof or r.sel < hof[r.key].sel):
                    hof[r.key] = r
            if len(hof) > H:
                hof = dict(sorted(hof.items(), key=lambda kv: kv[1].sel)[:H])
            best = pop[0]
            self.history.append({
                "generation": gen,
                "best_nmse": best.nmse,
                "best_violation": best.violation,
                "best_feasible": int(best.feasible),
                "best_size": best.tree.size(),
                "mean_size": float(np.mean([r.tree.size() for r in pop])),
                "feasible_share": (float(np.mean([r.valid and r.feasible for r in pop]))
                                   if self.mode != "none" else np.nan),
                "invalid_share": float(np.mean([not r.valid for r in pop])),
                "certified_share": (float(np.mean([r.extra.get("verdict") == "certified" for r in pop]))
                                    if self.mode == "verified" else np.nan),
                "undecided_share": (float(np.mean([r.extra.get("verdict") == "undecided" for r in pop]))
                                    if self.mode == "verified" else np.nan),
                "counterexamples": (0 if self.cx_X is None else len(self.cx_X)),
                "lam": self.lam,
                "elapsed_s": time.perf_counter() - t0,
            })
            if gen == self.spec["generations"]:
                break
            new_trees, keys = [best.tree.copy()], {best.key}
            cache = {r.key: r for r in pop}
            attempts = 0
            while len(new_trees) < len(pop) and attempts < 20 * len(pop):
                attempts += 1
                choice = int(self.rng.choice(len(probs), p=probs))
                p1 = self._tournament(pop).tree
                if choice == 0:
                    child = self._crossover(p1, self._tournament(pop).tree)
                elif choice == 1:
                    child = self._subtree_mutation(p1, n_vars)
                elif choice == 2:
                    child = self._point_mutation(p1, n_vars)
                elif choice == 3:
                    child = self._constant_mutation(p1)
                else:
                    child = p1.copy()
                if not self._ok(child):
                    child = p1.copy()
                k = child.key()
                if k in keys:                       # duplicate removal
                    child = self._random_tree(n_vars, int(self.rng.integers(2, 5)), full=False)
                    k = child.key()
                    if k in keys:
                        continue
                keys.add(k)
                new_trees.append(child)
            pop = [cache[t.key()] if t.key() in cache else self._evaluate(t) for t in new_trees]

        self._select_final(list(hof.values()), X_val, y_val, g_val)
        self._refit_scaling(X_full, y_full, g_full)
        self.info["train_time_s"] = time.perf_counter() - t0
        self.info["n_fitness_evaluations"] = self.n_evaluations
        if self.mode == "verified":
            self.info.update({f"verify_{k}": int(v) for k, v in self.vstats.items()})
        return self

    def _select_final(self, hof: list[Record], X_val, y_val, g_val):
        constrained = self.mode != "none"
        cands = [r for r in hof if r.valid]
        if not cands:
            raise RuntimeError("GP produced no valid individual; check the data and the function set")
        if self.spec.get("require_engine_size", False) and any(r.extra.get("has_engine") for r in cands):
            cands = [r for r in cands if r.extra.get("has_engine")]
        if constrained and any(r.feasible for r in cands):
            cands = [r for r in cands if r.feasible]
        elif self.mode == "verified" and any(r.extra.get("verdict") == "undecided" for r in cands):
            cands = [r for r in cands if r.extra.get("verdict") == "undecided"]
        if X_val is not None and len(y_val):
            for r in cands:
                pv = r.scaling.apply(r.tree.evaluate(np.asarray(X_val, float)), np.asarray(g_val, int))
                r.val_nmse = (float(np.mean((y_val - pv) ** 2) / self.y_var)
                              if np.all(np.isfinite(pv)) else np.inf)
            # Occam rule: the smallest equation whose validation error is within the tolerance of the best
            best_val = min(r.val_nmse for r in cands)
            tol = float(self.spec.get("selection_tolerance", 0.0))
            near = [r for r in cands if r.val_nmse <= best_val * (1.0 + tol)]
            chosen = min(near, key=lambda r: (r.tree.size(), r.val_nmse))
        else:
            chosen = min(cands, key=lambda r: r.sel)
        self.tree, self.scaling = chosen.tree.copy(), chosen.scaling
        self.info.update({
            "train_nmse": chosen.nmse, "val_nmse": chosen.val_nmse,
            "train_violation": chosen.violation, "feasible_on_constraint_set": bool(chosen.feasible),
            "hof_size": len(hof), "n_feasible_in_hof": int(sum(r.feasible for r in hof)),
            "uses_engine_size": bool(chosen.extra.get("has_engine")),
            "final_verdict": chosen.extra.get("verdict", ""),
            "selected_by": "validation" if X_val is not None else "training",
        })

    def _refit_scaling(self, X_full, y_full, g_full):
        """Refit a_k, b_k on fit + validation rows; keep the old ones if this breaks feasibility."""
        if X_full is None:
            self.info["scaling_refit"] = False
            return
        v = self.tree.evaluate(np.asarray(X_full, float))
        new = self._scale(v, np.asarray(y_full, float), np.asarray(g_full, int))
        keep_new = True
        if self.mode != "none" and self.info["feasible_on_constraint_set"]:
            if self.mode == "interval":
                c = check_interval(self.tree, new, self.LO, self.HI, self.gb, self.y_std, self.tol,
                                   constraints=self.constraints)
                keep_new = c.feasible
            elif self.mode == "verified":
                keep_new = self._verify(self.tree, new).status == "certified"
            else:
                c1 = check_points(self.tree, new, self.X, self.g, self.y_std, self.tol, constraints=self.constraints)
                ok = c1.feasible
                if self.mode in ("sample", "feasibility"):
                    ok = ok and check_points(self.tree, new, self.Xq, self.gq, self.y_std, self.tol,
                                             constraints=self.constraints).feasible
                keep_new = ok
        elif self.mode == "verified":
            # an undecided or refuted final model: refit only if it does not become refuted
            keep_new = self._verify(self.tree, new).status != "refuted"
        if keep_new:
            self.scaling = new
        self.info["scaling_refit"] = bool(keep_new)

    # ================================================================ prediction
    def predict(self, X, groups) -> np.ndarray:
        return self.scaling.apply(self.tree.evaluate(np.asarray(X, float)), np.asarray(groups, int))

    def slope(self, X, groups) -> np.ndarray:
        """Exact derivative of the prediction with respect to engine size (column 0)."""
        _, dv = self.tree.evaluate_dual(np.asarray(X, float), 0)
        _, b = self.scaling.coefficients(np.asarray(groups, int))
        return b * dv

    def certify(self, LO, HI, groups, tol=None) -> dict:
        """Plain interval arithmetic on a fixed partition (no subdivision)."""
        c = check_interval(self.tree, self.scaling, LO, HI, np.asarray(groups, int), 1.0,
                           self.tol if tol is None else tol, constraints=self.constraints)
        return {"certified": c.feasible, "share_boxes_failing": c.share_negative,
                "min_slope_lower_bound": c.min_slope}

    def verify_final(self, LO, HI, groups, max_boxes=200000, max_depth=16, tol=None) -> dict:
        """Branch-and-bound verdict for the final model (used for every GP method in evaluation)."""
        v = verify(self.tree, self.scaling, LO, HI, np.asarray(groups, int), self.tol if tol is None else tol,
                   max_boxes, max_depth, self.constraints)
        return {"bb_verdict": v.status, "bb_boxes": v.n_boxes, "bb_max_depth": v.max_depth,
                "bb_undecided_fraction": v.undecided_fraction,
                "bb_worst_slope": v.worst_slope if v.status == "refuted" else np.nan}

    def to_dict(self) -> dict:
        return {"tree": self.tree.to_dict(), "scaling": self.scaling.to_dict(),
                "n_groups": self.n_groups, "spec": self.spec, "seed": self.seed, "info": self.info}

    @staticmethod
    def from_dict(d) -> "GPRegressor":
        m = GPRegressor(d["spec"], d["seed"])
        m.tree = Node.from_dict(d["tree"])
        m.scaling = Scaling.from_dict(d["scaling"])
        m.n_groups = d["n_groups"]
        m.info = d.get("info", {})
        m.constraints = tuple((int(j), float(sg)) for j, sg in d["spec"].get("constraints", [[0, 1]]))
        return m
