"""Expression trees: evaluation, exact derivatives, interval bounds, export."""
from __future__ import annotations

import numpy as np

from . import primitives as P


class Node:
    __slots__ = ("op", "children", "value")

    def __init__(self, op: str, children=None, value=None):
        self.op = op                      # "const", "var" or a primitive name
        self.children = children or []
        self.value = value                # constant value, or variable index

    # ---------------------------------------------------------------- structure
    def copy(self) -> "Node":
        return Node(self.op, [c.copy() for c in self.children], self.value)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        return 1 + (max(c.depth() for c in self.children) if self.children else 0)

    def is_terminal(self) -> bool:
        return self.op in ("const", "var")

    def walk(self, parent=None, idx=None, out=None):
        """List of (node, parent, child_index) in prefix order."""
        if out is None:
            out = []
        out.append((self, parent, idx))
        for i, c in enumerate(self.children):
            c.walk(self, i, out)
        return out

    def variables(self) -> set[int]:
        if self.op == "var":
            return {self.value}
        s: set[int] = set()
        for c in self.children:
            s |= c.variables()
        return s

    def n_constants(self) -> int:
        return (1 if self.op == "const" else 0) + sum(c.n_constants() for c in self.children)

    # ---------------------------------------------------------------- evaluation
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        v = self._value(X)
        return np.broadcast_to(np.asarray(v, dtype=float), (X.shape[0],)).copy()

    def _value(self, X):
        if self.op == "const":
            return self.value
        if self.op == "var":
            return X[:, self.value]
        args = [c._value(X) for c in self.children]
        return P.value_only(self.op, *args)

    def evaluate_dual(self, X: np.ndarray, d_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Value and exact derivative with respect to column d_idx."""
        v, d = self._dual(X, d_idx)
        n = X.shape[0]
        return (np.broadcast_to(np.asarray(v, float), (n,)).copy(),
                np.broadcast_to(np.asarray(d, float), (n,)).copy())

    def _dual(self, X, d_idx):
        if self.op == "const":
            return self.value, 0.0
        if self.op == "var":
            return X[:, self.value], (1.0 if self.value == d_idx else 0.0)
        if len(self.children) == 1:
            a, da = self.children[0]._dual(X, d_idx)
            return P.dual(self.op, a, da)
        a, da = self.children[0]._dual(X, d_idx)
        b, db = self.children[1]._dual(X, d_idx)
        return P.dual(self.op, a, da, b, db)

    def evaluate_interval(self, LO: np.ndarray, HI: np.ndarray, d_idx: int):
        """Bounds (lo, hi, dlo, dhi) of value and derivative over boxes [LO, HI]."""
        out = self._interval(LO, HI, d_idx)
        n = LO.shape[0]
        return tuple(np.broadcast_to(np.asarray(x, float), (n,)).copy() for x in out)

    def _interval(self, LO, HI, d_idx):
        if self.op == "const":
            return self.value, self.value, 0.0, 0.0
        if self.op == "var":
            g = 1.0 if self.value == d_idx else 0.0
            return LO[:, self.value], HI[:, self.value], g, g
        if len(self.children) == 1:
            a = self.children[0]._interval(LO, HI, d_idx)
            return P.interval(self.op, *[np.asarray(x, float) for x in a])
        a = self.children[0]._interval(LO, HI, d_idx)
        b = self.children[1]._interval(LO, HI, d_idx)
        return P.interval(self.op, *[np.asarray(x, float) for x in a], *[np.asarray(x, float) for x in b])

    # ---------------------------------------------------------------- export
    def to_string(self, names: list[str] | None = None, digits: int = 6) -> str:
        if self.op == "const":
            return f"{self.value:.{digits}g}"
        if self.op == "var":
            return names[self.value] if names else f"x{self.value}"
        c = [ch.to_string(names, digits) for ch in self.children]
        if self.op in P.SYMBOL:
            return f"({c[0]} {P.SYMBOL[self.op]} {c[1]})"
        if self.op == "aq":
            return f"aq({c[0]}, {c[1]})"
        return f"{self.op}({c[0]})"

    def to_sympy(self, names: list[str]):
        import sympy as sp
        if self.op == "const":
            return sp.Float(self.value, 17)
        if self.op == "var":
            return sp.Symbol(names[self.value])
        c = [ch.to_sympy(names) for ch in self.children]
        if self.op == "add":
            return c[0] + c[1]
        if self.op == "sub":
            return c[0] - c[1]
        if self.op == "mul":
            return c[0] * c[1]
        if self.op == "aq":
            return c[0] / sp.sqrt(1 + c[1] ** 2)
        if self.op == "exp":
            return sp.exp(c[0])
        if self.op == "log":
            return sp.log(c[0])
        if self.op == "sqrt":
            return sp.sqrt(c[0])
        raise ValueError(self.op)

    def to_dict(self):
        d = {"op": self.op}
        if self.value is not None:
            d["value"] = float(self.value) if self.op == "const" else int(self.value)
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    @staticmethod
    def from_dict(d) -> "Node":
        return Node(d["op"], [Node.from_dict(c) for c in d.get("children", [])], d.get("value"))

    def key(self) -> str:
        return self.to_string(digits=10)


def const(v: float) -> Node:
    return Node("const", value=float(v))


def var(i: int) -> Node:
    return Node("var", value=int(i))


def op(name: str, *children: Node) -> Node:
    return Node(name, list(children))
