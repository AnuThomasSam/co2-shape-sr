# ============================================================
# Derivative-Constrained Symbolic Regression (DCSR)
# ============================================================

import copy
import time
import numpy as np


# ============================================================
# Protected mathematical operators
# ============================================================

EPS = 1e-10


def protected_div(a, b):
    """Division that avoids division by zero."""
    b_safe = np.where(
        np.abs(b) < EPS,
        np.where(b >= 0, EPS, -EPS),
        b
    )
    return a / b_safe


def protected_log(x):
    """Logarithm that avoids log(0)."""
    return np.log(np.maximum(np.abs(x), EPS))


def protected_exp(x):
    """Exponent with clipping to avoid overflow."""
    return np.exp(np.clip(x, -50, 50))


# ============================================================
# Operators
# ============================================================

BINARY_OPERATORS = [
    "add",
    "sub",
    "mul",
    "div"
]

UNARY_OPERATORS = [
    "exp",
    "log"
]


# ============================================================
# Tree node
# ============================================================

class Node:

    def __init__(
        self,
        node_type,
        value=None,
        left=None,
        right=None
    ):
        self.node_type = node_type
        self.value = value
        self.left = left
        self.right = right

    def clone(self):
        return copy.deepcopy(self)

    # --------------------------------------------------------
    # Evaluate equation
    # --------------------------------------------------------

    def evaluate(self, X):

        if self.node_type == "constant":
            return np.full(
                X.shape[0],
                self.value,
                dtype=float
            )

        if self.node_type == "variable":
            return X[:, self.value]

        if self.node_type == "add":
            return (
                self.left.evaluate(X)
                + self.right.evaluate(X)
            )

        if self.node_type == "sub":
            return (
                self.left.evaluate(X)
                - self.right.evaluate(X)
            )

        if self.node_type == "mul":
            return (
                self.left.evaluate(X)
                * self.right.evaluate(X)
            )

        if self.node_type == "div":
            return protected_div(
                self.left.evaluate(X),
                self.right.evaluate(X)
            )

        if self.node_type == "exp":
            return protected_exp(
                self.left.evaluate(X)
            )

        if self.node_type == "log":
            return protected_log(
                self.left.evaluate(X)
            )

        raise ValueError(
            f"Unknown node type: {self.node_type}"
        )

    # --------------------------------------------------------
    # Exact analytical derivative
    # --------------------------------------------------------

    def derivative(self, X, variable_index):

        if self.node_type == "constant":
            return np.zeros(X.shape[0])

        if self.node_type == "variable":

            if self.value == variable_index:
                return np.ones(X.shape[0])

            return np.zeros(X.shape[0])

        # ----------------------------------------------------
        # Addition
        # ----------------------------------------------------

        if self.node_type == "add":

            return (
                self.left.derivative(
                    X, variable_index
                )
                +
                self.right.derivative(
                    X, variable_index
                )
            )

        # ----------------------------------------------------
        # Subtraction
        # ----------------------------------------------------

        if self.node_type == "sub":

            return (
                self.left.derivative(
                    X, variable_index
                )
                -
                self.right.derivative(
                    X, variable_index
                )
            )

        # ----------------------------------------------------
        # Multiplication
        # ----------------------------------------------------

        if self.node_type == "mul":

            a = self.left.evaluate(X)
            b = self.right.evaluate(X)

            da = self.left.derivative(
                X, variable_index
            )

            db = self.right.derivative(
                X, variable_index
            )

            return da * b + a * db

        # ----------------------------------------------------
        # Division
        # ----------------------------------------------------

        if self.node_type == "div":

            a = self.left.evaluate(X)
            b = self.right.evaluate(X)

            da = self.left.derivative(
                X, variable_index
            )

            db = self.right.derivative(
                X, variable_index
            )

            b_safe = np.where(
                np.abs(b) < EPS,
                np.where(b >= 0, EPS, -EPS),
                b
            )

            return (
                (da * b_safe - a * db)
                / (b_safe ** 2)
            )

        # ----------------------------------------------------
        # exp
        #
        # d/dx exp(u) = exp(u) * du/dx
        # ----------------------------------------------------

        if self.node_type == "exp":

            a = self.left.evaluate(X)

            da = self.left.derivative(
                X, variable_index
            )

            return protected_exp(a) * da

        # ----------------------------------------------------
        # log
        #
        # d/dx log(|u|) = du/u
        # ----------------------------------------------------

        if self.node_type == "log":

            a = self.left.evaluate(X)

            da = self.left.derivative(
                X, variable_index
            )

            a_safe = np.where(
                np.abs(a) < EPS,
                np.where(a >= 0, EPS, -EPS),
                a
            )

            return da / a_safe

        raise ValueError(
            f"Unknown node type: {self.node_type}"
        )

    # --------------------------------------------------------
    # Equation length
    # --------------------------------------------------------

    def length(self):

        if self.node_type in [
            "constant",
            "variable"
        ]:
            return 1

        if self.node_type in [
            "exp",
            "log"
        ]:
            return 1 + self.left.length()

        return (
            1
            + self.left.length()
            + self.right.length()
        )

    # --------------------------------------------------------
    # Equation depth
    # --------------------------------------------------------

    def depth(self):

        if self.node_type in [
            "constant",
            "variable"
        ]:
            return 1

        if self.node_type in [
            "exp",
            "log"
        ]:
            return 1 + self.left.depth()

        return (
            1
            + max(
                self.left.depth(),
                self.right.depth()
            )
        )


# ============================================================
# Convert tree to readable equation
# ============================================================

def tree_to_string(node, feature_names):

    if node.node_type == "constant":
        return f"{node.value:.4f}"

    if node.node_type == "variable":
        return feature_names[node.value]

    if node.node_type == "add":
        return (
            "("
            + tree_to_string(node.left, feature_names)
            + " + "
            + tree_to_string(node.right, feature_names)
            + ")"
        )

    if node.node_type == "sub":
        return (
            "("
            + tree_to_string(node.left, feature_names)
            + " - "
            + tree_to_string(node.right, feature_names)
            + ")"
        )

    if node.node_type == "mul":
        return (
            "("
            + tree_to_string(node.left, feature_names)
            + " * "
            + tree_to_string(node.right, feature_names)
            + ")"
        )

    if node.node_type == "div":
        return (
            "("
            + tree_to_string(node.left, feature_names)
            + " / "
            + tree_to_string(node.right, feature_names)
            + ")"
        )

    if node.node_type == "exp":
        return (
            "exp("
            + tree_to_string(node.left, feature_names)
            + ")"
        )

    if node.node_type == "log":
        return (
            "log("
            + tree_to_string(node.left, feature_names)
            + ")"
        )

    raise ValueError(
        f"Unknown node type: {node.node_type}"
    )


# ============================================================
# Random terminal
# ============================================================

def random_terminal(n_features, rng):

    if rng.random() < 0.70:

        return Node(
            "variable",
            value=int(
                rng.integers(
                    0,
                    n_features
                )
            )
        )

    return Node(
        "constant",
        value=float(
            rng.uniform(-10, 10)
        )
    )


# ============================================================
# Random tree
# ============================================================

def random_tree(
    n_features,
    depth,
    rng,
    max_depth=5,
    min_depth=2
):

    # Stop at maximum depth
    if depth >= max_depth:

        return random_terminal(
            n_features,
            rng
        )

    # Sometimes create a terminal
    if (
        depth >= min_depth
        and rng.random() < 0.30
    ):

        return random_terminal(
            n_features,
            rng
        )

    # Binary operator
    if rng.random() < 0.70:

        operator = rng.choice(
            BINARY_OPERATORS
        )

        return Node(
            operator,
            left=random_tree(
                n_features,
                depth + 1,
                rng,
                max_depth,
                min_depth
            ),
            right=random_tree(
                n_features,
                depth + 1,
                rng,
                max_depth,
                min_depth
            )
        )

    # Unary operator
    operator = rng.choice(
        UNARY_OPERATORS
    )

    return Node(
        operator,
        left=random_tree(
            n_features,
            depth + 1,
            rng,
            max_depth,
            min_depth
        )
    )


# ============================================================
# Fitness calculation
# ============================================================

def calculate_dcsr_fitness(
    tree,
    X,
    y,
    engine_idx,
    lambda_penalty,
    complexity_penalty
):

    try:

        # ----------------------------------------------------
        # 1. Data loss
        # ----------------------------------------------------

        y_pred = tree.evaluate(X)

        if not np.all(
            np.isfinite(y_pred)
        ):
            return (
                np.inf,
                np.inf,
                np.inf,
                None
            )

        e_data = np.mean(
            (y - y_pred) ** 2
        )

        # ----------------------------------------------------
        # 2. Exact symbolic derivative
        # ----------------------------------------------------

        derivative = tree.derivative(
            X,
            engine_idx
        )

        if not np.all(
            np.isfinite(derivative)
        ):
            return (
                np.inf,
                e_data,
                np.inf,
                None
            )

        # ----------------------------------------------------
        # 3. Physics constraint
        #
        # d(CO2) / d(engine_size) >= 0
        # ----------------------------------------------------

        violations = np.maximum(
            0.0,
            -derivative
        )

        e_phys = (
            lambda_penalty
            * np.mean(violations)
        )

        # ----------------------------------------------------
        # 4. Complexity
        #
        # Used only as a small tie-breaking preference.
        # The physics and prediction errors remain dominant.
        # ----------------------------------------------------

        complexity = (
            complexity_penalty
            * tree.length()
        )

        # ----------------------------------------------------
        # 5. Combined fitness
        # ----------------------------------------------------

        fitness = (
            e_data
            + e_phys
            + complexity
        )

        return (
            fitness,
            e_data,
            e_phys,
            derivative
        )

    except Exception:

        return (
            np.inf,
            np.inf,
            np.inf,
            None
        )


# ============================================================
# Population evaluation
# ============================================================

def evaluate_population(
    population,
    X,
    y,
    engine_idx,
    lambda_penalty,
    complexity_penalty
):

    evaluated = []

    for tree in population:

        (
            fitness,
            e_data,
            e_phys,
            derivative
        ) = calculate_dcsr_fitness(
            tree,
            X,
            y,
            engine_idx,
            lambda_penalty,
            complexity_penalty
        )

        evaluated.append({

            "tree": tree,

            "fitness": fitness,

            "e_data": e_data,

            "e_phys": e_phys,

            "derivative": derivative,

            "length": tree.length(),

            "depth": tree.depth()
        })

    evaluated.sort(
        key=lambda item: item["fitness"]
    )

    return evaluated


# ============================================================
# Tournament selection
# ============================================================

def tournament_selection(
    evaluated_population,
    tournament_size,
    rng
):

    selected = rng.choice(
        evaluated_population,
        size=tournament_size,
        replace=False
    )

    winner = min(
        selected,
        key=lambda item: item["fitness"]
    )

    return winner["tree"].clone()


# ============================================================
# Get all nodes
# ============================================================

def get_nodes(node):

    nodes = [node]

    if node.left is not None:

        nodes.extend(
            get_nodes(node.left)
        )

    if node.right is not None:

        nodes.extend(
            get_nodes(node.right)
        )

    return nodes


# ============================================================
# Replace subtree
# ============================================================

def replace_node(
    root,
    target,
    replacement
):

    if root is target:
        return replacement

    if root.left is not None:

        root.left = replace_node(
            root.left,
            target,
            replacement
        )

    if root.right is not None:

        root.right = replace_node(
            root.right,
            target,
            replacement
        )

    return root


# ============================================================
# Crossover
# ============================================================

def subtree_crossover(
    parent1,
    parent2,
    rng
):

    child = parent1.clone()
    donor = parent2.clone()

    target = rng.choice(
        get_nodes(child)
    )

    replacement = rng.choice(
        get_nodes(donor)
    ).clone()

    return replace_node(
        child,
        target,
        replacement
    )


# ============================================================
# Mutation
# ============================================================

def subtree_mutation(
    tree,
    n_features,
    rng,
    max_depth=5
):

    mutated = tree.clone()

    target = rng.choice(
        get_nodes(mutated)
    )

    replacement = random_tree(
        n_features,
        0,
        rng,
        max_depth=max_depth,
        min_depth=1
    )

    return replace_node(
        mutated,
        target,
        replacement
    )


# ============================================================
# DCSR Regressor
# ============================================================

class DCSRRegressor:

    def __init__(
        self,
        population_size=500,
        generations=50,
        tournament_size=20,
        engine_col_idx=None,
        lambda_penalty=1e9,
        complexity_penalty=0.001,
        max_depth=5,
        crossover_probability=0.70,
        mutation_probability=0.30,
        random_state=42
    ):

        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size

        self.engine_col_idx = engine_col_idx
        self.lambda_penalty = lambda_penalty

        self.complexity_penalty = (
            complexity_penalty
        )

        self.max_depth = max_depth

        self.crossover_probability = (
            crossover_probability
        )

        self.mutation_probability = (
            mutation_probability
        )

        self.random_state = random_state

        # ----------------------------------------------------
        # Original names retained for notebook compatibility
        # ----------------------------------------------------

        self.best_tree_ = None

        self.best_fitness_ = np.inf

        self.best_e_data_ = np.inf

        self.best_e_phys_ = np.inf

        self.best_generation_ = None

        self.history_ = []

    # ========================================================
    # Fit
    # ========================================================

    def fit(self, X, y):

        X = np.asarray(
            X,
            dtype=float
        )

        y = np.asarray(
            y,
            dtype=float
        )

        if self.engine_col_idx is None:

            raise ValueError(
                "engine_col_idx must be specified."
            )

        rng = np.random.default_rng(
            self.random_state
        )

        n_features = X.shape[1]

        # ----------------------------------------------------
        # 1. Initial population
        # ----------------------------------------------------

        population = [

            random_tree(
                n_features,
                0,
                rng,
                max_depth=self.max_depth
            )

            for _ in range(
                self.population_size
            )
        ]

        # ----------------------------------------------------
        # 2. Evolution
        # ----------------------------------------------------

        for generation in range(
            self.generations
        ):

            start_time = time.perf_counter()

            # ------------------------------------------------
            # Evaluate population
            # ------------------------------------------------

            evaluated = evaluate_population(

                population,

                X,
                y,

                self.engine_col_idx,

                self.lambda_penalty,

                self.complexity_penalty
            )

            best = evaluated[0]

            # ------------------------------------------------
            # Save global best
            # ------------------------------------------------

            if (
                best["fitness"]
                < self.best_fitness_
            ):

                self.best_fitness_ = (
                    best["fitness"]
                )

                self.best_tree_ = (
                    best["tree"].clone()
                )

                self.best_e_data_ = (
                    best["e_data"]
                )

                self.best_e_phys_ = (
                    best["e_phys"]
                )

                self.best_generation_ = (
                    generation
                )

            # ------------------------------------------------
            # Save history
            # ------------------------------------------------

            elapsed = (
                time.perf_counter()
                - start_time
            )

            self.history_.append({

                "generation":
                    generation,

                "fitness":
                    best["fitness"],

                "e_data":
                    best["e_data"],

                "e_phys":
                    best["e_phys"],

                "length":
                    best["length"],

                "depth":
                    best["depth"],

                "time":
                    elapsed
            })

            # ------------------------------------------------
            # 3. Selection + evolution
            # ------------------------------------------------

            new_population = [

                best["tree"].clone()
            ]

            while len(
                new_population
            ) < self.population_size:

                parent1 = (
                    tournament_selection(
                        evaluated,
                        self.tournament_size,
                        rng
                    )
                )

                parent2 = (
                    tournament_selection(
                        evaluated,
                        self.tournament_size,
                        rng
                    )
                )

                # --------------------------------------------
                # Crossover
                # --------------------------------------------

                if (
                    rng.random()
                    < self.crossover_probability
                ):

                    child = (
                        subtree_crossover(
                            parent1,
                            parent2,
                            rng
                        )
                    )

                else:

                    child = parent1.clone()

                # --------------------------------------------
                # Mutation
                # --------------------------------------------

                if (
                    rng.random()
                    < self.mutation_probability
                ):

                    child = (
                        subtree_mutation(
                            child,
                            n_features,
                            rng,
                            self.max_depth
                        )
                    )

                # --------------------------------------------
                # Depth protection
                # --------------------------------------------

                if (
                    child.depth()
                    <= self.max_depth + 2
                ):

                    new_population.append(
                        child
                    )

            population = new_population

        return self

    # ========================================================
    # Prediction
    # ========================================================

    def predict(self, X):

        if self.best_tree_ is None:

            raise ValueError(
                "The model has not been fitted yet."
            )

        X = np.asarray(
            X,
            dtype=float
        )

        return self.best_tree_.evaluate(X)

    # ========================================================
    # Exact derivative
    # ========================================================

    def derivative(self, X):

        if self.best_tree_ is None:

            raise ValueError(
                "The model has not been fitted yet."
            )

        X = np.asarray(
            X,
            dtype=float
        )

        return self.best_tree_.derivative(
            X,
            self.engine_col_idx
        )

    # ========================================================
    # Equation
    # ========================================================

    def equation(self, feature_names):

        if self.best_tree_ is None:

            raise ValueError(
                "The model has not been fitted yet."
            )

        return tree_to_string(
            self.best_tree_,
            feature_names
        )

    # ========================================================
    # Equation length
    # ========================================================

    def equation_length(self):

        if self.best_tree_ is None:

            raise ValueError(
                "The model has not been fitted yet."
            )

        return self.best_tree_.length()

    # ========================================================
    # Equation depth
    # ========================================================

    def equation_depth(self):

        if self.best_tree_ is None:

            raise ValueError(
                "The model has not been fitted yet."
            )

        return self.best_tree_.depth()

    # ========================================================
    # Summary
    # ========================================================

    def summary(self):

        return {

            "best_fitness":
                self.best_fitness_,

            "data_mse":
                self.best_e_data_,

            "physics_penalty":
                self.best_e_phys_,

            "best_generation":
                self.best_generation_,

            "equation_length":
                self.equation_length(),

            "equation_depth":
                self.equation_depth()
        }