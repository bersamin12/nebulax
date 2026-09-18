"""Boosting-tier models from ``configs/model_ladder.yaml``: gradient-boosted trees, tree
ensembles, linear baselines, an interpretable rule learner, a stacked ensemble and a
conformal calibration wrapper.

Registers (name / input_kind / family / task, verbatim from the ladder):
    lgbm_residual            raw_window      residual_model      ad   (default)
    lgbm_three_regime        cycle_features  gradient_boosting   cls
    lgbm_envelope             window_stats    gradient_boosting   cls
    shallow_rule_learner      cycle_features  interpretable_rule  ad   (default)
    stacking_rf_xgb_logreg    cycle_features  stacking            cls
    random_forest_cycle       cycle_features  tree_ensemble       cls
    logreg_cycle              cycle_features  linear              cls
    logreg_envelope           window_stats    linear              cls
    conformal_threshold       window_stats    calibration         ad   (default)

Every model's tunables are plain constructor keywords with defaults documented on the class
docstring (``BaseModel`` records them verbatim in ``self.params`` for the results row). AD
models here (``lgbm_residual``, ``shallow_rule_learner``, ``conformal_threshold``) never read
``y`` - :class:`nebulax.bench.base.AnomalyDetector` does not even pass one to ``_fit``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from nebulax.bench.base import AnomalyDetector, BaseModel, Classifier
from nebulax.bench.registry import build as _build_registered
from nebulax.bench.registry import register

__all__ = [
    "LGBMResidual",
    "LGBMThreeRegime",
    "LGBMEnvelope",
    "ShallowRuleLearner",
    "StackingRFXGBLogReg",
    "RandomForestCycle",
    "LogRegCycle",
    "LogRegEnvelope",
    "ConformalThreshold",
]


# ----------------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------------


def _encode_labels(classes_: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Map raw labels to contiguous ``0..k-1`` codes in ``classes_`` order.

    Some boosted-tree libraries (xgboost in particular) require 0-indexed integer class
    codes; sklearn/lightgbm tolerate arbitrary label dtypes but are encoded the same way here
    for one consistent code path.
    """
    return np.searchsorted(classes_, y).astype(np.int64)


def _decode_labels(classes_: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_encode_labels`."""
    return classes_[np.asarray(codes, dtype=np.int64)]


#: Door/compressor cycle_features whose *name* marks a regime (case-insensitive substring
#: match), in priority order. Checked against the real column layouts before picking these:
#: the synthetic door table's ``opening_time``/``pos_open_max``, ``i_mean_cruise``/
#: ``i_rms_cruise``, ``closing_time``/``pos_close_max`` name their regime explicitly; Cranfield's
#: ``cycle_features`` (stroke-level mean/std/min/max of ``i_mean``, ``i_rms``, ``pos_err_max``,
#: ... ) name no regime at all, so it always falls back to the equal-slices approximation.
_REGIME_NAME_PATTERNS: tuple[tuple[str, str], ...] = (
    ("opening", "open"),
    ("cruise", "cruise"),
    ("closing", "close"),
)


def _name_based_regime_groups(feature_names: Sequence[str]) -> list[np.ndarray] | None:
    """Column indices per regime (opening/cruise/closing), built from ``feature_names``.

    A column whose (lower-cased) name contains a regime's marker substring is assigned to
    that regime; a column that matches none of them is a cycle-level/context feature (e.g. a
    temperature, a duty fraction, an un-phased current-profile bin) and is shared by every
    regime's sub-model instead of being dropped. Returns ``None`` (never an empty/degenerate
    grouping) when not a single column name marks a regime, so the caller can fall back to the
    equal-slices approximation.
    """
    lowered = [str(n).lower() for n in feature_names]
    matched: dict[str, list[int]] = {regime: [] for regime, _ in _REGIME_NAME_PATTERNS}
    unmatched: list[int] = []
    for i, name in enumerate(lowered):
        for regime, pattern in _REGIME_NAME_PATTERNS:
            if pattern in name:
                matched[regime].append(i)
                break
        else:
            unmatched.append(i)
    if not any(matched.values()):
        return None
    return [np.array(sorted(matched[regime] + unmatched), dtype=np.int64) for regime, _ in _REGIME_NAME_PATTERNS]


def _align_proba(classes_: np.ndarray, sub_classes: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Re-columns ``proba`` (as returned for ``sub_classes``) onto the full ``classes_`` order.

    Used when a sub-model saw only part of the training rows (e.g. a per-regime slice) and so
    may have a narrower ``classes_`` than the top-level model.
    """
    if sub_classes.size == classes_.size and np.array_equal(sub_classes, classes_):
        return proba
    out = np.zeros((proba.shape[0], classes_.size), dtype=np.float64)
    for j, cls in enumerate(sub_classes):
        out[:, int(np.searchsorted(classes_, cls))] = proba[:, j]
    return out


# ----------------------------------------------------------------------------------------
# lgbm_residual - AD, raw_window
# ----------------------------------------------------------------------------------------


@register("lgbm_residual", input_kind="raw_window", family="residual_model")
class LGBMResidual(AnomalyDetector):
    """Per-channel forecast-residual anomaly score over raw windows (ladder R32/R68).

    ``X`` is ``(n, L, c)``. For each channel ``ch`` this fits a LightGBM regressor that
    predicts the window's LAST timestep of that channel from every channel's values at every
    OTHER (earlier) timestep in the same window, flattened to one feature vector. The
    channel's own lags fall out naturally from that flattening (its own earlier timesteps are
    among the predictors), matching the ladder's "predict each channel from the others +
    lags". The anomaly score is the max, over channels, of the |residual| standardised by
    that channel's training-residual std - one badly predicted channel is enough to raise the
    window's score.

    Deviation: ``X`` at this layer carries no channel identity (analog vs digital are not
    distinguishable from a bare ``(n, L, c)`` array), so every channel is modelled the same
    way rather than analog-only. Pass ``channels`` (a sequence of channel indices) to restrict
    modelling to a known-analog subset when the caller has that mapping.

    Params:
        n_estimators: int = 150
        max_depth: int = 4
        learning_rate: float = 0.1
        min_child_samples: int = 20
        channels: Sequence[int] | None = None   -- default: model every channel
        n_jobs: int = -1
        seed: int = 0
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.n_estimators = int(params.get("n_estimators", 150))
        self.max_depth = int(params.get("max_depth", 4))
        self.learning_rate = float(params.get("learning_rate", 0.1))
        self.min_child_samples = int(params.get("min_child_samples", 20))
        self.channels = params.get("channels")
        self.n_jobs = int(params.get("n_jobs", -1))
        self.seed = int(params.get("seed", 0))

    def _make_regressor(self) -> LGBMRegressor:
        return LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            n_jobs=self.n_jobs,
            random_state=self.seed,
            verbose=-1,
        )

    @staticmethod
    def _features_targets(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, L, c = X.shape
        if L < 2:
            raise ValueError(f"lgbm_residual: window length L must be >= 2, got {L}")
        feats = X[:, :-1, :].reshape(n, (L - 1) * c).astype(np.float64)
        targets = X[:, -1, :].astype(np.float64)
        return feats, targets

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kwargs: Any) -> None:
        _, _, c = X.shape
        self.channels_ = list(self.channels) if self.channels is not None else list(range(c))
        feats, targets = self._features_targets(X)
        self.models_: dict[int, LGBMRegressor] = {}
        self.resid_std_: dict[int, float] = {}
        for ch in self.channels_:
            model = self._make_regressor()
            y = targets[:, ch]
            model.fit(feats, y)
            resid = y - model.predict(feats)
            self.models_[ch] = model
            self.resid_std_[ch] = float(np.std(resid)) + 1e-9

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        feats, targets = self._features_targets(X)
        z = np.zeros((feats.shape[0], len(self.channels_)), dtype=np.float64)
        for j, ch in enumerate(self.channels_):
            resid = targets[:, ch] - self.models_[ch].predict(feats)
            z[:, j] = np.abs(resid) / self.resid_std_[ch]
        return z.max(axis=1)


# ----------------------------------------------------------------------------------------
# lgbm_three_regime - CLS, cycle_features
# ----------------------------------------------------------------------------------------


@register("lgbm_three_regime", input_kind="cycle_features", family="gradient_boosting", task="cls")
class LGBMThreeRegime(Classifier):
    """Per-regime LightGBM ensemble over door-cycle features (ladder R64).

    A door cycle's ``cycle_features`` vector packs several phases (opening / cruise /
    closing) end to end. When the runner passes ``feature_names`` into :meth:`_fit` (the
    row-context channel, see :mod:`nebulax.bench.base`), regime column groups are built from
    the names themselves via :func:`_name_based_regime_groups`: a column whose name marks a
    regime (``"open"``/``"cruise"``/``"close"``, e.g. the synthetic door table's
    ``opening_time``, ``i_mean_cruise``, ``closing_time``) goes to that regime, and every
    unmarked column (context/cycle-level features with no phase of their own) is shared by all
    three regimes. When no ``feature_names`` are given, or none of them mark a regime (e.g.
    Cranfield's ``cycle_features``, which is all stroke-level mean/std/min/max with no
    open/cruise/close naming), this falls back to three CONTIGUOUS, near-equal slices of the
    feature vector instead. One LightGBM classifier is fit per regime group; predictions are
    combined by averaging the three regimes' ``predict_proba`` (soft voting). Which path was
    used is recorded on the fitted model as ``regime_source_``
    (``"regime_slices" | "feature_names" | "equal_slices"``).

    Deviation: pass ``regime_slices=[(start, end), (start, end), (start, end)]`` to force
    contiguous column blocks (skips both the name-based grouping and the equal-thirds
    fallback) when the real layout is known but not name-markable.

    Params:
        regime_slices: Sequence[tuple[int, int]] | None = None
        n_estimators: int = 150
        max_depth: int = 4
        learning_rate: float = 0.1
        min_child_samples: int = 10
        n_jobs: int = -1
        seed: int = 0

    Attributes (set by :meth:`_fit`)
    ---------------------------------
    regime_indices_ : list[np.ndarray]
        Column indices used for each of the three regime sub-models.
    regime_source_ : str
        How ``regime_indices_`` was built: ``"regime_slices"`` (explicit constructor
        override), ``"feature_names"`` (name-based grouping) or ``"equal_slices"`` (fallback).
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.regime_slices = params.get("regime_slices")
        self.n_estimators = int(params.get("n_estimators", 150))
        self.max_depth = int(params.get("max_depth", 4))
        self.learning_rate = float(params.get("learning_rate", 0.1))
        self.min_child_samples = int(params.get("min_child_samples", 10))
        self.n_jobs = int(params.get("n_jobs", -1))
        self.seed = int(params.get("seed", 0))

    def _regime_groups(
        self, n_features: int, feature_names: Sequence[str] | None
    ) -> tuple[list[np.ndarray], str]:
        """Column indices per regime, and which strategy built them - see the class docstring."""
        if self.regime_slices is not None:
            groups = [np.arange(int(a), int(b), dtype=np.int64) for a, b in self.regime_slices]
            return groups, "regime_slices"
        if feature_names is not None and len(feature_names) == n_features:
            groups = _name_based_regime_groups(feature_names)
            if groups is not None:
                return groups, "feature_names"
        edges = np.linspace(0, n_features, 4).round().astype(int)
        groups = [np.arange(int(edges[i]), int(edges[i + 1]), dtype=np.int64) for i in range(3)]
        return groups, "equal_slices"

    def _make_classifier(self) -> LGBMClassifier:
        return LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            n_jobs=self.n_jobs,
            random_state=self.seed,
            verbose=-1,
        )

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        n, f = X.shape
        codes = _encode_labels(self.classes_, y)
        self.regime_indices_, self.regime_source_ = self._regime_groups(f, feature_names)
        self.models_: list[LGBMClassifier | None] = []
        for idx in self.regime_indices_:
            if idx.size == 0:
                self.models_.append(None)
                continue
            model = self._make_classifier()
            model.fit(X[:, idx], codes)
            self.models_.append(model)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = []
        for idx, model in zip(self.regime_indices_, self.models_):
            if model is None:
                continue
            sub_classes = self.classes_[model.classes_]  # LightGBM's classes_ are code integers
            p = model.predict_proba(X[:, idx])
            probs.append(_align_proba(self.classes_, sub_classes, p))
        if not probs:
            raise RuntimeError("lgbm_three_regime: every regime group was empty")
        return np.mean(probs, axis=0)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        p = self._predict_proba(X)
        return self.classes_[np.argmax(p, axis=1)]


# ----------------------------------------------------------------------------------------
# lgbm_envelope - CLS, window_stats
# ----------------------------------------------------------------------------------------


@register("lgbm_envelope", input_kind="window_stats", family="gradient_boosting", task="cls")
class LGBMEnvelope(Classifier):
    """LightGBM classifier over bearing ``window_stats`` features (envelope-spectrum features
    are already folded into ``X`` upstream by the ``feature_set`` the loader selects; ladder
    R58/R59).

    Params:
        n_estimators: int = 200
        max_depth: int = -1   -- unlimited (LightGBM's convention)
        num_leaves: int = 31
        learning_rate: float = 0.05
        min_child_samples: int = 10
        n_jobs: int = -1
        seed: int = 0
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.n_estimators = int(params.get("n_estimators", 200))
        self.max_depth = int(params.get("max_depth", -1))
        self.num_leaves = int(params.get("num_leaves", 31))
        self.learning_rate = float(params.get("learning_rate", 0.05))
        self.min_child_samples = int(params.get("min_child_samples", 10))
        self.n_jobs = int(params.get("n_jobs", -1))
        self.seed = int(params.get("seed", 0))

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        codes = _encode_labels(self.classes_, y)
        self.model_ = LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            n_jobs=self.n_jobs,
            random_state=self.seed,
            verbose=-1,
        )
        self.model_.fit(X, codes)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        sub_classes = self.classes_[self.model_.classes_]
        return _align_proba(self.classes_, sub_classes, self.model_.predict_proba(X))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return _decode_labels(self.classes_, self.model_.predict(X))


# ----------------------------------------------------------------------------------------
# shallow_rule_learner - AD, cycle_features
# ----------------------------------------------------------------------------------------


class _RuleNode:
    """One node of the depth-<=``max_depth`` isolation rule tree. Leaf nodes carry only the
    training-row ``count`` that landed there; interior nodes carry the split."""

    __slots__ = ("count", "leaf", "feature", "threshold", "side", "extreme", "normal")

    def __init__(self, count: int) -> None:
        self.count = count
        self.leaf = True
        self.feature: int = -1
        self.threshold: float = 0.0
        self.side: int = 1
        self.extreme: "_RuleNode | None" = None
        self.normal: "_RuleNode | None" = None


@register("shallow_rule_learner", input_kind="cycle_features", family="interpretable_rule")
class ShallowRuleLearner(AnomalyDetector):
    """Unsupervised depth-<=2 "isolation rule" over cycle_features (ladder R93/R98; the ladder
    note allows depth<=3, kept at 2 here so the printed rule set stays to at most 3 splits / 4
    leaves).

    This never reads ``y`` (the AD contract forbids it - :class:`AnomalyDetector.fit` does not
    even accept one). At each node it standardises the node's own rows, picks the feature with
    the single most extreme z-score, and peels the most extreme ``tail_frac`` of that node's
    rows into an "extreme" child (on whichever side - upper or lower - holds the most extreme
    point), recursing one more level down that extreme branch only. The other, "normal" child
    is also recursed so every training row ends in some leaf.

    Anomaly score for a query row = ``-log((leaf_count + 1) / (n_train + 1))``: the rarer the
    training leaf a row's path lands it in, the higher its score. Human-readable rules are
    built once at fit time onto ``self.rules_`` (``list[str]``) and printed to stdout when
    ``verbose=True``.

    Params:
        max_depth: int = 2
        tail_frac: float = 0.05   -- fraction of a node's rows peeled into its extreme child
        verbose: bool = False
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.max_depth = int(params.get("max_depth", 2))
        self.tail_frac = float(params.get("tail_frac", 0.05))
        self.verbose = bool(params.get("verbose", False))

    def _build(self, X: np.ndarray, idx: np.ndarray, depth: int) -> _RuleNode:
        node = _RuleNode(count=int(idx.size))
        if depth >= self.max_depth or idx.size < 4:
            return node
        sub = X[idx]
        mu = sub.mean(axis=0)
        sd = sub.std(axis=0) + 1e-9
        z = np.abs((sub - mu) / sd)
        max_z_per_feature = z.max(axis=0)
        feature = int(np.argmax(max_z_per_feature))
        col = sub[:, feature]
        most_extreme_row = int(np.argmax(z[:, feature]))
        side = 1 if (col[most_extreme_row] - mu[feature]) >= 0 else -1
        if side >= 0:
            threshold = float(np.quantile(col, 1.0 - self.tail_frac))
            mask = col > threshold
        else:
            threshold = float(np.quantile(col, self.tail_frac))
            mask = col < threshold
        extreme_idx, normal_idx = idx[mask], idx[~mask]
        if extreme_idx.size == 0 or normal_idx.size == 0:
            return node
        node.leaf = False
        node.feature, node.threshold, node.side = feature, threshold, side
        node.extreme = self._build(X, extreme_idx, depth + 1)
        node.normal = self._build(X, normal_idx, depth + 1)
        return node

    def _assign_counts(self, node: _RuleNode, X: np.ndarray, mask: np.ndarray, out: np.ndarray) -> None:
        if node.leaf:
            out[mask] = node.count
            return
        col = X[:, node.feature]
        cond = (col > node.threshold) if node.side >= 0 else (col < node.threshold)
        self._assign_counts(node.extreme, X, mask & cond, out)
        self._assign_counts(node.normal, X, mask & ~cond, out)

    def _describe(self, node: _RuleNode, conds: list[str]) -> list[str]:
        if node.leaf:
            support = node.count / max(self.n_train_, 1)
            cond_str = " AND ".join(conds) if conds else "(root)"
            return [f"IF {cond_str} THEN leaf (support={support:.4f}, n={node.count})"]
        op_extreme = ">" if node.side >= 0 else "<"
        op_normal = "<=" if node.side >= 0 else ">="
        lines = self._describe(node.extreme, conds + [f"feature[{node.feature}] {op_extreme} {node.threshold:.6g}"])
        lines += self._describe(node.normal, conds + [f"feature[{node.feature}] {op_normal} {node.threshold:.6g}"])
        return lines

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kwargs: Any) -> None:
        X64 = X.astype(np.float64)
        self.n_train_ = int(X64.shape[0])
        self.tree_ = self._build(X64, np.arange(self.n_train_), depth=0)
        self.rules_ = self._describe(self.tree_, [])
        if self.verbose:
            for line in self.rules_:
                print(line)  # noqa: T201 - explicitly requested by the ladder ("rules printed")

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        X64 = X.astype(np.float64)
        counts = np.zeros(X64.shape[0], dtype=np.float64)
        self._assign_counts(self.tree_, X64, np.ones(X64.shape[0], dtype=bool), counts)
        return -np.log((counts + 1.0) / (self.n_train_ + 1.0))


# ----------------------------------------------------------------------------------------
# stacking_rf_xgb_logreg - CLS, cycle_features
# ----------------------------------------------------------------------------------------


@register("stacking_rf_xgb_logreg", input_kind="cycle_features", family="stacking", task="cls")
class StackingRFXGBLogReg(Classifier):
    """sklearn ``StackingClassifier``: RandomForest + XGBoost base learners, LogisticRegression
    meta-learner (ladder R68).

    Deviation: the ladder note attached to this row ("Spearman+VIF screening, F1-max
    threshold on validation only, SHAP audit, report PR-AUC with ROC-AUC") describes
    reporting/runner-level obligations around this model, not part of the
    ``BaseModel.fit``/``predict``/``predict_proba`` contract, so they are not implemented
    inside the estimator itself.

    Params:
        rf_n_estimators: int = 200
        rf_max_depth: int | None = None
        xgb_n_estimators: int = 150
        xgb_max_depth: int = 4
        xgb_learning_rate: float = 0.1
        stack_cv: int = 3      -- internal StackingClassifier folds for the meta-features
        n_jobs: int = -1       -- passed to RF, XGBoost and the StackingClassifier's own cv
        seed: int = 0
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.rf_n_estimators = int(params.get("rf_n_estimators", 200))
        self.rf_max_depth = params.get("rf_max_depth")
        self.xgb_n_estimators = int(params.get("xgb_n_estimators", 150))
        self.xgb_max_depth = int(params.get("xgb_max_depth", 4))
        self.xgb_learning_rate = float(params.get("xgb_learning_rate", 0.1))
        self.stack_cv = int(params.get("stack_cv", 3))
        self.n_jobs = int(params.get("n_jobs", -1))
        self.seed = int(params.get("seed", 0))

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        codes = _encode_labels(self.classes_, y)
        rf = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )
        xgb = XGBClassifier(
            n_estimators=self.xgb_n_estimators,
            max_depth=self.xgb_max_depth,
            learning_rate=self.xgb_learning_rate,
            random_state=self.seed,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=self.n_jobs,
        )
        clf = StackingClassifier(
            estimators=[("rf", rf), ("xgb", xgb)],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=self.stack_cv,
            n_jobs=self.n_jobs,
        )
        clf.fit(X, codes)
        self.model_ = clf

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        sub_classes = self.classes_[np.asarray(self.model_.classes_, dtype=np.int64)]
        return _align_proba(self.classes_, sub_classes, self.model_.predict_proba(X))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return _decode_labels(self.classes_, self.model_.predict(X))


# ----------------------------------------------------------------------------------------
# random_forest_cycle - CLS, cycle_features
# ----------------------------------------------------------------------------------------


@register("random_forest_cycle", input_kind="cycle_features", family="tree_ensemble", task="cls")
class RandomForestCycle(Classifier):
    """sklearn ``RandomForestClassifier`` over cycle_features (ladder R80/R68).

    Params:
        n_estimators: int = 300
        max_depth: int | None = None
        min_samples_leaf: int = 1
        n_jobs: int = -1
        seed: int = 0
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.n_estimators = int(params.get("n_estimators", 300))
        self.max_depth = params.get("max_depth")
        self.min_samples_leaf = int(params.get("min_samples_leaf", 1))
        self.n_jobs = int(params.get("n_jobs", -1))
        self.seed = int(params.get("seed", 0))

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        codes = _encode_labels(self.classes_, y)
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.seed,
        )
        self.model_.fit(X, codes)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        sub_classes = self.classes_[np.asarray(self.model_.classes_, dtype=np.int64)]
        return _align_proba(self.classes_, sub_classes, self.model_.predict_proba(X))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return _decode_labels(self.classes_, self.model_.predict(X))


# ----------------------------------------------------------------------------------------
# logreg_cycle / logreg_envelope - CLS, cycle_features / window_stats
# ----------------------------------------------------------------------------------------


class _LogRegBase(Classifier):
    """Shared body for the two standardise-then-logistic-regression baselines."""

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.C = float(params.get("C", 1.0))
        self.max_iter = int(params.get("max_iter", 1000))
        self.seed = int(params.get("seed", 0))

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        codes = _encode_labels(self.classes_, y)
        self.model_ = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=self.seed),
        )
        self.model_.fit(X, codes)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        clf = self.model_[-1]
        sub_classes = self.classes_[np.asarray(clf.classes_, dtype=np.int64)]
        return _align_proba(self.classes_, sub_classes, self.model_.predict_proba(X))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return _decode_labels(self.classes_, self.model_.predict(X))


@register("logreg_cycle", input_kind="cycle_features", family="linear", task="cls")
class LogRegCycle(_LogRegBase):
    """Standardise + ``LogisticRegression`` over cycle_features (ladder R80).

    Params: C: float = 1.0, max_iter: int = 1000, seed: int = 0
    """


@register("logreg_envelope", input_kind="window_stats", family="linear", task="cls")
class LogRegEnvelope(_LogRegBase):
    """Standardise + ``LogisticRegression`` over bearing window_stats+envelope features
    (ladder R117).

    Params: C: float = 1.0, max_iter: int = 1000, seed: int = 0
    """


# ----------------------------------------------------------------------------------------
# conformal_threshold - AD, window_stats (calibration wrapper)
# ----------------------------------------------------------------------------------------


@register("conformal_threshold", input_kind="window_stats", family="calibration")
class ConformalThreshold(AnomalyDetector):
    """Split-conformal calibration wrapper around another registered anomaly detector
    (ladder R42).

    IMPORTANT - "must only see validation": the runner's ``_fit_model`` hands every model's
    ``fit()`` the run's TRAIN slice only, never the run's real validation or test slices (that
    firewall lives in ``nebulax/bench/runner.py`` and this model does not - cannot - reach
    around it). So "calibrate on validation" is honoured here as classic split-conformal
    prediction: ``_fit`` itself carves its OWN train slice into an inner-fit part and a
    held-out "calibration" tail (``calibration_frac``, chronological by ``t`` when given, else
    the tail of the row order as passed in), fits ``inner`` on the fit part only, and scores
    the calibration tail once to build the nonconformity distribution. The genuine
    val/test slices the runner calibrates the operating threshold on are never touched by this
    class - that remains ``nebulax.bench.thresholds.calibrate``'s job.

    ``score(X)`` returns ``1 - p`` where ``p`` is the (one-sided, "large score is anomalous")
    conformal p-value of the inner score against the stored calibration distribution:
    ``p = (1 + #{calibration scores >= s}) / (1 + n_calibration)``. Higher output = smaller
    p-value = more anomalous, preserving the AD contract.

    Params:
        inner: str | nebulax.bench.base.AnomalyDetector = "isolation_forest" -- either the
            registered name of another window_stats AD model (built lazily via
            ``nebulax.bench.registry.build`` with ``inner_params``; the default names a
            ladder row that shares this row's ``window_stats`` input_kind) or an
            already-constructed ``AnomalyDetector`` instance (handy for tests that do not want
            to depend on the full model registry being populated).
        inner_params: dict | None = None   -- kwargs for ``build(inner, **inner_params)``
        calibration_frac: float = 0.2
        seed: int = 0
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.inner = params.get("inner", "isolation_forest")
        self.inner_params = dict(params.get("inner_params") or {})
        self.calibration_frac = float(params.get("calibration_frac", 0.2))
        self.seed = int(params.get("seed", 0))

    def _resolve_inner(self) -> AnomalyDetector:
        model = self.inner if isinstance(self.inner, BaseModel) else _build_registered(
            str(self.inner), **self.inner_params
        )
        if not isinstance(model, AnomalyDetector):
            raise TypeError(
                f"conformal_threshold: inner model {self.inner!r} is not an AnomalyDetector "
                f"(got {type(model).__name__})"
            )
        return model

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kwargs: Any) -> None:
        n = X.shape[0]
        n_cal = max(1, int(round(n * self.calibration_frac)))
        if n - n_cal < 1:
            raise ValueError(
                f"conformal_threshold: {n} training rows is too few for calibration_frac="
                f"{self.calibration_frac} (need >=1 fit row and >=1 calibration row)"
            )
        order = np.argsort(t) if t is not None else np.arange(n)
        fit_idx, cal_idx = order[: n - n_cal], order[n - n_cal :]
        self.inner_ = self._resolve_inner()
        t_fit = t[fit_idx] if t is not None else None
        t_cal = t[cal_idx] if t is not None else None
        self.inner_.fit(X[fit_idx], t_fit)
        cal_scores = self.inner_.score(X[cal_idx], t_cal)
        self.calib_scores_ = np.sort(np.asarray(cal_scores, dtype=np.float64))
        self.n_calibration_ = int(self.calib_scores_.size)

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        s = np.asarray(self.inner_.score(X, t), dtype=np.float64)
        # count(calib >= s) = n_cal - searchsorted(calib_sorted, s, side='left')
        n_ge = self.n_calibration_ - np.searchsorted(self.calib_scores_, s, side="left")
        p = (1.0 + n_ge) / (1.0 + self.n_calibration_)
        return 1.0 - p
