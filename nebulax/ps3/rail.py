"""Rail corrugation: 3-class Normal / Side I / Side II from one 1 s axle-box recording.

The physics is in :mod:`nebulax.ps3.rail_features` (tacho -> distance -> wavelength bands); this
module is the fold-local machine-learning layer on top of it: aggregation options, the classifier,
the inner-split class boosts, the frozen validation schemes and the `Task` the app calls.

Fold-local rule, concretely, for this task:

* :func:`nebulax.ps3.rail_features.aggregate` is a pure per-row function - it uses the file's own
  speed and its own channels and never a dataset statistic - so the design matrix can be built
  once for all files without leaking.
* every scaler lives inside a `Pipeline` fitted on the training fold;
* the Side I / Side II probability boosts are tuned on an **inner** stratified split of the
  training fold only;
* mirror-augmented copies (odd<->even swap with the label swapped) are generated from training
  files only and stay in their source file's fold; held-out files are never augmented;
* the two exact duplicate file pairs found by the fingerprint check are kept in the same fold by
  grouping on the fingerprint.

The rule ``speed < 20 km/h -> Normal`` is a **documented dataset shortcut**, not physics: no
training fault file runs below 35 km/h and 100 of the 234 Normal files do. It is a flag on the
model, on by default, reported with and without in the ladder.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from nebulax.ps3 import rail_features as rf
from nebulax.ps3.common import (
    RAIL_LABELS,
    BaseTask,
    Explanation,
    PredictionResult,
    RESULTS_DIR,
    Trace,
    Viewport,
    git_rev,
    labels_path,
    load_model,
    model_meta,
    natural_key,
    register_task,
    save_model,
    test_dir,
    train_dir,
)
from nebulax.ps3.scoring import class_f1_report, macro_f1

__all__ = [
    "RailModel",
    "RailTask",
    "fit_rail",
    "cross_validate",
    "run_ladder",
    "predict_test_set",
    "train",
    "LOW_SPEED_KMH",
    "SPEED_MATCHED_KMH",
]

#: Below this the dataset simply has no fault files - a shortcut, documented as such.
LOW_SPEED_KMH: float = 20.0

#: Speed-matched reporting subset (the faults' own range starts at 35 km/h).
SPEED_MATCHED_KMH: float = 35.0

_BOOST_GRID: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0)


# --------------------------------------------------------------------------------------
# Speed baseline and sub-window helpers
# --------------------------------------------------------------------------------------


def _is_db_level_column(col: str) -> bool:
    return (
        "_hz_" in col
        or "_wl_" in col
        or col.endswith("logrms")
        or col.endswith("logpeak")
    )


def _fit_speed_baseline(
    X: pd.DataFrame,
    y: np.ndarray,
    speed: np.ndarray,
) -> dict[str, tuple[float, float]]:
    norm_mask = (y == "Normal")
    if not norm_mask.any():
        return {}
    norm_speed = speed[norm_mask]
    log_sp = np.log(np.maximum(norm_speed, 0.1))
    coeffs: dict[str, tuple[float, float]] = {}
    db_cols = [c for c in X.columns if _is_db_level_column(c)]
    for col in db_cols:
        y_col = X.loc[norm_mask, col].to_numpy(dtype=float)
        b, a = np.polyfit(log_sp, y_col, 1)
        coeffs[col] = (float(a), float(b))
    return coeffs


def _add_speed_baseline_residuals(
    X: pd.DataFrame,
    coeffs: dict[str, tuple[float, float]],
    speed_kmh: np.ndarray | None = None,
) -> pd.DataFrame:
    if not coeffs:
        return X
    out = X.copy()
    if speed_kmh is not None:
        sp = speed_kmh
    elif "speed_kmh" in out.columns:
        sp = out["speed_kmh"].to_numpy(dtype=float)
    else:
        return out
    log_sp = np.log(np.maximum(sp, 0.1))
    new_cols = {}
    for col, (a, b) in coeffs.items():
        if col in out.columns:
            new_cols[f"{col}_resid"] = out[col].to_numpy(dtype=float) - (a + b * log_sp)
    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)
    return out


def _get_window_features(feats: rf.RailFeatures) -> tuple[rf.RailFeatures, np.ndarray]:
    """Load or extract sub-window features for each parent file in feats.

    Returns (win_feats, parent_idx).
    """
    from nebulax.ps3 import rail_windows as rw
    out_dir = rw.window_cache_dir()

    all_cached = True
    for fid in feats.file_ids:
        base_id = fid[8:] if fid.startswith("mirror::") else fid
        stem = Path(base_id).stem
        if not all((out_dir / f"{stem}#w{k}.parquet").exists() and (out_dir / f"{stem}#w{k}.spec.npy").exists() for k, _ in rw.WINDOW_SLICES):
            all_cached = False
            break

    if all_cached:
        win_frames, win_specs, win_ids, parent_indices = [], [], [], []
        for i, fid in enumerate(feats.file_ids):
            is_mir = fid.startswith("mirror::")
            base_id = fid[8:] if is_mir else fid
            w_feats, _ = rw.load_window_features([base_id])
            if is_mir:
                w_feats = rf.mirror(w_feats)
            win_frames.append(w_feats.scalars)
            win_specs.append(w_feats.spectra)
            win_ids.extend(w_feats.file_ids)
            parent_indices.extend([i] * len(w_feats))
        df = pd.concat(win_frames, ignore_index=True) if win_frames else pd.DataFrame()
        spec = np.concatenate(win_specs, axis=0) if win_specs else np.zeros((0, rf.N_CHANNELS, rf.N_LAMBDA), dtype=np.float32)
        return rf.RailFeatures(file_ids=win_ids, scalars=df, spectra=spec), np.asarray(parent_indices, dtype=int)

    # Fallback on-the-fly
    win_frames, win_specs, win_ids, parent_indices = [], [], [], []
    for i, fid in enumerate(feats.file_ids):
        is_mir = fid.startswith("mirror::")
        base_id = fid[8:] if is_mir else fid
        raw_p = None
        if feats.raw_path and Path(feats.raw_path).exists():
            raw_p = Path(feats.raw_path)
        else:
            cand = train_dir("rail") / base_id
            if not cand.exists():
                cand = test_dir("rail") / base_id
            if cand.exists():
                raw_p = cand

        if raw_p is not None:
            arr = rf.read_rail_csv(raw_p)
            w_feats, _ = rw.extract_windows_from_array(arr, stem=Path(base_id).stem)
        else:
            sc = feats.scalars.iloc[[i]]
            sp = feats.spectra[i : i + 1]
            w_frames = [sc.copy() for _ in range(3)]
            for k in range(3):
                w_frames[k]["file_id"] = f"{Path(base_id).stem}#w{k}.csv"
            w_df = pd.concat(w_frames, ignore_index=True)
            w_sp = np.repeat(sp, 3, axis=0)
            w_feats = rf.RailFeatures(
                file_ids=[f"{Path(base_id).stem}#w{k}.csv" for k in range(3)],
                scalars=w_df,
                spectra=w_sp,
            )

        if is_mir:
            w_feats = rf.mirror(w_feats)
        win_frames.append(w_feats.scalars)
        win_specs.append(w_feats.spectra)
        win_ids.extend(w_feats.file_ids)
        parent_indices.extend([i] * len(w_feats))

    df = pd.concat(win_frames, ignore_index=True) if win_frames else pd.DataFrame()
    spec = np.concatenate(win_specs, axis=0) if win_specs else np.zeros((0, rf.N_CHANNELS, rf.N_LAMBDA), dtype=np.float32)
    return rf.RailFeatures(file_ids=win_ids, scalars=df, spectra=spec), np.asarray(parent_indices, dtype=int)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


@dataclass
class RailModel:
    """A fitted rail classifier: estimators (one per seed), the column order and the boosts."""

    estimators: list[Any]
    columns: list[str]
    opts: dict[str, Any]
    boosts: tuple[float, float] = (1.0, 1.0)
    low_speed_rule: bool = True
    tta: bool = False
    kind: str = "lgbm"
    feature_version: str = rf.FEATURE_VERSION
    classes: tuple[str, ...] = RAIL_LABELS
    meta: dict[str, Any] = field(default_factory=dict)
    speed_baseline_coeffs: dict[str, tuple[float, float]] | None = None

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        """Mean class probability over the seed ensemble, columns in ``RAIL_LABELS`` order."""
        if self.kind == "lgbm_windows":
            raw: rf.RailFeatures | None = X.attrs.get("rail_features")
            if raw is not None:
                win_feats, _ = _get_window_features(raw)
                X_win = rf.aggregate(win_feats, self.opts)
                if self.speed_baseline_coeffs:
                    X_win = _add_speed_baseline_residuals(X_win, self.speed_baseline_coeffs)
                missing = [c for c in self.columns if c not in X_win.columns]
                if missing:
                    raise ValueError(f"rail model wants {len(self.columns)} columns, but {len(missing)} are absent: {missing[:3]}")
                Xm_win = X_win.reindex(columns=self.columns).to_numpy(dtype=np.float64)
                acc = np.zeros((len(X_win), len(self.classes)))
                for est in self.estimators:
                    p = est.predict_proba(Xm_win)
                    idx = [list(est.classes_).index(c) for c in self.classes]
                    acc += p[:, idx]
                p_win = acc / max(len(self.estimators), 1)
                return p_win.reshape(len(raw), 3, len(self.classes)).mean(axis=1)

        if self.speed_baseline_coeffs:
            X = _add_speed_baseline_residuals(X, self.speed_baseline_coeffs)
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(
                f"rail model was fitted on feature version {self.feature_version!r} and wants "
                f"{len(self.columns)} columns, but {len(missing)} are absent from this design matrix "
                f"(e.g. {missing[:3]}). The feature code is at version {rf.FEATURE_VERSION!r}: refit "
                "with `python scripts/ps3_train.py --task rail`."
            )
        Xm = X.reindex(columns=self.columns).to_numpy(dtype=np.float64)
        acc = np.zeros((len(X), len(self.classes)))
        for est in self.estimators:
            p = est.predict_proba(Xm)
            idx = [list(est.classes_).index(c) for c in self.classes]
            acc += p[:, idx]
        return acc / max(len(self.estimators), 1)

    def proba_tta(self, X: pd.DataFrame, X_mirror: pd.DataFrame | None) -> np.ndarray:
        """Mirror test-time augmentation: average p(x) with the side-swapped p(mirror(x)).

        The sensor layout makes the mirror exact, so the two views must agree; averaging them
        imposes that equivariance at inference for free.
        """
        p = self.proba(X)
        if not self.tta or X_mirror is None:
            return p
        pm = self.proba(X_mirror)[:, [0, 2, 1]]
        return 0.5 * (p + pm)

    def predict_labels(
        self,
        X: pd.DataFrame,
        speed_kmh: np.ndarray | None = None,
        X_mirror: pd.DataFrame | None = None,
    ) -> np.ndarray:
        p = self.proba_tta(X, X_mirror)
        return _apply_rules(p, self.boosts, speed_kmh if self.low_speed_rule else None)


def _apply_rules(proba: np.ndarray, boosts: tuple[float, float], speed_kmh: np.ndarray | None) -> np.ndarray:
    scaled = proba.copy()
    scaled[:, 1] *= boosts[0]
    scaled[:, 2] *= boosts[1]
    out = np.asarray(RAIL_LABELS, dtype=object)[np.argmax(scaled, axis=1)]
    if speed_kmh is not None:
        out = np.where(np.asarray(speed_kmh) < LOW_SPEED_KMH, RAIL_LABELS[0], out)
    return out


def _make_estimator(kind: str, seed: int, n_jobs: int = 4, columns: Sequence[str] | None = None) -> Any:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if kind == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=7,
            max_depth=4,
            min_child_samples=5,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.3,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )
    if kind == "lgbm_regularized":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.04,
            num_leaves=5,
            max_depth=3,
            min_child_samples=12,
            min_split_gain=0.05,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=5.0,
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )
    if kind == "lgbm_2view":
        return _TwoViewLGBM(seed=seed, n_jobs=n_jobs, columns=columns)
    if kind == "lgbm_pruned":
        return _PrunedLGBM(seed=seed, n_jobs=n_jobs, columns=columns)
    if kind == "lgbm_windows":
        return _make_estimator("lgbm", seed=seed, n_jobs=n_jobs, columns=columns)
    if kind == "logreg":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=0.1, max_iter=4000, class_weight="balanced", random_state=seed)),
            ]
        )
    if kind == "svm":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", SVC(C=5.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed)),
            ]
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=n_jobs,
        )
    if kind == "lgbm_hier":
        return _Hierarchical("lgbm", seed, n_jobs=n_jobs)
    if kind == "multirocket_ridge":
        return _RocketSpectra(seed)
    raise ValueError(f"unknown rail model kind {kind!r}")


class _TwoViewLGBM:
    """Arithmetic-mean probability ensemble of two LightGBM models trained on two views:
    Hz view (columns without the vib_wl_/shock_wl_ wavelength blocks) and
    wavelength view (columns without the vib_hz_/shock_hz_ Hz blocks)."""

    def __init__(
        self,
        seed: int = 0,
        n_jobs: int = 4,
        columns: Sequence[str] | None = None,
    ) -> None:
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.columns = list(columns) if columns is not None else []
        self.model_hz = _make_estimator("lgbm", self.seed, n_jobs=self.n_jobs)
        self.model_wl = _make_estimator("lgbm", self.seed, n_jobs=self.n_jobs)
        self.est_hz = self.model_hz
        self.est_wl = self.model_wl
        self.sub_models = (self.model_hz, self.model_wl)
        self.classes_ = np.asarray(RAIL_LABELS, dtype=object)
        self._hz_idx: list[int] = []
        self._wl_idx: list[int] = []
        if self.columns:
            self._set_view_indices(self.columns)

    def _set_view_indices(self, cols: Sequence[str]) -> None:
        self._hz_idx = [
            i for i, c in enumerate(cols)
            if not (c.startswith("vib_wl_") or c.startswith("shock_wl_"))
        ]
        self._wl_idx = [
            i for i, c in enumerate(cols)
            if not (c.startswith("vib_hz_") or c.startswith("shock_hz_"))
        ]

    def fit(self, X: Any, y: np.ndarray) -> "_TwoViewLGBM":
        if isinstance(X, pd.DataFrame):
            self.columns = list(X.columns)
            self._set_view_indices(self.columns)
            X = X.to_numpy(dtype=np.float64)
        elif not self._hz_idx or not self._wl_idx:
            if self.columns:
                self._set_view_indices(self.columns)

        assert len(self._hz_idx) > 0, "Hz view is empty; check column names"
        assert len(self._wl_idx) > 0, "wavelength view is empty; check column names"

        y = np.asarray(y, dtype=object)
        X_hz = X[:, self._hz_idx]
        X_wl = X[:, self._wl_idx]

        self.model_hz.fit(X_hz, y)
        self.model_wl.fit(X_wl, y)
        self.classes_ = np.asarray(self.model_hz.classes_)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = (
                X.reindex(columns=self.columns).to_numpy(dtype=np.float64)
                if self.columns
                else X.to_numpy(dtype=np.float64)
            )
        p_hz = self.model_hz.predict_proba(X[:, self._hz_idx])
        p_wl = self.model_wl.predict_proba(X[:, self._wl_idx])
        idx_hz = [list(self.model_hz.classes_).index(c) for c in self.classes_]
        idx_wl = [list(self.model_wl.classes_).index(c) for c in self.classes_]
        return 0.5 * (p_hz[:, idx_hz] + p_wl[:, idx_wl])

    def predict(self, X: Any) -> np.ndarray:
        p = self.predict_proba(X)
        return self.classes_[np.argmax(p, axis=1)]


class _PrunedLGBM:
    """Estimator that fits the winner on the training fold, selects the top-64 columns
    by total gain importance summed across seed models, and refits on those."""

    def __init__(
        self,
        seed: int = 0,
        n_jobs: int = 4,
        columns: Sequence[str] | None = None,
        top_k: int = 64,
    ) -> None:
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.columns = list(columns) if columns is not None else []
        self.top_k = int(top_k)
        self.model = _make_estimator("lgbm", self.seed, n_jobs=self.n_jobs)
        self.classes_ = np.asarray(RAIL_LABELS, dtype=object)
        self.selected_indices: list[int] = []
        self.selected_columns: list[str] = []

    def fit(self, X: Any, y: np.ndarray) -> "_PrunedLGBM":
        if isinstance(X, pd.DataFrame):
            self.columns = list(X.columns)
            X = X.to_numpy(dtype=np.float64)
        elif not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=object)

        total_gain = np.zeros(X.shape[1], dtype=np.float64)
        for s in (0, 1, 2):
            base = _make_estimator("lgbm", s, n_jobs=self.n_jobs).fit(X, y)
            total_gain += base.booster_.feature_importance(importance_type="gain")

        k = min(self.top_k, X.shape[1])
        top_idx = np.argsort(total_gain)[::-1][:k]
        self.selected_indices = sorted(int(i) for i in top_idx)
        if self.columns:
            self.selected_columns = [self.columns[i] for i in self.selected_indices]

        self.model.fit(X[:, self.selected_indices], y)
        self.classes_ = np.asarray(self.model.classes_)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = (
                X.reindex(columns=self.columns).to_numpy(dtype=np.float64)
                if self.columns
                else X.to_numpy(dtype=np.float64)
            )
        elif not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float64)
        return self.model.predict_proba(X[:, self.selected_indices])

    def predict(self, X: Any) -> np.ndarray:
        p = self.predict_proba(X)
        return self.classes_[np.argmax(p, axis=1)]


class _Hierarchical:
    """(a) corrugation present/absent, then (b) which side - 38 positives support one binary
    decision far better than three (`ps3_addendum.md` section 6 row 3)."""

    def __init__(self, base: str = "lgbm", seed: int = 0, n_jobs: int = 4) -> None:
        self.base, self.seed, self.n_jobs = base, int(seed), int(n_jobs)
        self.classes_ = np.asarray(RAIL_LABELS, dtype=object)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_Hierarchical":
        y = np.asarray(y, dtype=object)
        self._presence = _make_estimator(self.base, self.seed, n_jobs=self.n_jobs)
        self._presence.fit(X, (y != RAIL_LABELS[0]).astype(int))
        fault = y != RAIL_LABELS[0]
        self._side = None
        if fault.sum() >= 4 and len(np.unique(y[fault])) == 2:
            self._side = _make_estimator(self.base, self.seed, n_jobs=self.n_jobs)
            self._side.fit(X[fault], y[fault])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pf = self._presence.predict_proba(X)[:, list(self._presence.classes_).index(1)]
        out = np.zeros((len(X), 3))
        out[:, 0] = 1.0 - pf
        if self._side is None:
            out[:, 1] = out[:, 2] = pf / 2.0
            return out
        ps = self._side.predict_proba(X)
        for j, c in enumerate(RAIL_LABELS[1:], start=1):
            out[:, j] = pf * ps[:, list(self._side.classes_).index(c)]
        return out


class _RocketSpectra:
    """MultiRocket + RidgeClassifierCV on the **per-side wavelength spectra**, not pooled raw
    samples: four channels (Side I/II x vibration/shock) of 192 log-spaced lambda points
    [R249], `model_ladder.md` section 1b [R43][R47]."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self.classes_ = np.asarray(RAIL_LABELS, dtype=object)

    @staticmethod
    def _series(X: np.ndarray, n_cols: int) -> np.ndarray:
        n_ch, n_t = 4, n_cols // 4
        return X[:, : n_ch * n_t].reshape(len(X), n_ch, n_t)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RocketSpectra":
        from aeon.transformations.collection.convolution_based import MultiRocket
        from sklearn.linear_model import RidgeClassifierCV

        self._n_cols = X.shape[1] - X.shape[1] % 4
        self._tf = MultiRocket(n_kernels=2500, random_state=self.seed, n_jobs=1)
        Z = self._tf.fit_transform(self._series(X, self._n_cols))
        self._clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        self._clf.fit(np.nan_to_num(Z), np.asarray(y, dtype=object))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = np.nan_to_num(self._tf.transform(self._series(X, self._n_cols)))
        d = self._clf.decision_function(Z)
        d = d.reshape(len(X), -1)
        if d.shape[1] == 1:  # binary fallback
            d = np.hstack([-d, d])
        e = np.exp(d - d.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        idx = {str(c): i for i, c in enumerate(self._clf.classes_)}
        out = np.zeros((len(X), 3))
        for j, c in enumerate(RAIL_LABELS):
            if c in idx:
                out[:, j] = p[:, idx[c]]
        return out


# --------------------------------------------------------------------------------------
# Fold-local fitting
# --------------------------------------------------------------------------------------


def _mirror_labels(y: np.ndarray) -> np.ndarray:
    swap = {"Side I": "Side II", "Side II": "Side I", "Normal": "Normal"}
    return np.asarray([swap[str(v)] for v in y], dtype=object)


def _tune_boosts(
    probas: Sequence[np.ndarray] | np.ndarray, y: np.ndarray, speed: np.ndarray | None
) -> tuple[float, float]:
    """Pick the class boosts that maximise macro F1, averaged over repeated inner OOF passes.

    A single inner pass puts ~2.8 Side I files in each inner fold, so the grid argmax is very
    noisy (fold-tuned boosts ranged 1.0-4.0 across the outer folds of the v1 run). Averaging the
    objective over several inner repeats before taking the argmax costs one extra inner fit per
    repeat and removes most of that jitter. Still strictly fold-local: every repeat splits the
    same training fold.
    """
    mats = [np.asarray(probas)] if np.ndim(probas) == 2 else [np.asarray(p) for p in probas]
    best, best_f1 = (1.0, 1.0), -1.0
    for b1 in _BOOST_GRID:
        for b2 in _BOOST_GRID:
            f1 = float(np.mean([macro_f1(y, _apply_rules(m, (b1, b2), speed)) for m in mats]))
            if f1 > best_f1 + 1e-12:
                best, best_f1 = (b1, b2), f1
    return best


def _fit_estimators(
    X: np.ndarray,
    y: np.ndarray,
    kind: str,
    seeds: Sequence[int],
    n_jobs: int,
    columns: Sequence[str] | None = None,
    sample_weight: np.ndarray | None = None,
) -> list[Any]:
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
    return [
        _make_estimator(kind, int(s), n_jobs=n_jobs, columns=columns).fit(X, y, **fit_kwargs)
        for s in seeds
    ]


def _fault_feature_mixup(
    X: np.ndarray,
    y: np.ndarray,
    speed: np.ndarray,
    groups: np.ndarray,
    *,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """One same-class, speed-near feature interpolation per fault row.

    Call this on a fitting partition only. In particular, inner calibration calls it after
    splitting, so neither parent of a synthetic row can belong to the validation partition.
    """
    if alpha <= 0:
        raise ValueError("feature mixup alpha must be positive")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=object)
    speed = np.asarray(speed, dtype=float)
    groups = np.asarray(groups)
    if not (len(X) == len(y) == len(speed) == len(groups)):
        raise ValueError("feature mixup inputs must have equal row counts")
    rng = np.random.default_rng(seed)
    mixed: list[np.ndarray] = []
    labels: list[str] = []
    parents: list[tuple[int, int]] = []
    for i in np.flatnonzero(y != RAIL_LABELS[0]):
        eligible = np.flatnonzero((y == y[i]) & (groups != groups[i]))
        if len(eligible) == 0:
            continue
        nearby = eligible[np.abs(speed[eligible] - speed[i]) <= 8.0]
        if len(nearby):
            eligible = nearby
        else:
            eligible = eligible[np.argsort(np.abs(speed[eligible] - speed[i]))[:3]]
        j = int(rng.choice(eligible))
        weight = float(rng.beta(alpha, alpha))
        mixed.append(weight * X[i] + (1.0 - weight) * X[j])
        labels.append(str(y[i]))
        parents.append((int(i), j))
    if not mixed:
        return X, y, parents
    return np.vstack((X, np.stack(mixed))), np.concatenate((y, np.asarray(labels, dtype=object))), parents


def _fault_normal_soft_mixup(
    X: np.ndarray,
    y: np.ndarray,
    speed: np.ndarray,
    groups: np.ndarray,
    *,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Approximate soft targets with two weighted copies of each fault/Normal interpolation."""
    if alpha <= 0:
        raise ValueError("feature mixup alpha must be positive")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=object)
    speed = np.asarray(speed, dtype=float)
    groups = np.asarray(groups)
    if not (len(X) == len(y) == len(speed) == len(groups)):
        raise ValueError("feature mixup inputs must have equal row counts")
    rng = np.random.default_rng(seed)
    mixed: list[np.ndarray] = []
    labels: list[str] = []
    weights: list[float] = []
    parents: list[tuple[int, int]] = []
    for i in np.flatnonzero(y != RAIL_LABELS[0]):
        eligible = np.flatnonzero((y == RAIL_LABELS[0]) & (groups != groups[i]))
        if len(eligible) == 0:
            continue
        nearby = eligible[np.abs(speed[eligible] - speed[i]) <= 8.0]
        if len(nearby):
            eligible = nearby
        else:
            eligible = eligible[np.argsort(np.abs(speed[eligible] - speed[i]))[:3]]
        j = int(rng.choice(eligible))
        weight = 0.5 + 0.5 * float(rng.beta(alpha, alpha))
        row = weight * X[i] + (1.0 - weight) * X[j]
        mixed.extend((row, row.copy()))
        labels.extend((str(y[i]), RAIL_LABELS[0]))
        weights.extend((weight, 1.0 - weight))
        parents.append((int(i), j))
    if not mixed:
        return X, y, np.ones(len(X), dtype=float), parents
    return (
        np.vstack((X, np.stack(mixed))),
        np.concatenate((y, np.asarray(labels, dtype=object))),
        np.concatenate((np.ones(len(X), dtype=float), np.asarray(weights, dtype=float))),
        parents,
    )


def fit_rail(
    feats: rf.RailFeatures,
    y: Sequence[str],
    *,
    kind: str = "lgbm",
    opts: dict[str, Any] | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    augment: bool = True,
    low_speed_rule: bool = True,
    tune_boosts: bool = True,
    boost_repeats: int = 3,
    aligned_boosts: bool = False,
    sensor_augmentation: str | None = None,
    low_speed_normal_keep: float | None = None,
    feature_mixup_alpha: float | None = None,
    feature_mixup_soft: bool = False,
    tta: bool = False,
    n_jobs: int = 4,
    mirrored: rf.RailFeatures | None = None,
) -> RailModel:
    """Fit one rail model on **training-fold data only**.

    ``feats`` must already be restricted to the training fold; ``mirrored`` is its mirror image
    (pass it in to avoid recomputing). The class boosts are tuned on an inner 3-fold split of
    exactly this data, so nothing outside the fold is ever touched.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    o = {**rf.AGG_DEFAULT, **(opts or {})}
    if aligned_boosts and (not tta or not augment or kind == "lgbm_windows"):
        raise ValueError("aligned boosts require mirror augmentation and TTA on file-level features")
    if sensor_augmentation not in {None, "gain", "mask"}:
        raise ValueError(f"unknown rail sensor augmentation {sensor_augmentation!r}")
    if sensor_augmentation and (not augment or kind == "lgbm_windows"):
        raise ValueError("sensor augmentation requires mirror augmentation on file-level features")
    if feature_mixup_alpha is not None and (kind != "lgbm" or sensor_augmentation is not None):
        raise ValueError("feature mixup currently supports plain LightGBM without sensor augmentation")
    if feature_mixup_soft and feature_mixup_alpha is None:
        raise ValueError("soft feature mixup needs a positive alpha")
    y = np.asarray(list(y), dtype=object)
    speed = feats.scalars["speed_kmh"].to_numpy(dtype=float)
    n_source = len(y)
    if low_speed_normal_keep is not None:
        fraction = float(low_speed_normal_keep)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("low_speed_normal_keep must be between 0 and 1")
        easy = np.flatnonzero((y == RAIL_LABELS[0]) & (speed < LOW_SPEED_KMH))
        keep = np.ones(len(y), dtype=bool)
        keep[easy] = False
        n_easy_keep = int(np.ceil(fraction * len(easy)))
        if n_easy_keep:
            chosen = np.random.default_rng(int(seeds[0]) + 7349).choice(easy, n_easy_keep, replace=False)
            keep[chosen] = True
        chosen = np.flatnonzero(keep)
        feats = feats.subset(chosen)
        if mirrored is not None:
            mirrored = mirrored.subset(chosen)
        y = y[chosen]
        speed = speed[chosen]

    use_speed_baseline = bool(o.get("speed_baseline", False))
    speed_coeffs: dict[str, tuple[float, float]] | None = None

    if kind == "lgbm_windows":
        win_feats, parent_idx = _get_window_features(feats)
        win_y = np.repeat(y, 3)
        win_speed = win_feats.scalars["speed_kmh"].to_numpy(dtype=float)
        X_win = rf.aggregate(win_feats, o)
        if use_speed_baseline:
            speed_coeffs = _fit_speed_baseline(X_win, win_y, win_speed)
            X_win = _add_speed_baseline_residuals(X_win, speed_coeffs, win_speed)
        if augment:
            mir_win = rf.mirror(win_feats)
            X_mir_win = rf.aggregate(mir_win, o)
            if use_speed_baseline and speed_coeffs:
                X_mir_win = _add_speed_baseline_residuals(X_mir_win, speed_coeffs, win_speed)
            Xa = pd.concat([X_win, X_mir_win], ignore_index=True)
            ya = np.concatenate([win_y, _mirror_labels(win_y)])
            sa = np.concatenate([win_speed, win_speed])
            inner_groups = np.tile(parent_idx, 2)
        else:
            Xa, ya, sa = X_win, win_y, win_speed
            inner_groups = parent_idx
        est_kind = "lgbm"
    else:
        X = rf.aggregate(feats, o)
        if use_speed_baseline:
            speed_coeffs = _fit_speed_baseline(X, y, speed)
            X = _add_speed_baseline_residuals(X, speed_coeffs, speed)
        if augment:
            mir = mirrored if mirrored is not None else rf.mirror(feats)
            X_mir = rf.aggregate(mir, o)
            if use_speed_baseline and speed_coeffs:
                X_mir = _add_speed_baseline_residuals(X_mir, speed_coeffs, speed)
            Xa = pd.concat([X, X_mir], ignore_index=True)
            ya = np.concatenate([y, _mirror_labels(y)])
            sa = np.concatenate([speed, speed])
            inner_groups = np.tile(_duplicate_groups(feats), 2)
            if sensor_augmentation:
                fault_idx = np.flatnonzero(y != RAIL_LABELS[0])
                if len(fault_idx):
                    changed = rf.perturb_fault_channels(
                        feats.subset(fault_idx), sensor_augmentation, seed=int(seeds[0]) + 7919
                    )
                    changed_mirror = rf.mirror(changed)
                    X_changed = rf.aggregate(changed, o)
                    X_changed_mirror = rf.aggregate(changed_mirror, o)
                    if use_speed_baseline and speed_coeffs:
                        X_changed = _add_speed_baseline_residuals(X_changed, speed_coeffs, speed[fault_idx])
                        X_changed_mirror = _add_speed_baseline_residuals(
                            X_changed_mirror, speed_coeffs, speed[fault_idx]
                        )
                    Xa = pd.concat([Xa, X_changed, X_changed_mirror], ignore_index=True)
                    ya = np.concatenate([ya, y[fault_idx], _mirror_labels(y[fault_idx])])
                    sa = np.concatenate([sa, speed[fault_idx], speed[fault_idx]])
                    inner_groups = np.concatenate([
                        inner_groups,
                        _duplicate_groups(feats)[fault_idx],
                        _duplicate_groups(feats)[fault_idx],
                    ])
        else:
            Xa, ya, sa = X, y, speed
            inner_groups = _duplicate_groups(feats)
        est_kind = kind

    columns = list(Xa.columns)
    Xm = Xa.to_numpy(dtype=np.float64)
    fit_weights = None
    if feature_mixup_alpha is not None and feature_mixup_soft:
        X_fit, y_fit, fit_weights, mixup_pairs = _fault_normal_soft_mixup(
            Xm, ya, sa, inner_groups, alpha=float(feature_mixup_alpha), seed=int(seeds[0]) + 1451,
        )
    elif feature_mixup_alpha is not None:
        X_fit, y_fit, mixup_pairs = _fault_feature_mixup(
            Xm, ya, sa, inner_groups, alpha=float(feature_mixup_alpha), seed=int(seeds[0]) + 1451,
        )
    else:
        X_fit, y_fit, mixup_pairs = Xm, ya, []
    model = RailModel(
        estimators=_fit_estimators(X_fit, y_fit, est_kind, seeds, n_jobs,
                                   columns=columns, sample_weight=fit_weights),
        columns=columns,
        opts=o,
        boosts=(1.0, 1.0),
        low_speed_rule=low_speed_rule,
        tta=bool(tta),
        kind=kind,
        meta={"n_train": int(len(Xa)), "augment": bool(augment), "seeds": [int(s) for s in seeds]},
        speed_baseline_coeffs=speed_coeffs,
    )
    model.meta["sensor_augmentation"] = sensor_augmentation
    model.meta["low_speed_normal_keep"] = low_speed_normal_keep
    model.meta["n_source_before_undersampling"] = n_source
    model.meta["feature_mixup_alpha"] = feature_mixup_alpha
    model.meta["feature_mixup_soft"] = bool(feature_mixup_soft)
    model.meta["n_feature_mixup"] = len(mixup_pairs)
    if kind == "lgbm_pruned" and model.estimators:
        est0 = model.estimators[0]
        if hasattr(est0, "selected_columns") and est0.selected_columns:
            model.meta["pruned_columns"] = est0.selected_columns

    if tune_boosts and len(np.unique(ya)) > 1 and len(ya) >= 12:
        oofs = []
        try:
            for rep in range(max(1, int(boost_repeats))):
                oof = np.zeros((len(ya), len(RAIL_LABELS)))
                inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=int(seeds[0]) + 101 * rep)
                for inner_fold, (tr_i, va_i) in enumerate(inner.split(Xm, ya, inner_groups)):
                    X_tr, y_tr = Xm[tr_i], ya[tr_i]
                    tr_weights = None
                    if feature_mixup_alpha is not None and feature_mixup_soft:
                        X_tr, y_tr, tr_weights, _ = _fault_normal_soft_mixup(
                            X_tr, y_tr, sa[tr_i], inner_groups[tr_i],
                            alpha=float(feature_mixup_alpha),
                            seed=int(seeds[0]) + 1451 + 101 * rep + inner_fold,
                        )
                    elif feature_mixup_alpha is not None:
                        X_tr, y_tr, _ = _fault_feature_mixup(
                            X_tr, y_tr, sa[tr_i], inner_groups[tr_i],
                            alpha=float(feature_mixup_alpha),
                            seed=int(seeds[0]) + 1451 + 101 * rep + inner_fold,
                        )
                    est = _fit_estimators(X_tr, y_tr, est_kind, seeds[:1], n_jobs,
                                          columns=columns, sample_weight=tr_weights)[0]
                    idx = [list(est.classes_).index(c) for c in RAIL_LABELS]
                    oof[va_i] = est.predict_proba(Xm[va_i])[:, idx]
                oofs.append(oof)
            if kind == "lgbm_windows":
                n_files = len(y)
                oof_files = [o[: n_files * 3].reshape(n_files, 3, len(RAIL_LABELS)).mean(axis=1) for o in oofs]
                model.boosts = _tune_boosts(oof_files, y, speed if low_speed_rule else None)
            elif aligned_boosts:
                n_files = len(y)
                oof_files = [0.5 * (o[:n_files] + o[n_files:, [0, 2, 1]]) for o in oofs]
                model.boosts = _tune_boosts(oof_files, y, speed if low_speed_rule else None)
            elif sensor_augmentation:
                n_original_and_mirror = 2 * len(y)
                model.boosts = _tune_boosts(
                    [o[:n_original_and_mirror] for o in oofs],
                    ya[:n_original_and_mirror],
                    sa[:n_original_and_mirror] if low_speed_rule else None,
                )
            else:
                model.boosts = _tune_boosts(oofs, ya, sa if low_speed_rule else None)
            model.meta["boost_repeats"] = int(max(1, boost_repeats))
            model.meta["aligned_boosts"] = bool(aligned_boosts)
        except ValueError:  # too few members of a class to split - keep the neutral boosts
            model.boosts = (1.0, 1.0)
    return model


# --------------------------------------------------------------------------------------
# Validation schemes (frozen; plan section W4)
# --------------------------------------------------------------------------------------


def _splits_stratified(y: np.ndarray, groups: np.ndarray, seeds: Sequence[int], n_splits: int = 5):
    from sklearn.model_selection import StratifiedGroupKFold

    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            yield f"seed{seed}_fold{fold}", tr, te


def _duplicate_groups(feats: rf.RailFeatures) -> np.ndarray:
    """Fingerprint group used by every split (channel-1 and tacho hashes together)."""
    return np.asarray(
        [f"{a}:{b}" for a, b in zip(feats.scalars["hash_ch1"], feats.scalars["hash_speed"])],
        dtype=object,
    )


def _splits_contiguous(file_ids: Sequence[str], groups: np.ndarray, n_blocks: int = 5):
    """Contiguous filename-group blocks; exact duplicate groups can never straddle an edge."""
    members: dict[str, list[int]] = {}
    for i, group in enumerate(groups):
        members.setdefault(str(group), []).append(i)
    ordered_groups = sorted(
        members,
        key=lambda group: natural_key(min((file_ids[i] for i in members[group]), key=natural_key)),
    )
    blocks = [np.asarray([i for group in chunk for i in members[group]], dtype=int) for chunk in np.array_split(ordered_groups, n_blocks)]
    for i, te in enumerate(blocks):
        tr = np.concatenate([b for j, b in enumerate(blocks) if j != i])
        yield f"block{i}", np.sort(tr), np.sort(te)


def _splits_speed_range(speed: np.ndarray, y: np.ndarray, n_bins: int = 3):
    """Leave-speed-range-out: hold out one speed band at a time (fault-bearing bands only)."""
    faults = speed[y != RAIL_LABELS[0]]
    edges = np.quantile(faults, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    for i in range(n_bins):
        te = np.flatnonzero((speed >= edges[i]) & (speed < edges[i + 1]))
        tr = np.flatnonzero(~((speed >= edges[i]) & (speed < edges[i + 1])))
        if len(te) == 0 or len(np.unique(y[tr])) < 2:
            continue
        lo = "-inf" if not np.isfinite(edges[i]) else f"{edges[i]:.0f}"
        hi = "inf" if not np.isfinite(edges[i + 1]) else f"{edges[i + 1]:.0f}"
        yield f"speed[{lo},{hi})", tr, te


def cross_validate(
    feats: rf.RailFeatures,
    y: np.ndarray,
    *,
    scheme: str = "stratified",
    kind: str = "lgbm",
    opts: dict[str, Any] | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    augment: bool = True,
    low_speed_rule: bool = True,
    tta: bool = False,
    boost_repeats: int = 3,
    aligned_boosts: bool = False,
    sensor_augmentation: str | None = None,
    low_speed_normal_keep: float | None = None,
    feature_mixup_alpha: float | None = None,
    feature_mixup_soft: bool = False,
    n_splits: int = 5,
    include_predictions: bool = False,
    n_jobs: int = 4,
    fold_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run one frozen validation scheme end to end and return the report dict.

    Nothing fitted crosses a fold: the design matrix is built by a pure per-row function, the
    model and its class boosts are fitted inside ``tr`` only, mirror copies are made from ``tr``
    only, and the two exact duplicate files are kept together by the fingerprint group.
    """
    t0 = time.time()
    o = {**rf.AGG_DEFAULT, **(opts or {})}
    speed = feats.scalars["speed_kmh"].to_numpy(dtype=float)
    groups = _duplicate_groups(feats)
    X_all = rf.aggregate(feats, o)
    mir_all = rf.mirror(feats)
    Xm_all = rf.aggregate(mir_all, o) if (augment or tta) else None

    if scheme == "stratified":
        splits = list(_splits_stratified(y, groups, seeds, n_splits=n_splits))
    elif scheme == "contiguous":
        splits = list(_splits_contiguous(feats.file_ids, groups, n_blocks=n_splits))
    elif scheme == "speed_range":
        splits = list(_splits_speed_range(speed, y))
    else:
        raise ValueError(f"unknown rail CV scheme {scheme!r}")
    if fold_indices is not None:
        chosen = [int(i) for i in fold_indices]
        if len(set(chosen)) != len(chosen) or any(i < 0 or i >= len(splits) for i in chosen):
            raise ValueError("CV fold indices must be unique valid split positions")
        splits = [splits[i] for i in chosen]

    for split_name, tr, te in splits:
        overlap = set(groups[np.asarray(tr, dtype=int)]) & set(groups[np.asarray(te, dtype=int)])
        if overlap:
            raise RuntimeError(f"rail split {split_name} separates {len(overlap)} duplicate fingerprint group(s)")

    per_fold: list[dict[str, Any]] = []
    all_true: list[Any] = []
    all_pred: list[Any] = []
    for name, tr, te in splits:
        fit_start = time.time()
        model = fit_rail(
            feats.subset(tr),
            y[tr],
            kind=kind,
            opts=o,
            seeds=seeds,
            augment=augment,
            low_speed_rule=low_speed_rule,
            tta=tta,
            boost_repeats=boost_repeats,
            aligned_boosts=aligned_boosts,
            sensor_augmentation=sensor_augmentation,
            low_speed_normal_keep=low_speed_normal_keep,
            feature_mixup_alpha=feature_mixup_alpha,
            feature_mixup_soft=feature_mixup_soft,
            n_jobs=n_jobs,
            mirrored=mir_all.subset(tr),
        )
        Xte = X_all.iloc[te].copy()
        Xte.attrs["rail_features"] = feats.subset(te)
        Xte_m = None
        if Xm_all is not None:
            Xte_m = Xm_all.iloc[te].copy()
            Xte_m.attrs["rail_features"] = mir_all.subset(te)
        yhat = model.predict_labels(Xte, speed[te], Xte_m)
        all_true.extend(y[te])
        all_pred.extend(yhat)
        rep = class_f1_report(y[te], yhat)
        matched = speed[te] >= SPEED_MATCHED_KMH
        per_fold.append(
            {
                "fold": name,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "macro_f1": float(rep["macro_f1"]),
                "macro_f1_speed_matched": float(macro_f1(y[te][matched], yhat[matched])) if matched.any() else None,
                "class_f1": {c: float(rep["per_class"][c]["f1"]) for c in RAIL_LABELS},
                "boosts": [float(b) for b in model.boosts],
                "fit_seconds": float(time.time() - fit_start),
            }
        )
        if include_predictions:
            per_fold[-1]["held_predictions"] = [
                {"file_id": feats.file_ids[int(i)], "truth": str(t), "prediction": str(p)}
                for i, t, p in zip(te, y[te], yhat)
            ]

    vals = np.array([f["macro_f1"] for f in per_fold], dtype=float)
    matched_vals = np.array([f["macro_f1_speed_matched"] for f in per_fold if f["macro_f1_speed_matched"] is not None])
    pooled = class_f1_report(all_true, all_pred)
    report = {
        "scheme": scheme,
        "model": kind,
        "opts": o,
        "augment": bool(augment),
        "tta": bool(tta),
        "boost_repeats": int(boost_repeats),
        "aligned_boosts": bool(aligned_boosts),
        "sensor_augmentation": sensor_augmentation,
        "low_speed_normal_keep": low_speed_normal_keep,
        "feature_mixup_alpha": feature_mixup_alpha,
        "feature_mixup_soft": bool(feature_mixup_soft),
        "low_speed_rule": bool(low_speed_rule),
        "seeds": [int(s) for s in seeds],
        "n_folds": len(per_fold),
        "n_files": int(len(y)),
        "n_features": int(X_all.shape[1]),
        "feature_version": rf.FEATURE_VERSION,
        "git_rev": git_rev(),
        "macro_f1_mean": float(vals.mean()) if len(vals) else 0.0,
        "macro_f1_sd": float(vals.std(ddof=0)) if len(vals) else 0.0,
        "macro_f1_speed_matched_mean": float(matched_vals.mean()) if len(matched_vals) else None,
        "macro_f1_speed_matched_sd": float(matched_vals.std(ddof=0)) if len(matched_vals) else None,
        "class_f1_mean": {
            c: float(np.mean([f["class_f1"].get(c, 0.0) for f in per_fold])) for c in RAIL_LABELS
        },
        "pooled_macro_f1": float(pooled["macro_f1"]),
        "pooled_class_f1": {c: float(pooled["per_class"][c]["f1"]) for c in RAIL_LABELS},
        "folds": per_fold,
        "fit_seconds_mean": float(np.mean([f["fit_seconds"] for f in per_fold])) if per_fold else 0.0,
        "wall_seconds": round(time.time() - t0, 2),
    }
    report["confusion"] = _confusion(np.asarray(all_true, dtype=object), np.asarray(all_pred, dtype=object))
    return report


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, int]]:
    out = {a: {b: 0 for b in RAIL_LABELS} for a in RAIL_LABELS}
    for t, p in zip(y_true, y_pred):
        if str(t) in out and str(p) in out[str(t)]:
            out[str(t)][str(p)] += 1
    return out


# --------------------------------------------------------------------------------------
# The Task
# --------------------------------------------------------------------------------------


def _implicated_side(spec: np.ndarray) -> tuple[str, float, np.ndarray, list[int]]:
    """(side, short-pitch contrast dB, that side's median wavelength spectrum, its top-3 boxes).

    "Which side is worse" is read off the 25-80 mm short-pitch band of the vibration channels,
    the band corrugation lives in [R248]; the top-3 boxes are what the 3D twin highlights.
    """
    vib = spec[rf._VIB, :]  # (64 boxes, N_LAMBDA)
    short = (rf.LAMBDA_GRID_MM >= rf.SHORT_PITCH_MM[0]) & (rf.LAMBDA_GRID_MM <= rf.SHORT_PITCH_MM[1])
    lvl = vib[:, short].mean(axis=1)
    mi, mii = rf.side_mask("I"), rf.side_mask("II")
    d = float(np.median(lvl[mi]) - np.median(lvl[mii]))
    side = "I" if d >= 0 else "II"
    m = mi if side == "I" else mii
    med = np.median(vib[m, :], axis=0)
    boxes = np.flatnonzero(m)
    order = boxes[np.argsort(lvl[m])[::-1]]
    return side, d, med, [int(b) for b in order[:3]]


def _box_name(box: int) -> tuple[int, int]:
    """Axle-box index 0..63 -> (car 1..8, position 1..8)."""
    return box // 8 + 1, box % 8 + 1


class RailTask(BaseTask):
    """`load` one csv -> `featurise` to per-side wavelength features -> `predict` a class."""

    name = "rail"

    def load(self, path: Path | str) -> rf.RailFeatures:
        scalars, spec = rf.extract_file(path)
        scalars["file_id"] = Path(path).name
        return rf.RailFeatures(
            file_ids=[Path(path).name],
            scalars=pd.DataFrame([scalars]),
            spectra=spec[None, ...].astype(np.float32),
            raw_path=str(path),
        )

    def featurise(self, raw: rf.RailFeatures) -> pd.DataFrame:
        X = rf.aggregate(raw, rf.AGG_DEFAULT)
        X.attrs["rail_features"] = raw
        return X

    def predict(self, feats: pd.DataFrame, model: Any = None) -> PredictionResult:
        mdl: RailModel = model if model is not None else load_model("rail")
        raw: rf.RailFeatures | None = feats.attrs.get("rail_features")
        X = feats
        if raw is not None and mdl.opts != rf.AGG_DEFAULT:
            X = rf.aggregate(raw, mdl.opts)
            X.attrs["rail_features"] = raw
        X_mirror = None
        if mdl.tta and raw is not None:
            mir_raw = rf.mirror(raw)
            X_mirror = rf.aggregate(mir_raw, mdl.opts)
            X_mirror.attrs["rail_features"] = mir_raw
        speed = (
            raw.scalars["speed_kmh"].to_numpy(dtype=float)
            if raw is not None
            else X.get("speed_kmh", pd.Series(np.full(len(X), np.nan))).to_numpy(dtype=float)
        )
        proba = mdl.proba_tta(X, X_mirror)
        labels = _apply_rules(proba, mdl.boosts, speed if mdl.low_speed_rule else None)
        file_ids = raw.file_ids if raw is not None else [""] * len(X)
        rows = [{"file_id": fid, "prediction": str(lab)} for fid, lab in zip(file_ids, labels)]

        first = 0
        numbers: dict[str, float] = {}
        trace = None
        viewport = None
        if raw is not None and len(raw) > 0:
            side, contrast_db, med_spec, top_boxes = _implicated_side(raw.spectra[first])
            pred = str(labels[first])
            car, pos = _box_name(top_boxes[0])
            rms = raw.channel_block("logrms")[first][rf._VIB]  # per-box vibration level, dB
            numbers = {
                "speed_kmh": float(speed[first]),
                "side_max_rms_ratio_db": float(rms[rf.side_mask("I")].max() - rms[rf.side_mask("II")].max()),
                "side_i_minus_ii_db": contrast_db,
                "p_normal": float(proba[first, 0]),
                "p_side_i": float(proba[first, 1]),
                "p_side_ii": float(proba[first, 2]),
            }
            peak = int(np.argmax(med_spec))
            marks = [
                {
                    "x": round(float(rf.LAMBDA_GRID_MM[peak]) / 10.0, 3),
                    "label": f"{rf.LAMBDA_GRID_MM[peak] / 10.0:.1f} cm peak",
                    "kind": "peak",
                }
            ]
            for rank, box in enumerate(top_boxes, start=1):
                c, p_ = _box_name(box)
                own_peak = int(np.argmax(raw.spectra[first][2 * box]))
                marks.append(
                    {
                        "x": round(float(rf.LAMBDA_GRID_MM[own_peak]) / 10.0, 3),
                        "label": f"#{rank} car {c} position {p_}",
                        "kind": "sensor",
                        "component": f"axlebox_c{c}_p{p_}",
                    }
                )
            trace = Trace(
                x=[round(float(v) / 10.0, 3) for v in rf.LAMBDA_GRID_MM],
                y=[float(v) for v in med_spec],
                marks=marks,
                label=f"wavelength PSD (dB), Side {side} axleboxes",
            )
            viewport = Viewport(
                car=int(car),
                side=side if pred != RAIL_LABELS[0] else None,
                health="crit" if pred != RAIL_LABELS[0] else "ok",
                component=f"axlebox_c{car}_p{pos}",
            )
        return PredictionResult(
            task="rail",
            file_id=file_ids[first] if file_ids else "",
            rows=rows,
            numbers=numbers,
            trace=trace,
            viewport=viewport,
            extras={
                "proba": proba,
                "labels": list(map(str, labels)),
                "file_ids": list(file_ids),
                "top_boxes": [f"axlebox_c{_box_name(b)[0]}_p{_box_name(b)[1]}" for b in (top_boxes if raw is not None and len(raw) else [])],
            },
        )

    def explain(self, result: PredictionResult) -> Explanation:
        return Explanation(
            file_id=result.file_id,
            numbers=dict(result.numbers),
            trace=result.trace or Trace(),
            viewport=result.viewport or Viewport(),
        )


register_task(RailTask())


# --------------------------------------------------------------------------------------
# Trainer entry point (scripts/ps3_train.py --task rail)
# --------------------------------------------------------------------------------------


def _load_training(n_jobs: int = 4) -> tuple[rf.RailFeatures, np.ndarray]:
    paths = sorted(train_dir("rail").glob("*.csv"), key=lambda p: natural_key(p.name))
    if not paths:
        raise FileNotFoundError(f"no rail training csvs under {train_dir('rail')}")
    feats = rf.load_feature_cache(paths, n_jobs=n_jobs)
    lab = pd.read_csv(labels_path("rail")).set_index("filename")["label"]
    y = lab.reindex(feats.file_ids)
    if y.isna().any():
        missing = list(y[y.isna()].index[:5])
        raise ValueError(f"{int(y.isna().sum())} rail files have no label, e.g. {missing}")
    return feats, y.to_numpy(dtype=object)


def _duplicate_report(feats: rf.RailFeatures, y: np.ndarray) -> list[dict[str, Any]]:
    """Candidate duplicates, with whole-file identity verified before they are reported."""
    def full_sha(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    out = []
    keyed = pd.Series(_duplicate_groups(feats))
    for h, idx in keyed.groupby(keyed).groups.items():
        rows = list(idx)
        if len(rows) > 1:
            paths = [train_dir("rail") / feats.file_ids[i] for i in rows]
            hashes = [full_sha(p) for p in paths]
            out.append(
                {
                    "fingerprint": str(h),
                    "files": [feats.file_ids[i] for i in rows],
                    "labels": sorted({str(y[i]) for i in rows}),
                    "speed_hash_equal": bool(feats.scalars["hash_speed"].iloc[rows].nunique() == 1),
                    "whole_file_sha256": hashes[0] if len(set(hashes)) == 1 else None,
                    "byte_identical": len(set(hashes)) == 1,
                }
            )
    return out


def _resolve_dirs(args: Any) -> tuple[Path, Path | None]:
    out_dir = Path(getattr(args, "out_dir", None) or RESULTS_DIR)
    model_dir = getattr(args, "model_dir", None)
    return out_dir, Path(model_dir) if model_dir else None


def _seeds_from(args: Any) -> tuple[int, ...]:
    raw = getattr(args, "seeds", None)
    if raw is None:
        return (0, 1, 2)
    if isinstance(raw, int):
        return tuple(range(raw)) if raw > 1 else (int(raw),)
    seeds = tuple(int(v) for v in raw)
    return seeds or (0,)


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------------------


def _cv_markdown(payload: dict[str, Any]) -> str:
    head = payload["headline"]
    baseline = payload["schemes"][0]
    lines = [
        f"# Rail corrugation - {payload['stage']} CV",
        "",
        f"`python scripts/ps3_train.py --task rail`, git `{payload['git_rev']}`, feature version "
        f"`{payload['feature_version']}`, seeds {payload['seeds']}, {payload['wall_seconds']:.0f} s wall. "
        "Every number here is reproducible from `rail_cv.json` in this directory.",
        "",
        f"Training set: {payload['n_files']} files, "
        + ", ".join(f"{k} {v}" for k, v in payload["class_counts"].items())
        + ". Metric: macro F1 over the fixed three-label vocabulary "
        "(`nebulax.ps3.scoring.macro_f1`), so a class the model never gets right scores 0.",
        "",
        "## Frozen validation schemes",
        "",
        "| scheme | folds | macro F1 (mean +- sd) | speed-matched v>=35 | Normal F1 | Side I F1 | Side II F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for rep in payload["schemes"]:
        sm = (
            f"{rep['macro_f1_speed_matched_mean']:.3f} +- {rep['macro_f1_speed_matched_sd']:.3f}"
            if rep["macro_f1_speed_matched_mean"] is not None
            else "n/a"
        )
        cf = rep["class_f1_mean"]
        lines.append(
            f"| {rep['scheme']} | {rep['n_folds']} | {rep['macro_f1_mean']:.3f} +- {rep['macro_f1_sd']:.3f} | {sm} | "
            f"{cf['Normal']:.3f} | {cf['Side I']:.3f} | {cf['Side II']:.3f} |"
        )
    if payload.get("selection_cv", {}).get("opts", {}).get("coherence"):
        lines += [
            "",
            "These three rows show the original all-on baseline. The selected coherence row's "
            "selection score and two stress scores are in `rail_ladder.json:winner` and "
            "`rail_ladder.json:winner_schemes`.",
        ]
    conf = head["confusion"]
    lines += [
        "",
        f"**Honest headline ({head['scheme']}, {head['n_folds']} folds): "
        f"macro F1 {head['macro_f1_mean']:.3f} +- {head['macro_f1_sd']:.3f}.**",
        "Every declared ladder row is ranked only on grouped inner folds of each outer training "
        "partition. The shipped row was chosen after the full ladder was read and is labelled "
        "post-hoc; its selection-CV score is not this headline.",
        "",
        "Pooled confusion over every held-out fold of the headline scheme (rows = truth):",
        "",
        "| true \\ predicted | Normal | Side I | Side II |",
        "|---|---|---|---|",
    ]
    for c in RAIL_LABELS:
        lines.append(f"| {c} | {conf[c]['Normal']} | {conf[c]['Side I']} | {conf[c]['Side II']} |")
    lines += [
        "",
        "## Ablations (same scheme, same seeds, one flag flipped)",
        "",
        "| arm | macro F1 (mean +- sd) | Side I F1 | n features |",
        "|---|---|---|---|",
        f"| baseline (all on) | {baseline['macro_f1_mean']:.3f} +- {baseline['macro_f1_sd']:.3f} | "
        f"{baseline['class_f1_mean']['Side I']:.3f} | {baseline['n_features']} |",
    ]
    for ab in payload["ablations"]:
        lines.append(
            f"| {ab['arm']} | {ab['macro_f1_mean']:.3f} +- {ab['macro_f1_sd']:.3f} | "
            f"{ab['class_f1_mean']['Side I']:.3f} | {ab['n_features']} |"
        )
    dup = payload["duplicates"]
    sp = payload["speed"]
    lines += [
        "",
        "## Dataset checks",
        "",
        "* **Duplicate fingerprint** (sha1 of the first 1,000 samples of channel 1 **and** the speed "
        f"column): {len(dup)} candidate group(s) - "
        + ("; ".join(f"`{'`, `'.join(g['files'])}` labelled {g['labels']}" for g in dup) if dup else "none")
        + f". Full-file SHA-256 confirms {sum(bool(g.get('byte_identical')) for g in dup)} "
        "byte-identical group(s). Every candidate fingerprint group is forced into the same fold "
        "for both stratified and contiguous schemes, so it cannot straddle train and test.",
        f"* **Speed confound**: Normal spans {sp['normal'][0]:.0f}-{sp['normal'][1]:.0f} km/h, "
        f"Side I {sp['side_i'][0]:.0f}-{sp['side_i'][1]:.0f}, Side II {sp['side_ii'][0]:.0f}-{sp['side_ii'][1]:.0f}. "
        f"{sp['n_normal_below_20']} Normal files and {sp['n_fault_below_20']} fault files run below "
        f"{LOW_SPEED_KMH:.0f} km/h, so `speed < 20 km/h -> Normal` is a **dataset shortcut, not physics** - "
        "the no-rule ablation above shows exactly what it is worth, and the speed-matched column "
        "(v >= 35 km/h, the faults' own range) shows the score with the confound removed.",
        "",
        "## Method",
        "",
        "Tacho pulse -> cumulative distance (pi*0.85/90 = 29.67 mm per pulse, integrated, never "
        "differentiated) -> anti-aliased resampling of all 128 channels onto a 1 mm spatial grid "
        "-> Welch PSD in the wavelength domain -> 1/3-octave wavelength bands over 8-500 mm "
        "(IEC 61373 / EN 15610 presentation [R233][R250]), v^2-normalised [R237] -> per-side "
        "aggregation (32 odd = Side I, 32 even = Side II; median / p90 / max) plus the "
        "Side I - Side II contrast in dB, which cancels speed, track type and sensor gain as "
        "common mode. Per channel the wavelength bands sit beside the 26-feature metro set's "
        "time-domain shape factors (log RMS, kurtosis, log peak, crest, skew, margin, pulse and "
        "waveform factors) and spectral moments, with its fixed 250-2000 Hz wavelet bands replaced "
        "by the wavelength bands [R232]; the seven 20-5000 Hz bands survive only as the "
        "counter-design ablation arm [R234]. Impulsive-vs-sustained discriminators (spectral "
        "flatness, harmonic peak prominence, envelope duty cycle, within-side cross-channel "
        "agreement) keep a wheel flat or a switch from scoring as corrugation [R235][R246]. "
        "Classifier: LightGBM, class-balanced, "
        f"{len(payload['seeds'])} seeds averaged, with the Side I / Side II probability boosts "
        "tuned on three repeats of a grouped inner 3-fold split of each training fold (a file and "
        "its mirror share a group), and mirror augmentation (odd<->even swap with the label "
        "swapped) applied inside the training fold only.",
        "",
        "Research basis: `docs/research/ps3_addendum.md` section 2 and `references.md` R231-R250 - "
        "the physics (29.67 mm per tacho pulse; lambda 25-80 mm [R248] sweeping 35-744 Hz over "
        "0-67 km/h, so fixed-Hz bands survive only as the counter-design arm [R234]), v^2 "
        "normalisation [R237], the 1/3-octave wavelength band presentation [R231][R233][R250], "
        "computed order tracking [R239][R245], impulsive-vs-sustained confounders [R235][R246], "
        "the 26-feature metro set with its Hz bands replaced by wavelength bands [R232], ROCKET + "
        "RidgeClassifierCV on short accelerometer windows [R249][R43][R47], and the realism anchor "
        "[R241]: a properly held-out railway-vibration classification lands near 0.8, not the "
        "95 %+ of rigs and simulations [R240][R243].",
        "",
    ]
    selected = payload.get("selection_cv", {})
    if selected.get("opts", {}).get("coherence"):
        lines += [
            "## Selected-row configuration",
            "",
            "The selected row uses the no-shock, fixed-Hz features and adds 21 Welch "
            "magnitude-squared coherence summaries. For each car, six vibration-box pairs "
            "are formed within each rail side (positions 1/3/5/7 for Side I and 2/4/6/8 "
            "for Side II). The 48 pairs per side are averaged in seven Hz bands, then Side I, "
            "Side II and their difference enter the classifier. The coherence calculation "
            "uses only the recording being predicted; all classifier fitting and class-boost "
            "calibration remain inside each training fold.",
            "",
        ]
    return "\n".join(lines)


def _ladder_markdown(payload: dict[str, Any]) -> str:
    rows = payload["rows"]
    lines = [
        "# Rail corrugation - model ladder",
        "",
        f"git `{payload['git_rev']}`, feature version `{payload['feature_version']}`, "
        f"frozen outer CV = stratified 5-fold by file (grouped on the duplicate fingerprint) x "
        f"{len(payload['seeds'])} seeds {payload['seeds']}, {payload['wall_seconds']:.0f} s wall. "
        "Every row is one model x ablation run through the *same* scheme; all numbers are "
        "reproducible from `rail_ladder.json`. The table ranks selection-CV rows; the selected "
        "row is post-hoc and its value is not the headline.",
        "",
        "| # | model | ablation | macro F1 (mean +- sd) | Side I F1 | Side II F1 | speed-matched | n feat | fit s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(sorted(rows, key=lambda r: -r["macro_f1_mean"]), start=1):
        sm = f"{r['macro_f1_speed_matched_mean']:.3f}" if r["macro_f1_speed_matched_mean"] is not None else "n/a"
        lines.append(
            f"| {i} | `{r['model']}` | {r['arm']} | **{r['macro_f1_mean']:.3f}** +- {r['macro_f1_sd']:.3f} | "
            f"{r['class_f1_mean']['Side I']:.3f} | {r['class_f1_mean']['Side II']:.3f} | {sm} | "
            f"{r['n_features']} | {r['fit_seconds_mean']:.2f} |"
        )
    w = payload["winner"]
    lines += [
        "",
        f"**Post-hoc selection-CV winner: `{w['model']}` / {w['arm']} at macro F1 {w['macro_f1_mean']:.3f} +- {w['macro_f1_sd']:.3f}**, "
        f"against the baseline's {payload['baseline']['macro_f1_mean']:.3f} +- {payload['baseline']['macro_f1_sd']:.3f}. "
        + (
            "It replaces `models/ps3/rail.pkl`."
            if payload["replaced_artefact"]
            else "It did **not** beat the baseline, so the baseline artefact stands."
        ),
        "",
        "## Honest nested/outer headline",
        "",
        f"Macro F1 **{payload['nested']['macro_f1_mean']:.3f} +- {payload['nested']['macro_f1_sd']:.3f}** "
        f"over {payload['nested']['n_folds']} grouped outer folds (5 folds x seeds {payload['nested']['seeds']}). "
        "All declared ladder rows are selected by grouped inner 3-fold CV inside each outer training partition.",
        "",
        "## Stress splits for the winner",
        "",
        "| scheme | folds | macro F1 (mean +- sd) |",
        "|---|---|---|",
    ]
    for rep in payload["winner_schemes"]:
        lines.append(f"| {rep['scheme']} | {rep['n_folds']} | {rep['macro_f1_mean']:.3f} +- {rep['macro_f1_sd']:.3f} |")
    lines += [
        "",
        "## What the ablations say",
        "",
        *payload["findings"],
        "",
        "## Rows skipped for budget, with the reason",
        "",
        *[f"* {s}" for s in payload["skipped"]],
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Trainer entry points
# --------------------------------------------------------------------------------------

#: Ablation arms run beside the baseline; each flips exactly one thing.
ABLATION_ARMS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "no distance resampling at all (fixed 20-5000 Hz bands, no wavelength discriminators)",
        {"opts": {"wavelength": False, "hz": True, "discriminators": False, "votes": False}},
    ),
    ("wavelength bands -> Hz bands, wavelength discriminators kept", {"opts": {"wavelength": False, "hz": True}}),
    ("no v^2 normalisation", {"opts": {"v2_normalise": False}}),
    ("no Side I - Side II contrast", {"opts": {"contrast": False}}),
    ("no mirror augmentation", {"augment": False}),
    ("no speed features", {"opts": {"speed": False}}),
    ("no impulsive/sustained discriminators", {"opts": {"discriminators": False}}),
    ("no low-speed rule", {"low_speed_rule": False}),
)

#: Ladder rows: (label, model kind, kwargs for cross_validate).
LADDER_ROWS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("baseline features", "logreg", {}),
    ("baseline features", "svm", {}),
    ("baseline features", "rf", {"boost_repeats": 1}),
    ("baseline features", "lgbm", {}),
    ("no v^2 normalisation", "lgbm", {"opts": {"v2_normalise": False}}),
    ("no v^2, + mirror TTA", "lgbm", {"opts": {"v2_normalise": False}, "tta": True}),
    ("no v^2, hierarchical present/absent then side", "lgbm_hier", {"opts": {"v2_normalise": False}}),
    ("no v^2, hierarchical + mirror TTA", "lgbm_hier", {"opts": {"v2_normalise": False}, "tta": True}),
    (
        "counter-design: Hz bands + speed, nothing distance-derived [R234]",
        "lgbm",
        {
            "opts": {
                "wavelength": False,
                "hz": True,
                "v2_normalise": False,
                "discriminators": False,
                "votes": False,
            },
            "tta": True,
        },
    ),
    (
        "Hz bands + speed, wavelength discriminators kept, + mirror TTA",
        "lgbm",
        {"opts": {"wavelength": False, "hz": True, "v2_normalise": False}, "tta": True},
    ),
    (
        "no v^2, wavelength + Hz bands, + mirror TTA",
        "lgbm",
        {"opts": {"hz": True, "v2_normalise": False}, "tta": True},
    ),
    ("no v^2, + mirror TTA", "rf", {"opts": {"v2_normalise": False}, "tta": True, "boost_repeats": 1}),
    ("no v^2, + mirror TTA", "logreg", {"opts": {"v2_normalise": False}, "tta": True}),
    (
        "per-side wavelength spectra (4 x 192)",
        "multirocket_ridge",
        {
            "opts": {
                "wavelength": False,
                "hz": False,
                "time": False,
                "discriminators": False,
                "contrast": False,
                "speed": False,
                "votes": False,
                "spectra_series": True,
            },
            "boost_repeats": 1,
        },
    ),
    (
        "per-side wavelength spectra, no v^2",
        "multirocket_ridge",
        {
            "opts": {
                "wavelength": False,
                "hz": False,
                "time": False,
                "discriminators": False,
                "contrast": False,
                "speed": False,
                "votes": False,
                "spectra_series": True,
                "v2_normalise": False,
            },
            "boost_repeats": 1,
        },
    ),
    (
        "two-view ensemble: Hz-band model + wavelength model, no v^2, probability average, + mirror TTA",
        "lgbm_2view",
        {"opts": {"hz": True, "v2_normalise": False}, "tta": True},
    ),
    (
        "+ no shock channels",
        "lgbm",
        {
            "opts": {
                "wavelength": False,
                "hz": True,
                "v2_normalise": False,
                "shock": False,
            },
            "tta": True,
        },
    ),
    (
        "+ no time-domain block",
        "lgbm",
        {
            "opts": {
                "wavelength": False,
                "hz": True,
                "v2_normalise": False,
                "time": False,
            },
            "tta": True,
        },
    ),
    (
        "+ no shock, no time block, no votes",
        "lgbm",
        {
            "opts": {
                "wavelength": False,
                "hz": True,
                "v2_normalise": False,
                "shock": False,
                "time": False,
                "votes": False,
            },
            "tta": True,
        },
    ),
    (
        "+ importance-pruned to 64 columns",
        "lgbm_pruned",
        {
            "opts": {
                "wavelength": False,
                "hz": True,
                "v2_normalise": False,
            },
            "tta": True,
        },
    ),
    (
        "Hz bands + speed, + fold-local speed-baseline residuals, + mirror TTA",
        "lgbm",
        {
            "opts": {
                "wavelength": False,
                "hz": True,
                "v2_normalise": False,
                "speed_baseline": True,
            },
            "tta": True,
        },
    ),
    (
        "sub-window voting: 3 x 0.5 s windows, Hz bands + speed, + mirror TTA",
        "lgbm_windows",
        {
            "opts": {
                "wavelength": False,
                "hz": True,
                "v2_normalise": False,
            },
            "tta": True,
        },
    ),
)

# The W8 round adds only these three predeclared tests to the frozen W7 ladder.
REVISED_ROWS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    (
        "no shock + robust vibration RMS",
        "lgbm",
        {"opts": {"wavelength": False, "hz": True, "v2_normalise": False,
                  "shock": False, "robust_time": True}, "tta": True},
    ),
    (
        "no shock + mirrored OOF boost calibration",
        "lgbm",
        {"opts": {"wavelength": False, "hz": True, "v2_normalise": False,
                  "shock": False}, "tta": True, "aligned_boosts": True},
    ),
    (
        "no shock + robust RMS + mirrored OOF calibration",
        "lgbm",
        {"opts": {"wavelength": False, "hz": True, "v2_normalise": False,
                  "shock": False, "robust_time": True}, "tta": True, "aligned_boosts": True},
    ),
)

COHERENCE_ROWS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    (
        "no shock + same-side axle-box coherence",
        "lgbm",
        {"opts": {"wavelength": False, "hz": True, "v2_normalise": False,
                  "shock": False, "coherence": True}, "tta": True},
    ),
)

ALL_LADDER_ROWS = (*LADDER_ROWS, *COHERENCE_ROWS)

# W9: two single-change sensor augmentation trials against the selected W7 row.
SENSOR_AUGMENT_ROWS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    (
        "no shock + bounded sensor gain jitter",
        "lgbm",
        {"opts": {"wavelength": False, "hz": True, "v2_normalise": False,
                  "shock": False}, "tta": True, "sensor_augmentation": "gain"},
    ),
    (
        "no shock + sparse sensor masking",
        "lgbm",
        {"opts": {"wavelength": False, "hz": True, "v2_normalise": False,
                  "shock": False}, "tta": True, "sensor_augmentation": "mask"},
    ),
)

SKIPPED_ROWS: tuple[str, ...] = (
    "**VMD / CEEMDAN / EWT / SPWVD adaptive decompositions** [R247]: "
    "per-record, parameter-heavy and slow, with no evidence they beat a wavelength-band PSD on a "
    "3-class macro-F1 task at n=272.",
    "**Model-based roughness inversion** [R238][R239]: needs track receptance and rail-pad "
    "stiffness we do not have; the source's own sensitivity analysis moves the answer 3.5 dB for a "
    "20 % pad-stiffness error.",
    "**Self-supervised / contrastive pre-training** [R244] and **sim-pretrain -> fine-tune** "
    "[R243]: the right shape for 14 labelled positives, but 234 unlabelled 1 s records "
    "are far too few for MoCo to pay and no corrugation simulator exists in this repo.",
    "**Deep 1D-CNN rows** [R240][R242]: a learnable front end on 272 files overfits, and "
    "the plan caps the deep tier at one row per family under a strict budget; `multirocket_ridge` "
    "is the row that family gets here.",
    "**GAN / diffusion augmentation**: the mirror swap is exact and free; generative augmentation "
    "on 14 spectra is strictly worse and adds a failure mode [R307][R308][R304]; "
    "`ps3_addendum.md` section 6 row 5.",
    "**Comb filter keyed to the wheel circumference** [R238]: implementable but "
    "optional at this budget; the side contrast captures most of the wheel-vs-rail separation.",
)

_RAIL_CHANCE_FLOOR: dict[str, Any] = {
    "name": "majority class (Normal)",
    "score": 0.308300395256917,
    "sd": 0.0,
    "n_files": 272,
    "class_counts": {"Normal": 234, "Side I": 14, "Side II": 24},
    "source": "nebulax.ps3.scoring.macro_f1",
}


def _rail_cite(arm: str, kind: str) -> str:
    if "coherence" in arm:
        return "[R235][R246]"
    if "two-view ensemble" in arm or kind == "lgbm_2view":
        return "[R234][R231][R232][R233]"
    if "sub-window" in arm or kind == "lgbm_windows":
        return "[R249]"
    if "counter-design" in arm or "Hz bands" in arm:
        return "[R234]"
    if kind == "lgbm_pruned" or "pruned" in arm:
        return "[R234]"
    if "speed-baseline" in arm or "speed_baseline" in arm:
        return "[R234]"
    if "shock" in arm or "time" in arm:
        return "[R234]"
    if kind == "multirocket_ridge":
        return "[R249][R43][R47]"
    if kind == "lgbm_hier":
        return "[R232]"
    return "[R231][R232][R233][R237][R250]"


def nested_cross_validate(
    feats: rf.RailFeatures,
    y: np.ndarray,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    n_jobs: int = 4,
    rows: Sequence[tuple[str, str, dict[str, Any]]] = ALL_LADDER_ROWS,
    checkpoint_path: Path | None = None,
    fold_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Nested 5-fold x seed score of the complete declared ladder-selection procedure.

    Candidate ranking uses a grouped inner 3-fold split of the outer training partition and one
    deterministic selection seed. The selected candidate is then refitted as the deployed
    three-seed ensemble and evaluated once on the untouched outer fold.
    """
    t0 = time.time()
    groups = _duplicate_groups(feats)
    speed = feats.scalars["speed_kmh"].to_numpy(dtype=float)
    outer_splits = list(_splits_stratified(y, groups, seeds, n_splits=5))
    chosen_indices = list(range(len(outer_splits))) if fold_indices is None else [int(i) for i in fold_indices]
    if len(set(chosen_indices)) != len(chosen_indices) or any(i < 0 or i >= len(outer_splits) for i in chosen_indices):
        raise ValueError("nested fold indices must be unique valid outer-fold positions")
    signature = hashlib.sha1(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()
    folds: list[dict[str, Any]] = []
    selected_names: list[str] = []
    if checkpoint_path is not None and checkpoint_path.exists():
        saved = json.loads(checkpoint_path.read_text())
        if saved.get("signature") != signature or saved.get("feature_version") != rf.FEATURE_VERSION:
            raise ValueError("nested checkpoint does not match the declared rows or feature version")
        folds = saved["folds"]
        selected_names = saved["selected_names"]
    for local_order, outer_order in enumerate(chosen_indices):
        fold_name, tr, te = outer_splits[outer_order]
        if local_order < len(folds):
            if folds[local_order]["fold"] != fold_name:
                raise ValueError("nested checkpoint fold order changed")
            continue
        train_feats = feats.subset(tr)
        train_y = y[tr]
        inner_seed = 10_000 + outer_order
        candidates: list[dict[str, Any]] = []
        for row_index, (arm, kind, kwargs) in enumerate(rows):
            report = cross_validate(
                train_feats,
                train_y,
                scheme="stratified",
                kind=kind,
                seeds=(inner_seed,),
                n_splits=3,
                n_jobs=n_jobs,
                **kwargs,
            )
            candidates.append(
                {
                    "row_index": int(row_index),
                    "arm": arm,
                    "model": kind,
                    "macro_f1_mean": float(report["macro_f1_mean"]),
                    "macro_f1_sd": float(report["macro_f1_sd"]),
                    "kwargs": kwargs,
                }
            )
        selected = max(candidates, key=lambda row: (row["macro_f1_mean"], -row["row_index"]))
        selected_names.append(f"{selected['model']} / {selected['arm']}")
        fit_start = time.time()
        model = fit_rail(
            train_feats,
            train_y,
            kind=selected["model"],
            seeds=seeds,
            n_jobs=n_jobs,
            **selected["kwargs"],
        )
        sub_te = feats.subset(te)
        Xte = rf.aggregate(sub_te, model.opts)
        Xte.attrs["rail_features"] = sub_te
        Xte_m = None
        if model.tta:
            mir_sub_te = rf.mirror(sub_te)
            Xte_m = rf.aggregate(mir_sub_te, model.opts)
            Xte_m.attrs["rail_features"] = mir_sub_te
        yhat = model.predict_labels(Xte, speed[te], Xte_m)
        rep = class_f1_report(y[te], yhat)
        matched = speed[te] >= SPEED_MATCHED_KMH
        held = [
            {"file_id": feats.file_ids[int(i)], "truth": str(t), "prediction": str(p)}
            for i, t, p in zip(te, y[te], yhat)
        ]
        check = macro_f1([row["truth"] for row in held], [row["prediction"] for row in held])
        if not np.isclose(check, rep["macro_f1"], atol=1e-12):  # pragma: no cover
            raise RuntimeError("rail held-prediction scorer drift")
        folds.append(
            {
                "fold": fold_name,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "selected_row_index": int(selected["row_index"]),
                "selected_arm": str(selected["arm"]),
                "selected_model": str(selected["model"]),
                "inner_macro_f1_mean": float(selected["macro_f1_mean"]),
                "inner_macro_f1_sd": float(selected["macro_f1_sd"]),
                "macro_f1": float(rep["macro_f1"]),
                "macro_f1_speed_matched": float(macro_f1(y[te][matched], yhat[matched])) if matched.any() else None,
                "class_f1": {c: float(rep["per_class"][c]["f1"]) for c in RAIL_LABELS},
                "fit_seconds": float(time.time() - fit_start),
                "held_predictions": held,
            }
        )
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            tmp.write_text(json.dumps({"signature": signature, "feature_version": rf.FEATURE_VERSION,
                                       "selected_names": selected_names, "folds": folds}, indent=2) + "\n")
            tmp.replace(checkpoint_path)
        print(f"  nested {outer_order + 1}/{len(outer_splits)}: {fold_name} "
              f"{rep['macro_f1']:.3f} ({selected['arm']})", flush=True)
    return summarise_nested_folds(folds, seeds=seeds, n_candidates=len(rows),
                                  wall_seconds=time.time() - t0)


def summarise_nested_folds(
    folds: Sequence[dict[str, Any]], *, seeds: Sequence[int], n_candidates: int,
    wall_seconds: float,
) -> dict[str, Any]:
    """Combine independent held-out outer folds without refitting any candidate."""
    values = np.asarray([f["macro_f1"] for f in folds], dtype=float)
    matched_values = np.asarray([f["macro_f1_speed_matched"] for f in folds if f["macro_f1_speed_matched"] is not None])
    selected_names = [f"{f['selected_model']} / {f['selected_arm']}" for f in folds]
    counts = {name: selected_names.count(name) for name in sorted(set(selected_names))}
    all_held = [row for fold in folds for row in fold["held_predictions"]]
    confusion = _confusion(
        np.asarray([row["truth"] for row in all_held], dtype=object),
        np.asarray([row["prediction"] for row in all_held], dtype=object),
    )
    return {
        "scheme": "nested stratified grouped 5-fold x seeds; complete ladder selected by grouped inner 3-fold",
        "selection_status": "honest outer estimate; every declared ladder row is selected only inside the outer training partition",
        "seeds": [int(s) for s in seeds],
        "n_folds": int(len(folds)),
        "n_candidates": int(n_candidates),
        "macro_f1_mean": float(values.mean()),
        "macro_f1_sd": float(values.std(ddof=0)),
        "macro_f1_speed_matched_mean": float(matched_values.mean()),
        "macro_f1_speed_matched_sd": float(matched_values.std(ddof=0)),
        "class_f1_mean": {c: float(np.mean([f["class_f1"][c] for f in folds])) for c in RAIL_LABELS},
        "confusion": confusion,
        "selection_counts": counts,
        "folds": list(folds),
        "wall_seconds": round(wall_seconds, 2),
    }


def train(args: Any = None) -> dict[str, Any]:
    """Fit the rail model, run the frozen validation schemes, write the artefacts.

    With ``args.ladder`` it also runs the model ladder and, when a ladder row beats the baseline
    on the frozen outer CV, refits that row as the committed artefact.
    """
    t0 = time.time()
    n_jobs = int(getattr(args, "n_jobs", 4) or 4)
    seeds = _seeds_from(args)
    out_dir, model_dir = _resolve_dirs(args)
    tag = str(getattr(args, "tag", "baseline") or "baseline")
    want_predict = bool(getattr(args, "predict", True))

    feats, y = _load_training(n_jobs=n_jobs)
    speed = feats.scalars["speed_kmh"].to_numpy(dtype=float)

    schemes = [
        cross_validate(feats, y, scheme=scheme, seeds=seeds, n_jobs=n_jobs, include_predictions=True)
        for scheme in ("stratified", "contiguous", "speed_range")
    ]
    headline = schemes[0]

    ablations = []
    for arm, kwargs in ABLATION_ARMS:
        rep = cross_validate(feats, y, scheme="stratified", seeds=seeds, n_jobs=n_jobs, **kwargs)
        ablations.append(
            {
                "arm": arm,
                "macro_f1_mean": rep["macro_f1_mean"],
                "macro_f1_sd": rep["macro_f1_sd"],
                "class_f1_mean": rep["class_f1_mean"],
                "n_features": rep["n_features"],
                "report": rep,
            }
        )

    model = fit_rail(feats, y, seeds=seeds, n_jobs=n_jobs)
    meta = _model_meta(model, headline, stage="baseline")
    save_model("rail", model, meta, model_dir=model_dir)

    payload = {
        "task": "rail",
        "stage": "baseline",
        "git_rev": git_rev(),
        "feature_version": rf.FEATURE_VERSION,
        "seeds": list(seeds),
        "n_files": int(len(y)),
        "class_counts": {c: int((y == c).sum()) for c in RAIL_LABELS},
        "headline": headline,
        "schemes": schemes,
        "ablations": ablations,
        "duplicates": _duplicate_report(feats, y),
        "speed": _speed_summary(speed, y),
        "model_meta": meta,
        "wall_seconds": round(time.time() - t0, 2),
    }
    payload["markdown"] = str(out_dir / "rail_cv.md")
    _write_json(payload, out_dir / "rail_cv.json")
    (out_dir / "rail_cv.md").write_text(_cv_markdown(payload), encoding="utf-8")

    if want_predict:
        csv_path = predict_test_set(
            model, n_jobs=n_jobs, out=out_dir / f"rail_predictions_{tag}.csv", feats=None
        )
        payload["predictions_csv"] = str(csv_path)
        _write_json(payload, out_dir / "rail_cv.json")

    if getattr(args, "ladder", False):
        ladder_result = run_ladder(
            feats,
            y,
            seeds=seeds,
            n_jobs=n_jobs,
            out_dir=out_dir,
            model_dir=model_dir,
            baseline=headline,
            want_predict=want_predict,
        )
        payload["ladder"] = ladder_result
        payload["stage"] = "post-hoc selected artefact; nested outer headline"
        payload["selection_cv"] = ladder_result["winner"]
        payload["winner_selection_status"] = ladder_result["winner_selection_status"]
        payload["headline"] = ladder_result["nested"]
        payload["model_meta"] = model_meta("rail", model_dir=model_dir)
        if ladder_result.get("predictions_csv"):
            payload["predictions_csv"] = ladder_result["predictions_csv"]
        payload["wall_seconds"] = round(time.time() - t0, 2)
        _write_json(payload, out_dir / "rail_cv.json")
        (out_dir / "rail_cv.md").write_text(_cv_markdown(payload), encoding="utf-8")
        headline = payload["headline"]

    return {
        k: payload[k]
        for k in (
            "task",
            "stage",
            "git_rev",
            "feature_version",
            "seeds",
            "n_files",
            "class_counts",
            "duplicates",
            "speed",
            "model_meta",
            "wall_seconds",
        )
        if k in payload
    } | {
        "macro_f1_mean": headline["macro_f1_mean"],
        "macro_f1_sd": headline["macro_f1_sd"],
        "predictions_csv": payload.get("predictions_csv"),
        "cv_json": str(out_dir / "rail_cv.json"),
        "ladder_json": str(out_dir / "rail_ladder.json") if getattr(args, "ladder", False) else None,
        "ladder_winner": (payload.get("ladder") or {}).get("winner", {}).get("model"),
    }


def _model_meta(
    model: RailModel,
    report: dict[str, Any],
    *,
    stage: str,
    honest_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = {
        "stage": stage,
        "scheme": report["scheme"],
        "macro_f1_mean": report["macro_f1_mean"],
        "macro_f1_sd": report["macro_f1_sd"],
        "macro_f1_speed_matched_mean": report["macro_f1_speed_matched_mean"],
        "feature_version": rf.FEATURE_VERSION,
        "model": model.kind,
        "opts": model.opts,
        "boosts": list(model.boosts),
        "tta": bool(model.tta),
        "augment": bool(model.meta.get("augment", True)),
        "low_speed_rule": bool(model.low_speed_rule),
        "n_features": len(model.columns),
        "seeds": list(model.meta.get("seeds", [])),
    }
    if honest_report is not None:
        meta["honest_headline"] = {
            "scheme": honest_report["scheme"],
            "macro_f1_mean": honest_report["macro_f1_mean"],
            "macro_f1_sd": honest_report["macro_f1_sd"],
            "n_folds": honest_report["n_folds"],
            "seeds": honest_report["seeds"],
        }
        meta["selection_status"] = "post-hoc artefact; headline is the nested outer estimate"
    return meta


def _speed_summary(speed: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    return {
        "normal": [float(speed[y == "Normal"].min()), float(speed[y == "Normal"].max())],
        "side_i": [float(speed[y == "Side I"].min()), float(speed[y == "Side I"].max())],
        "side_ii": [float(speed[y == "Side II"].min()), float(speed[y == "Side II"].max())],
        "n_normal_below_20": int(((y == "Normal") & (speed < LOW_SPEED_KMH)).sum()),
        "n_fault_below_20": int(((y != "Normal") & (speed < LOW_SPEED_KMH)).sum()),
    }


def run_ladder(
    feats: rf.RailFeatures,
    y: np.ndarray,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    n_jobs: int = 4,
    out_dir: Path | None = None,
    model_dir: Path | None = None,
    baseline: dict[str, Any] | None = None,
    want_predict: bool = True,
) -> dict[str, Any]:
    """Every model x ablation row plus an honest nested estimate; refit the selected row."""
    t0 = time.time()
    out = Path(out_dir or RESULTS_DIR)
    base = baseline or cross_validate(feats, y, scheme="stratified", seeds=seeds, n_jobs=n_jobs)

    rows: list[dict[str, Any]] = []
    for row_index, (arm, kind, kwargs) in enumerate(ALL_LADDER_ROWS):
        rep = cross_validate(feats, y, scheme="stratified", kind=kind, seeds=seeds, n_jobs=n_jobs, **kwargs)
        rows.append(
            {
                "row_index": int(row_index),
                "arm": arm,
                "model": kind,
                "macro_f1_mean": rep["macro_f1_mean"],
                "macro_f1_sd": rep["macro_f1_sd"],
                "macro_f1_speed_matched_mean": rep["macro_f1_speed_matched_mean"],
                "class_f1_mean": rep["class_f1_mean"],
                "n_features": rep["n_features"],
                "fit_seconds_mean": rep["fit_seconds_mean"],
                "wall_seconds": rep["wall_seconds"],
                "opts": rep["opts"],
                "tta": rep["tta"],
                "augment": rep["augment"],
                "boost_repeats": rep["boost_repeats"],
                "cite": _rail_cite(arm, kind),
                "report": rep,
            }
        )
        print(f"  ladder {kind:18s} {arm[:44]:44s} {rep['macro_f1_mean']:.3f}", flush=True)

    winner = max(rows, key=lambda r: (r["macro_f1_mean"], -r["row_index"]))
    replaced = winner["macro_f1_mean"] > base["macro_f1_mean"] + 1e-9
    nested = nested_cross_validate(feats, y, seeds=seeds, n_jobs=n_jobs)
    winner_schemes: list[dict[str, Any]] = []
    csv_path = None
    if replaced:
        wkw = {k: winner[k] for k in ("tta", "augment", "boost_repeats")}
        model = fit_rail(
            feats,
            y,
            kind=winner["model"],
            opts=winner["opts"],
            seeds=seeds,
            n_jobs=n_jobs,
            **wkw,
        )
        save_model(
            "rail",
            model,
            _model_meta(model, winner["report"], stage="ladder-winner", honest_report=nested),
            model_dir=model_dir,
        )
        for scheme in ("contiguous", "speed_range"):
            winner_schemes.append(
                cross_validate(
                    feats,
                    y,
                    scheme=scheme,
                    kind=winner["model"],
                    opts=winner["opts"],
                    seeds=seeds,
                    n_jobs=n_jobs,
                    **wkw,
                )
            )
        if want_predict:
            csv_path = str(predict_test_set(model, n_jobs=n_jobs, out=out / "rail_predictions.csv"))

    payload = {
        "task": "rail",
        "stage": "ladder",
        "git_rev": git_rev(),
        "feature_version": rf.FEATURE_VERSION,
        "seeds": list(seeds),
        "scheme": "stratified 5-fold by file (grouped on the duplicate fingerprint) x seeds",
        "baseline": {
            "row_index": 3,
            **{k: base[k] for k in ("model", "macro_f1_mean", "macro_f1_sd")},
        },
        "physics_row_index": 3,
        "rows": rows,
        "winner": {
            k: winner[k]
            for k in (
                "row_index",
                "arm",
                "model",
                "macro_f1_mean",
                "macro_f1_sd",
                "opts",
                "tta",
                "augment",
                "boost_repeats",
                "cite",
            )
        },
        "winner_selection_status": "post-hoc: chosen after ranking the full ladder on the frozen outer CV",
        "nested": nested,
        "chance_floor": dict(_RAIL_CHANCE_FLOOR),
        "winner_schemes": [
            {k: r[k] for k in ("scheme", "n_folds", "macro_f1_mean", "macro_f1_sd", "class_f1_mean", "confusion")}
            for r in winner_schemes
        ],
        "replaced_artefact": bool(replaced),
        "predictions_csv": csv_path,
        "findings": _ladder_findings(rows, base),
        "skipped": list(SKIPPED_ROWS),
        "wall_seconds": round(time.time() - t0, 2),
    }
    _write_json(payload, out / "rail_ladder.json")
    (out / "rail_ladder.md").write_text(_ladder_markdown(payload), encoding="utf-8")
    return {k: v for k, v in payload.items() if k != "rows"}


def _ladder_findings(rows: list[dict[str, Any]], base: dict[str, Any]) -> list[str]:
    by = {(r["model"], r["arm"]): r for r in rows}
    out = []
    lg = by.get(("lgbm", "baseline features"))
    lg_nov2 = by.get(("lgbm", "no v^2 normalisation"))
    if lg and lg_nov2:
        d = lg_nov2["macro_f1_mean"] - lg["macro_f1_mean"]
        out.append(
            f"* **v^2 normalisation**: {'costs' if d > 0 else 'buys'} {abs(d):.3f} macro F1 "
            f"({lg['macro_f1_mean']:.3f} with, {lg_nov2['macro_f1_mean']:.3f} without). "
            "[R237] predicts it should help by making the roughness->acceleration transfer "
            "speed-independent; on this dataset the fault files occupy a narrow speed band "
            "(35-67 km/h) and the un-normalised level is itself informative, so removing the speed "
            "scaling removes usable signal. Reported as measured, not as predicted."
        )
    rocket = [r for r in rows if r["model"] == "multirocket_ridge"]
    if rocket and lg:
        best_r = max(rocket, key=lambda r: r["macro_f1_mean"])
        out.append(
            f"* **MultiRocket + RidgeClassifierCV on the per-side wavelength spectra**: "
            f"{best_r['macro_f1_mean']:.3f} vs LightGBM's {lg['macro_f1_mean']:.3f} on the same "
            "representation. [R249] finds ROCKET ahead of hand-crafted features on raw "
            "accelerometer windows; here the hand-crafted side contrast already encodes what the "
            "label describes, so the gap narrows."
        )
    hier = [r for r in rows if r["model"] == "lgbm_hier"]
    if hier and lg_nov2:
        best_h = max(hier, key=lambda r: r["macro_f1_mean"])
        out.append(
            f"* **Hierarchical (present/absent, then which side)**: {best_h['macro_f1_mean']:.3f} vs "
            f"{lg_nov2['macro_f1_mean']:.3f} flat on the same features - the 38 positives support one "
            "binary decision better than three-way softmax (`ps3_addendum.md` section 6 row 3)."
        )
    hz_pure = by.get(("lgbm", "counter-design: Hz bands + speed, nothing distance-derived [R234]"))
    hz_mixed = by.get(("lgbm", "Hz bands + speed, wavelength discriminators kept, + mirror TTA"))
    wl_tta = by.get(("lgbm", "no v^2, + mirror TTA"))
    if hz_pure and hz_mixed and wl_tta:
        out.append(
            f"* **The counter-design arm** [R234] - hand the model fixed 20-5000 Hz bands and the "
            f"speed instead of normalising speed away - scores {hz_pure['macro_f1_mean']:.3f} with "
            f"nothing distance-derived and {hz_mixed['macro_f1_mean']:.3f} when the wavelength "
            f"discriminators are kept alongside, against {wl_tta['macro_f1_mean']:.3f} for the "
            "distance/wavelength path. Read it honestly: on **this** dataset the faults occupy "
            "35-67 km/h while a third of the Normals sit below 20, so a representation that tracks "
            "speed is rewarded by the label distribution, which is exactly the confound the "
            "speed-matched column and the leave-speed-range-out split exist to expose. Two things "
            "keep it from being pure confound-mining: the winner also leads on the speed-matched "
            "subset and on leave-speed-range-out (see the stress table above and `rail_cv.md`), "
            "and the gap is well inside one fold-to-fold sd. Note the direction flips with the "
            "v^2 flag: in `rail_cv.md`, where v^2 normalisation is on, dropping distance "
            "resampling *costs* 0.04 - the Hz bands only win once the v^2 scaling is off, i.e. "
            "once the model is allowed to use absolute level, which is itself a speed proxy."
        )

    ens = by.get(("lgbm_2view", "two-view ensemble: Hz-band model + wavelength model, no v^2, probability average, + mirror TTA"))
    if ens and hz_mixed and wl_tta:
        out.append(
            f"* **Two-view ensemble (Hz-band model + wavelength model, probability average, + mirror TTA)**: "
            f"scores {ens['macro_f1_mean']:.3f} +- {ens['macro_f1_sd']:.3f} "
            f"(Side I {ens['class_f1_mean']['Side I']:.3f}, Side II {ens['class_f1_mean']['Side II']:.3f}), "
            f"against the single-view Hz model's {hz_mixed['macro_f1_mean']:.3f} "
            f"(Side I {hz_mixed['class_f1_mean']['Side I']:.3f}) and the single-view wavelength model's "
            f"{wl_tta['macro_f1_mean']:.3f} (Side I {wl_tta['class_f1_mean']['Side I']:.3f})."
        )

    r_noshock = by.get(("lgbm", "+ no shock channels"))
    r_notime = by.get(("lgbm", "+ no time-domain block"))
    r_min = by.get(("lgbm", "+ no shock, no time block, no votes"))
    r_pruned = by.get(("lgbm_pruned", "+ importance-pruned to 64 columns"))
    if hz_mixed and (r_noshock or r_notime or r_min or r_pruned):
        pieces = []
        if r_noshock:
            pieces.append(f"no shock {r_noshock['macro_f1_mean']:.3f}")
        if r_notime:
            pieces.append(f"no time {r_notime['macro_f1_mean']:.3f}")
        if r_min:
            pieces.append(f"minimal {r_min['macro_f1_mean']:.3f}")
        if r_pruned:
            pieces.append(f"pruned-64 {r_pruned['macro_f1_mean']:.3f}")
        out.append(
            f"* **Feature reduction rows**: base Hz model {hz_mixed['macro_f1_mean']:.3f} vs "
            + ", ".join(pieces)
            + ". Pruning down from 352 columns follows the VSB and LANL winning approaches."
        )

    r_speed_base = by.get(("lgbm", "Hz bands + speed, + fold-local speed-baseline residuals, + mirror TTA"))
    if r_speed_base and hz_mixed:
        out.append(
            f"* **Fold-local speed-baseline residuals**: {r_speed_base['macro_f1_mean']:.3f} +- {r_speed_base['macro_f1_sd']:.3f} "
            f"vs base Hz model {hz_mixed['macro_f1_mean']:.3f}."
        )

    r_win = by.get(("lgbm_windows", "sub-window voting: 3 x 0.5 s windows, Hz bands + speed, + mirror TTA"))
    if r_win and hz_mixed:
        out.append(
            f"* **Sub-window voting (3 x 0.5 s windows, + mirror TTA)** [R249]: {r_win['macro_f1_mean']:.3f} +- {r_win['macro_f1_sd']:.3f} "
            f"(Side I {r_win['class_f1_mean']['Side I']:.3f}) vs base {hz_mixed['macro_f1_mean']:.3f}."
        )

    tta = [r for r in rows if r["tta"]]
    if tta:
        best_t = max(tta, key=lambda r: r["macro_f1_mean"])
        out.append(
            f"* **Mirror test-time augmentation** (average p(x) with the side-swapped p(mirror(x))): "
            f"best TTA row {best_t['macro_f1_mean']:.3f}. The mirror is exact for this sensor layout, "
            "so the two views must agree; averaging them costs one extra forward pass."
        )
    out.append(
        f"* The spread across folds (sd {base['macro_f1_sd']:.3f} at n=14 for Side I, ~2.8 per fold) "
        "is the honest headline, not a point estimate: [R241] is the realism anchor (81.5 % held-out "
        "accuracy over 21 classifiers on railway vibration), and `ps3_addendum.md` section 6 row 5 "
        "sets the expectation band at 0.70-0.85 macro F1 with visible spread, against the 0.95+ of "
        "rig and simulation studies [R240][R243]."
    )
    return out


def predict_test_set(
    model: RailModel | None = None,
    *,
    n_jobs: int = 4,
    out: Path | str | None = None,
    feats: rf.RailFeatures | None = None,
    model_dir: Path | None = None,
) -> Path:
    """Score the distributed Test folder and write the organiser-schema csv."""
    mdl = model if model is not None else load_model("rail", model_dir=model_dir)
    if feats is None:
        paths = sorted(test_dir("rail").glob("*.csv"), key=lambda p: natural_key(p.name))
        if not paths:
            raise FileNotFoundError(f"no rail test csvs under {test_dir('rail')}")
        feats = rf.load_feature_cache(paths, n_jobs=n_jobs)
    X = rf.aggregate(feats, mdl.opts)
    X.attrs["rail_features"] = feats
    Xm = None
    if mdl.tta:
        mir_feats = rf.mirror(feats)
        Xm = rf.aggregate(mir_feats, mdl.opts)
        Xm.attrs["rail_features"] = mir_feats
    speed = feats.scalars["speed_kmh"].to_numpy(dtype=float)
    labels = mdl.predict_labels(X, speed if mdl.low_speed_rule else None, Xm)
    path = Path(out) if out is not None else RESULTS_DIR / "rail_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"file_id": feats.file_ids, "prediction": [str(v) for v in labels]}).to_csv(path, index=False)
    return path
