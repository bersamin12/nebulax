"""Run the frozen benchmark winners over the whole demo fleet.

``docs/app_contract.md`` section 3 is the contract this module implements. Nothing here
re-implements the benchmark: the winners' knobs come out of ``results/runs.parquet``, the
feature matrices out of :mod:`nebulax.bench.data`, the folds out of
:mod:`nebulax.bench.splits`, and fit / calibrate / score are the runner's own
:func:`~nebulax.bench.runner._fit_model`, :func:`~nebulax.bench.runner._calibrate` and
:func:`~nebulax.bench.runner._score`, called in that order so a test row can never reach a
threshold.

What is different from a benchmark run, and only this
-----------------------------------------------------
* **Every run is scored.** The sweep capped the sim blocks at ``max_runs`` 5-8 runs; the demo
  scores all 10 pneumatic / 20 bearing runs, so ``data_kwargs["max_runs"]`` is
  dropped from the winner spec (:attr:`WinnerSpec.data_kwargs` keeps what is left).
* **Every fold is used.** ``sim_loo_unit`` over the whole fleet is ten folds, one per train,
  which is exactly the contract's leave-one-train-out protocol.
* **The scores are kept.** The benchmark throws the per-row scores away after computing its
  metrics; here they are the product.

Everything else - the training regime, the contamination, ``H``, the episode definition
(``k_consecutive`` / ``merge_gap_s`` / ``max_step_s``) and the false-alarm budget - is read
off the winner's results row and passed through unchanged.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from nebulax import schema as S
from nebulax.bench import data as D
from nebulax.bench import metrics as M
from nebulax.bench import runner as R
from nebulax.bench.data import BenchData
from nebulax.bench.registry import build
from nebulax.bench.splits import Split, make_split

__all__ = [
    "WINNERS",
    "WinnerSpec",
    "FittedWinner",
    "FleetIndex",
    "RUNS_PARQUET",
    "SIM_ROOT",
    "SCORES_DIR",
    "load_winners",
    "load_fleet",
    "fit_winner",
    "load_fitted",
    "save_fitted",
    "score_run",
    "score_frames",
    "score_held_out",
    "episodes_from_scores",
    "scoreable_from_fault_log",
    "faults_detected",
    "event_summary",
    "empty_episodes",
    "EPISODE_COLUMNS",
    "top_signals",
]

LOGGER = logging.getLogger(__name__)

#: The benchmark results frame the winners are read from (contract section 2).
RUNS_PARQUET: Path = Path("results/runs.parquet")
#: The synthetic fleet root.
SIM_ROOT: Path = Path("data/sim")
#: Where the batch script writes (contract section 3).
SCORES_DIR: Path = Path("data/scores")
#: The loader feature cache the benchmark uses.
CACHE_DIR: Path = Path("data/features")

#: Consistency constant turning a median-absolute-deviation into a standard-deviation
#: estimate for a normal sample, so ``top_signals``' ``z`` is on the usual scale.
MAD_TO_SIGMA: float = 1.4826
#: Floor on the MAD denominator, so a constant training feature cannot produce an infinite z.
_Z_EPS: float = 1e-9
#: ``top_signals_json`` holds at most this many entries (contract section 3).
MAX_TOP_SIGNALS: int = 5

#: The demo's subsystem keys and the ``(dataset, dataset_subsystem, model, input_kind,
#: window, peer_norm)`` row each one is joined to in ``results/runs.parquet``, verbatim from
#: ``docs/app_contract.md`` section 2.
#:
#: ``contamination`` is part of the selector although the contract's join key does not name
#: it: on door and bearing the six-key join matches **two** ok rows that differ only in
#: contamination (0.0 and 0.05), and the leaderboard's "Selected per subsystem" row - the one
#: the contract quotes the numbers of (door 4/12, bearing 10/12) - is the clean
#: ``contamination=0.0`` fit. Without this the join would be ambiguous.
WINNER_KEYS: dict[str, dict[str, Any]] = {
    "door": {
        "dataset": "sim",
        "dataset_subsystem": "door",
        "model": "cusum_cycle_scalar",
        "input_kind": "cycle_features",
        "window": None,
        "peer_norm": True,
        "contamination": 0.0,
    },
    "pneumatic": {
        "dataset": "sim",
        "dataset_subsystem": "pneumatic",
        "model": "sparse_autoencoder",
        "input_kind": "window_stats",
        "window": 360,
        "peer_norm": False,
        "contamination": 0.0,
    },
    "bearing": {
        "dataset": "sim",
        "dataset_subsystem": "bearing",
        "model": "cusum_cycle_scalar",
        "input_kind": "cycle_features",
        "window": None,
        "peer_norm": True,
        "contamination": 0.0,
    },
    "metropt3": {
        "dataset": "metropt3",
        "dataset_subsystem": "pneumatic",
        "model": "lgbm_residual",
        "input_kind": "raw_window",
        "window": 360,
        "peer_norm": False,
        "contamination": 0.0,
        "split": "metropt_temporal",
    },
}

#: The train id the MetroPT-3 unit is published under (contract sections 1 and 3).
METROPT_TRAIN_ID: str = "MP3"
METROPT_CAR: int = 0
METROPT_COMPONENT: str = "apu_1"


# --------------------------------------------------------------------------------------
# WinnerSpec
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WinnerSpec:
    """One frozen winner: every knob its results row carries, and nothing invented here."""

    key: str
    dataset: str
    dataset_subsystem: str
    model: str
    params: dict[str, Any]
    input_kind: str
    window: int | None
    peer_norm: bool
    split: str
    feature_set: str
    train_regime: str
    contamination: float
    H: float
    seed: int
    max_minutes: float
    config_hash: str
    #: Episode definition, as the benchmark computed it for this run.
    k_consecutive: int
    merge_gap_s: float
    max_step_s: float
    window_seconds: float
    #: The published operating point, for cross-checking a refit (never used as the threshold:
    #: every held-out train is calibrated afresh on its own validation carve).
    published_threshold: float
    #: Loader keywords the row carried that are not ablation axes, minus ``max_runs`` - the
    #: demo scores the whole fleet (see the module docstring).
    data_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def subsystem(self) -> str:
        """The sim subsystem the loader needs (``""`` for MetroPT-3, which holds one)."""
        return self.dataset_subsystem if self.dataset == "sim" else ""

    def to_run_spec(self) -> R.RunSpec:
        """The :class:`~nebulax.bench.runner.RunSpec` the runner helpers take."""
        return R.RunSpec(
            dataset=self.dataset,
            model=self.model,
            task="ad",
            subsystem=self.subsystem,
            input_kind=self.input_kind,
            split=self.split,
            params=dict(self.params),
            window=self.window,
            feature_set=self.feature_set,
            train_regime=self.train_regime,
            H=self.H,
            peer_norm=self.peer_norm,
            contamination=self.contamination,
            seed=self.seed,
            max_minutes=self.max_minutes,
        )

    def loader_kwargs(self, *, root: Path | None = None) -> dict[str, Any]:
        """Loader keywords for :func:`nebulax.bench.data.load_bench`.

        The demo scores the whole fleet, and "whole" has to be said out loud on the streamed
        path: ``load_sim(input_kind="window_stats")`` with no ``max_runs`` falls back to **4
        runs** (its sampling default, because the raw telemetry is 808 M rows), which silently
        turned the pneumatic block into a four-train fleet. The per-cycle path already reads
        every run when ``run_ids`` is ``None``, so it is left alone - and left alone is also
        what keeps its (expensive) feature-cache key stable.
        """
        kw: dict[str, Any] = {**self.data_kwargs, **self.to_run_spec().loader_kwargs()}
        if root is not None:
            kw["raw_dir"] = str(root)
        if self.dataset == "sim" and self.input_kind != "cycle_features":
            kw["run_ids"] = D._sim_run_ids(Path(root) if root else SIM_ROOT, self.subsystem)
        return kw

    def as_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "dataset": self.dataset,
            "dataset_subsystem": self.dataset_subsystem,
            "model": self.model,
            "params": dict(self.params),
            "input_kind": self.input_kind,
            "window": self.window,
            "peer_norm": self.peer_norm,
            "split": self.split,
            "feature_set": self.feature_set,
            "train_regime": self.train_regime,
            "contamination": self.contamination,
            "H": self.H,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "k_consecutive": self.k_consecutive,
            "merge_gap_s": self.merge_gap_s,
            "max_step_s": self.max_step_s,
            "window_seconds": self.window_seconds,
            "published_threshold": self.published_threshold,
            "data_kwargs": dict(self.data_kwargs),
        }


def _as_window(v: Any) -> int | None:
    if v is None:
        return None
    f = float(v)
    if not np.isfinite(f):
        return None
    return int(f)


def _match_winner(runs: pd.DataFrame, key: str, want: Mapping[str, Any]) -> pd.Series:
    """The single results row for ``key``, or a ``ValueError`` saying exactly what is wrong."""
    mask = pd.Series(True, index=runs.index)
    for col, value in want.items():
        if col == "window":
            col_vals = runs["window"].map(_as_window)
            mask &= col_vals.eq(value) if value is not None else col_vals.isna()
        elif col == "peer_norm":
            mask &= runs["peer_norm"].astype(bool).eq(bool(value))
        elif col == "contamination":
            mask &= np.isclose(runs["contamination"].astype(float), float(value))
        else:
            mask &= runs[col].astype(str).eq(str(value))
    hit = runs[mask]
    if hit.empty:
        raise ValueError(
            f"demo.scoring: no benchmark row for winner {key!r} matching {dict(want)}. "
            f"docs/app_contract.md section 2 froze this pick; either the results frame is not "
            f"the one the contract was written against, or the contract needs editing first."
        )
    ok = hit[hit["status"].astype(str).eq("ok")]
    if ok.empty:
        raise ValueError(
            f"demo.scoring: the benchmark row(s) for winner {key!r} ({list(hit['config_hash'])}) "
            f"have status={sorted(set(hit['status'].astype(str)))}, not 'ok'; a winner must be a "
            f"run that finished"
        )
    if len(ok) > 1:
        raise ValueError(
            f"demo.scoring: winner {key!r} matches {len(ok)} ok benchmark rows "
            f"({list(ok['config_hash'])}); the selector in WINNER_KEYS is ambiguous and would "
            f"publish one row's thresholds under another row's name"
        )
    return ok.iloc[0]


def _spec_from_row(key: str, row: pd.Series) -> WinnerSpec:
    params = json.loads(row["params"]) if isinstance(row["params"], str) else dict(row["params"] or {})
    raw_dk = json.loads(row["data_kwargs"]) if isinstance(row["data_kwargs"], str) else dict(row["data_kwargs"] or {})
    # max_runs capped the sweep, not the demo: the app scores every run of the fleet.
    data_kwargs = {k: v for k, v in raw_dk.items() if k != "max_runs"}
    return WinnerSpec(
        key=key,
        dataset=str(row["dataset"]),
        dataset_subsystem=str(row["dataset_subsystem"]),
        model=str(row["model"]),
        params=params,
        input_kind=str(row["input_kind"]),
        window=_as_window(row["window"]),
        peer_norm=bool(row["peer_norm"]),
        split=str(row["split"]),
        feature_set=str(row["feature_set"]),
        train_regime=str(row["train_regime"]),
        contamination=float(row["contamination"]),
        seed=int(row["seed"]),
        H=float(row["H"]),
        max_minutes=float(row["max_minutes"]),
        config_hash=str(row["config_hash"]),
        k_consecutive=int(row["k_consecutive"]),
        merge_gap_s=float(row["merge_gap_s"]),
        max_step_s=float(row["max_step_s"]),
        window_seconds=float(row["window_seconds"]),
        published_threshold=float(row["threshold_budget"]),
        data_kwargs=data_kwargs,
    )


_WINNER_CACHE: dict[str, dict[str, WinnerSpec]] = {}


def load_winners(runs_path: str | Path | None = None, *, keys: Sequence[str] | None = None) -> dict[str, WinnerSpec]:
    """``{subsystem key: WinnerSpec}`` read from the benchmark results frame.

    Fails loudly when a winner's row is missing, is not ``status="ok"``, or when the
    selector matches more than one row - the contract's picks are frozen, so a silent
    substitution would publish one model's numbers under another model's name.
    """
    path = Path(runs_path) if runs_path is not None else RUNS_PARQUET
    # The old Door winner remains addressable for historical test fixtures, but the active
    # research fleet contains only pneumatic and bearing runs.
    wanted = tuple(keys) if keys is not None else tuple(k for k in WINNER_KEYS if k != "door")
    stamp = path.stat().st_mtime_ns if path.exists() else 0
    cache_key = f"{path.resolve() if path.exists() else path}|{stamp}|{','.join(wanted)}"
    hit = _WINNER_CACHE.get(cache_key)
    if hit is not None:
        return dict(hit)
    if not path.exists():
        raise FileNotFoundError(
            f"demo.scoring.load_winners: {path} does not exist; the demo winners are read from the "
            f"benchmark results frame (docs/app_contract.md section 2). Run scripts/run_bench.py, "
            f"or pass runs_path=."
        )
    runs = pd.read_parquet(path)
    out = {key: _spec_from_row(key, _match_winner(runs, key, WINNER_KEYS[key])) for key in wanted}
    _WINNER_CACHE[cache_key] = dict(out)
    return out


def __getattr__(name: str) -> Any:
    """``WINNERS`` is resolved on first use, so importing this module needs no parquet."""
    if name == "WINNERS":
        return load_winners()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --------------------------------------------------------------------------------------
# FleetIndex - the loaded table + its folds, shared by the ten leave-one-train-out fits
# --------------------------------------------------------------------------------------


@dataclass
class FleetIndex:
    """One winner's loaded :class:`BenchData` plus its folds.

    Built once per subsystem and handed to :func:`fit_winner` as ``index=`` so ten
    leave-one-train-out fits do not reload a 1.4 M-row feature table ten times.
    """

    spec: WinnerSpec
    data: BenchData
    splits: list[Split]
    root: Path

    @property
    def held_out_trains(self) -> list[str]:
        return [str(s.meta.get("held_out_group", s.fold)) for s in self.splits]

    def split_for(self, exclude_train: str | None) -> Split:
        if len(self.splits) == 1:
            # A single-split preset (MetroPT-3's metropt_temporal): there is nothing to hold
            # out, the split already reserves the test slice.
            return self.splits[0]
        if exclude_train is None:
            if len(self.splits) != 1:
                raise ValueError(
                    f"FleetIndex.split_for(None): {self.spec.key!r} has {len(self.splits)} folds; "
                    f"name the held-out train (one of {self.held_out_trains})"
                )
            return self.splits[0]
        for s in self.splits:
            if str(s.meta.get("held_out_group", "")) == str(exclude_train):
                return s
        raise ValueError(
            f"FleetIndex.split_for({exclude_train!r}): no fold holds out that train; "
            f"folds hold out {self.held_out_trains}"
        )


def load_fleet(
    subsystem: str,
    *,
    root: str | Path | None = None,
    runs_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    spec: WinnerSpec | None = None,
) -> FleetIndex:
    """Load the whole table one winner scores, and build its folds.

    ``root`` overrides the dataset root (``data/sim`` / ``data/raw/metropt3``) - the tests
    point it at a fixture fleet in ``tmp_path``.
    """
    spec = spec or load_winners(runs_path, keys=[subsystem])[subsystem]
    cdir = Path(cache_dir) if cache_dir is not None else (CACHE_DIR if root is None else None)
    base = Path(root) if root is not None else (SIM_ROOT if spec.dataset == "sim" else None)
    data = D.load_bench(spec.dataset, cache_dir=cdir, **spec.loader_kwargs(root=base))
    if spec.dataset == "sim":
        _assert_whole_fleet(spec, data, base or SIM_ROOT)
    splits = make_split(spec.split, data, seed=spec.seed)
    return FleetIndex(spec=spec, data=data, splits=list(splits), root=Path(base) if base else Path("."))


def _assert_whole_fleet(spec: WinnerSpec, data: BenchData, root: Path) -> None:
    """Every run of the subsystem must be in the table the demo scores.

    The loaders have sampling defaults (``max_runs``, the streamed path's fallback of four
    runs) that exist to keep a sweep affordable and are exactly wrong here: a missing run is a
    train with no scores, which the app would render as ``nodata`` on a fleet the contract says
    is fully instrumented. Cheap to check, and silent otherwise.
    """
    index = Path(root) / "index.json"
    if not index.exists():  # a trimmed fixture; nothing to check against
        return
    try:
        runs = json.loads(index.read_text(encoding="utf-8"))["runs"]
    except Exception:  # pragma: no cover
        return
    want = {str(r["run_id"]) for r in runs if str(r.get("subsystem")) == spec.dataset_subsystem}
    got = {str(x) for x in pd.unique(np.asarray(data.labels["run_id"], dtype=object))} if "run_id" in data.labels else set()
    missing = sorted(want - got)
    if missing:
        raise ValueError(
            f"demo.scoring.load_fleet({spec.key!r}): the loaded table is missing {len(missing)} of "
            f"{len(want)} run(s) of this subsystem ({missing[:6]}...). The demo scores the whole "
            f"fleet; check the loader keywords (max_runs / run_ids) in WinnerSpec.loader_kwargs."
        )


# --------------------------------------------------------------------------------------
# FittedWinner
# --------------------------------------------------------------------------------------


@dataclass
class FittedWinner:
    """One winner fitted on one leave-one-train-out fold, with its operating point.

    Picklable: ``scripts/score_for_demo.py`` writes one per held-out train to
    ``data/scores/models/<subsystem>/<train>.pkl`` so ``POST /api/sim/inject`` can re-score a
    regenerated component without refitting (contract section 3).
    """

    subsystem: str
    model_name: str
    params: dict[str, Any]
    input_kind: str
    window: int | None
    peer_norm: bool
    threshold: float
    feature_names: list[str]
    train_median: np.ndarray
    train_mad: np.ndarray
    held_out_train: str
    fitted_at: str

    # --- everything below is bookkeeping the contract does not name but the app needs ---
    model: Any = None
    spec: WinnerSpec | None = None
    thresholds: dict[str, float] = field(default_factory=dict)
    attribution: str = "robust_z"
    fit_seconds: float = 0.0
    n_train_rows: int = 0
    n_val_rows: int = 0
    calibration: dict[str, Any] = field(default_factory=dict)

    # --- episode definition, carried so the inject path counts episodes identically ---
    k_consecutive: int = M.DEFAULT_K_CONSECUTIVE
    merge_gap_s: float = M.DEFAULT_MERGE_GAP_S
    max_step_s: float = float("inf")
    window_seconds: float = 0.0
    H: float = 3 * 86400.0

    @property
    def config_hash(self) -> str:
        return self.spec.config_hash if self.spec is not None else ""

    def robust_z(self, X: np.ndarray) -> np.ndarray:
        """``(x - median) / (1.4826 * MAD)`` per feature, against the fitted training rows."""
        arr = np.asarray(X, dtype=np.float64)
        med = np.asarray(self.train_median, dtype=np.float64)
        mad = np.asarray(self.train_mad, dtype=np.float64)
        denom = MAD_TO_SIGMA * mad
        denom = np.where(denom > _Z_EPS, denom, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (arr - med) / denom
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _pickle_path(subsystem: str, held_out_train: str, *, scores_dir: Path) -> Path:
    return Path(scores_dir) / "models" / subsystem / f"{held_out_train}.pkl"


def save_fitted(fw: FittedWinner, *, scores_dir: str | Path = SCORES_DIR) -> Path:
    """Pickle ``fw`` to ``<scores_dir>/models/<subsystem>/<held_out_train>.pkl``."""
    import pickle

    path = _pickle_path(fw.subsystem, fw.held_out_train, scores_dir=Path(scores_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    _to_cpu(fw.model)
    with open(path, "wb") as fh:
        pickle.dump(fw, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_fitted(subsystem: str, held_out_train: str, *, scores_dir: str | Path = SCORES_DIR) -> FittedWinner:
    """Read back a :class:`FittedWinner` written by :func:`save_fitted`."""
    import pickle

    path = _pickle_path(subsystem, held_out_train, scores_dir=Path(scores_dir))
    if not path.exists():
        raise FileNotFoundError(
            f"demo.scoring.load_fitted({subsystem!r}, {held_out_train!r}): {path} does not exist; "
            f"run scripts/score_for_demo.py first"
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _to_cpu(model: Any) -> None:
    """Move a torch model's parameters to the CPU before pickling.

    A CUDA tensor pickles with its device, so a pickle written on this box would only load on
    a box with a GPU - and ``POST /sim/inject`` scores a handful of windows, which the CPU
    does instantly. Silently a no-op for every non-torch model.
    """
    mods = getattr(model, "_modules_", None)
    if not mods:
        return
    try:
        import torch

        for m in mods:
            m.to("cpu")
        model.device = torch.device("cpu")
    except Exception as exc:  # pragma: no cover - torch absent or a model without .to()
        LOGGER.warning("demo.scoring: could not move %s to the CPU for pickling (%s)", type(model).__name__, exc)


# --------------------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------------------


def _train_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature median and MAD of the rows the model was fitted on.

    A 3-D ``raw_window`` matrix is reduced to its **last timestep** per channel, which is the
    quantity ``lgbm_residual`` predicts and therefore the one a per-channel z is about.
    """
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[:, -1, :]
    med = np.nanmedian(arr, axis=0)
    mad = np.nanmedian(np.abs(arr - med), axis=0)
    return np.nan_to_num(med, nan=0.0), np.nan_to_num(mad, nan=0.0)


def _attribution_mode(spec: WinnerSpec) -> str:
    """``"model"`` when the winner exposes a per-feature attribution, else ``"robust_z"``.

    ``lgbm_residual`` is the only one that does: it fits one regressor per channel and its
    score *is* the max standardised residual, so the per-channel residuals are the model's own
    statement about which channel is anomalous. ``cusum_cycle_scalar`` reduces the feature
    vector to a scalar before the CUSUM accumulates, and ``sparse_autoencoder`` exposes only
    the row MSE, so both use the contract's robust z.
    """
    return "model" if spec.model == "lgbm_residual" else "robust_z"


def fit_winner(
    subsystem: str,
    *,
    exclude_train: str | None = None,
    index: FleetIndex | None = None,
    root: str | Path | None = None,
    runs_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> FittedWinner:
    """Fit the winner for ``subsystem`` on every train **except** ``exclude_train``.

    The sequence is the runner's, in the runner's order: ``train_rows`` (normal-only, the
    row's contamination) -> ``_fit_model`` -> ``_calibrate`` on the validation carve of the
    training trains. The held-out train's rows are never touched here; :func:`score_held_out`
    scores them afterwards.
    """
    idx = index or load_fleet(subsystem, root=root, runs_path=runs_path, cache_dir=cache_dir)
    spec = idx.spec
    data = idx.data
    split = idx.split_for(exclude_train)
    run_spec = spec.to_run_spec()

    R._import_models()
    tr = D.train_rows(
        data,
        split.train,
        train_regime=spec.train_regime,
        contamination=spec.contamination,
        seed=spec.seed,
        task="ad",
    )
    if tr.size == 0:
        raise ValueError(
            f"fit_winner({subsystem!r}, exclude_train={exclude_train!r}): fold {split.fold} has no "
            f"training rows under train_regime={spec.train_regime!r}"
        )
    model = build(spec.model, **spec.params)
    t0 = time.perf_counter()
    model = R._fit_model(model, data, run_spec, tr, budget_s=spec.max_minutes * 60.0)
    fit_seconds = time.perf_counter() - t0

    if split.val.size == 0:
        raise ValueError(
            f"fit_winner({subsystem!r}, exclude_train={exclude_train!r}): fold {split.fold} has an "
            f"empty validation slice, so there is no operating point to calibrate"
        )
    thr, cal_info = R._calibrate(model, data, run_spec, split.val)
    cal_info.pop("_calibrated_on_index", None)
    if thr is None:
        raise ValueError(
            f"fit_winner({subsystem!r}, exclude_train={exclude_train!r}): uncalibrated - "
            f"{cal_info.get('uncalibrated_reason', 'no reason given')}"
        )

    med, mad = _train_stats(data.X[tr])
    return FittedWinner(
        subsystem=spec.key,
        model_name=spec.model,
        params=dict(spec.params),
        input_kind=spec.input_kind,
        window=spec.window,
        peer_norm=spec.peer_norm,
        threshold=float(thr.primary.value),
        feature_names=list(data.feature_names),
        train_median=med,
        train_mad=mad,
        held_out_train=str(exclude_train)
        if exclude_train is not None
        else (METROPT_TRAIN_ID if spec.dataset == "metropt3" else ""),
        fitted_at=pd.Timestamp.utcnow().isoformat(),
        model=model,
        spec=spec,
        thresholds={name: float(t.value) for name, t in thr.items()},
        attribution=_attribution_mode(spec),
        fit_seconds=float(fit_seconds),
        n_train_rows=int(tr.size),
        n_val_rows=int(split.val.size),
        calibration={
            **{k: _jsonable(v) for k, v in cal_info.items()},
            "budget_val_episodes": int(thr.primary.val_episodes),
            "budget_val_train_days": float(thr.primary.val_train_days),
            "budget_val_far_per_train_day": float(thr.primary.val_far_per_train_day),
            "budget_method": str(thr.primary.method),
            "n_calibrated_on": int(np.asarray(thr.calibrated_on_index).size),
        },
        k_consecutive=int(spec.k_consecutive),
        merge_gap_s=float(spec.merge_gap_s),
        # Measured on THIS table, not copied off the results row: R._calibrate derives its own
        # contiguity budget from the data it is handed, and the demo loads 20 runs where the
        # sweep loaded 8, so the frozen number would count validation episodes under one rule
        # and test episodes under another. spec.max_step_s stays in the manifest beside it.
        max_step_s=float(R._max_step_seconds(data)),
        window_seconds=float(data.window_seconds),
        H=float(spec.H),
    )


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


# --------------------------------------------------------------------------------------
# Score
# --------------------------------------------------------------------------------------


def top_signals(fw: FittedWinner, X: np.ndarray, *, extra_z: np.ndarray | None = None) -> list[str]:
    """``top_signals_json`` for every row of ``X``: at most 5 ``{signal, z, value}``, |z| desc.

    ``extra_z`` overrides the robust z with the model's own per-feature attribution
    (``lgbm_residual``'s per-channel standardised residual); the reported ``value`` is always
    the feature's own value on that row.
    """
    arr = np.asarray(X, dtype=np.float64)
    values = arr[:, -1, :] if arr.ndim == 3 else arr
    z = fw.robust_z(values) if extra_z is None else np.asarray(extra_z, dtype=np.float64)
    names = list(fw.feature_names)
    if z.shape != values.shape:
        raise ValueError(f"top_signals: z has shape {z.shape} but the value matrix has {values.shape}")
    if len(names) != values.shape[1]:
        raise ValueError(f"top_signals: {len(names)} feature names for {values.shape[1]} columns")
    k = min(MAX_TOP_SIGNALS, values.shape[1])
    if k == 0:
        return ["[]"] * values.shape[0]
    order = np.argsort(-np.abs(z), axis=1, kind="stable")[:, :k]
    out: list[str] = []
    for i in range(values.shape[0]):
        row = [
            {"signal": names[j], "z": round(float(z[i, j]), 6), "value": round(float(values[i, j]), 6)}
            for j in order[i]
            if np.isfinite(z[i, j])
        ]
        out.append(json.dumps(row))
    return out


def _model_attribution(fw: FittedWinner, X: np.ndarray) -> np.ndarray | None:
    """``lgbm_residual``'s per-channel **signed** standardised residual, or ``None``.

    This is the model's own score decomposition: it predicts each modelled channel's last
    timestep from every channel's earlier timesteps and takes the max |residual| / residual
    std, so the per-channel residuals are exactly what drove the row's score. Channels the
    model was told not to model get 0.
    """
    if fw.attribution != "model":
        return None
    model = fw.model
    if model is None or not hasattr(model, "models_"):
        return None
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 3:
        return None
    feats, targets = model._features_targets(arr)
    z = np.zeros(targets.shape, dtype=np.float64)
    for ch in model.channels_:
        resid = targets[:, ch] - model.models_[ch].predict(feats)
        z[:, ch] = resid / model.resid_std_[ch]
    return z


def _as_bench(fw: FittedWinner, X: np.ndarray, t_end: np.ndarray, series: np.ndarray, unit: np.ndarray) -> BenchData:
    """A minimal :class:`BenchData` so :func:`nebulax.bench.runner._score` can be reused.

    ``_score`` needs ``X``, ``t_end``, ``series``, ``unit`` and ``feature_names``; the labels
    frame only has to have the right number of rows, and it is never shown to a model.
    """
    n = int(np.asarray(X).shape[0])
    return BenchData(
        dataset="sim" if fw.subsystem != "metropt3" else "metropt3",
        subsystem=fw.subsystem if fw.subsystem != "metropt3" else "pneumatic",
        input_kind=fw.input_kind,
        X=np.asarray(X, dtype=np.float32),
        feature_names=list(fw.feature_names),
        t_start=np.asarray(t_end),
        t_end=np.asarray(t_end),
        unit=np.asarray(unit, dtype=object),
        group=np.asarray(unit, dtype=object),
        labels=pd.DataFrame({"is_faulty": np.zeros(n, dtype=bool)}),
        events=[],
        window_seconds=float(fw.window_seconds),
        series=np.asarray(series, dtype=object),
    )


def _alert_flags(
    fw: FittedWinner, scores: np.ndarray, t_end: np.ndarray, series: np.ndarray, *, scoreable: np.ndarray | None = None
) -> np.ndarray:
    """``alert`` per row: the row belongs to an alarm episode (contract section 3).

    Rows are assumed sorted by ``(series, t_end)`` so an episode's ``[i_start, i_end]`` is the
    contiguous block of rows it covers. ``scoreable`` restricts episode detection to the rows
    the benchmark measures on (the sim blanks everything after a component's ``t_failure``);
    excluded rows keep their score and are never part of an episode.
    """
    n = int(np.asarray(scores).size)
    keep = np.ones(n, dtype=bool) if scoreable is None else np.asarray(scoreable, dtype=bool)
    idx = np.flatnonzero(keep)
    alert = np.zeros(n, dtype=bool)
    if idx.size == 0:
        return alert
    eps = M.episodes(
        np.asarray(scores)[idx],
        np.asarray(t_end)[idx],
        fw.threshold,
        k=fw.k_consecutive,
        merge_gap_s=fw.merge_gap_s,
        units=np.asarray(series, dtype=object)[idx],
        max_step_s=fw.max_step_s,
    )
    for e in eps:
        alert[idx[e.i_start : e.i_end + 1]] = True
    return alert


def _scores_frame(
    fw: FittedWinner,
    *,
    t_end: np.ndarray,
    train_id: np.ndarray,
    car: np.ndarray,
    subsystem: np.ndarray,
    component_id: np.ndarray,
    scores: np.ndarray,
    alert: np.ndarray,
    top: Sequence[str],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(pd.Series(t_end), utc=True),
            "train_id": np.asarray(train_id, dtype=object),
            "car": np.asarray(car, dtype=np.int16),
            "subsystem": np.asarray(subsystem, dtype=object),
            "component_id": np.asarray(component_id, dtype=object),
            "model": np.full(len(scores), fw.model_name, dtype=object),
            "score": np.asarray(scores, dtype=np.float64),
            "threshold": np.full(len(scores), float(fw.threshold), dtype=np.float64),
            "alert": np.asarray(alert, dtype=bool),
            "top_signals_json": list(top),
        }
    )
    return S.coerce_scores(frame)


def _sorted_by_series_time(series: np.ndarray, t_end: np.ndarray) -> np.ndarray:
    """Row order ``(series, t_end)`` - what :func:`_alert_flags` assumes."""
    s = np.asarray(series, dtype=object).astype(str)
    t = M.to_epoch_seconds(t_end)
    return np.lexsort((t, s))


def score_held_out(fw: FittedWinner, index: FleetIndex, exclude_train: str | None = None) -> pd.DataFrame:
    """Score the held-out train's rows of ``index`` with ``fw``; ``SCORES_COLUMNS``.

    This is the batch path: the table is the fleet's, so peer features are the real ones
    (a door's sibling door is in the table) and the episode contiguity budget is the run's.
    """
    data = index.data
    split = index.split_for(fw.held_out_train or exclude_train)
    test = np.asarray(split.test, dtype=np.int64)
    if test.size == 0:
        return S.empty_scores()
    scores = R._score(fw.model, data, test)
    order = _sorted_by_series_time(data.series[test], data.t_end[test])
    test = test[order]
    scores = scores[order]

    labels = data.labels.iloc[test]
    series = np.asarray(data.series, dtype=object)[test]
    scoreable = data.masks["scoreable"][test]
    alert = _alert_flags(fw, scores, data.t_end[test], series, scoreable=scoreable)
    X = data.X[test]
    top = top_signals(fw, X, extra_z=_model_attribution(fw, X))
    if index.spec.dataset == "metropt3":
        # MetroPT-3 is one real APU published as the extra unit MP3 (contract sections 1, 3);
        # its loader carries no component or car columns, because it has exactly one of each.
        train_ids = np.full(test.size, METROPT_TRAIN_ID, dtype=object)
        cars = np.full(test.size, METROPT_CAR, dtype=np.int16)
        comps = np.full(test.size, METROPT_COMPONENT, dtype=object)
    else:
        train_ids = labels["train_id"].astype(str).to_numpy()
        cars = _car_column(labels, index.root, len(test))
        comps = labels["component_id"].astype(str).to_numpy()
    return _scores_frame(
        fw,
        t_end=data.t_end[test],
        train_id=train_ids,
        car=cars,
        subsystem=np.full(test.size, index.spec.dataset_subsystem, dtype=object),
        component_id=comps,
        scores=scores,
        alert=alert,
        top=top,
    )


def _car_column(labels: pd.DataFrame, root: Path, n: int) -> np.ndarray:
    """The ``car`` of every row, from the labels frame or the fleet index.

    ``cycle_features`` carries ``car`` per row; the streamed ``window_stats`` path does not,
    so it is looked up from ``<root>/index.json`` by ``run_id`` (the pneumatic runs are all
    unit-level, ``car=0``).
    """
    if "car" in labels.columns:
        return labels["car"].to_numpy(dtype=np.int16)
    cars = _run_cars(root)
    if "run_id" in labels.columns and cars:
        return labels["run_id"].astype(str).map(cars).fillna(0).to_numpy(dtype=np.int16)
    return np.zeros(n, dtype=np.int16)


def _run_cars(root: Path) -> dict[str, int]:
    path = Path(root) / "index.json"
    if not path.exists():
        return {}
    try:
        runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
    except Exception:  # pragma: no cover - a malformed index is not worth failing a score run
        return {}
    return {str(r["run_id"]): int(r.get("car", 0)) for r in runs}


# --------------------------------------------------------------------------------------
# The in-memory path: score one run's frames (POST /sim/inject)
# --------------------------------------------------------------------------------------


def _align(mat: np.ndarray, names: Sequence[str], want: Sequence[str]) -> np.ndarray:
    """Reorder ``mat``'s columns to ``want``; a column ``names`` does not have becomes NaN."""
    col = {str(n): i for i, n in enumerate(names)}
    n = mat.shape[0]
    blocks = [
        mat[:, col[str(w)]] if str(w) in col else np.full(n, np.nan, dtype=np.float64) for w in want
    ]
    return np.stack(blocks, axis=1)


def _cycle_matrix(fw: FittedWinner, features: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """``X`` for a ``cycle_features`` winner from one run's ``features.parquet`` frame.

    Peer columns are built with :func:`nebulax.features.cycles.peer_normalise` under the
    subsystem's own :class:`~nebulax.bench.data.PeerGrouping`, so a **bearing** run - whose
    eight axle boxes are all in the one run - gets exactly the values the fleet loader
    computes.

    **A lone door has no peers.** The loader refuses that case outright
    (``_sim_peer_normalise`` raises on singleton groups, deliberately, so an ablation cannot
    silently measure nothing); here refusing is not an option, because ``POST /sim/inject``
    regenerates one door leaf and must still score it. The documented fallback is the loader's
    own downstream behaviour: the peer columns come back all-NaN and
    :func:`nebulax.bench.data._finite` turns them into 0.0, i.e. "this door is exactly average
    against its peers". Scores on an injected lone door are therefore driven by its absolute
    features only and are not identical to the fleet-scored ones; pass the sibling run's rows
    in ``features`` (both doors of the train) to get the real peer comparison.
    """
    feats = features.reset_index(drop=True).copy()
    if fw.peer_norm:
        from nebulax.features.cycles import peer_normalise

        grouping = D._SIM_PEER_GROUPS.get(fw.subsystem)
        if grouping is None:
            raise ValueError(
                f"score_frames: peer_norm=True but subsystem {fw.subsystem!r} has no peer grouping"
            )
        missing = [c for c in grouping.group_cols if c not in feats.columns]
        if missing:
            raise ValueError(f"score_frames: peer grouping needs column(s) {missing}")
        value_cols = [c for c in S.feature_columns(feats) if pd.api.types.is_numeric_dtype(feats[c])]
        feats = peer_normalise(
            feats, grouping.group_cols, value_cols=value_cols, same_side=grouping.same_side
        ).drop(columns=["_peer_side"], errors="ignore")
        feats = D._drop_undefined_peer_columns(features, feats, fw.subsystem, grouping.members_per_group)
    mat, names = D._numeric_matrix(feats, S.feature_columns(feats))
    return D._finite(_align(mat, names, fw.feature_names).astype(np.float32)), feats


def _window_matrix(fw: FittedWinner, long: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """``X`` and its key columns for a ``window_stats`` winner from one run's long telemetry.

    Built exactly the way :func:`nebulax.bench.data._sim_windows` builds it: ``to_wide`` over
    the subsystem's registered signals, ``make_windows`` at the run's native rate with the
    loader's gap budget, then :func:`nebulax.features.stats.window_stats`.
    """
    from nebulax.features.stats import feature_names as stat_feature_names
    from nebulax.features.stats import window_stats
    from nebulax.features.windows import make_windows

    sub = fw.subsystem
    fs = D._SIM_FS[sub]
    sigs = list(S.SIGNALS[sub])
    L = int(fw.window or 0)
    if L < 2:
        raise ValueError(f"score_frames: window_stats needs a window >= 2, got {fw.window!r}")
    wide = S.to_wide(long, sub, signals=sigs)
    if wide.empty:
        raise ValueError(f"score_frames: the long frame has no {sub!r} rows")
    W = make_windows(wide, sigs, L=L, stride=max(1, L // 2), fs=fs, max_gap_s=max(3.0 / fs, 5.0))
    if len(W) == 0:
        raise ValueError(
            f"score_frames: no windows survived gap splitting (L={L}, {len(wide)} samples); the "
            f"injected window is shorter than one model window"
        )
    Xw = np.ascontiguousarray(W.X, dtype=np.float32)
    mat = window_stats(Xw)
    names = stat_feature_names(sigs)
    keys = pd.DataFrame(
        {
            "t_end": D._as_ms(W.t_end),
            "train_id": np.asarray(W.train_id, dtype=object).astype(str),
            "component_id": np.asarray(W.component_id, dtype=object).astype(str),
            "run_id": (np.asarray(W.run_id, dtype=object).astype(str) if W.run_id is not None else ""),
            "car": (np.asarray(W.car, dtype=object) if W.car is not None else 0),
        }
    )
    return D._finite(_align(mat, names, fw.feature_names).astype(np.float32)), keys


def score_frames(
    fw: FittedWinner,
    long: pd.DataFrame | None,
    features: pd.DataFrame | None,
    *,
    train_id: str,
    car: int,
    component_id: str,
) -> pd.DataFrame:
    """Score one run's in-memory frames with a fitted winner; returns ``SCORES_COLUMNS``.

    This is the path ``POST /api/sim/inject`` takes: the simulator hands back a freshly
    generated run's ``long`` and ``features`` frames and they are scored without touching the
    disk and without refitting.

    ``train_id`` / ``car`` / ``component_id`` are the **defaults**: whenever the frames carry
    those key columns per row (they always do for a run written by
    :func:`nebulax.schema.write_dataset`) the per-row values win, so a bearing run's eight
    axle boxes come back as eight series rather than one.
    """
    if fw.input_kind == "cycle_features":
        if features is None or features.empty:
            raise ValueError("score_frames: a cycle_features winner needs the run's features frame")
        X, feats = _cycle_matrix(fw, features)
        t_end = D._as_ms(feats["t_end"])
        keys = feats
    elif fw.input_kind == "window_stats":
        if long is None or long.empty:
            raise ValueError("score_frames: a window_stats winner needs the run's long telemetry")
        X, keys = _window_matrix(fw, long)
        t_end = np.asarray(keys["t_end"])
    else:
        raise NotImplementedError(
            f"score_frames: input_kind={fw.input_kind!r} has no in-memory path. The MetroPT-3 "
            f"winner (raw_window) is scored by scripts/score_for_demo.py from the real recording; "
            f"the simulator cannot regenerate it."
        )

    n = X.shape[0]
    tid = keys["train_id"].astype(str).to_numpy() if "train_id" in keys else np.full(n, str(train_id), dtype=object)
    cid = (
        keys["component_id"].astype(str).to_numpy()
        if "component_id" in keys
        else np.full(n, str(component_id), dtype=object)
    )
    cars = keys["car"].to_numpy(dtype=np.int16) if "car" in keys else np.full(n, int(car), dtype=np.int16)
    # The alarm series is the component, and on this fleet a component is
    # (train, car, component_id): two bogies of one train carry the same eight axle-box names.
    series = np.asarray([f"{a}/{c}/{b}" for a, b, c in zip(tid, cid, cars)], dtype=object)

    order = _sorted_by_series_time(series, t_end)
    X, t_end, tid, cid, cars, series = X[order], t_end[order], tid[order], cid[order], cars[order], series[order]

    bench = _as_bench(fw, X, t_end, series, tid)
    scores = R._score(fw.model, bench, np.arange(n, dtype=np.int64))
    alert = _alert_flags(fw, scores, t_end, series)
    top = top_signals(fw, X, extra_z=_model_attribution(fw, X))
    return _scores_frame(
        fw,
        t_end=t_end,
        train_id=tid,
        car=cars,
        subsystem=np.full(n, fw.subsystem, dtype=object),
        component_id=cid,
        scores=scores,
        alert=alert,
        top=top,
    )


def score_run(fw: FittedWinner, run_dir: str | Path) -> pd.DataFrame:
    """Score one written run partition (``source=sim/run_id=<run>/``) with ``fw``.

    Reads only the tables the winner needs - ``features.parquet`` for a ``cycle_features``
    winner, ``telemetry.parquet`` for a ``window_stats`` one - and delegates to
    :func:`score_frames`, so the two paths cannot drift apart. The run is read **in
    isolation**: see :func:`_cycle_matrix` for what that means for a lone door's peer columns.
    """
    part = Path(run_dir)
    feats: pd.DataFrame | None = None
    long: pd.DataFrame | None = None
    if fw.input_kind == "cycle_features":
        fpath = part / "features.parquet"
        if not fpath.exists():
            raise FileNotFoundError(f"score_run: {fpath} does not exist")
        feats = pd.read_parquet(fpath, engine="pyarrow")
    else:
        lpath = part / "telemetry.parquet"
        if not lpath.exists():
            raise FileNotFoundError(f"score_run: {lpath} does not exist")
        long = pd.read_parquet(lpath, engine="pyarrow")
    ref = feats if feats is not None else long
    if ref is None or ref.empty:
        return S.empty_scores()
    return score_frames(
        fw,
        long,
        feats,
        train_id=str(ref["train_id"].iloc[0]),
        car=int(ref["car"].iloc[0]) if "car" in ref.columns else 0,
        component_id=str(ref["component_id"].iloc[0]),
    )


# --------------------------------------------------------------------------------------
# Episodes
# --------------------------------------------------------------------------------------

#: ``episodes.parquet`` (contract section 3).
EPISODE_COLUMNS: tuple[str, ...] = (
    "train_id",
    "car",
    "subsystem",
    "component_id",
    "model",
    "t_start",
    "t_end",
    "peak_score",
    "n_rows",
    "episode_id",
    "fault_type",
    "t_onset",
    "t_failure",
    "lead_to_failure_h",
    "matched",
    #: The detection horizon this row's ``matched`` was decided with, in seconds, so one file
    #: cannot silently mix the sim's 72 h and MetroPT-3's 48 h (contract section 3).
    "H_s",
)


def empty_episodes() -> pd.DataFrame:
    """An empty, correctly typed ``episodes.parquet`` frame."""
    return pd.DataFrame(
        {
            "train_id": pd.Series(dtype="object"),
            "car": pd.Series(dtype="int16"),
            "subsystem": pd.Series(dtype="object"),
            "component_id": pd.Series(dtype="object"),
            "model": pd.Series(dtype="object"),
            "t_start": pd.Series(dtype=S.TIMESTAMP_DTYPE),
            "t_end": pd.Series(dtype=S.TIMESTAMP_DTYPE),
            "peak_score": pd.Series(dtype="float32"),
            "n_rows": pd.Series(dtype="int32"),
            "episode_id": pd.Series(dtype="object"),
            "fault_type": pd.Series(dtype="object"),
            "t_onset": pd.Series(dtype=S.TIMESTAMP_DTYPE),
            "t_failure": pd.Series(dtype=S.TIMESTAMP_DTYPE),
            "lead_to_failure_h": pd.Series(dtype="float32"),
            "matched": pd.Series(dtype="bool"),
            "H_s": pd.Series(dtype="float64"),
        }
    )


#: How :mod:`nebulax.bench.data` blanks rows from ``masks["scoreable"]``, per dataset. Both
#: rules the demo's winners meet are a function of the fault log alone, so a published
#: ``scores.parquet`` can be re-measured without reloading the feature table.
SCOREABLE_RULES: tuple[str, ...] = ("post_failure", "post_repair", "all")


def scoreable_from_fault_log(
    scores: pd.DataFrame,
    fault_log: pd.DataFrame | None,
    *,
    rule: str = "post_failure",
    post_repair_s: float = D.POST_REPAIR_S,
) -> np.ndarray:
    """The rows the benchmark measures on, re-derived from a scores frame and the fault log.

    * ``post_failure`` (the synthetic fleet) - every row of a component **after** its
      ``t_failure`` is blanked. The simulator models no repair, so the blanking runs to the end
      of the run (:func:`nebulax.bench.data._sim_post_failure_mask`).
    * ``post_repair`` (MetroPT-3) - the ``post_repair_s`` (24 h) after a failure
      (:func:`nebulax.bench.data._metro_masks`). MetroPT additionally blanks ``is_transition``
      windows, which are a property of the raw digital channels and **cannot** be recovered
      from a scores frame; the caller must say so where it matters.
    * ``all`` - nothing is blanked.

    Returns a bool array aligned with ``scores``' own row order.
    """
    if rule not in SCOREABLE_RULES:
        raise ValueError(f"scoreable_from_fault_log: unknown rule {rule!r}; expected one of {list(SCOREABLE_RULES)}")
    n = int(len(scores))
    keep = np.ones(n, dtype=bool)
    if n == 0 or rule == "all" or fault_log is None or len(fault_log) == 0:
        return keep
    t = M.to_epoch_seconds(pd.to_datetime(scores["timestamp"], utc=True).to_numpy())
    key = _series_key(scores)
    fl = fault_log
    f_key = _series_key(fl)
    f_fail = M.to_epoch_seconds(pd.to_datetime(fl["t_failure"], utc=True).to_numpy())
    for i in range(len(fl)):
        if not np.isfinite(f_fail[i]):
            continue
        same = key == f_key[i]
        if rule == "post_failure":
            keep &= ~(same & (t > f_fail[i]))
        else:
            keep &= ~(same & (t > f_fail[i]) & (t <= f_fail[i] + float(post_repair_s)))
    return keep


def _series_key(frame: pd.DataFrame) -> np.ndarray:
    """``train_id/car/subsystem/component_id`` per row - the series an alarm is raised on.

    The alarm series of this fleet is the component, and a component is
    ``(train, car, component_id)``: the two bogies of one train carry the same eight axle-box
    names, so ``car`` is part of the key (it is what tells the loader's ``run_id/component_id``
    series apart once the rows are published under train ids).
    """
    car = frame["car"].to_numpy()
    car = np.asarray(car, dtype=np.int64) if len(frame) else np.zeros(0, dtype=np.int64)
    return np.asarray(
        [
            f"{a}/{int(c)}/{sub}/{b}"
            for a, c, sub, b in zip(
                frame["train_id"].astype(str),
                car,
                frame["subsystem"].astype(str),
                frame["component_id"].astype(str),
            )
        ],
        dtype=object,
    )


def episodes_from_scores(
    scores: pd.DataFrame,
    fault_log: pd.DataFrame | None,
    *,
    H: float = 3 * 86400.0,
    k: int = M.DEFAULT_K_CONSECUTIVE,
    merge_gap_s: float = M.DEFAULT_MERGE_GAP_S,
    max_step_s: float | None = None,
    scoreable: Any = None,
) -> pd.DataFrame:
    """One row per alarm episode, joined to the ground-truth fault log.

    The episode rule is the contract's and the benchmark's, applied by
    :func:`nebulax.bench.metrics.episodes` itself: ``score > threshold`` for ``>= k`` rows
    **consecutive in time** on one series, runs cut wherever the step exceeds ``max_step_s``,
    qualifying runs merged when they are less than ``merge_gap_s`` apart. Nothing here
    re-derives an episode from array adjacency: a door's ~5 h nightly out-of-service gap sits
    between two array-adjacent rows, and two alarm blocks either side of it are two episodes,
    not one.

    Parameters
    ----------
    k, merge_gap_s, max_step_s
        The fold's own episode definition - :attr:`FittedWinner.k_consecutive`,
        ``merge_gap_s`` and ``max_step_s`` (``episode_max_step_s`` in ``manifest.json``), the
        same numbers :func:`_alert_flags` wrote the ``alert`` column with. ``max_step_s=None``
        measures it from the frame itself (:func:`nebulax.bench.metrics.max_step_seconds` on
        the per-series row pitch), which is what the in-memory ``POST /api/sim/inject`` path
        gets.
    scoreable
        Bool array aligned with ``scores``: the rows the benchmark measures on. ``None``
        re-derives it from the frame - a row **above** its threshold that is not flagged
        ``alert`` was excluded from episode detection when the column was written, and
        dropping it cannot change the episode structure (such rows only ever form blocks
        shorter than ``k``). Pass the real mask (see :func:`scoreable_from_fault_log`) when
        you have it.

    ``matched`` is the benchmark's own rule (:func:`nebulax.bench.metrics.event_metrics`): the
    episode overlaps a fault's detection window ``[t_onset - H, t_failure]`` on the **same
    component**. ``lead_to_failure_h = (t_failure - t_start) / 3600`` for a matched episode,
    NaN otherwise - and NaN too when the matched fault has no failure time. ``H_s`` records
    the ``H`` the row was decided with.
    """
    if scores is None or scores.empty:
        return empty_episodes()
    df = scores.copy()
    for col in ("train_id", "subsystem", "component_id", "model"):
        df[col] = df[col].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    order = np.lexsort(
        (
            M.to_epoch_seconds(df["timestamp"].to_numpy()),
            df["component_id"].to_numpy().astype(str),
            df["subsystem"].to_numpy().astype(str),
            df["car"].to_numpy(dtype=np.int64),
            df["train_id"].to_numpy().astype(str),
        )
    )
    keep_all = np.ones(len(df), dtype=bool) if scoreable is None else np.asarray(scoreable, dtype=bool).reshape(-1)
    if keep_all.size != len(df):
        raise ValueError(f"episodes_from_scores: scoreable has {keep_all.size} rows for {len(df)} scores")
    df = df.iloc[order].reset_index(drop=True)
    keep_all = keep_all[order]

    score = df["score"].to_numpy(dtype=np.float64)
    thr = df["threshold"].to_numpy(dtype=np.float64)
    above = score > thr
    alert = df["alert"].to_numpy(dtype=bool)
    if scoreable is None:
        # An above-threshold row the writer did not flag was not in the population episodes
        # were detected on; see the docstring.
        keep_all = ~(above & ~alert)

    idx = np.flatnonzero(keep_all)
    if idx.size == 0:
        return empty_episodes()
    series = _series_key(df)
    t = M.to_epoch_seconds(df["timestamp"].to_numpy())
    step = (
        M.max_step_seconds(None, M.row_pitch_seconds(t[idx], series[idx]))
        if max_step_s is None
        else float(max_step_s)
    )
    # ``above`` is evaluated per row against that row's OWN threshold (the ten
    # leave-one-train-out folds each calibrate their own), so the flag - not the score - is
    # what metrics.episodes thresholds. peak_score is read back off the real scores.
    eps = M.episodes(
        above[idx].astype(np.float64),
        t[idx],
        0.5,
        k=int(k),
        merge_gap_s=float(merge_gap_s),
        units=series[idx],
        max_step_s=step,
    )

    rows: list[dict[str, Any]] = []
    for e in eps:
        block = idx[e.i_start : e.i_end + 1]
        rows.append(
            {
                "train_id": str(df["train_id"].iloc[block[0]]),
                "car": int(df["car"].iloc[block[0]]),
                "subsystem": str(df["subsystem"].iloc[block[0]]),
                "component_id": str(df["component_id"].iloc[block[0]]),
                "model": str(df["model"].iloc[block[0]]),
                "t_start": df["timestamp"].iloc[block[0]],
                "t_end": df["timestamp"].iloc[block[-1]],
                "peak_score": float(score[block].max()),
                "n_rows": int(e.n_windows),
                "_sort": float(e.t_start),
            }
        )
    if not rows:
        return empty_episodes()
    eps_df = pd.DataFrame(rows).sort_values(
        ["train_id", "car", "subsystem", "component_id", "_sort"], kind="stable"
    ).reset_index(drop=True)
    counter: dict[tuple, int] = {}
    ids: list[str] = []
    for r in eps_df.itertuples():
        key = (r.train_id, int(r.car), r.subsystem, r.component_id)
        n_seen = counter.get(key, 0)
        counter[key] = n_seen + 1
        ids.append(f"{r.train_id}-{r.component_id}-{n_seen}")
    eps_df["episode_id"] = ids
    return _join_fault_log(eps_df.drop(columns=["_sort"]), fault_log, H=H)


def _join_fault_log(eps: pd.DataFrame, fault_log: pd.DataFrame | None, *, H: float) -> pd.DataFrame:
    n = len(eps)
    fault_type = np.full(n, "", dtype=object)
    t_onset = np.full(n, np.datetime64("NaT", "ms"), dtype="datetime64[ms]")
    t_failure = np.full(n, np.datetime64("NaT", "ms"), dtype="datetime64[ms]")
    lead = np.full(n, np.nan, dtype=np.float64)
    matched = np.zeros(n, dtype=bool)

    if n and fault_log is not None and len(fault_log):
        fl = fault_log.copy()
        for col in ("train_id", "subsystem", "component_id"):
            fl[col] = fl[col].astype(str)
        fl["t_onset"] = pd.to_datetime(fl["t_onset"], utc=True)
        fl["t_failure"] = pd.to_datetime(fl["t_failure"], utc=True)
        fl_car = fl["car"].to_numpy(dtype=np.int64) if "car" in fl.columns else np.zeros(len(fl), dtype=np.int64)
        e_start = M.to_epoch_seconds(eps["t_start"].to_numpy())
        e_end = M.to_epoch_seconds(eps["t_end"].to_numpy())
        f_on = M.to_epoch_seconds(fl["t_onset"].to_numpy())
        f_fail = M.to_epoch_seconds(fl["t_failure"].to_numpy())
        for f in range(len(fl)):
            same = (
                (eps["train_id"].to_numpy() == fl["train_id"].iloc[f])
                & (eps["car"].to_numpy().astype(np.int64) == fl_car[f])
                & (eps["subsystem"].to_numpy() == fl["subsystem"].iloc[f])
                & (eps["component_id"].to_numpy() == fl["component_id"].iloc[f])
            )
            lo = f_on[f] - float(H)
            hi = f_fail[f] if np.isfinite(f_fail[f]) else np.inf
            hit = same & (e_end >= lo) & (e_start <= hi) & ~matched
            if not hit.any():
                continue
            matched |= hit
            fault_type[hit] = str(fl["fault_type"].iloc[f])
            t_onset[hit] = np.datetime64(pd.Timestamp(fl["t_onset"].iloc[f]).tz_localize(None), "ms")
            if pd.notna(fl["t_failure"].iloc[f]):
                t_failure[hit] = np.datetime64(pd.Timestamp(fl["t_failure"].iloc[f]).tz_localize(None), "ms")
                lead[hit] = (f_fail[f] - e_start[hit]) / 3600.0

    out = eps.copy()
    out["fault_type"] = fault_type
    out["t_onset"] = pd.to_datetime(pd.Series(t_onset), utc=True)
    out["t_failure"] = pd.to_datetime(pd.Series(t_failure), utc=True)
    out["lead_to_failure_h"] = lead.astype(np.float32)
    out["matched"] = matched
    # One file carries both horizons (sim 72 h, MetroPT-3 48 h); record which one decided
    # this row rather than leaving a reader to guess from the subsystem.
    out["H_s"] = np.full(n, float(H), dtype=np.float64)
    if out.empty:
        return empty_episodes()
    out["car"] = out["car"].astype("int16")
    out["n_rows"] = out["n_rows"].astype("int32")
    out["peak_score"] = out["peak_score"].astype("float32")
    out["t_start"] = pd.to_datetime(out["t_start"], utc=True).astype(S.TIMESTAMP_DTYPE)
    out["t_end"] = pd.to_datetime(out["t_end"], utc=True).astype(S.TIMESTAMP_DTYPE)
    return out[list(EPISODE_COLUMNS)].reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Manifest helpers (used by scripts/score_for_demo.py)
# --------------------------------------------------------------------------------------


def faults_detected(episodes: pd.DataFrame, fault_log: pd.DataFrame, *, H: float) -> np.ndarray:
    """One bool per fault-log row: some episode on that component overlaps ``[t_onset-H, t_failure]``.

    Computed independently of ``episodes["matched"]``, which assigns each episode to at most
    one fault so that its ``fault_type`` column is unambiguous. Four MetroPT-3 leaks share one
    component, so an episode spanning two of their windows would otherwise credit only the
    first, and the recall denominator and numerator would disagree.
    """
    n = int(len(fault_log))
    out = np.zeros(n, dtype=bool)
    if n == 0 or episodes is None or len(episodes) == 0:
        return out
    e_start = M.to_epoch_seconds(episodes["t_start"].to_numpy())
    e_end = M.to_epoch_seconds(episodes["t_end"].to_numpy())
    e_key = list(
        zip(
            episodes["train_id"].astype(str),
            episodes["car"].astype(int),
            episodes["subsystem"].astype(str),
            episodes["component_id"].astype(str),
        )
    )
    f_on = M.to_epoch_seconds(pd.to_datetime(fault_log["t_onset"], utc=True).to_numpy())
    f_fail = M.to_epoch_seconds(pd.to_datetime(fault_log["t_failure"], utc=True).to_numpy())
    cars = fault_log["car"].to_numpy(dtype=np.int64) if "car" in fault_log.columns else np.zeros(n, dtype=np.int64)
    for i in range(n):
        key = (
            str(fault_log["train_id"].iloc[i]),
            int(cars[i]),
            str(fault_log["subsystem"].iloc[i]),
            str(fault_log["component_id"].iloc[i]),
        )
        same = np.asarray([k == key for k in e_key], dtype=bool)
        hi = f_fail[i] if np.isfinite(f_fail[i]) else np.inf
        out[i] = bool((same & (e_end >= f_on[i] - float(H)) & (e_start <= hi)).any())
    return out


def event_summary(
    episodes: pd.DataFrame,
    fault_log: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    H: float = 3 * 86400.0,
    scoreable: Any = None,
) -> dict[str, Any]:
    """Events total / detected, false episodes and scored train-days for one subsystem.

    ``scored_train_days`` is :func:`nebulax.bench.metrics.train_days` on the **scoreable**
    rows - the population the benchmark's own ``train_days`` (and therefore the false-alarm
    budget in ``results/runs.parquet``) is measured on - counted per train, never per
    component. ``scoreable=None`` counts every row, which is only right for a frame that has
    no blanked rows.
    """
    n_events = int(len(fault_log))
    detected = int(faults_detected(episodes, fault_log, H=H).sum())
    n_false = int((~episodes["matched"].astype(bool)).sum()) if len(episodes) else 0
    rows = scores
    if scoreable is not None and len(scores):
        keep = np.asarray(scoreable, dtype=bool).reshape(-1)
        if keep.size != len(scores):
            raise ValueError(f"event_summary: scoreable has {keep.size} rows for {len(scores)} scores")
        rows = scores[keep]
    days = (
        float(M.train_days(rows["timestamp"].to_numpy(), rows["train_id"].astype(str).to_numpy()))
        if len(rows)
        else 0.0
    )
    return {
        "events_total": n_events,
        "events_detected": detected,
        "episodes_total": int(len(episodes)),
        "episodes_matched": int(episodes["matched"].astype(bool).sum()) if len(episodes) else 0,
        "episodes_false": n_false,
        "n_rows_scoreable": int(len(rows)),
        "scored_train_days": days,
        "false_episodes_per_scored_train_day": (n_false / days) if days > 0 else float("nan"),
    }


def fleet_fault_log(root: str | Path = SIM_ROOT, *, subsystem: str | None = None) -> pd.DataFrame:
    """The fleet's canonical ground truth, optionally one subsystem."""
    path = Path(root) / D.SIM_FAULT_LOG
    if not path.exists():
        raise FileNotFoundError(f"demo.scoring.fleet_fault_log: {path} does not exist")
    fl = pd.read_parquet(path)
    if subsystem is not None:
        fl = fl[fl["subsystem"].astype(str) == str(subsystem)]
    return fl.reset_index(drop=True)


def git_rev() -> str:
    return R.git_rev()
