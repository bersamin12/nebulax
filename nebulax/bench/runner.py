"""The benchmark runner: config -> run specs -> one parquet row per run.

What the runner guarantees (plan "Evaluation protocol", ``configs/model_ladder.yaml``)
--------------------------------------------------------------------------------------
* **Test data cannot reach calibration or fitting.** :func:`run_one` calls
  :func:`_fit_model` with training indices, then :func:`_calibrate` with validation indices,
  and only *afterwards* computes any test score. The test score array does not exist as a
  Python object while either of those runs, and both take explicit index arrays so a test can
  monkeypatch them and assert it (``tests/test_bench_runner.py``).
* **One honest run per config.** ``runs_per_config: 1``; no best-of-N, no point adjustment,
  no re-thresholding on test.
* **Nothing is lost.** A crashed run becomes ``status="failed"`` with its traceback in the
  row; a run past its wall-clock cap becomes ``status="timeout"``. Both still produce a row,
  so the leaderboard can say "this model did not finish" instead of silently omitting it.
* **Reproducible.** Every row carries the full :class:`RunSpec`, its ``config_hash``, the
  seed and the git revision, plus the split audit parquet the row was computed from.

Execution model
---------------
Classical runs go into a bounded pool of worker **processes** (each run in its own process,
so a segfaulting C extension costs one row, not the sweep) with a per-run wall-clock cap
enforced by ``Process.join(timeout)`` + ``terminate()``. Deep / GPU runs are serialised one
at a time - the box has a single RTX A4500 - and are additionally handed a ``budget_s``
training budget and a 50 k-window training cap with early stopping left to the model.
``max_workers=0`` runs everything inline in this process (used by the tests and by
``--dry-run``); the wall-clock cap is then advisory, because you cannot preempt a C loop.
"""

from __future__ import annotations

import copy

import hashlib
import itertools
import json
import logging
import multiprocessing as mp
import os
import resource
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from nebulax.bench import base as B
from nebulax.bench import metrics as M
from nebulax.bench import thresholds as T
from nebulax.bench.data import BenchData, load_bench, majority_vote, train_rows
from nebulax.bench.registry import build, get_spec
from nebulax.bench.splits import Split, check_preset_dataset, make_split, splits_to_frame

__all__ = [
    "FORBIDDEN_COMBOS",
    "RunSpec",
    "expand",
    "run_one",
    "run",
    "load_config",
    "git_rev",
    "DEEP_FAMILIES",
    "MAX_DEEP_TRAIN_WINDOWS",
    "ROW_COLUMNS",
]

LOGGER = logging.getLogger(__name__)

#: Families the runner treats as "deep": serialised on the GPU, budgeted, train-capped.
DEEP_FAMILIES: frozenset[str] = frozenset(
    {
        "autoencoder",
        "tsfm_ad",
        "tsfm",
        "deep_sequence",
        "transformer",
        "foundation_model",
        "graph_neural_network",
        "reconstruction_deep",
        "forecast_deep",
    }
)
#: ``deep models train on <= 50k windows with early stopping``.
MAX_DEEP_TRAIN_WINDOWS: int = 50_000

_TASKS = ("ad", "cls", "cpd", "rul")
_TRAIN_REGIMES = ("normal_only", "all")

#: Keys of a run block that are ablation axes: a list value means "one run per element".
ABLATION_KEYS: tuple[str, ...] = (
    "model",
    "input_kind",
    "split",
    "window",
    "feature_set",
    "train_regime",
    "H",
    "peer_norm",
    "contamination",
    "seed",
    "subsystem",
    "task",
)

#: The loader keywords that an ablation axis owns. A ``data:`` block may not set any of them:
#: the results row is written from the :class:`RunSpec` axis, so a ``data:`` override would
#: publish one value and load another. ``peer_norm`` is the dangerous one - it is the axis
#: :func:`_check_peer_norm_split` reads - but ``H``, ``feature_set`` and ``window`` matter too:
#: ``H`` drives ``labels['alarm_window']`` inside the loader while the metrics use ``spec.H``,
#: so a mismatch puts two different horizons inside a single run.
_AXIS_LOADER_KEYS: frozenset[str] = frozenset(
    {"input_kind", "feature_set", "H", "peer_norm", "subsystem", "window"}
)


# --------------------------------------------------------------------------------------
# RunSpec
# --------------------------------------------------------------------------------------


#: ``(dataset, input_kind, task)`` combinations that are refused outright, with the reason
#: printed in the error. These are not merely weak configurations - they produce numbers that
#: look excellent for a reason that has nothing to do with fault detection.
FORBIDDEN_COMBOS: dict[tuple[str, str, str], str] = {
    ("metropt3", "cycle_features", "ad"): (
        "metropt3 + cycle_features + task=ad is a degenerate pointwise target and is refused. "
        "Only 6 of the ~10,395 compressor cycles carry is_faulty=True, because an air-leak "
        "episode collapses into one multi-day cycle; the pointwise label therefore means "
        "'this cycle is very long', and the single features tower_switches / purge_count reach "
        "AUROC ~0.999 against it. MetroPT-3 is an event-scored anomaly-detection dataset: use "
        "input_kind=window_stats or raw_window (the 10 s aggregate table) for task=ad, and keep "
        "cycle_features for per-cycle diagnostics, not for a published AD metric."
    ),
    # task=cpd runs through the identical pointwise code path against the identical y_binary
    ("metropt3", "cycle_features", "cpd"): (
        "metropt3 + cycle_features + task=cpd is a degenerate pointwise target and is refused. "
        "Only 6 of the ~10,395 compressor cycles carry is_faulty=True, because an air-leak "
        "episode collapses into one multi-day cycle; the pointwise label therefore means "
        "'this cycle is very long', and the single features tower_switches / purge_count reach "
        "AUROC ~0.999 against it. MetroPT-3 is an event-scored anomaly-detection dataset: use "
        "input_kind=window_stats or raw_window (the 10 s aggregate table) for task=cpd, and keep "
        "cycle_features for per-cycle diagnostics, not for a published AD metric."
    ),
}


def _check_registry_contract(spec: RunSpec) -> None:
    """The spec and the model's registry row must agree on ``task`` and ``input_kind``.

    ``@register(name, input_kind, family, task)`` is the contract that keeps the config, the
    registry and ``configs/model_ladder.yaml`` in step, and the results row is written from the
    **spec**. Nothing checked that the two matched, so a detector registered as
    ``ad``/``raw_window`` could run to ``status="ok"`` and be filed in the leaderboard under
    ``cpd``/``window_stats`` - the numbers would be real and the label on them wrong, which is
    worse than a failure.
    """
    try:
        registered = get_spec(spec.model)
    except ValueError:
        return  # unregistered names are the registry's error to raise, at build() time
    if registered.task != spec.task:
        raise ValueError(
            f"run_one: model {spec.model!r} is registered for task={registered.task!r} but the spec "
            f"says task={spec.task!r}; the results row would be filed under the wrong task"
        )
    if registered.input_kind != spec.input_kind:
        raise ValueError(
            f"run_one: model {spec.model!r} is registered for input_kind={registered.input_kind!r} but "
            f"the spec says {spec.input_kind!r}; use input_kind: auto, or register a separate name"
        )


#: Datasets whose loaders need a subsystem named explicitly, because they hold several and the
#: loader would otherwise pick its own default while the results row kept the empty string.
_SUBSYSTEM_REQUIRED: frozenset[str] = frozenset({"sim"})


def _check_peer_norm_split(dataset: str, subsystem: str, split: str, peer_norm: bool) -> None:
    """Refuse ``peer_norm=True`` under a split that holds out a unit the peer groups span.

    Peer features are built on the whole table, before the split. That is only safe while
    every peer group sits inside the unit the split holds out - see
    ``data.PeerGrouping.safe_units``. The door grouping is ``(train_id, cycle_id)``: two doors
    on one train share a dwell, and the fleet puts those two doors in *different runs*, so the
    group spans ``run_id``. Under ``sim_loo_unit`` (holds out ``train_id``) that is fine;
    under ``sim_run_kfold`` (holds out ``run_id``) a training row's peer would be a row from
    the held-out fold, which is leakage.
    """
    if not peer_norm or dataset != "sim":
        return
    from nebulax.bench.data import _SIM_PEER_GROUPS
    from nebulax.bench.splits import PRESETS

    grouping = _SIM_PEER_GROUPS.get(subsystem)
    if grouping is None:
        raise ValueError(
            f"RunSpec: peer_norm=True is not defined for sim subsystem {subsystem!r} "
            f"(no concurrent siblings); defined for {sorted(_SIM_PEER_GROUPS)}"
        )
    preset = PRESETS.get(split)
    if preset is None:
        raise ValueError(f"RunSpec: unknown split preset {split!r}")
    if preset.kind in ("temporal", "temporal_fracs"):
        # Safe because the SPLIT enforces it, not because the timestamps make it likely.
        # data.PEER_GROUP_COL travels with the rows and splits._atomic_peer_supports widens each
        # row's support to its whole peer group, so temporal_split places a group entirely on one
        # side of every cut or purges it. A training row's peer therefore cannot be a test row.
        # (data._assert_peer_groups_time_coincident is a separate, weaker check that the grouping
        # means "concurrent siblings" at all; it is not what makes this safe.)
        return
    held = preset.params.get("group_col")
    if held is None:
        raise ValueError(
            f"RunSpec: peer_norm=True with split {split!r} cannot be checked - the preset holds out no "
            f"named group column, so there is no way to show a peer group does not cross its fence. "
            f"Peer normalisation is only allowed under a temporal split or one that holds out a "
            f"column in {list(grouping.safe_units)}."
        )
    if held not in grouping.safe_units:
        raise ValueError(
            f"RunSpec: peer_norm=True with split {split!r} would leak on sim/{subsystem}: the peer "
            f"groups {grouping.group_cols} span {held!r}, which this split holds out, so a training "
            f"row's peer mean could include a held-out row. Peer normalisation for {subsystem!r} is "
            f"only safe under a split that holds out one of {list(grouping.safe_units)}."
        )


@dataclass(frozen=True)
class RunSpec:
    """One benchmark run: exactly the knobs the results table is indexed by.

    ``config_hash`` is a stable 16-hex digest of every field except ``block`` (a config-file
    label that touches nothing), and is the parquet file name, the skip-existing key and the
    join key between the results row and its split audit file. ``max_minutes`` **is** part of
    it - see :attr:`identity` for why a budget is identity rather than scheduling.
    """

    dataset: str
    model: str
    task: str = "ad"
    subsystem: str = ""
    input_kind: str = "window_stats"
    split: str = "temporal_fracs"
    params: dict[str, Any] = field(default_factory=dict)
    window: int | float | None = None
    feature_set: str = "default"
    train_regime: str = "normal_only"
    H: float = 3 * 86400.0
    peer_norm: bool = False
    contamination: float = 0.0
    seed: int = 0
    data_kwargs: dict[str, Any] = field(default_factory=dict)
    block: str = ""
    max_minutes: float = 20.0

    def __post_init__(self) -> None:
        if self.task not in _TASKS:
            raise ValueError(f"RunSpec: unknown task {self.task!r}; expected one of {list(_TASKS)}")
        if self.train_regime not in _TRAIN_REGIMES:
            raise ValueError(
                f"RunSpec: unknown train_regime {self.train_regime!r}; expected one of {list(_TRAIN_REGIMES)}"
            )
        check_preset_dataset(self.split, self.dataset)
        if self.dataset in _SUBSYSTEM_REQUIRED and not self.subsystem:
            raise ValueError(
                f"RunSpec: dataset={self.dataset!r} needs an explicit subsystem (door | pneumatic | "
                f"bearing). Without one load_sim quietly returns its own default and the results row "
                f"still says subsystem=''"
            )
        clash = sorted(_AXIS_LOADER_KEYS & set(self.data_kwargs))
        if clash:
            raise ValueError(
                f"RunSpec: data_kwargs may not set {clash} - those loader keywords are ablation "
                f"axes and are written to the results row from the spec. Setting them under "
                f"'data:' would publish one value and load another (and, for peer_norm, walk "
                f"straight past the split safety check). Put them on the axis instead."
            )
        _check_peer_norm_split(self.dataset, self.subsystem, self.split, self.peer_norm)
        if (self.dataset, self.input_kind, self.task) in FORBIDDEN_COMBOS:
            raise ValueError(f"RunSpec: {FORBIDDEN_COMBOS[(self.dataset, self.input_kind, self.task)]}")

    @property
    def identity(self) -> dict[str, Any]:
        """The fields ``config_hash`` is taken over: **every field except ``block``**.

        ``block`` is a label for the config file and never touches a number, so it is the only
        thing dropped. In particular ``max_minutes`` is kept, for two reasons that together
        cover both kinds of model: a **deep** run is handed the remaining budget as ``_fit``'s
        ``budget_s`` and stops early on it, so 1 minute and 99 minutes train different models;
        and **any** run can finish at one cap and time out at another, which changes both the
        row and what ``skip_existing`` considers done.

        Deliberately computed from the dataclass alone. An earlier version kept
        ``max_minutes`` only when ``is_deep`` was true - but ``is_deep`` reads the registry, so
        the same spec hashed differently depending on whether the model happened to be imported
        yet, and a worker process (which imports models separately) could disagree with its
        parent about which runs were already done.
        """
        d = asdict(self)
        d.pop("block", None)
        return d

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.identity, sort_keys=True, default=str).encode()).hexdigest()[:16]

    @property
    def is_deep(self) -> bool:
        """True when the model's registered family is in :data:`DEEP_FAMILIES` or it declares
        ``needs_gpu=True`` in the registry."""
        try:
            spec = get_spec(self.model)
        except ValueError:
            return False
        return spec.family in DEEP_FAMILIES or bool(spec.meta.get("needs_gpu", False))

    def loader_kwargs(self) -> dict[str, Any]:
        """Loader keywords for this spec: ``data_kwargs`` first, **the ablation axes last**.

        The order matters and is a leakage guard, not a style choice. ``data_kwargs`` is a free
        passthrough for loader keywords the results table is not indexed by (``raw_dir``,
        ``stride``, ``run_ids``, ...). It used to be applied *last*, which let a config write
        ``peer_norm: false`` on the axis and ``data: {peer_norm: true}`` underneath it: the run
        loaded peer-normalised features, :func:`_check_peer_norm_split` (which reads
        ``self.peer_norm``) never fired, and the published row claimed ``peer_norm=False``. On
        the door fleet under ``sim_run_kfold`` that put ~40 k two-member peer groups astride the
        train/test fence with the held-out value exactly recoverable from its training partner.
        The same override silently falsified ``H``, ``feature_set`` and ``window``.

        Two things stop it now: :meth:`__post_init__` refuses a ``data_kwargs`` key that names an
        ablation axis (:data:`_AXIS_LOADER_KEYS`), and - belt and braces, for a spec built by
        some other route - the axis values are merged **over** ``data_kwargs`` here, so the
        column in the results row is always the value the loader was given.
        """
        axis_kw: dict[str, Any] = {
            "input_kind": self.input_kind,
            "feature_set": self.feature_set,
            "H": self.H,
            "peer_norm": self.peer_norm,
        }
        if self.subsystem:
            axis_kw["subsystem"] = self.subsystem
        if self.window is not None:
            axis_kw["window"] = self.window
        return {**self.data_kwargs, **axis_kw}

    def as_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["params"] = json.dumps(d["params"], sort_keys=True, default=str)
        d["data_kwargs"] = json.dumps(d["data_kwargs"], sort_keys=True, default=str)
        d["config_hash"] = self.config_hash
        return d


# --------------------------------------------------------------------------------------
# Config -> specs
# --------------------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a benchmark config yaml. See :func:`expand` for the schema."""
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"load_config: {path} must contain a mapping at the top level")
    return cfg


def _listify(v: Any) -> list[Any]:
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _param_grid(params: Any) -> list[dict[str, Any]]:
    """``{"n_estimators": [100, 200], "max_samples": 256}`` -> two param dicts."""
    if not params:
        return [{}]
    if isinstance(params, list):
        return [dict(p) for p in params]
    keys = sorted(params)
    values = [_listify(params[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _make_spec(*, kw: Mapping[str, Any] | None = None, **fields: Any) -> RunSpec:
    """Build one :class:`RunSpec` for :func:`expand`, refusing leftover ablation keys.

    ``kw`` is what remains of the block's axis dict after every known axis has been popped;
    anything still in it is a key :class:`RunSpec` would have dropped on the floor.
    """
    if kw:
        raise ValueError(f"unconsumed ablation key(s) {sorted(kw)}")
    return RunSpec(**fields)


def expand(config: Mapping[str, Any]) -> list[RunSpec]:
    """Expand a benchmark config into the cartesian product of its ablation lists.

    Schema (every key, with allowed values)::

        version: 1                       # int, informational
        defaults: {<any run key>: <scalar>}   # merged under every block
        runs:                            # list of ablation blocks
          - name: metropt_ad             # str, copied to RunSpec.block
            dataset: metropt3            # metropt3 | cranfield | ottawa | sim   (required)
            subsystem: pneumatic         # door | pneumatic | bearing | ""  (sim: required)
            task: ad                     # ad | cls | cpd | rul
            model: [iforest, lof]        # str | [str] | [{name: str, params: {...}}]
            params: {n_estimators: [100, 200]}   # grid, applied to every model in the block
            input_kind: auto             # auto | window_stats | raw_window | cycle_features
            split: [metropt_temporal]    # a key of nebulax.bench.splits.PRESETS, and it must
                                         # list this dataset (splits.check_preset_dataset)
            window: [60, 360]            # int steps (metropt/sim) or float seconds (ottawa)
            feature_set: [all]           # dataset-specific; see the loaders
            train_regime: [normal_only]  # normal_only | all
            H: [259200]                  # float seconds, the detection horizon
            peer_norm: [false]           # bool
            contamination: [0.0, 0.05]   # float, fraction of faulty rows kept in training
            seed: [0]                    # int
            max_minutes: 20              # float, per-run wall-clock cap
            data: {raw_dir: null, stride: null, max_runs: 4, decimate: 4}  # extra loader kwargs

    Any of the keys in :data:`ABLATION_KEYS` may be a scalar or a list; a list means one run
    per element. ``params`` is a grid of its own (list-valued entries multiply out), and
    ``model`` entries may carry their own ``params`` which are merged **over** the block grid.
    ``input_kind: auto`` (the default) takes each model's registered ``input_kind`` from the
    registry, which is what keeps the config and ``configs/model_ladder.yaml`` in step.

    Two classes of combination are refused here rather than silently benchmarked, because both
    produce numbers that look good for reasons unrelated to fault detection:

    * a ``split`` preset that is not declared for this ``dataset``
      (:func:`nebulax.bench.splits.check_preset_dataset`) - e.g. ``dataset: ottawa`` with
      ``split: temporal_fracs`` puts the same bearing in train and test;
    * anything in :data:`FORBIDDEN_COMBOS` - currently ``metropt3 + cycle_features`` with task ``ad`` or ``cpd``,
      whose pointwise target is degenerate;
    * a spec whose explicit ``task`` or ``input_kind`` contradicts the model's ``@register``
      row (:func:`_check_registry_contract`). This is checked here, not only in
      :func:`run_one`, so ``--dry-run`` reports it in a second instead of a sweep discovering
      it after loading a 6 GB dataset. Models that are not registered yet are left to
      ``build()`` to reject.

    Returns the specs in config order, de-duplicated by ``config_hash``. Raises ``ValueError``
    naming the offending block for any of the above.
    """
    defaults = dict(config.get("defaults", {}) or {})
    blocks = config.get("runs") or config.get("blocks") or []
    if not isinstance(blocks, list):
        raise ValueError("expand: config['runs'] must be a list of ablation blocks")

    out: list[RunSpec] = []
    seen: set[str] = set()
    for i, raw_block in enumerate(blocks):
        if not isinstance(raw_block, Mapping):
            raise ValueError(f"expand: runs[{i}] must be a mapping, got {type(raw_block).__name__}")
        block = {**defaults, **dict(raw_block)}
        name = str(block.pop("name", f"block{i}"))
        if "dataset" not in block:
            raise ValueError(f"expand: runs[{i}] ({name!r}) has no 'dataset'")
        data_kwargs = dict(block.pop("data", {}) or {})
        clash = sorted(_AXIS_LOADER_KEYS & set(data_kwargs))
        if clash:
            raise ValueError(
                f"expand: runs[{i}] ({name!r}) sets {clash} under 'data:'. Those loader keywords "
                f"are ablation axes: the results row publishes them from the spec, so a 'data:' "
                f"override would label the run with one value and load another (and 'peer_norm' "
                f"there also bypasses the peer-group/split safety check). Move them out of "
                f"'data:' and set them as block keys, where they become real ablation axes."
            )
        base_params = block.pop("params", {}) or {}
        max_minutes = float(block.pop("max_minutes", 20.0))
        dataset = str(block.pop("dataset"))

        models: list[tuple[str, dict[str, Any]]] = []
        for entry in _listify(block.pop("model", [])):
            if isinstance(entry, Mapping):
                models.append((str(entry["name"]), dict(entry.get("params", {}) or {})))
            else:
                models.append((str(entry), {}))
        if not models:
            raise ValueError(f"expand: runs[{i}] ({name!r}) lists no models")

        axes = {k: _listify(block.pop(k)) for k in ABLATION_KEYS if k in block}
        unknown = set(block) - {"comment", "note", "notes"}
        if unknown:
            raise ValueError(
                f"expand: runs[{i}] ({name!r}) has unknown key(s) {sorted(unknown)}; "
                f"allowed = {sorted(set(ABLATION_KEYS) | {'name', 'dataset', 'data', 'params', 'max_minutes'})}"
            )
        axis_keys = [k for k in ABLATION_KEYS if k in axes]
        combos = list(itertools.product(*[axes[k] for k in axis_keys])) or [()]

        for model_name, model_params in models:
            for grid in _param_grid(base_params):
                for combo in combos:
                    kw = dict(zip(axis_keys, combo))
                    input_kind = kw.pop("input_kind", "auto")
                    if input_kind in (None, "auto"):
                        input_kind = get_spec(model_name).input_kind
                    try:
                        spec = _make_spec(
                            dataset=dataset,
                            model=model_name,
                            task=str(kw.pop("task", defaults.get("task", "ad"))),
                            subsystem=str(kw.pop("subsystem", "") or ""),
                            input_kind=str(input_kind),
                            split=str(kw.pop("split", "temporal_fracs")),
                            params={**grid, **model_params},
                            window=kw.pop("window", None),
                            feature_set=str(kw.pop("feature_set", "default")),
                            train_regime=str(kw.pop("train_regime", "normal_only")),
                            H=float(kw.pop("H", 3 * 86400.0)),
                            peer_norm=bool(kw.pop("peer_norm", False)),
                            contamination=float(kw.pop("contamination", 0.0)),
                            seed=int(kw.pop("seed", 0)),
                            data_kwargs=data_kwargs,
                            block=name,
                            max_minutes=max_minutes,
                            kw=kw,
                        )
                        _check_registry_contract(spec)
                    except ValueError as exc:
                        raise ValueError(f"expand: runs[{i}] ({name!r}): {exc}") from None
                    if spec.config_hash not in seen:
                        seen.add(spec.config_hash)
                        out.append(spec)
    return out


# --------------------------------------------------------------------------------------
# The two gated entry points (see the module docstring and the leakage test)
# --------------------------------------------------------------------------------------


def _slice(data: BenchData, idx: np.ndarray) -> np.ndarray:
    return data.X[np.asarray(idx, dtype=np.int64)]


def _fit_model(model: Any, data: BenchData, spec: RunSpec, train_idx: np.ndarray, *, budget_s: float | None = None) -> Any:
    """Fit ``model`` on ``train_idx`` **only**. The single place the runner calls ``fit``.

    Deep models are capped at :data:`MAX_DEEP_TRAIN_WINDOWS` rows (seeded subsample) and, when
    their ``_fit`` declares a ``budget_s`` parameter, handed the remaining wall-clock budget so
    they can stop early instead of being killed by the subprocess timeout.
    """
    idx = np.asarray(train_idx, dtype=np.int64)
    if spec.is_deep and idx.size > MAX_DEEP_TRAIN_WINDOWS:
        idx = np.sort(np.random.default_rng(spec.seed).choice(idx, MAX_DEEP_TRAIN_WINDOWS, replace=False))
    X = _slice(data, idx)
    kwargs: dict[str, Any] = {}
    if budget_s is not None and _accepts(model, "budget_s"):
        kwargs["budget_s"] = float(budget_s)
    kwargs.update(_context(model._fit, data, idx))
    if spec.task in ("cls", "rul"):
        return model.fit(X, data.target(spec.task)[idx], **kwargs)
    return model.fit(X, data.t_end[idx], **kwargs)


def _accepts(model: Any, param: str) -> bool:
    return B.accepts_keyword(model._fit, param)


def _context(fn: Any, data: BenchData, idx: np.ndarray) -> dict[str, Any]:
    """The row context (``base.CONTEXT_KEYS``) a model method declares, sliced to ``idx``.

    Feature names, series (component) ids and unit ids are the runner's to give: a model only
    ever sees arrays, so physics rows that need "the TP2 column" or "the sibling door"
    declare the keyword and get exactly the rows they are fitting or scoring. Never labels.
    """
    i = np.asarray(idx, dtype=np.int64)
    out: dict[str, Any] = {}
    if B.accepts_keyword(fn, "feature_names"):
        out["feature_names"] = list(data.feature_names)
    if B.accepts_keyword(fn, "series"):
        out["series"] = np.asarray(data.series, dtype=object)[i]
    if B.accepts_keyword(fn, "unit"):
        out["unit"] = np.asarray(data.unit, dtype=object)[i]
    if B.accepts_keyword(fn, "fs_hz"):
        fs = _delivered_fs_hz(data)
        if fs is not None:
            out["fs_hz"] = fs
    return out


def _delivered_fs_hz(data: BenchData) -> float | None:
    """Sampling rate of the rows' samples as delivered (post-decimation), from loader meta."""
    for key in ("fs_hz_out", "fs_hz"):
        v = data.meta.get(key)
        if v is not None and np.isfinite(float(v)) and float(v) > 0:
            return float(v)
    return None


def _score(model: Any, data: BenchData, idx: np.ndarray) -> np.ndarray:
    """Score ``idx`` rows. ``SEQUENTIAL`` models are scored one series at a time in time order.

    A running statistic (EWMA, CUSUM, Page-Hinkley, ADWIN) must not carry over from one
    component to the next when a slice interleaves two doors or sixteen axle boxes, so the
    runner groups the rows by ``data.series``, sorts each group by ``t_end`` and scores the
    groups separately. Scores are returned in the caller's row order.
    """
    i = np.asarray(idx, dtype=np.int64)
    t = data.t_end[i]
    if not getattr(model, "SEQUENTIAL", False):
        ctx = _context(model._score, data, i)
        return np.asarray(model.score(_slice(data, i), t, **ctx), dtype=np.float64)
    series = np.asarray(data.series, dtype=object)[i]
    out = np.full(i.size, np.nan, dtype=np.float64)
    for s in pd.unique(series):
        sel = np.flatnonzero(series == s)
        order = sel[np.argsort(M.to_epoch_seconds(t[sel]), kind="stable")]
        ctx = _context(model._score, data, i[order])
        # A FRESH COPY of the fitted model per series and per slice: a streaming detector
        # (half-space trees, drift detectors) learns inside _score, so scoring one component
        # must not leave state behind for the next one, and the validation pass used for
        # calibration must not change the state the test pass is scored with. `model` itself
        # is never mutated by scoring.
        fresh = copy.deepcopy(model)
        out[order] = np.asarray(fresh.score(_slice(data, i[order]), t[order], **ctx), dtype=np.float64)
    return out


def _val_alarm_window_rows(data: BenchData, idx: np.ndarray, H: float) -> int:
    """How many validation rows sit inside some event's ``[t_onset - H, t_failure]`` window.

    Zero is the clean case. A non-zero count means the operating point was calibrated over
    rising pre-failure scores, which raises the budget threshold and *depresses* test recall -
    conservative, but the report must be able to say so, hence the column.
    """
    if idx.size == 0 or not data.events:
        return 0
    t = M.to_epoch_seconds(data.t_end)[idx]
    u = np.asarray(data.series)[idx]  # events are keyed by series (component), like episodes
    inside = np.zeros(idx.size, dtype=bool)
    for ev in data.events:
        end = ev.t_failure if np.isfinite(ev.t_failure) else np.inf
        inside |= (u == ev.unit) & (t >= ev.t_onset - float(H)) & (t <= end)
    return int(inside.sum())


def _calibrate(
    model: Any, data: BenchData, spec: RunSpec, val_idx: np.ndarray
) -> tuple[T.ThresholdSet | None, dict[str, Any]]:
    """Score the **validation** slice and calibrate. The single place the runner calibrates.

    Returns ``(threshold_set, info)``. ``threshold_set`` is ``None`` - and
    ``info["uncalibrated"]`` is ``True`` with a reason - when there is nothing to calibrate on;
    :func:`run_one` then marks the whole row uncalibrated so the leaderboard cannot present a
    thresholdless run as a clean result.

    Only rows in ``masks["scoreable"]`` are calibrated on, which is exactly the population
    ``_ad_fold_metrics`` reports against. Calibrating a false-alarm budget over rows that are
    then excluded from the test measurement would count the budget against a different
    population than the one it is published on.

    ``info`` also carries the validation-slice threshold-free metrics (``val_auroc``,
    ``val_auprc``, ``val_vus_pr``). They exist so that model selection has a non-test number
    to select on - see ``report._selected_per_subsystem``.
    """
    idx = np.asarray(val_idx, dtype=np.int64)
    info: dict[str, Any] = {"uncalibrated": False, "uncalibrated_reason": ""}
    keep = data.masks["scoreable"][idx]
    idx = idx[keep]
    info["n_val_scoreable_rows"] = int(idx.size)
    info["n_val_excluded_rows"] = int(np.asarray(val_idx).size - idx.size)
    if idx.size == 0:
        info.update({"uncalibrated": True, "uncalibrated_reason": "no scoreable validation rows"})
        return None, info

    scores = _score(model, data, idx)
    y_val = data.y_binary[idx]
    info.update(
        {
            "val_auroc": M.auroc(y_val, scores),
            "val_auprc": M.auprc(y_val, scores),
            "val_vus_pr": M.vus_pr(y_val, scores, max_window=_vus_window(data)),
            "val_alarm_window_rows": _val_alarm_window_rows(data, idx, spec.H),
            "val_faulty_rows": int(y_val.sum()),
            "val_positive_rate": float(y_val.mean()) if y_val.size else float("nan"),
        }
    )
    bundle = T.ValidationScores(
        scores=scores, times=data.t_end[idx], units=data.unit[idx], index=idx, part="val",
        series=data.series[idx],
    )
    thr = T.calibrate(
        bundle,
        window_seconds=data.window_seconds,
        k_consecutive=M.DEFAULT_K_CONSECUTIVE,
        merge_gap_s=M.DEFAULT_MERGE_GAP_S,
        max_step_s=_max_step_seconds(data),  # the run's single contiguity budget, shared with the test folds
        budget_applicable=not _fabricated_timeline(data),
    )
    info["threshold_budget_silent"] = bool(thr["budget"].meta.get("silent", False))
    info["budget_applicable"] = bool(thr["budget"].meta.get("budget_applicable", True))
    # Validation-side flag only. The TEST-side `episodes_impossible` (written by
    # _ad_fold_metrics) is the one that drives the operational columns; this one merely
    # explains a silent threshold on a table whose validation slice cannot form an episode.
    info["val_episodes_impossible"] = bool(_episodes_impossible(data, idx, M.DEFAULT_K_CONSECUTIVE))
    info["_calibrated_on_index"] = thr.calibrated_on_index
    return thr, info


# --------------------------------------------------------------------------------------
# Metric assembly
# --------------------------------------------------------------------------------------


def _ad_fold_metrics(
    data: BenchData, spec: RunSpec, test_idx: np.ndarray, scores: np.ndarray, thr: T.ThresholdSet | None
) -> dict[str, Any]:
    """Threshold-free + per-threshold operational metrics on one test fold."""
    keep = data.masks["scoreable"][test_idx]
    idx = test_idx[keep]
    s = scores[keep]
    y = data.y_binary[idx]
    t_scored = data.t_end[idx]
    vus_pitch = _row_pitch_seconds(data)
    vus_w = _vus_window(data)
    # Dropping non-scoreable rows leaves holes in the timeline, so episodes() is told what a
    # nominal step is - three above-threshold rows either side of a 24 h repair blanking are
    # array-adjacent but not consecutive in time. The step is the larger of the window and
    # the per-series row pitch of the WHOLE table (the same number calibration used).
    max_step_s = _max_step_seconds(data)
    out: dict[str, Any] = {
        "n_test_rows": int(test_idx.size),
        "n_scored_rows": int(idx.size),
        "n_excluded_rows": int(test_idx.size - idx.size),
        # The no-skill floor of AUPRC / VUS-PR is the positive rate of the scored slice; a
        # 0.92-positive slice makes a random scorer look like a 0.92 detector.
        "n_test_faulty_rows": int(y.sum()),
        "test_positive_rate": float(y.mean()) if y.size else float("nan"),
        "auroc": M.auroc(y, s),
        "auprc": M.auprc(y, s),
        "vus_pr": M.vus_pr(y, s, max_window=vus_w),
        "vus_buffer_windows": int(vus_w),
        "vus_buffer_seconds": float(vus_w * vus_pitch),
        "vus_row_pitch_s": float(vus_pitch),
        "episode_max_step_s": float(max_step_s),
        "episode_row_pitch_s": float(vus_pitch),
        "monotonicity": M.monotonicity(s, data.stage[idx]),
    }
    events = data.events_in(idx, spec.H)
    fabricated = _fabricated_timeline(data)
    n_days = float("nan") if fabricated else M.train_days(data.t_end[idx], data.unit[idx])
    impossible = _episodes_impossible(data, idx, M.DEFAULT_K_CONSECUTIVE)
    out["n_events"] = len(events)
    # An impossible fold contributes no operational measurement: its train-days are NaN too,
    # so the fold aggregation's numerator and denominator cover the same (measurable) folds.
    out["train_days"] = float("nan") if impossible else n_days
    out["far_not_applicable_reason"] = "fabricated timeline" if fabricated else ""
    out["episodes_impossible"] = bool(impossible)
    out["max_rows_per_series"] = int(_max_rows_per_series(data, idx))
    if thr is not None:
        for name, t in thr.items():
            if impossible:
                # k consecutive rows cannot exist on any series of this slice (Cranfield's
                # one-row-per-test table): the operational columns are a property of the
                # table, not a measurement of the model, so they are NaN and flagged.
                out[f"{name}_threshold"] = float(t.value)
                out[f"{name}_event_recall"] = float("nan")
                out[f"{name}_n_events"] = float("nan")
                out[f"{name}_n_events_detected"] = float("nan")
                out[f"{name}_false_alarms_per_train_day"] = float("nan")
                out[f"{name}_n_false_alarms"] = float("nan")
                out[f"{name}_median_lead_time_s"] = float("nan")
                out[f"{name}_n_episodes"] = float("nan")
                continue
            eps = M.episodes(
                s,
                data.t_end[idx],
                t.value,
                k=M.DEFAULT_K_CONSECUTIVE,
                merge_gap_s=M.DEFAULT_MERGE_GAP_S,
                units=data.series[idx],  # per component: never pooled across sibling doors / boxes
                max_step_s=max_step_s,
            )
            em = M.event_metrics(eps, events, spec.H, n_train_days=n_days)
            out[f"{name}_threshold"] = float(t.value)
            out[f"{name}_event_recall"] = em["event_recall"]
            out[f"{name}_n_events"] = em["n_events"]
            out[f"{name}_n_events_detected"] = em["n_events_detected"]
            out[f"{name}_false_alarms_per_train_day"] = em["false_alarms_per_train_day"]
            out[f"{name}_n_false_alarms"] = em["n_false_alarms"]
            out[f"{name}_median_lead_time_s"] = em["median_lead_time_s"]
            out[f"{name}_n_episodes"] = em["n_episodes"]
            if name == "budget":
                out["lead_times_s"] = em["lead_times_s"]
    return out


def _check_axes_against_loader(spec: RunSpec, data: BenchData) -> None:
    """Every ablation axis on the row must be the value the loader was actually called with.

    ``dataset`` / ``input_kind`` / ``subsystem`` are checked against the :class:`BenchData`
    fields just above this; ``peer_norm``, ``H``, ``feature_set`` and ``window`` have no
    ``BenchData`` field, so they are checked against ``data.meta["loader_kwargs"]`` - the
    keywords :func:`nebulax.bench.data.load_bench` recorded when it built (or cache-keyed) the
    table. Without this a caller that hands ``run_one`` a pre-built ``data=`` could publish a
    row saying ``peer_norm=False, H=3 d`` over features built with ``peer_norm=True, H=0``.
    Missing keys are not an error: a ``BenchData`` constructed by hand in a test carries none.
    """
    seen = data.meta.get("loader_kwargs")
    if not isinstance(seen, Mapping):
        return
    want = spec.loader_kwargs()
    # Every data: passthrough key must have reached the loader: one the loader never saw
    # (a typo, or a keyword of another dataset's loader) would be published in the row as if
    # it had shaped the table.
    unseen = sorted(k for k in spec.data_kwargs if k not in seen)
    if unseen:
        raise ValueError(
            f"run_one: data_kwargs {unseen} were never seen by the loader that built this table "
            f"(loader_kwargs = {sorted(seen)}); refusing to publish a mislabelled run"
        )
    for axis in sorted(_AXIS_LOADER_KEYS):
        if axis not in seen or axis not in want:
            continue
        a, b = seen[axis], want[axis]
        same = (a == b) or (
            isinstance(a, (int, float))
            and isinstance(b, (int, float))
            and not isinstance(a, bool)
            and not isinstance(b, bool)
            and float(a) == float(b)
        )
        if not same:
            raise ValueError(
                f"run_one: the results row would say {axis}={b!r} but the loaded table was built "
                f"with {axis}={a!r}; refusing to publish a mislabelled run"
            )


#: The tolerance VUS-PR is computed at, in seconds. One hour: the operational question is
#: "did the detector fire around this time", and an hour is the same gap the episode merge rule
#: uses. It is a *time*, deliberately - see :func:`_vus_window`.
VUS_BUFFER_SECONDS: float = 3600.0


def _fabricated_timeline(data: BenchData) -> bool:
    """True for datasets whose timestamps are adapter-placed anchors (see ``BenchData.meta``)."""
    return str(data.meta.get("timeline", "")) == "fabricated"


def _max_rows_per_series(data: BenchData, idx: np.ndarray) -> int:
    i = np.asarray(idx, dtype=np.int64)
    if i.size == 0:
        return 0
    return int(pd.Series(np.asarray(data.series, dtype=object)[i]).value_counts().iloc[0])


def _episodes_impossible(data: BenchData, idx: np.ndarray, k: int) -> bool:
    """No series in ``idx`` has ``k`` rows, so the ``>= k`` consecutive-window rule can never fire."""
    return _max_rows_per_series(data, idx) < int(k)


def _row_pitch_seconds(data: BenchData) -> float:
    """Seconds between two consecutive rows of one series - see ``metrics.row_pitch_seconds``.

    ``vus_pr(max_window=w)`` measures ``w`` in **rows**, not seconds, and rows are spaced by
    the *stride*, not by the window length: ``load_metropt3(window=60)`` makes 600 s windows
    every 300 s. The same pitch sets the episode contiguity budget (``metrics.max_step_seconds``).
    Measured ONCE on the whole table, per series (component), so calibration, validation
    metrics and every test fold count episodes and VUS buffers under one definition; a pitch
    measured per slice differed between validation and test on real loaders (Cranfield
    folds, sim/door) and put the budget and the published recall under different rules.
    Falls back to ``window_seconds`` when the table is too short or has no finite gaps.
    """
    # Memoised on the instance, NOT in data.meta: meta is copied by BenchData.subset and
    # persisted by the cache, either of which would carry a parent's pitch onto a different table.
    cached = getattr(data, "_row_pitch_cache", None)
    if cached is not None:
        return float(cached)
    pitch = M.row_pitch_seconds(data.t_end, data.series, fallback=float(data.window_seconds or 0.0))
    data._row_pitch_cache = pitch
    return pitch


def _max_step_seconds(data: BenchData) -> float:
    """The run's single episode contiguity budget: ``max(window, pitch)`` with the protocol slack."""
    return M.max_step_seconds(data.window_seconds, _row_pitch_seconds(data))


def _vus_window(data: BenchData) -> int:
    """VUS buffer in **rows**, covering :data:`VUS_BUFFER_SECONDS` of real time (max 20 rows).

    The floor is 1 row, not 4: a 4-row floor at a 1800 s row pitch is a 2 h tolerance published
    against a 1 h label, and it made the buffer span 200 s .. 7200 s across the ``window``
    ablation. Pair every use with :func:`_row_pitch_seconds` so the ``vus_buffer_seconds``
    column is that buffer times that pitch.
    """
    pitch = _row_pitch_seconds(data)
    if pitch <= 0:
        return 10
    return int(np.clip(round(VUS_BUFFER_SECONDS / pitch), 1, 20))


def _cls_fold_metrics(data: BenchData, spec: RunSpec, test_idx: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """Classification metrics, scored at the level the loader declares.

    When the loader sets ``meta["vote_group"]`` - Cranfield's ``raw_window`` does, naming
    ``meta_test_id`` - the unit of evaluation is the **test**, not the window. A test that
    happens to yield 40 windows would otherwise count 40 times against one that yields 10, so
    the macro-F1 would be a weighted average over recording lengths rather than over tests,
    and it would not be comparable with the ``cycle_features`` runs on the same dataset (one
    row per test by construction). Window-level numbers are kept beside the voted ones as
    ``window_macro_f1`` / ``window_balanced_accuracy`` rather than thrown away.
    """
    y = data.target("cls")[test_idx]
    labels = sorted(set(data.target("cls").tolist()))
    out = M.cls_metrics(y, pred, labels=labels)
    row: dict[str, Any] = {
        "n_test_rows": int(test_idx.size),
        "macro_f1": out["macro_f1"],
        "balanced_accuracy": out["balanced_accuracy"],
        "accuracy": out["accuracy"],
        "confusion_matrix": out["confusion_matrix"],
        "class_labels": out["labels"],
        "per_class_f1": out["per_class_f1"],
        "voted": False,
    }

    vote_col = data.meta.get("vote_group")
    if not vote_col or vote_col not in data.labels.columns:
        return row
    groups = data.labels[vote_col].to_numpy(dtype=object)[test_idx]
    gid_pred, voted_pred = majority_vote(pred, groups)
    gid_true, voted_true = majority_vote(y, groups)
    assert list(gid_pred) == list(gid_true), "majority_vote must preserve group order"
    voted = M.cls_metrics(voted_true, voted_pred, labels=labels)
    row.update(
        {
            "voted": True,
            "vote_group": str(vote_col),
            "n_test_groups": int(len(gid_pred)),
            "window_macro_f1": out["macro_f1"],
            "window_balanced_accuracy": out["balanced_accuracy"],
            "window_accuracy": out["accuracy"],
            "macro_f1": voted["macro_f1"],
            "balanced_accuracy": voted["balanced_accuracy"],
            "accuracy": voted["accuracy"],
            "confusion_matrix": voted["confusion_matrix"],
            "class_labels": voted["labels"],
            "per_class_f1": voted["per_class_f1"],
        }
    )
    return row


def _rul_fold_metrics(data: BenchData, spec: RunSpec, test_idx: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y = data.target("rul")[test_idx]
    ok = np.isfinite(y) & np.isfinite(pred)
    if ok.sum() == 0:
        return {"n_test_rows": int(test_idx.size), "rmse": float("nan"), "mae": float("nan"), "monotonicity": float("nan")}
    err = pred[ok] - y[ok]
    return {
        "n_test_rows": int(test_idx.size),
        "n_scored_rows": int(ok.sum()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "monotonicity": M.monotonicity(-pred[ok], data.stage[test_idx][ok]),
    }


_SKIP_AGG = {"confusion_matrix", "class_labels", "per_class_f1", "lead_times_s"}
#: Metrics that are **counts**: summed over folds, never averaged. Averaging them is what
#: produces the nonsense "recall 0.80 (0/0)" cell when half the folds hold no event.
_SUM_SUFFIXES: tuple[str, ...] = (
    "n_events",
    "n_events_detected",
    "n_episodes",
    "n_false_alarms",
    "n_test_rows",
    "n_test_groups",
    "n_scored_rows",
    "n_excluded_rows",
    "n_val_rows",
    "train_days",
    "train_rows",
    "val_rows",
    # Validation-side counts. These are counts of ROWS, exactly like the test-side ones, and
    # averaging them made the results row contradict the audit trail beside it: a 5-fold run
    # reported calibrated_on_n = 100 while <config_hash>.calibration.parquet held 500 rows.
    "n_val_scoreable_rows",
    "n_val_excluded_rows",
    "val_alarm_window_rows",
    # Positive-row counts on both slices: the positive RATE is re-derived from these sums, so
    # averaging them halves the rate on a two-fold run (Cranfield loo_profile: 0.46 for 0.92).
    "n_test_faulty_rows",
    "val_faulty_rows",
    "val_faulty_rows",
    "calibrated_on_n",
    "val_episodes",
)
#: Fold values that are **extremes**, not quantities: taking a mean of two row indices names a
#: row that was never calibrated on. Aggregated with min / max over the folds.
_MIN_KEYS: tuple[str, ...] = ("calibrated_on_min",)
_MAX_KEYS: tuple[str, ...] = ("calibrated_on_max",)


#: Boolean per-fold flags: OR-ed across folds rather than averaged.
_ANY_FLAGS: tuple[str, ...] = (
    "uncalibrated",
    "threshold_budget_silent",
    "degenerate_labels",
    "voted",
    "val_episodes_impossible",
)
#: String per-fold values kept verbatim: the distinct values, joined with ";" in fold order.
#: (The numeric aggregator drops non-numbers, which silently lost the calibration digest.)
_STR_KEYS: tuple[str, ...] = ("calibrated_on_digest", "vote_group", "stop_reason")


def _is_count(key: str) -> bool:
    return any(key == suf or key.endswith("_" + suf) for suf in _SUM_SUFFIXES)


def _aggregate_folds(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fold aggregation: counts are **summed**, rates are **micro-averaged** from those sums,
    everything else numeric is the mean over folds, and the structured values (confusion
    matrices, lead-time lists) are kept per fold as JSON.

    Micro-averaging the operational rates matters: with leave-one-unit-out over a fleet where
    only some units ever fail, a macro-averaged recall is an average over folds that had
    nothing to find, and it no longer matches the ``k/N`` the report prints beside it.
    """
    if not folds:
        return {}
    out: dict[str, Any] = {"n_folds": len(folds)}
    # The UNION of keys, in first-seen order - not folds[0]'s. A key that only appears on a
    # later fold (the threshold columns of the first fold that managed to calibrate, say) was
    # silently dropped when the first fold happened not to have it.
    keys = list(dict.fromkeys(k for f in folds for k in f if k not in _SKIP_AGG))
    for k in keys:
        vals = [f[k] for f in folds if k in f and isinstance(f[k], (int, float, np.floating, np.integer))]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        if _is_count(k):
            # A count that no fold could measure (train_days on a fabricated timeline, the
            # false-alarm count when episodes are impossible) stays NaN: nansum would print 0.
            out[k] = float(np.nansum(arr)) if np.isfinite(arr).any() else float("nan")
        else:
            with np.errstate(invalid="ignore"):
                out[k] = float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")

    # Index extremes are min/max-ed over the folds; the mean of two row indices is a row that
    # was never calibrated on at all.
    for key, fn in [(k, np.nanmin) for k in _MIN_KEYS] + [(k, np.nanmax) for k in _MAX_KEYS]:
        vals = [float(f[key]) for f in folds if isinstance(f.get(key), (int, float, np.number))]
        if vals:
            out[key] = float(fn(np.asarray(vals, dtype=np.float64)))

    # Boolean warning flags are OR-ed, never averaged: one uncalibrated fold makes the whole
    # row uncalibrated, and "0.5 uncalibrated" is not a thing a report can act on.
    for flag in _ANY_FLAGS:
        if any(flag in f for f in folds):
            out[flag] = bool(any(bool(f.get(flag, False)) for f in folds))
    if any("budget_applicable" in f for f in folds):  # applicable only if every fold's budget was
        out["budget_applicable"] = bool(all(bool(f.get("budget_applicable", True)) for f in folds))
    n_sc, n_pos = out.get("n_scored_rows"), out.get("n_test_faulty_rows")
    if n_sc is not None and n_pos is not None and np.isfinite(n_sc) and n_sc > 0:
        out["test_positive_rate"] = float(n_pos) / float(n_sc)
    n_v, n_vp = out.get("n_val_scoreable_rows"), out.get("val_faulty_rows")
    if n_v is not None and n_vp is not None and np.isfinite(n_v) and n_v > 0:
        out["val_positive_rate"] = float(n_vp) / float(n_v)
    if any("episodes_impossible" in f for f in folds):
        # Operational columns are aggregated over the folds that COULD form an episode (their
        # per-fold counts are NaN otherwise and nansum skips them); the run as a whole is
        # "impossible" only when every fold was, and the number of excluded folds is published.
        flags = [bool(f.get("episodes_impossible", False)) for f in folds]
        out["episodes_impossible"] = bool(all(flags))
        out["n_folds_episodes_impossible"] = int(sum(flags))
    for key in _STR_KEYS:
        vals = [str(f[key]) for f in folds if key in f]
        if vals:
            out[key] = ";".join(dict.fromkeys(vals))
    reasons = sorted({str(f.get("uncalibrated_reason", "")) for f in folds if f.get("uncalibrated")})
    if any("uncalibrated" in f for f in folds):
        out["uncalibrated_reason"] = "; ".join(r for r in reasons if r)
    if any("far_not_applicable_reason" in f for f in folds):
        out["far_not_applicable_reason"] = "; ".join(
            sorted({str(f.get("far_not_applicable_reason", "")) for f in folds} - {""})
        )
    # Re-derive the rates from the summed counts so they agree with what the report prints.
    # Counts of folds that could not form an episode are NaN per fold and were skipped by the
    # sum above; if every fold was impossible the sums are NaN and so are the rates.
    for prefix in {k.rsplit("_n_events", 1)[0] for k in out if k.endswith("_n_events")}:
        n_ev, n_det = out.get(f"{prefix}_n_events"), out.get(f"{prefix}_n_events_detected")
        if n_ev is not None and n_det is not None and np.isfinite(n_ev) and np.isfinite(n_det):
            out[f"{prefix}_event_recall"] = (n_det / n_ev) if n_ev else float("nan")
        elif n_ev is not None and not np.isfinite(n_ev):
            out[f"{prefix}_event_recall"] = float("nan")
        n_fa, days = out.get(f"{prefix}_n_false_alarms"), out.get("train_days")
        if n_fa is not None and days is not None and np.isfinite(days) and days:
            out[f"{prefix}_false_alarms_per_train_day"] = n_fa / days
        elif f"{prefix}_false_alarms_per_train_day" in out and (days is None or not np.isfinite(days)):
            out[f"{prefix}_false_alarms_per_train_day"] = float("nan")  # fabricated timeline: no rate

    # The same micro-average on the validation side, so the false-alarm rate the report filters
    # on reconciles with the episode count printed beside it.
    for prefix in {k.rsplit("_val_episodes", 1)[0] for k in out if k.endswith("_val_episodes")}:
        n_eps, days = out.get(f"{prefix}_val_episodes"), out.get(f"{prefix}_val_train_days")
        if n_eps is not None and days is not None and np.isfinite(days) and days:
            out[f"{prefix}_val_far_per_train_day"] = n_eps / days
        elif f"{prefix}_val_far_per_train_day" in out and (days is None or not np.isfinite(days)):
            out[f"{prefix}_val_far_per_train_day"] = float("nan")

    for k in _SKIP_AGG:
        if any(k in f for f in folds):
            out[k] = json.dumps([f.get(k) for f in folds], default=str)
    out["per_fold_json"] = json.dumps(
        [{k: v for k, v in f.items() if k not in _SKIP_AGG} for f in folds], default=str
    )
    return out


# --------------------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------------------


def git_rev() -> str:
    """Short git revision of the repo this code lives in (``"unknown"`` outside a checkout)."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _n_params(model: Any) -> float:
    """Parameter count: the model's own ``n_params`` if it has one, else a torch/sklearn probe."""
    for attr in ("n_params", "n_params_"):
        v = getattr(model, attr, None)
        if callable(v):
            v = v()
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
    try:  # torch
        params = getattr(model, "parameters", None)
        if callable(params):
            return float(sum(int(np.prod(p.shape)) for p in params()))
    except Exception:  # pragma: no cover
        pass
    total = 0.0
    for name in dir(model):
        if not name.endswith("_") or name.startswith("_"):
            continue
        try:
            v = getattr(model, name)
        except Exception:  # pragma: no cover
            continue
        if isinstance(v, np.ndarray):
            total += float(v.size)
    return total if total else float("nan")


#: What a deep model's ``train_info_`` calls each field, and the results column it becomes.
#: ``deep.train_loop`` returns ``{"epochs", "best_loss", "stop_reason", "val_rows"}``.
_TRAIN_INFO_COLUMNS: tuple[tuple[str, str], ...] = (
    ("epochs", "epochs_run"),
    ("best_loss", "best_val_loss"),
    ("stop_reason", "stop_reason"),
)


def _deep_train_info(model: Any) -> dict[str, Any]:
    """``epochs_run`` / ``best_val_loss`` / ``stop_reason`` for a deep row, else ``{}``.

    Without these a 1.68 s fit that scores at chance is indistinguishable from a network that
    trained and lost: patience firing at epoch 1, the wall-clock budget cutting the first epoch
    short and a genuine 200-epoch run all look the same in ``fit_seconds``. An ensemble
    (``classifier_ensemble``) reports its members' mean epochs and loss and the distinct stop
    reasons, so the row still says whether anything trained.
    """
    info = getattr(model, "train_info_", None)
    if not isinstance(info, Mapping) or not info:
        return {}
    members = info.get("members")
    infos = [m for m in members if isinstance(m, Mapping)] if isinstance(members, (list, tuple)) else [info]
    out: dict[str, Any] = {}
    for src, dst in _TRAIN_INFO_COLUMNS:
        vals = [m[src] for m in infos if m.get(src) is not None]
        if not vals:
            continue
        if dst == "stop_reason":
            out[dst] = ";".join(dict.fromkeys(str(v) for v in vals))
        else:
            nums = [float(v) for v in vals if isinstance(v, (int, float, np.integer, np.floating))]
            if nums:
                out[dst] = float(np.mean(nums))
    return out


def _peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _peak_vram_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / 1e6
    except Exception:  # pragma: no cover
        pass
    return 0.0


def run_one(
    spec: RunSpec,
    *,
    cache_dir: str | Path | None = "data/features",
    splits_dir: str | Path | None = None,
    data: BenchData | None = None,
) -> dict[str, Any]:
    """Execute one :class:`RunSpec` and return the results row.

    The row always has :data:`ROW_COLUMNS` plus the task's metric columns:
    ``status`` is ``"ok"`` / ``"failed"`` / ``"timeout"``, and on failure ``traceback`` holds
    the formatted exception. Never raises for a model-side error - a sweep must finish.

    ``data`` lets a caller (or a test) inject a pre-built :class:`BenchData` instead of
    hitting the loaders; ``splits_dir`` writes the split audit parquet for this run.

    ``peak_rss_mb`` is the **process** peak (``ru_maxrss``), which is per-run only in the
    normal subprocess mode; with ``max_workers=0`` every inline row reports the same
    high-water mark of the shared parent process. ``peak_vram_mb`` is
    ``torch.cuda.max_memory_allocated`` when torch and a GPU are present, else 0.
    """
    t_start = time.perf_counter()
    row: dict[str, Any] = {
        **spec.as_row(),
        "is_deep": spec.is_deep,
        "status": "ok",
        "traceback": "",
        "uncalibrated": False,
        "git_rev": git_rev(),
        "started_at": pd.Timestamp.utcnow().isoformat(),
        "fit_seconds": float("nan"),
        "score_seconds_per_10k": float("nan"),
        "n_params": float("nan"),
        "peak_rss_mb": float("nan"),
        "peak_vram_mb": float("nan"),
        "wall_seconds": float("nan"),
    }
    try:
        if data is None:
            data = load_bench(spec.dataset, cache_dir=cache_dir, **spec.loader_kwargs())
        if data.input_kind != spec.input_kind:
            raise ValueError(
                f"run_one: spec asks for input_kind={spec.input_kind!r} but the loader returned "
                f"{data.input_kind!r}"
            )
        if data.dataset != spec.dataset:
            raise ValueError(
                f"run_one: spec asks for dataset={spec.dataset!r} but the loader returned {data.dataset!r}"
            )
        if spec.subsystem and data.subsystem and data.subsystem != spec.subsystem:
            raise ValueError(
                f"run_one: spec asks for subsystem={spec.subsystem!r} but the loader returned "
                f"{data.subsystem!r} - the results row would name a subsystem the numbers do not "
                f"come from"
            )
        _check_axes_against_loader(spec, data)
        _check_registry_contract(spec)
        row.update({"n_rows": len(data), "n_input_features": data.n_features, "dataset_subsystem": data.subsystem})

        splits = make_split(spec.split, data, seed=spec.seed)
        if splits_dir is not None:
            Path(splits_dir).mkdir(parents=True, exist_ok=True)
            frame = splits_to_frame(splits)
            frame["config_hash"] = spec.config_hash
            frame.to_parquet(Path(splits_dir) / f"{spec.config_hash}.parquet", engine="pyarrow", index=False)

        fold_rows: list[dict[str, Any]] = []
        calibration_audit: list[pd.DataFrame] = []
        fit_seconds = 0.0
        score_seconds = 0.0
        scored_rows = 0
        n_params = float("nan")
        budget_s = spec.max_minutes * 60.0

        for split in splits:
            tr = train_rows(
                data,
                split.train,
                train_regime=spec.train_regime,
                contamination=spec.contamination,
                seed=spec.seed,
                task=spec.task,
            )
            if tr.size == 0:
                raise ValueError(f"run_one: split {split.name} fold {split.fold} has no training rows")
            model = build(spec.model, **spec.params)
            remaining = max(0.0, budget_s - (time.perf_counter() - t_start))
            model = _fit_model(model, data, spec, tr, budget_s=remaining)
            fit_seconds += float(getattr(model, "fit_seconds", float("nan")))
            n_params = _n_params(model)

            thr: T.ThresholdSet | None = None
            cal_info: dict[str, Any] = {}
            if spec.task in ("ad", "cpd"):
                if split.val.size:
                    thr, cal_info = _calibrate(model, data, spec, split.val)
                else:
                    cal_info = {
                        "uncalibrated": True,
                        "uncalibrated_reason": f"split {split.name} fold {split.fold} has an empty validation slice",
                    }
                if cal_info.get("uncalibrated"):
                    LOGGER.warning(
                        "run %s: %s - no operating point, threshold metrics will be NaN",
                        spec.config_hash,
                        cal_info.get("uncalibrated_reason", ""),
                    )

            # --- test scores exist only from here on ---
            t0 = time.perf_counter()
            if spec.task in ("ad", "cpd"):
                test_out = _score(model, data, split.test)
            else:
                test_out = np.asarray(model.predict(_slice(data, split.test)))
            score_seconds += time.perf_counter() - t0
            scored_rows += int(split.test.size)

            if spec.task in ("ad", "cpd"):
                fold_rows.append(_ad_fold_metrics(data, spec, split.test, test_out, thr))
            elif spec.task == "cls":
                fold_rows.append(_cls_fold_metrics(data, spec, split.test, test_out))
            else:
                fold_rows.append(_rul_fold_metrics(data, spec, split.test, test_out))
            if thr is not None:
                fold_rows[-1].update(thr.as_dict())
            cal_index = cal_info.pop("_calibrated_on_index", None)
            if cal_index is not None:
                calibration_audit.append(
                    pd.DataFrame(
                        {
                            "fold": int(split.fold),
                            "split": split.name,
                            "part": "calibrated_on",
                            "row_index": np.asarray(cal_index, dtype=np.int64),
                        }
                    )
                )
            fold_rows[-1].update(cal_info)
            fold_rows[-1].update({"train_rows": int(tr.size), "val_rows": int(split.val.size)})
            # Deep rows only: how long the network actually trained and why it stopped.
            fold_rows[-1].update(_deep_train_info(model))

        # The rows the thresholds were actually calibrated on, joinable to this results row by
        # config_hash. thresholds.ThresholdSet.as_dict only carries their count and digest.
        if splits_dir is not None and calibration_audit:
            frame = pd.concat(calibration_audit, ignore_index=True)
            frame["config_hash"] = spec.config_hash
            frame.to_parquet(
                Path(splits_dir) / f"{spec.config_hash}.calibration.parquet", engine="pyarrow", index=False
            )

        row.update(_aggregate_folds(fold_rows))
        row["fit_seconds"] = fit_seconds
        row["score_seconds_per_10k"] = (score_seconds / scored_rows * 10_000.0) if scored_rows else float("nan")
        row["n_params"] = n_params
    except Exception:
        row["status"] = "failed"
        row["traceback"] = traceback.format_exc()
        LOGGER.warning("run %s (%s/%s) failed:\n%s", spec.config_hash, spec.dataset, spec.model, row["traceback"])
    row["peak_rss_mb"] = _peak_rss_mb()
    row["peak_vram_mb"] = _peak_vram_mb()
    row["wall_seconds"] = time.perf_counter() - t_start
    row["finished_at"] = pd.Timestamp.utcnow().isoformat()
    return row


#: Columns every row carries regardless of task (metrics are added on top).
ROW_COLUMNS: tuple[str, ...] = (
    "dataset",
    "model",
    "task",
    "subsystem",
    "input_kind",
    "split",
    "params",
    "window",
    "feature_set",
    "train_regime",
    "H",
    "peer_norm",
    "contamination",
    "seed",
    "data_kwargs",
    "block",
    "max_minutes",
    "config_hash",
    "is_deep",
    "status",
    "traceback",
    "git_rev",
    "fit_seconds",
    "score_seconds_per_10k",
    "n_params",
    "peak_rss_mb",
    "peak_vram_mb",
    "wall_seconds",
)


# --------------------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------------------


def _worker(spec: RunSpec, cache_dir: Any, splits_dir: Any, q: Any) -> None:  # pragma: no cover - subprocess
    try:
        _import_models()
        q.put(run_one(spec, cache_dir=cache_dir, splits_dir=splits_dir))
    except Exception:
        q.put({**spec.as_row(), "status": "failed", "traceback": traceback.format_exc(), "git_rev": git_rev()})


def _import_models() -> None:
    """Populate the registry: import ``nebulax.models`` plus anything named in the
    ``NEBULAX_MODEL_MODULES`` environment variable (comma-separated).

    The env hook exists because worker processes are spawned, not forked: a model registered
    only inside the parent's ``__main__`` would not exist in the child. Anything the sweep
    must be able to build has to be importable by module path, and this is how a caller says
    where from. ``nebulax.models`` being absent is not an error - it does not exist yet during
    W1 - it just means the registry holds whatever was registered by hand.
    """
    import importlib

    for name in ["nebulax.models", *(m.strip() for m in os.environ.get("NEBULAX_MODEL_MODULES", "").split(","))]:
        if not name:
            continue
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if name == "nebulax.models":
                LOGGER.debug("bench.runner: nebulax.models not importable yet (%s)", exc)
            else:
                LOGGER.warning("bench.runner: NEBULAX_MODEL_MODULES entry %r is not importable (%s)", name, exc)


def _timeout_row(spec: RunSpec, seconds: float) -> dict[str, Any]:
    return {
        **spec.as_row(),
        "is_deep": spec.is_deep,
        "status": "timeout",
        "traceback": f"run exceeded its wall-clock cap of {seconds / 60.0:.1f} min and was terminated",
        "git_rev": git_rev(),
        "fit_seconds": float("nan"),
        "score_seconds_per_10k": float("nan"),
        "n_params": float("nan"),
        "peak_rss_mb": float("nan"),
        "peak_vram_mb": float("nan"),
        "wall_seconds": float(seconds),
    }


def _run_batch(
    specs: Sequence[RunSpec], cache_dir: Any, splits_dir: Any, caps: Sequence[float]
) -> list[dict[str, Any]]:
    """Run ``specs`` concurrently, one subprocess each, and collect a row for every one.

    Each process is joined against its own wall-clock cap counted from the moment the batch
    started; a process still alive at its deadline is terminated (then killed) and reported as
    ``status="timeout"``. A worker that dies without putting a row still yields a
    ``status="failed"`` row naming its exit code, so ``len(rows) == len(specs)`` always.
    """
    ctx = mp.get_context("spawn")
    jobs = []
    t0 = time.perf_counter()
    for spec, cap in zip(specs, caps):
        q = ctx.Queue()
        proc = ctx.Process(target=_worker, args=(spec, cache_dir, splits_dir, q), daemon=False)
        proc.start()
        jobs.append((spec, proc, q, float(cap)))

    rows: list[dict[str, Any]] = []
    for spec, proc, q, cap in jobs:
        # A row can be sitting in the pipe while the child is still shutting down, so read
        # first (with the remaining budget as the timeout) and only then join.
        remaining = max(1.0, cap - (time.perf_counter() - t0))
        row: dict[str, Any] | None = None
        try:
            row = q.get(timeout=remaining)
        except Exception:
            row = None
        if row is None and proc.is_alive():
            proc.terminate()
            proc.join(10)
            if proc.is_alive():  # pragma: no cover
                proc.kill()
            rows.append(_timeout_row(spec, cap))
            continue
        proc.join(30)
        if row is None:  # pragma: no cover - worker died before putting a row
            row = {
                **spec.as_row(),
                "is_deep": spec.is_deep,
                "status": "failed",
                "traceback": f"worker exited with code {proc.exitcode} without returning a row",
                "git_rev": git_rev(),
            }
        rows.append(row)
    return rows


def run(
    config: Mapping[str, Any] | str | Path,
    out_dir: str | Path = "results/runs",
    *,
    max_workers: int = 4,
    max_minutes_per_run: float | None = None,
    skip_existing: bool = True,
    limit: int | None = None,
    dry_run: bool = False,
    cache_dir: str | Path | None = "data/features",
) -> pd.DataFrame:
    """Run a whole sweep: one parquet per run under ``out_dir``, plus a combined
    ``<out_dir>/../runs.parquet``.

    ``max_minutes_per_run`` overrides every spec's ``max_minutes``, so the model's own early
    stopping budget and the subprocess timeout are always the same number (and, for deep specs,
    the one in the config hash). ``skip_existing`` skips any spec whose
    ``<config_hash>.parquet`` already exists, so an interrupted sweep resumes exactly where it
    stopped. ``dry_run`` expands and prints the
    specs without running anything. ``max_workers=0`` runs inline (no subprocess, so the
    per-run cap is advisory); ``>=1`` runs classical specs in that many worker processes and
    deep/GPU specs strictly one at a time.
    """
    cfg = load_config(config) if isinstance(config, (str, Path)) else dict(config)
    # Before expand(), and before the dry-run return: `input_kind: auto` resolves each model's
    # kind from the registry, which is empty in a fresh process until the model modules are
    # imported. Expanding first made --dry-run fail on exactly the configs it exists to check.
    _import_models()
    specs = expand(cfg)
    if max_minutes_per_run is not None:
        # Apply the override to the SPECS, not just to the subprocess cap. run_one hands
        # spec.max_minutes to a deep model's `budget_s` for early stopping, so overriding only
        # the external timeout left the model training to the config's budget and then killed
        # it at the CLI's - two different numbers, and the row recorded the wrong one.
        specs = [replace(sp, max_minutes=float(max_minutes_per_run)) for sp in specs]
    if limit is not None:
        specs = specs[: int(limit)]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits_dir = out.parent / "splits"
    combined_path = out.parent / "runs.parquet"

    if dry_run:
        LOGGER.info("bench.runner: dry run, %d spec(s)", len(specs))
        return pd.DataFrame([s.as_row() for s in specs])

    pending = [s for s in specs if not (skip_existing and (out / f"{s.config_hash}.parquet").exists())]
    n_skipped = len(specs) - len(pending)
    LOGGER.info("bench.runner: %d spec(s), %d already done, %d to run", len(specs), n_skipped, len(pending))

    deep = [s for s in pending if s.is_deep]
    classical = [s for s in pending if not s.is_deep]
    rows: list[dict[str, Any]] = []

    def _finish(row: Mapping[str, Any]) -> None:
        df = pd.DataFrame([dict(row)])
        df.to_parquet(out / f"{row['config_hash']}.parquet", engine="pyarrow", index=False)
        rows.append(dict(row))

    if max_workers <= 0:
        for spec in classical + deep:
            _finish(run_one(spec, cache_dir=cache_dir, splits_dir=splits_dir))
    else:
        for group, workers in ((classical, max(1, int(max_workers))), (deep, 1)):
            for i in range(0, len(group), workers):
                batch = group[i : i + workers]
                caps = [sp.max_minutes * 60.0 for sp in batch]  # already overridden above
                for row in _run_batch(batch, cache_dir, splits_dir, caps):
                    _finish(row)

    combined = _combine(out)
    if len(combined):
        combined.to_parquet(combined_path, engine="pyarrow", index=False)
    LOGGER.info("bench.runner: wrote %d row(s) to %s", len(combined), combined_path)
    return combined


def _combine(out_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(out_dir.glob("*.parquet")):
        try:
            frames.append(pd.read_parquet(p, engine="pyarrow"))
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("bench.runner: skipping unreadable run file %s (%s)", p, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
