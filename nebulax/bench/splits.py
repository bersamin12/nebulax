"""Train / validation / test splits for the benchmark, and the named presets the protocol
fixes per dataset.

Three rules this module exists to enforce
-----------------------------------------
1. **Disjoint.** ``train``, ``val`` and ``test`` never share a row. :meth:`Split.check` is
   called by every constructor here, so a broken split raises at build time, not after a
   12-hour sweep.
2. **Grouped.** Where the protocol says leave-one-*group*-out, no group id appears on both
   sides of the fence - that is what stops device-level leakage inflating a result
   (``configs/model_ladder.yaml:evaluation.rul_splits``).
3. **Auditable.** Every split serialises to a tidy frame (:meth:`Split.to_frame`) that the
   runner writes next to the results, so any published number can be traced back to the exact
   row indices that produced it.

Named presets (``configs/core.yaml: split:``)
---------------------------------------------
``metropt_temporal``          train 1 Feb - 31 Mar 2020, val 1-10 Apr, test 11 Apr onward
``metropt_contaminated``      train 1 Feb - 31 May, val 1-3 Jun, test 4 Jun onward
``cranfield_rep``             reps 0-6 train / rep 7 val / reps 8-9 test (by rep rank)
``cranfield_loo_load``        leave-one-load-out (20 kg / 40 kg / -40 kg)
``cranfield_loo_profile``     leave-one-motion-profile-out (sinusoidal / trapezoidal)
``ottawa_sgkf5``              StratifiedGroupKFold(5), groups = bearing id, strata = fault type
``sim_loo_unit``              leave-one-unit-out by ``train_id``
``sim_run_kfold``             GroupKFold(5) by ``run_id``
``temporal_fracs``            generic 60/20/20 chronological split, for a dataset with no preset
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import logging

import numpy as np
import pandas as pd

from nebulax.bench.data import PEER_GROUP_COL

LOGGER = logging.getLogger(__name__)

from nebulax.bench.metrics import to_epoch_seconds

__all__ = [
    "Split",
    "temporal_split",
    "group_kfold",
    "leave_one_group_out",
    "leave_one_unit_out",
    "PRESETS",
    "SplitPreset",
    "make_split",
    "splits_to_frame",
    "random_group_holdout",
    "check_preset_dataset",
    "KNOWN_DATASETS",
]


@dataclass(frozen=True)
class Split:
    """One train/val/test partition of a row-indexed dataset.

    Index arrays are ``int64`` positions into the benchmark table, sorted ascending.
    ``fold`` is 0 for a single split and 0..k-1 for a cross-validated preset. ``meta``
    records how the split was made (cut dates, held-out group, ...) and is written to the
    audit parquet verbatim.
    """

    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    n_rows: int
    fold: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "train", np.asarray(self.train, dtype=np.int64).reshape(-1))
        object.__setattr__(self, "val", np.asarray(self.val, dtype=np.int64).reshape(-1))
        object.__setattr__(self, "test", np.asarray(self.test, dtype=np.int64).reshape(-1))
        self.check()

    def check(self, *, require_cover: bool = False) -> None:
        """Raise unless the three parts are disjoint and in range; optionally also covering."""
        parts = {"train": self.train, "val": self.val, "test": self.test}
        for name, idx in parts.items():
            if idx.size and (idx.min() < 0 or idx.max() >= self.n_rows):
                raise ValueError(
                    f"Split {self.name!r} fold {self.fold}: {name} index out of range "
                    f"[0, {self.n_rows}) (min={idx.min()}, max={idx.max()})"
                )
            if idx.size != np.unique(idx).size:
                raise ValueError(f"Split {self.name!r} fold {self.fold}: {name} has duplicate indices")
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = np.intersect1d(parts[a], parts[b], assume_unique=True)
            if overlap.size:
                raise ValueError(
                    f"Split {self.name!r} fold {self.fold}: {a} and {b} overlap on "
                    f"{overlap.size} row(s), first={int(overlap[0])}"
                )
        if require_cover:
            covered = np.concatenate([self.train, self.val, self.test]) if self.n_rows else np.empty(0, np.int64)
            if np.unique(covered).size != self.n_rows:
                raise ValueError(
                    f"Split {self.name!r} fold {self.fold}: covers {np.unique(covered).size} of "
                    f"{self.n_rows} rows"
                )

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": int(self.train.size), "val": int(self.val.size), "test": int(self.test.size)}

    def to_frame(self) -> pd.DataFrame:
        """Tidy audit frame: one row per (split, fold, part, row index)."""
        frames = [
            pd.DataFrame({"split": self.name, "fold": self.fold, "part": part, "row": idx})
            for part, idx in (("train", self.train), ("val", self.val), ("test", self.test))
            if idx.size
        ]
        if not frames:
            return pd.DataFrame({"split": [], "fold": [], "part": [], "row": []})
        return pd.concat(frames, ignore_index=True)


def splits_to_frame(splits: Sequence[Split]) -> pd.DataFrame:
    """Concatenate :meth:`Split.to_frame` over several folds (what the runner persists)."""
    frames = [s.to_frame() for s in splits]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        {"split": [], "fold": [], "part": [], "row": []}
    )


# --------------------------------------------------------------------------------------
# Primitive splitters
# --------------------------------------------------------------------------------------


def temporal_split(
    times: Any,
    train_end: Any,
    val_end: Any,
    *,
    name: str = "temporal",
    test_end: Any = None,
    train_start: Any = None,
    t_start: Any = None,
) -> Split:
    """Chronological split: ``train = [train_start, train_end)``, ``val = [train_end,
    val_end)``, ``test = [val_end, test_end)``.

    Cuts are **half-open on the right**, so a row exactly at ``train_end`` is validation.
    ``train_start`` / ``test_end`` are optional outer bounds (rows outside them belong to no
    part - that is how the MetroPT "contaminated" variant drops nothing but is still able to
    start the training window in February). Rows with a NaT timestamp are dropped from every
    part rather than silently landing in train.

    ``t_start`` **purges rows that straddle a cut**, and should always be passed when the rows
    are overlapping windows. A row is placed by its ``times`` value (its window *end*), so
    without this a MetroPT window at ``window=60, stride=30`` puts 300 s of raw samples on both
    sides of a boundary: the last validation window and the first test window are computed from
    partly the same observations, and the threshold is then calibrated on data the test slice
    also contains. With ``t_start`` given, a row joins a part only if its whole support
    ``[t_start, t_end]`` lies inside it, and the straddling rows belong to no part at all -
    the standard purge. ``meta["n_purged_straddling"]`` records how many were dropped, so the
    cost is visible rather than inferred.
    """
    t = to_epoch_seconds(times).reshape(-1)
    n = t.size
    ts = None if t_start is None else to_epoch_seconds(t_start).reshape(-1)
    if ts is not None and ts.size != n:
        raise ValueError(f"temporal_split: times has {n} rows but t_start has {ts.size}")
    lo = -np.inf if train_start is None else to_epoch_seconds(np.asarray([pd.Timestamp(train_start)]))[0]
    a = to_epoch_seconds(np.asarray([pd.Timestamp(train_end)]))[0]
    b = to_epoch_seconds(np.asarray([pd.Timestamp(val_end)]))[0]
    hi = np.inf if test_end is None else to_epoch_seconds(np.asarray([pd.Timestamp(test_end)]))[0]
    if not (lo <= a <= b <= hi):
        raise ValueError(
            f"temporal_split: cuts must be ordered train_start <= train_end <= val_end <= test_end, "
            f"got {train_start} / {train_end} / {val_end} / {test_end}"
        )
    ok = np.isfinite(t)
    idx = np.arange(n, dtype=np.int64)
    # A row's support starts at t_start when we know it, and at its own end otherwise (a point
    # observation). ``begins`` is what decides whether a row has crossed a cut.
    begins = t if ts is None else ts
    if ts is not None:
        ok &= np.isfinite(ts)
    train = idx[ok & (begins >= lo) & (t < a)]
    val = idx[ok & (begins >= a) & (t < b)]
    test = idx[ok & (begins >= b) & (t < hi)]
    n_purged = int(ok.sum()) - (train.size + val.size + test.size)
    if ts is not None and n_purged:
        LOGGER.info(
            "temporal_split(%s): purged %d window(s) straddling a cut (their raw support spans "
            "the boundary, so they cannot belong to either side)",
            name,
            n_purged,
        )
    return Split(
        name=name,
        train=train,
        val=val,
        test=test,
        n_rows=n,
        meta={
            "kind": "temporal",
            "train_start": str(train_start),
            "train_end": str(train_end),
            "val_end": str(val_end),
            "test_end": str(test_end),
            "n_dropped_nat": int((~np.isfinite(t)).sum()),
            "purged_by_support": ts is not None,
            "n_purged_straddling": max(0, n_purged),
        },
    )


def _carve_val(train_groups: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a set of training group ids into (train, val) group id arrays, by whole groups.

    Returns an **empty** validation array when there is only one group left to carve from;
    callers must then fall back to :func:`_carve_val_rows`, because an AD run with no
    validation slice has no operating point at all.
    """
    uniq = np.asarray(pd.unique(train_groups))
    if uniq.size < 2:
        return uniq, np.asarray([], dtype=uniq.dtype)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq.size)
    n_val = max(1, int(round(val_frac * uniq.size)))
    n_val = min(n_val, uniq.size - 1)
    return uniq[perm[n_val:]], uniq[perm[:n_val]]


def _carve_val_rows(
    available: np.ndarray, val_frac: float, seed: int, fine_groups: np.ndarray | None = None
) -> tuple[np.ndarray, str]:
    """Validation fallback when a single split group is all that remains outside the test fold
    (leave-one-motion-profile-out on Cranfield's two profiles is the case the protocol
    contains). Returns ``(val_rows, carved_by)``.

    Carving is by the **finest series** available (``fine_groups``, the ``BenchData.series``
    column: one recording, one component) whenever ``available`` spans at least two of them,
    so a 4 s window of a recording is never calibrated against while its neighbours from the
    same 80 s recording are being fitted on - that was the leak: 385/385 Cranfield recordings
    straddled train and validation under the old row-level fallback. Only a single-series
    remainder falls back to a seeded random ``val_frac`` of rows (``carved_by="row"``), which
    is still better than an empty validation slice that produces no threshold, no event recall
    and no false-alarm rate. The fold records ``val_carved_by`` in its meta so the weakening
    is visible in the split audit parquet.
    """
    a = np.asarray(available, dtype=np.int64).reshape(-1)
    if a.size < 2:
        return np.asarray([], dtype=np.int64), "none"
    if fine_groups is not None:
        fg = np.asarray(fine_groups, dtype=object).reshape(-1)[a]
        if pd.unique(fg).size >= 2:
            _, va_groups = _carve_val(fg, val_frac, seed)
            rows = a[np.isin(fg, va_groups)]
            if rows.size and rows.size < a.size:
                return np.sort(rows), "series"
    rng = np.random.default_rng(seed)
    n_val = int(np.clip(round(float(val_frac) * a.size), 1, a.size - 1))
    return np.sort(a[rng.permutation(a.size)[:n_val]]), "row"


def group_kfold(
    groups: Any,
    k: int = 5,
    *,
    stratify: Any = None,
    seed: int = 0,
    val_frac: float = 0.2,
    name: str | None = None,
    fine_groups: Any = None,
) -> list[Split]:
    """``k``-fold cross-validation by whole groups; validation is carved out of the training
    groups (again by whole groups), never out of the test fold. ``fine_groups`` (the series
    column) is what the validation fallback carves by when only one group remains.

    ``stratify`` (one label per row) switches sklearn's ``StratifiedGroupKFold`` on, which is
    what the protocol asks for on Ottawa (5 folds, groups = bearing, strata = fault type).
    Without it, plain ``GroupKFold``. Every fold's ``train | val | test`` covers all rows.
    """
    g = np.asarray(groups, dtype=object).reshape(-1)
    n = g.size
    idx = np.arange(n, dtype=np.int64)
    k = int(k)
    n_groups = pd.unique(g).size
    if k < 2 or k > n_groups:
        raise ValueError(f"group_kfold: k must be in [2, n_groups={n_groups}], got {k}")
    label = name or ("group_sgkf" if stratify is not None else "group_kfold")

    if stratify is not None:
        from sklearn.model_selection import StratifiedGroupKFold

        y = pd.factorize(np.asarray(stratify).reshape(-1))[0]
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        folds = list(splitter.split(np.zeros((n, 1)), y, groups=g))
    else:
        from sklearn.model_selection import GroupKFold

        folds = list(GroupKFold(n_splits=k).split(np.zeros((n, 1)), None, groups=g))

    out: list[Split] = []
    for f, (tr_idx, te_idx) in enumerate(folds):
        tr_groups, va_groups = _carve_val(g[tr_idx], val_frac, seed + f)
        in_val = np.isin(g, va_groups)
        va_rows = idx[tr_idx][in_val[tr_idx]]
        carved_by = "group"
        if va_rows.size == 0:  # single training group: carve by series, else rows (see _carve_val_rows)
            va_rows, carved_by = _carve_val_rows(idx[tr_idx], val_frac, seed + f, fine_groups)
        is_val = np.zeros(n, dtype=bool)
        is_val[va_rows] = True
        out.append(
            Split(
                name=label,
                fold=f,
                n_rows=n,
                train=np.sort(idx[tr_idx][~is_val[tr_idx]]),
                val=np.sort(va_rows),
                test=np.sort(idx[te_idx]),
                meta={
                    "kind": "stratified_group_kfold" if stratify is not None else "group_kfold",
                    "k": k,
                    "seed": seed,
                    "val_frac": val_frac,
                    "test_groups": [str(x) for x in pd.unique(g[te_idx])],
                    "val_groups": [str(x) for x in va_groups],
                    "val_carved_by": carved_by,
                },
            )
        )
    return out


def leave_one_group_out(
    groups: Any,
    *,
    seed: int = 0,
    val_frac: float = 0.2,
    name: str = "logo",
    only: Sequence[Any] | None = None,
    fine_groups: Any = None,
) -> list[Split]:
    """One fold per distinct group: that group is the test set, the rest is train, and a
    random subset of the *remaining* groups is validation.

    ``only`` restricts which groups are ever held out (the protocol's leave-one-load-out and
    leave-one-motion-profile-out both use every level, but the sim's leave-one-unit-out gets
    expensive at 10 trains x every model)."""
    g = np.asarray(groups, dtype=object).reshape(-1)
    n = g.size
    idx = np.arange(n, dtype=np.int64)
    uniq = list(pd.unique(g))
    held = [x for x in uniq if only is None or x in set(only)]
    if len(uniq) < 2:
        raise ValueError(f"leave_one_group_out: need >= 2 groups, got {uniq}")
    out: list[Split] = []
    for f, held_out in enumerate(held):
        is_test = g == held_out
        rest = g[~is_test]
        tr_groups, va_groups = _carve_val(rest, val_frac, seed + f)
        in_val = np.isin(g, va_groups) & ~is_test
        carved_by = "group"
        if not in_val.any():  # only one non-held-out group: carve by series, else rows
            rows, carved_by = _carve_val_rows(idx[~is_test], val_frac, seed + f, fine_groups)
            in_val = np.zeros(n, dtype=bool)
            in_val[rows] = True
        out.append(
            Split(
                name=name,
                fold=f,
                n_rows=n,
                train=np.sort(idx[~is_test & ~in_val]),
                val=np.sort(idx[in_val]),
                test=np.sort(idx[is_test]),
                meta={
                    "kind": "leave_one_group_out",
                    "held_out_group": str(held_out),
                    "val_groups": [str(x) for x in va_groups],
                    "val_carved_by": carved_by,
                    "seed": seed,
                    "val_frac": val_frac,
                },
            )
        )
    return out


def leave_one_unit_out(units: Any, **kwargs: Any) -> list[Split]:
    """:func:`leave_one_group_out` with ``units`` (train / bearing / run ids) as the groups -
    the protocol's RUL and synthetic-fleet rule, named the way the plan names it."""
    kwargs.setdefault("name", "leave_one_unit_out")
    return leave_one_group_out(units, **kwargs)


def random_group_holdout(
    groups: Any,
    *,
    seed: int = 0,
    test_frac: float = 0.2,
    val_frac: float = 0.2,
    name: str = "random_group",
) -> list[Split]:
    """A single random split by whole groups - the "random" arm of the CLS generalisation gap
    (``F1(random) - F1(held-out group)``), so both arms use the same grouping machinery."""
    g = np.asarray(groups, dtype=object).reshape(-1)
    n = g.size
    idx = np.arange(n, dtype=np.int64)
    uniq = np.asarray(pd.unique(g))
    rng = np.random.default_rng(seed)
    perm = uniq[rng.permutation(uniq.size)]
    n_test = max(1, int(round(test_frac * uniq.size)))
    n_val = max(1, int(round(val_frac * uniq.size)))
    if n_test + n_val >= uniq.size:
        n_test, n_val = 1, 1
    test_g, val_g = set(perm[:n_test].tolist()), set(perm[n_test : n_test + n_val].tolist())
    is_test = np.array([x in test_g for x in g])
    is_val = np.array([x in val_g for x in g])
    return [
        Split(
            name=name,
            fold=0,
            n_rows=n,
            train=idx[~is_test & ~is_val],
            val=idx[is_val],
            test=idx[is_test],
            meta={
                "kind": "random_group_holdout",
                "seed": seed,
                "test_groups": [str(x) for x in sorted(test_g, key=str)],
                "val_groups": [str(x) for x in sorted(val_g, key=str)],
            },
        )
    ]


# --------------------------------------------------------------------------------------
# Named presets
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SplitPreset:
    """A named split from the evaluation protocol: what it needs and what it produces."""

    name: str
    kind: str
    datasets: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)
    doc: str = ""


PRESETS: dict[str, SplitPreset] = {
    "metropt_temporal": SplitPreset(
        "metropt_temporal",
        "temporal",
        ("metropt3",),
        {"train_end": "2020-04-01", "val_end": "2020-04-11"},
        "train 1 Feb - 31 Mar 2020, val 1-10 Apr, test 11 Apr onward",
    ),
    "metropt_contaminated": SplitPreset(
        "metropt_contaminated",
        "temporal",
        ("metropt3",),
        {"train_end": "2020-06-01", "val_end": "2020-06-04"},
        "contaminated variant: train Feb-May, val 1-3 Jun, test 4 Jun onward",
    ),
    "cranfield_rep": SplitPreset(
        "cranfield_rep",
        "rep_rank",
        ("cranfield",),
        {"group_col": "meta_rep", "train_ranks": [0, 1, 2, 3, 4, 5, 6], "val_ranks": [7], "test_ranks": [8, 9]},
        "reps 0-6 train / rep 7 val / reps 8-9 test, by rank of the sorted distinct rep ids",
    ),
    "cranfield_loo_load": SplitPreset(
        "cranfield_loo_load",
        "leave_one_group_out",
        ("cranfield",),
        {"group_col": "load_kg"},
        "leave-one-load-out (20 kg / 40 kg / -40 kg)",
    ),
    "cranfield_loo_profile": SplitPreset(
        "cranfield_loo_profile",
        "leave_one_group_out",
        ("cranfield",),
        {"group_col": "meta_motion_profile"},
        "leave-one-motion-profile-out (sinusoidal / trapezoidal)",
    ),
    "cranfield_random_rep": SplitPreset(
        "cranfield_random_rep",
        "random_group",
        ("cranfield",),
        {"group_col": "meta_rep"},
        "random arm of the generalisation gap, grouped by rep",
    ),
    "ottawa_sgkf5": SplitPreset(
        "ottawa_sgkf5",
        "group_kfold",
        ("ottawa",),
        {"group_col": "meta_bearing_id", "stratify_col": "fault_type", "k": 5},
        "StratifiedGroupKFold(5) by bearing id, stratified on fault type",
    ),
    "sim_loo_unit": SplitPreset(
        "sim_loo_unit",
        "leave_one_group_out",
        ("sim",),
        {"group_col": "train_id"},
        "leave-one-unit-out by train_id across the synthetic fleet",
    ),
    "sim_run_kfold": SplitPreset(
        "sim_run_kfold",
        "group_kfold",
        ("sim",),
        {"group_col": "run_id", "k": 5},
        "GroupKFold(5) by run_id",
    ),
    "sim_random_unit": SplitPreset(
        "sim_random_unit",
        "random_group",
        ("sim",),
        {"group_col": "train_id"},
        "random arm of the generalisation gap, grouped by train_id",
    ),
    "temporal_fracs": SplitPreset(
        "temporal_fracs",
        "temporal_fracs",
        ("metropt3", "sim"),
        {"train_frac": 0.6, "val_frac": 0.2},
        "chronological 60/20/20 fallback - time-ordered datasets ONLY (see KNOWN_DATASETS)",
    ),
}

#: The datasets :func:`make_split` polices. A ``BenchData`` whose ``dataset`` is one of these
#: may only use a preset that lists it; anything else (a test fixture, a new dataset) is left
#: alone. See :func:`check_preset_dataset`.
KNOWN_DATASETS: frozenset[str] = frozenset({"metropt3", "cranfield", "ottawa", "sim"})


def check_preset_dataset(preset: str, dataset: Any) -> None:
    """Raise unless ``preset`` is declared for ``dataset``.

    Presets are not interchangeable: they encode *which* grouping the protocol requires, and
    a chronological split of a dataset whose rows are repetitions of a rig test is not a
    weaker split, it is a broken one. ``temporal_fracs`` on Ottawa puts the same bearing in
    train and test - and Ottawa's CLS target is constant per bearing, so that is direct label
    leakage - while leaving whole fault classes out of the test slice. On Cranfield it leaves
    every ``meta_rep`` on both sides. Both were accepted configs before this check existed.

    ``dataset`` values outside :data:`KNOWN_DATASETS` (test fixtures, datasets added later)
    are not policed - there is nothing to check them against.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown split preset {preset!r}; known = {sorted(PRESETS)}")
    name = None if dataset is None else str(dataset)
    if name is None or name not in KNOWN_DATASETS:
        return
    allowed = PRESETS[preset].datasets
    if name not in allowed:
        ok = sorted(k for k, v in PRESETS.items() if name in v.datasets)
        raise ValueError(
            f"split preset {preset!r} is not defined for dataset {name!r} "
            f"(it is for {list(allowed)}); use one of {ok}"
        )


def _column(data: Any, col: str) -> np.ndarray:
    """Fetch a grouping column from a :class:`~nebulax.bench.data.BenchData`-like object.

    Looks in ``data.labels`` (the per-row label/metadata frame) first, then falls back to the
    ``unit`` / ``group`` attributes for the two names that are not table columns.
    """
    labels = getattr(data, "labels", None)
    if labels is not None and col in getattr(labels, "columns", []):
        return np.asarray(labels[col].to_numpy(), dtype=object)
    if col in ("unit", "train_id") and getattr(data, "unit", None) is not None:
        return np.asarray(data.unit, dtype=object)
    if col == "group" and getattr(data, "group", None) is not None:
        return np.asarray(data.group, dtype=object)
    raise ValueError(
        f"make_split: grouping column {col!r} not found; available label columns = "
        f"{sorted(getattr(labels, 'columns', []))}"
    )


def _atomic_peer_supports(data: Any, t_start: Any, t_end: Any) -> tuple[Any, Any]:
    """Widen each row's support to its whole **peer group**, so a temporal cut cannot split one.

    Peer features are computed on the full table, before the split (see
    ``nebulax.bench.data.PEER_GROUP_COL``). A leave-one-out peer statistic is therefore a real
    leak the moment a chronological cut falls *inside* a peer group: the earlier member is a
    training row whose features already contain the later member, which is a test row.

    Arguing that this is unlikely - "peers start within a second of each other, and the cuts
    are day boundaries" - is not a proof, and the two door peers of one dwell really do end up
    to 9.3 s apart on the real fleet. So instead of arguing, make it impossible: report each
    row's support as its group's ``[min t_start, max t_end]``. ``temporal_split`` already
    purges anything whose support straddles a cut, so a peer group now lands whole in one
    partition or is dropped from all three, and ``meta["n_purged_straddling"]`` counts the cost.

    A no-op (the arrays come back unchanged) when the table has no peer-group column, i.e.
    whenever ``peer_norm`` is off.
    """
    labels = getattr(data, "labels", None)
    col = PEER_GROUP_COL
    if labels is None or col not in getattr(labels, "columns", []) or t_start is None:
        return t_start, t_end
    g = np.asarray(labels[col].to_numpy())
    ts = to_epoch_seconds(t_start).reshape(-1)
    te = to_epoch_seconds(t_end).reshape(-1)
    frame = pd.DataFrame({"g": g, "ts": ts, "te": te})
    grouped = frame.groupby("g", sort=False, observed=True)
    lo = grouped["ts"].transform("min").to_numpy()
    hi = grouped["te"].transform("max").to_numpy()
    widened = int(np.count_nonzero((lo < ts) | (hi > te)))
    if widened:
        LOGGER.info(
            "make_split: widening %d row support(s) to their peer group so no temporal cut can "
            "split a group whose features were computed before the split",
            widened,
        )
    return lo, hi


def make_split(preset: str, data: Any, *, seed: int = 0, val_frac: float = 0.2) -> list[Split]:
    """Build the folds of a named preset for a :class:`~nebulax.bench.data.BenchData`.

    Always returns a **list** of :class:`Split` (length 1 for a single split), so the runner
    has one code path. Raises ``ValueError`` for an unknown preset name, naming the known ones.
    """
    if preset not in PRESETS:
        raise ValueError(f"make_split: unknown split preset {preset!r}; known = {sorted(PRESETS)}")
    try:
        check_preset_dataset(preset, getattr(data, "dataset", None))
    except ValueError as exc:
        raise ValueError(f"make_split: {exc}") from None
    p = PRESETS[preset]
    times = getattr(data, "t_end")
    n = len(np.asarray(times).reshape(-1))

    t_start = getattr(data, "t_start", None)
    if p.kind in ("temporal", "temporal_fracs"):
        t_start, times = _atomic_peer_supports(data, t_start, times)
    if p.kind == "temporal":
        return [
            temporal_split(times, p.params["train_end"], p.params["val_end"], name=p.name, t_start=t_start)
        ]

    if p.kind == "temporal_fracs":
        t = to_epoch_seconds(times).reshape(-1)
        finite = t[np.isfinite(t)]
        if finite.size == 0:
            raise ValueError("make_split(temporal_fracs): no finite timestamps")
        a = float(np.quantile(finite, p.params["train_frac"]))
        b = float(np.quantile(finite, p.params["train_frac"] + p.params["val_frac"]))
        # Same support-purge as the named temporal presets - see temporal_split's docstring.
        sp = temporal_split(
            times,
            pd.Timestamp(a, unit="s"),
            pd.Timestamp(b, unit="s"),
            name=p.name,
            t_start=t_start,
        )
        sp.meta.update({"kind": "temporal_fracs", **p.params})
        return [sp]

    groups = _column(data, p.params["group_col"])
    fine = getattr(data, "series", None)

    if p.kind == "leave_one_group_out":
        return leave_one_group_out(groups, seed=seed, val_frac=val_frac, name=p.name, fine_groups=fine)
    if p.kind == "random_group":
        return random_group_holdout(groups, seed=seed, val_frac=val_frac, name=p.name)
    if p.kind == "group_kfold":
        strat = _column(data, p.params["stratify_col"]) if p.params.get("stratify_col") else None
        return group_kfold(
            groups, p.params.get("k", 5), stratify=strat, seed=seed, val_frac=val_frac, name=p.name, fine_groups=fine
        )
    if p.kind == "rep_rank":
        ranks = {v: i for i, v in enumerate(sorted(pd.unique(groups), key=lambda x: (str(type(x)), x)))}
        r = np.array([ranks[v] for v in groups])
        idx = np.arange(n, dtype=np.int64)
        tr, va, te = (set(p.params[key]) for key in ("train_ranks", "val_ranks", "test_ranks"))
        return [
            Split(
                name=p.name,
                n_rows=n,
                train=idx[np.isin(r, list(tr))],
                val=idx[np.isin(r, list(va))],
                test=idx[np.isin(r, list(te))],
                meta={
                    "kind": "rep_rank",
                    "group_col": p.params["group_col"],
                    "rank_of_group": {str(kk): int(vv) for kk, vv in ranks.items()},
                },
            )
        ]
    raise ValueError(f"make_split: preset {preset!r} has unhandled kind {p.kind!r}")
