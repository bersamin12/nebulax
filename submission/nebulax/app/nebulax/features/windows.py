"""Fixed-length windows cut from per-component telemetry.

``make_windows`` is the entry point every other module in this package sits on top of: it
turns a nebulax long telemetry frame (or an already-pivoted wide frame) into a
:class:`Windows` bundle whose ``X`` array is exactly the ``"raw_window"`` ``(n, L, c)``
``input_kind`` shape :mod:`nebulax.bench.base` expects, plus enough key columns
(``train_id``, ``component_id`` and, when present in the input, ``run_id``/``car``/``source``)
to trace every window straight back to :data:`nebulax.schema.FEATURE_KEY_COLUMNS`.

Windowing never crosses a ``(train_id, component_id, ...)`` group boundary **and never
crosses a logger outage**: rows are sorted by time and split wherever the sample-to-sample
gap exceeds ``max_gap_s`` (default ``3 / fs``, i.e. three nominal sample periods), so a
window is always a contiguous stretch of real acquisition rather than a slice by row
position. That guard is not hypothetical - the simulators do emit a regular grid, but
MetroPT-3 (``nebulax.adapters.metropt3``) has **190 gaps longer than 30 minutes** in its
10 s-cadence analogue log (docs/parameters.md, pneumatic section), and slicing it by row
position alone would have welded the two sides of an outage into one "continuous" raw window
whose stats and spectra are meaningless. Rows carrying a NaT/non-finite timestamp are treated
as a gap on both sides for the same reason. Genuine short dropouts *within* an acquisition
still show up as NaN *values*, not missing rows, and are left to the feature code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np
import pandas as pd

__all__ = ["Windows", "make_windows"]

#: Ticks per second for each datetime64/timedelta64 resolution pandas can hand us.
_PER_SECOND: Final[dict[str, float]] = {"s": 1.0, "ms": 1.0e3, "us": 1.0e6, "ns": 1.0e9}

#: Extra grouping/key columns carried through when present, beyond the mandatory
#: ``train_id``/``component_id``.
_OPTIONAL_KEYS: Final[tuple[str, ...]] = ("run_id", "car", "source", "subsystem")


@dataclass(slots=True)
class Windows:
    """One subsystem/signal-set's worth of fixed-length windows.

    ``X`` : ``(n, L, c)`` float32, channel order matches :attr:`signals`.
    ``t_end`` : ``(n,)`` - the timestamp of each window's last sample.
    ``train_id`` / ``component_id`` : ``(n,)`` object arrays, one value per window.
    ``run_id`` / ``car`` / ``source`` / ``subsystem`` : ``(n,)`` object arrays when the input
    frame carried that column, else ``None``.
    ``max_gap_s`` : the outage threshold the windows were cut with (seconds); every window is
    guaranteed to be free of any sample-to-sample gap larger than it.
    """

    X: np.ndarray
    t_end: np.ndarray
    train_id: np.ndarray
    component_id: np.ndarray
    signals: tuple[str, ...]
    fs: float
    L: int
    stride: int
    max_gap_s: float = float("inf")
    run_id: np.ndarray | None = None
    car: np.ndarray | None = None
    source: np.ndarray | None = None
    subsystem: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.X.shape[0])


def _is_long(df: pd.DataFrame) -> bool:
    return "signal" in df.columns and "value" in df.columns


def _sorted_index_values(index: pd.Index) -> np.ndarray:
    """``index`` (a DatetimeIndex, tz-aware or not) as a plain numpy array."""
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        return index.tz_convert("UTC").to_numpy()
    return index.to_numpy()


def _index_seconds(index: pd.Index) -> np.ndarray | None:
    """``index`` as float seconds for gap arithmetic, or ``None`` if it is not a time axis.

    NaT / non-finite entries come back as NaN so they can be treated as gaps. Works for
    tz-aware and tz-naive ``DatetimeIndex`` alike (``asi8`` is always UTC).
    """
    if isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        # ``asi8`` is in the index's own resolution (pandas 2 keeps s/ms/us/ns), so scale by
        # it rather than assuming nanoseconds - dividing a millisecond index by 1e9 would
        # shrink every gap 1000x and hide exactly the outages this exists to catch.
        # ``index.unit`` is the only reliable source: a tz-aware index's dtype is a pandas
        # ``DatetimeTZDtype`` (which has ``.unit``) but a tz-NAIVE one is a plain numpy
        # ``datetime64[ms]`` dtype, which does not, so ``getattr(dtype, "unit", "ns")``
        # silently reports nanoseconds for every non-ns tz-naive trace.
        unit = getattr(index, "unit", None) or np.datetime_data(index.dtype)[0]
        per_second = _PER_SECOND.get(unit, 1.0e9)
        i8 = index.asi8.astype(np.float64)
        return np.where(np.asarray(index.isna()), np.nan, i8 / per_second)
    if pd.api.types.is_numeric_dtype(index.dtype):
        return index.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
    return None


def _segment_bounds(index: pd.Index, max_gap_s: float) -> np.ndarray:
    """``(n_seg, 2)`` ``[start, stop)`` row ranges of ``index`` with no gap > ``max_gap_s``.

    ``index`` must already be sorted ascending. Datetimes and plain numeric
    timestamps-in-seconds are both understood; a NaT/non-finite timestamp is a gap on both
    sides of itself, so it can never sit inside a window. Any other dtype (an opaque object
    index) yields one segment - there is nothing to measure a gap with.
    """
    T = int(index.shape[0])
    if T == 0:
        return np.empty((0, 2), dtype=np.int64)
    seconds = _index_seconds(index) if np.isfinite(max_gap_s) else None
    if T < 2 or seconds is None:
        return np.array([[0, T]], dtype=np.int64)

    # A NaN difference (either endpoint unusable) compares False, so or it in explicitly.
    bad = ~np.isfinite(seconds)
    with np.errstate(invalid="ignore"):
        is_gap = np.diff(seconds) > max_gap_s
    is_gap |= bad[:-1] | bad[1:]
    cuts = np.flatnonzero(is_gap) + 1
    starts = np.concatenate(([0], cuts))
    stops = np.concatenate((cuts, [T]))
    return np.stack((starts, stops), axis=1).astype(np.int64)


def make_windows(
    long_or_wide: pd.DataFrame,
    signals: Sequence[str],
    L: int,
    stride: int,
    fs: float,
    *,
    max_gap_s: float | None = None,
) -> Windows:
    """Slice ``long_or_wide`` into ``(L, len(signals))`` windows, ``stride`` samples apart.

    Parameters
    ----------
    long_or_wide
        Either nebulax long telemetry (columns ``timestamp, signal, value`` plus at least
        ``train_id, component_id``) or a wide frame (``timestamp`` plus one column per name
        in ``signals``, plus at least ``train_id, component_id``). ``run_id``, ``car``,
        ``source`` and ``subsystem`` are used as extra grouping keys and carried through
        whenever present.
    signals
        Channel names, in the order they should appear in ``X``'s last axis. A channel
        missing for a given group is filled with NaN for that group rather than dropping it,
        so windows from different components stay comparable.
    L
        Window length in samples.
    stride
        Hop size in samples between consecutive window starts (``stride <= L`` overlaps,
        ``stride == L`` tiles, ``stride > L`` skips samples).
    fs
        Nominal sample rate in Hz, stored on the result for downstream feature code
        (:mod:`nebulax.features.vibration` needs it); not used to resample here. It also sets
        the default ``max_gap_s``.
    max_gap_s
        Largest sample-to-sample time step, in seconds, that may sit *inside* a window.
        Defaults to ``3 / fs`` - three nominal sample periods, loose enough to absorb jitter
        and the odd dropped sample, tight enough to catch a real logger outage. Anything
        larger starts a new window run, so no window ever bridges an acquisition gap (see the
        module docstring: MetroPT-3 has 190 gaps > 30 min). Pass ``float("inf")`` for the old
        slice-by-row-position behaviour. Rows with a NaT/non-finite timestamp always split.

    Returns
    -------
    Windows
        ``n = sum over groups, over gap-free segments of that group, of
        max(0, (T_seg - L) // stride + 1)`` windows; 0 if every segment is shorter than ``L``.
    """
    if L <= 0 or stride <= 0:
        raise ValueError(f"make_windows: L and stride must be positive ints, got L={L}, stride={stride}")
    if fs <= 0:
        raise ValueError(f"make_windows: fs must be > 0, got {fs}")
    gap_s = (3.0 / float(fs)) if max_gap_s is None else float(max_gap_s)
    if not (gap_s > 0):
        raise ValueError(f"make_windows: max_gap_s must be > 0, got {max_gap_s}")
    sig_tuple = tuple(signals)
    if not sig_tuple:
        raise ValueError("make_windows: signals must be non-empty")
    if "timestamp" not in long_or_wide.columns:
        raise ValueError("make_windows: frame must carry a 'timestamp' column")
    if "train_id" not in long_or_wide.columns or "component_id" not in long_or_wide.columns:
        raise ValueError("make_windows: frame must carry 'train_id' and 'component_id' columns")

    is_long = _is_long(long_or_wide)
    extra_keys = [c for c in _OPTIONAL_KEYS if c in long_or_wide.columns]
    group_cols = ["train_id", "component_id", *extra_keys]

    Xs: list[np.ndarray] = []
    t_ends: list[np.ndarray] = []
    key_lists: dict[str, list[np.ndarray]] = {k: [] for k in group_cols}

    for key_vals, g in long_or_wide.groupby(group_cols, sort=False, observed=True):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        if is_long:
            wide = g.pivot_table(index="timestamp", columns="signal", values="value", aggfunc="first")
            wide = wide.reindex(columns=sig_tuple)
        else:
            wide = g.set_index("timestamp")
            for s in sig_tuple:
                if s not in wide.columns:
                    wide[s] = np.float32("nan")
            wide = wide[list(sig_tuple)]
        wide = wide.sort_index(kind="stable")
        arr = wide.to_numpy(dtype=np.float32, copy=True)
        if arr.shape[0] < L:
            continue
        ts = _sorted_index_values(wide.index)

        # Cut the group at every acquisition gap first, then window each gap-free segment on
        # its own: a window that straddled an outage would be contiguous in row position but
        # not in time.
        for seg_start, seg_stop in _segment_bounds(wide.index, gap_s):
            seg = arr[seg_start:seg_stop]
            if seg.shape[0] < L:
                continue
            # (T, C) -> (T - L + 1, C, L) sliding windows, then keep every `stride`-th one and
            # move the window axis to sit between n and c: (n_win, L, C).
            windows = np.lib.stride_tricks.sliding_window_view(seg, L, axis=0)[::stride]
            windows = np.ascontiguousarray(np.moveaxis(windows, 1, 2))
            n_win = windows.shape[0]

            end_idx = seg_start + np.arange(L - 1, L - 1 + n_win * stride, stride)
            t_ends.append(ts[end_idx])
            Xs.append(windows.astype(np.float32, copy=False))
            for col, val in zip(group_cols, key_vals):
                key_lists[col].append(np.full(n_win, val, dtype=object))

    if not Xs:
        empty_keys = {k: np.empty(0, dtype=object) for k in group_cols}
        return Windows(
            X=np.empty((0, L, len(sig_tuple)), dtype=np.float32),
            t_end=np.empty(0, dtype="datetime64[ms]"),
            train_id=empty_keys["train_id"],
            component_id=empty_keys["component_id"],
            signals=sig_tuple,
            fs=float(fs),
            L=int(L),
            stride=int(stride),
            max_gap_s=gap_s,
            run_id=empty_keys.get("run_id"),
            car=empty_keys.get("car"),
            source=empty_keys.get("source"),
            subsystem=empty_keys.get("subsystem"),
        )

    X = np.concatenate(Xs, axis=0)
    t_end = np.concatenate(t_ends, axis=0)
    keys = {k: np.concatenate(v) for k, v in key_lists.items()}
    return Windows(
        X=X,
        t_end=t_end,
        train_id=keys["train_id"],
        component_id=keys["component_id"],
        signals=sig_tuple,
        fs=float(fs),
        L=int(L),
        stride=int(stride),
        max_gap_s=gap_s,
        run_id=keys.get("run_id"),
        car=keys.get("car"),
        source=keys.get("source"),
        subsystem=keys.get("subsystem"),
    )
