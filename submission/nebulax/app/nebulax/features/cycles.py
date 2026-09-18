"""Join per-cycle/window feature tables with the fault log, and normalise features against
sibling components ("peers") - the two pieces of glue between raw feature extraction
(:mod:`nebulax.features.stats`/:mod:`nebulax.features.vibration`) and the frozen feature-table
spec (:data:`nebulax.schema.FEATURE_KEY_COLUMNS` + :data:`nebulax.schema.LABEL_COLUMNS`).

Label reconstruction is deliberately schema-driven rather than simulator-specific: it reads
only ``t_onset, t_failure, gamma, shape, fault_type`` off :data:`nebulax.schema.FAULT_LOG_COLUMNS`,
so it works unchanged on simulator output, on the Cranfield ordinal-severity fault log, and on
the MetroPT hard-coded failure windows, as long as they all speak the one fault-log schema.
"""

from __future__ import annotations

from typing import Final, Sequence

import numpy as np
import pandas as pd

from nebulax.schema import FEATURE_KEY_DTYPES, LABEL_COLUMNS, coerce_features

__all__ = ["label_from_fault_log", "peer_normalise", "DEFAULT_ALARM_WINDOW_S"]

#: Default "alarm_window_3d" horizon, in seconds. The name is historical (the plan's 3-day
#: pre-failure alarm window); pass ``alarm_window_s`` to use a different one.
DEFAULT_ALARM_WINDOW_S: Final[float] = 3.0 * 86_400.0


def _to_ns(s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """A timestamp column as ``(int64 ns-since-epoch, is_nat mask)``, tz-naive-safe.

    The mask is returned, not folded into the values, because ``NaT`` casts to the int64
    sentinel ``-2**63``: arithmetic on it silently overflows into a finite-looking number
    (that is exactly how an unknown ``t_failure`` used to produce ``rul_s = 0`` and a true
    3-day alarm on every row of a statically labelled dataset). Every caller must consult the
    mask before using the values.
    """
    dt = pd.DatetimeIndex(pd.to_datetime(s, utc=True))
    naive = dt.tz_localize(None)
    return naive.to_numpy(dtype="datetime64[ns]").view("int64"), np.asarray(naive.isna())


def label_from_fault_log(
    features: pd.DataFrame,
    fault_log: pd.DataFrame,
    *,
    alarm_window_s: float = DEFAULT_ALARM_WINDOW_S,
) -> pd.DataFrame:
    """Attach ``fault_type, severity, rul_s, is_faulty, alarm_window_3d`` to ``features``.

    For each feature row, keyed on ``(run_id, subsystem, component_id)``, the *latest*
    fault-log row of that key whose ``t_onset <= t_end`` governs the label (later injected
    faults supersede earlier ones on the same component). Severity is recomputed from
    ``t_onset, t_failure, gamma, shape`` with the same law as
    :meth:`nebulax.sim.common.DegradationTrajectory.severity` -
    ``((t_end - t_onset) / (t_failure - t_onset)) ** gamma`` for ``shape="power"``, ``1.0``
    once ``t_end >= t_onset`` for ``shape="step"``. ``shape="shock"`` has no closed form from
    four scalar columns alone (the true law needs the sampled shock series), so it falls back
    to the same power-law with ``gamma`` as read from the log - documented here as an
    approximation, not a re-derivation of the shock trajectory.

    Rows with no covering fault (including an entirely empty ``fault_log``) get
    ``fault_type="healthy"``, ``severity=0``, ``is_faulty=False``, ``rul_s=NaN`` (undefined -
    there is no failure to count down to) and ``alarm_window_3d=False``.
    ``alarm_window_3d`` is ``True`` exactly when ``is_faulty`` and ``rul_s <= alarm_window_s``.

    **Unknown timestamps are unknown, not zero.** A statically labelled dataset has no
    run-to-failure axis at all: :mod:`nebulax.adapters.cranfield` writes one ``shape="step"``
    row per recording with ``t_onset`` = the first sample and ``t_failure=NaT``, because the
    bench fixture holds its seeded condition for the whole test and never fails. Such a row
    gets ``rul_s=NaN`` and ``alarm_window_3d=False`` - never a finite RUL. ``severity`` still
    follows the shape rule where the rule does not need the failure time (``"step"`` -> 1.0
    from ``t_onset`` on) and is ``NaN`` where it does (``"power"``/``"shock"``, whose ``u`` has
    no denominator without ``t_failure``); ``is_faulty`` is ``True`` in both cases, since the
    log says the component *is* in that condition. Fault-log rows whose ``t_onset`` is ``NaT``
    are unusable - there is no time at which they start - and are ignored entirely; feature
    rows with a ``NaT`` ``t_end`` cannot be placed on any fault's timeline and stay healthy.
    """
    out = features.reset_index(drop=True).copy()
    n = len(out)

    fault_type = np.full(n, "healthy", dtype=object)
    severity = np.zeros(n, dtype=np.float64)
    rul_s = np.full(n, np.nan, dtype=np.float64)
    is_faulty = np.zeros(n, dtype=bool)

    if n and len(fault_log):
        req_fl = {"run_id", "subsystem", "component_id", "t_onset", "t_failure", "gamma", "shape", "fault_type"}
        missing_fl = req_fl - set(fault_log.columns)
        if missing_fl:
            raise ValueError(f"label_from_fault_log: fault_log missing column(s) {sorted(missing_fl)}")
        req_f = {"run_id", "subsystem", "component_id", "t_end"}
        missing_f = req_f - set(out.columns)
        if missing_f:
            raise ValueError(f"label_from_fault_log: features missing column(s) {sorted(missing_f)}")

        t_end_ns, t_end_nat = _to_ns(out["t_end"])
        f_key = np.asarray(
            list(zip(out["run_id"].astype(str), out["subsystem"].astype(str), out["component_id"].astype(str))),
            dtype=object,
        )

        fl = fault_log.reset_index(drop=True)
        fl_onset_ns, fl_onset_nat = _to_ns(fl["t_onset"])
        fl_fail_ns, fl_fail_nat = _to_ns(fl["t_failure"])
        fl_gamma = fl["gamma"].to_numpy(dtype=np.float64)
        fl_shape = fl["shape"].astype(str).to_numpy()
        fl_type = fl["fault_type"].astype(str).to_numpy()
        fl_key = np.asarray(
            list(zip(fl["run_id"].astype(str), fl["subsystem"].astype(str), fl["component_id"].astype(str))),
            dtype=object,
        )

        fl_groups: dict[tuple, np.ndarray] = {}
        for k in {tuple(row) for row in fl_key}:
            fl_groups[k] = np.flatnonzero(np.all(fl_key == np.array(k, dtype=object), axis=1))

        for k in {tuple(row) for row in f_key}:
            fidx = fl_groups.get(k)
            if fidx is None:
                continue
            # A fault with no onset has no timeline to sit on: drop it rather than let the
            # int64 NaT sentinel sort to -2**63 and "cover" every row.
            fidx = fidx[~fl_onset_nat[fidx]]
            if fidx.size == 0:
                continue
            order = np.argsort(fl_onset_ns[fidx], kind="stable")
            fidx = fidx[order]
            onset = fl_onset_ns[fidx]
            fail = fl_fail_ns[fidx]
            fail_unknown = fl_fail_nat[fidx]
            gamma = fl_gamma[fidx]
            shape = fl_shape[fidx]
            ftype = fl_type[fidx]

            rows = np.flatnonzero(np.all(f_key == np.array(k, dtype=object), axis=1))
            rows = rows[~t_end_nat[rows]]
            if rows.size == 0:
                continue
            te = t_end_ns[rows]
            pos = np.searchsorted(onset, te, side="right") - 1
            active = pos >= 0
            if not active.any():
                continue

            r = rows[active]
            p = pos[active]
            o = onset[p]
            fa = fail[p]
            fa_unknown = fail_unknown[p]
            g = gamma[p]
            sh = shape[p]
            ft = ftype[p]
            te_a = te[active]

            # Never do arithmetic on the NaT sentinel: substitute a dummy span, then mask the
            # results it fed back out to NaN.
            fa_safe = np.where(fa_unknown, o + 1, fa)
            span_ns = np.maximum(fa_safe - o, 1)
            u = np.clip((te_a - o) / span_ns, 0.0, 1.0)
            is_step = sh == "step"
            sev = np.clip(np.where(is_step, (te_a >= o).astype(np.float64), u**g), 0.0, 1.0)
            # Only the step law is closed-form without a failure time; the power/shock law is
            # not, so its severity is unknown rather than 1.0.
            sev = np.where(fa_unknown & ~is_step, np.nan, sev)

            # `pos >= 0` already guarantees `te_a >= o`, so an unknown-severity row is a row
            # whose fault is active with an unquantified magnitude - faulty, not healthy.
            active_fault = (sev > 0.0) | np.isnan(sev)
            severity[r] = sev
            is_faulty[r] = active_fault
            fault_type[r] = np.where(active_fault, ft, "healthy")
            rul = np.where(fa_unknown, np.nan, np.maximum((fa_safe - te_a) / 1.0e9, 0.0))
            rul_s[r] = np.where(active_fault, rul, np.nan)

    alarm = is_faulty & np.isfinite(rul_s) & (rul_s <= alarm_window_s)

    out["fault_type"] = pd.Series(fault_type).astype("string").astype("category")
    out["severity"] = severity.astype(np.float32)
    out["rul_s"] = rul_s.astype(np.float32)
    out["is_faulty"] = is_faulty
    out["alarm_window_3d"] = alarm
    return coerce_features(out)


def peer_normalise(
    features: pd.DataFrame,
    group_cols: Sequence[str],
    *,
    value_cols: Sequence[str] | None = None,
    same_side: bool = False,
    component_col: str = "component_id",
    min_peers: int = 1,
    delta_suffix: str = "_peer_delta",
    z_suffix: str = "_peer_z",
) -> pd.DataFrame:
    """Leave-one-out peer comparison: for each row, compare its features against the *other*
    rows sharing ``group_cols`` (e.g. ``["run_id", "car"]`` for boxes on the same bogie at the
    same window, or ``["run_id", "cycle_id"]`` for sibling doors at the same dwell).

    ``same_side=True`` (per [rail_phm 4.3.4]: bearing peers must be the *same-side* boxes, not
    all eight on a car - a motor-car bearing runs measurably hotter than its opposite-side
    twin) adds one more implicit grouping key: the trailing ``L``/``R`` letter parsed out of
    ``component_col`` (``"axlebox_2R"`` -> ``"R"``; ``"door_L3"`` -> ``"L"``).

    Adds two columns per value column: ``<col>_peer_delta`` (``x - mean(others)``) and
    ``<col>_peer_z`` (``<col>_peer_delta / std(others)``, NaN where the peer std is 0).
    Rows whose group has fewer than ``min_peers`` *other* members get NaN in both new columns
    (not enough peers to compare against). ``value_cols`` defaults to every numeric column
    that is not a feature-key or label column.
    """
    out = features.copy()
    cols = list(group_cols)
    side_col = None
    if same_side:
        side_col = "_peer_side"
        out[side_col] = out[component_col].astype(str).str.extract(r"([LR])\d*$", expand=False)
        cols = [*cols, side_col]

    if value_cols is None:
        exclude = set(FEATURE_KEY_DTYPES) | set(LABEL_COLUMNS) | {side_col}
        value_cols = [c for c in out.columns if c not in exclude and pd.api.types.is_numeric_dtype(out[c])]
    value_cols = list(value_cols)
    if not value_cols:
        raise ValueError("peer_normalise: no numeric feature columns found to normalise")
    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise ValueError(f"peer_normalise: group column(s) {missing} not found in features")

    grp = out.groupby(cols, observed=True, dropna=False)
    n = grp[value_cols[0]].transform("size").to_numpy(dtype=np.float64)
    loo_n = np.maximum(n - 1.0, 0.0)
    enough = loo_n >= min_peers

    for col in value_cols:
        x = out[col].to_numpy(dtype=np.float64)
        aux = out[[*cols]].copy()
        aux["_x"] = x
        aux["_x2"] = x * x
        g_aux = aux.groupby(cols, observed=True, dropna=False)
        s1 = g_aux["_x"].transform("sum").to_numpy(dtype=np.float64)
        s2 = g_aux["_x2"].transform("sum").to_numpy(dtype=np.float64)

        with np.errstate(divide="ignore", invalid="ignore"):
            loo_mean = np.where(loo_n > 0, (s1 - x) / np.where(loo_n == 0, np.nan, loo_n), np.nan)
            loo_var = (s2 - x * x) / np.where(loo_n == 0, np.nan, loo_n) - loo_mean**2
        loo_var = np.clip(loo_var, 0.0, None)
        loo_std = np.sqrt(loo_var)
        delta = x - loo_mean
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(loo_std > 0, delta / np.where(loo_std == 0, np.nan, loo_std), np.nan)

        out[f"{col}{delta_suffix}"] = np.where(enough, delta, np.nan).astype(np.float32)
        out[f"{col}{z_suffix}"] = np.where(enough, z, np.nan).astype(np.float32)

    if side_col is not None:
        out = out.drop(columns=[side_col])
    return out
