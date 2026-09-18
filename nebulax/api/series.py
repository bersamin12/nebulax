"""Telemetry readers for the chart routes.

Two rules, both from ``docs/app_contract.md`` section 4:

* never ``read_parquet`` a whole telemetry file - one 30-day door run is 64 MB / 23 M rows,
  so every read goes through ``pyarrow.parquet.read_table(path, filters=[...])`` on
  ``component_id``/``signal``/``timestamp`` (and ``car`` where it disambiguates);
* the browser gets at most ``max_points`` points, picked by **min/max bucketing** so a spike
  that lives in one sample survives the downsample instead of being averaged away.

The train-context channels (``speed``, ``load_frac``, ``T_amb``, ``in_service``) are stored
once per run under ``component_id="train"``, ``car=0`` - not under the door leaf or the axle
box. Asking a component for one of them therefore reads the *train* rows of that component's
run, and the answer says so (``source_component: "train"``).

Door waveforms are not resampled at all: ``/cycle`` returns one *stored* cycle verbatim
(the simulator keeps every ``store_every``-th cycle plus everything in the two days before a
failure), found by splitting the filtered rows on their inter-cycle gaps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from nebulax import schema as S
from nebulax.api.state import MP3_TRAIN_ID, FleetState, iso, parse_ts

__all__ = [
    "DOOR_CYCLE_SIGNALS",
    "signal_unit",
    "available_signals",
    "read_series",
    "read_cycle",
    "downsample_minmax",
]

#: The waveform channels ``/cycle`` returns, in contract order (``vel`` is a bonus channel).
DOOR_CYCLE_SIGNALS: tuple[str, ...] = ("pos", "pos_ref", "current", "pwm", "vel")

#: Gap between two samples that means "a new door cycle starts here". Inside one cycle the only
#: gap is the dwell between the opening and closing waveforms (25-45 s by ``ServiceParams``);
#: between two cycles there is at least one run segment (90-150 s). 60 s sits between the two,
#: so opening + closing stay one cycle and two cycles never merge.
CYCLE_GAP_S = 60.0

#: How far around ``ts`` ``/cycle`` looks for a stored cycle, widening until it finds one.
_CYCLE_WINDOWS_S: tuple[float, ...] = (3600.0, 6 * 3600.0, 24 * 3600.0, 7 * 86400.0, 40 * 86400.0)

_UNIT_FALLBACK: dict[str, str] = {"speed": "m/s", "load_frac": "frac", "T_amb": "degC", "in_service": "bool"}


def signal_unit(subsystem: str, signal: str) -> str:
    """Engineering unit of one channel, from :data:`nebulax.schema.SIGNAL_SPECS`."""
    spec = S.SIGNAL_SPECS.get((subsystem, signal))
    if spec is not None:
        return spec.unit
    for (sub, name), sp in S.SIGNAL_SPECS.items():  # same name on another subsystem
        if name == signal:
            return sp.unit
    return _UNIT_FALLBACK.get(signal, "")


def available_signals(subsystem: str) -> list[str]:
    """Channel names of a subsystem (plus the train-context channels)."""
    return list(S.SIGNALS.get(subsystem, ()))


def run_dir(state: FleetState, run: dict[str, Any]) -> Path:
    """Partition directory of an ``index.json`` run entry."""
    rel = run.get("path") or f"source=sim/run_id={run.get('run_id')}"
    return state.settings.sim_dir / rel


def _filters(
    component_id: str,
    signals: Sequence[str],
    t_from: pd.Timestamp | None,
    t_to: pd.Timestamp | None,
    car: int | None,
) -> list[tuple[str, str, Any]]:
    f: list[tuple[str, str, Any]] = [("component_id", "==", component_id)]
    if len(signals) == 1:
        f.append(("signal", "==", signals[0]))
    elif signals:
        f.append(("signal", "in", list(signals)))
    if car is not None:
        f.append(("car", "==", np.int8(car)))
    if t_from is not None:
        f.append(("timestamp", ">=", t_from.to_pydatetime()))
    if t_to is not None:
        f.append(("timestamp", "<=", t_to.to_pydatetime()))
    return f


def _read_long(
    path: Path,
    component_id: str,
    signals: Sequence[str],
    *,
    t_from: pd.Timestamp | None = None,
    t_to: pd.Timestamp | None = None,
    car: int | None = None,
) -> pd.DataFrame:
    """``timestamp, signal, value`` rows of one component, pushed down to the parquet reader."""
    import pyarrow.parquet as pq

    table = pq.read_table(
        path,
        columns=["timestamp", "signal", "value"],
        filters=_filters(component_id, signals, t_from, t_to, car),
    )
    df = table.to_pandas()
    if len(df):
        df["signal"] = df["signal"].astype(str)
        df = df.sort_values(["signal", "timestamp"], kind="stable")
    return df


def _overlay_long(
    state: FleetState,
    train_id: str,
    component_id: str,
    car: int | None,
) -> pd.DataFrame | None:
    """The injected telemetry of this component, if ``POST /sim/inject`` produced any."""
    for key, df in state.overlay.long.items():
        if key[0] != train_id:
            continue
        if car is not None and int(key[1]) != int(car) and component_id != S.TRAIN_COMPONENT_ID:
            continue
        if component_id in (key[3], S.TRAIN_COMPONENT_ID):
            sub = df[df["component_id"].astype(str) == component_id]
            if len(sub):
                return sub
    return None


def downsample_minmax(ts_ns: np.ndarray, values: np.ndarray, max_points: int) -> list[list[Any]]:
    """Bucket ``(ts, value)`` into ``max_points//2`` equal-count buckets and keep each bucket's
    min and max in time order, so peaks survive. Returns ``[[iso_ts, value], ...]``."""
    n = len(ts_ns)
    if max_points <= 0:
        raise ValueError(f"downsample_minmax: max_points must be >= 1, got {max_points}")
    if n == 0:
        return []
    if n <= max_points:
        keep = np.arange(n)
    else:
        n_buckets = max(int(max_points) // 2, 1)
        edges = np.linspace(0, n, n_buckets + 1).astype(int)
        picks: list[int] = []
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            chunk = values[a:b]
            finite = np.isfinite(chunk)
            if not finite.any():
                picks.append(a)
                continue
            lo = a + int(np.nanargmin(np.where(finite, chunk, np.inf)))
            hi = a + int(np.nanargmax(np.where(finite, chunk, -np.inf)))
            picks.extend((lo, hi) if lo <= hi else (hi, lo))
        keep = np.unique(np.asarray(picks, dtype=int)) if picks else np.arange(min(n, max_points))
    out: list[list[Any]] = []
    for i in keep:
        v = float(values[i])
        out.append([iso(pd.Timestamp(int(ts_ns[i]), unit="ns", tz="UTC")), None if not np.isfinite(v) else v])
    return out


#: The MetroPT-3 CSV, relative to the data root; the adapter owns the column names.
MP3_CSV = Path("raw") / "metropt3" / "MetroPT3(AirCompressor).csv"


def read_mp3_channel(state: FleetState, signal: str) -> tuple[np.ndarray, np.ndarray]:
    """One MetroPT-3 channel as ``(epoch_ns, values)``, cached on the state after first read.

    MetroPT-3 ships as one 218 MB CSV with no parquet mirror in this repo, so the first request
    for a channel costs ~2 s (two columns only) and every later one is free. Nothing is written
    to disk: ``data/`` belongs to the dataset, not to the API.
    """
    if signal in state.mp3_cache:
        return state.mp3_cache[signal]
    if signal not in S.METROPT3_SIGNALS:
        raise ValueError(
            f"series: {signal!r} is not a MetroPT-3 channel; expected one of {list(S.METROPT3_SIGNALS)}"
        )
    path = state.settings.data_dir / MP3_CSV
    if not path.exists():
        raise FileNotFoundError(f"series: {path} does not exist (run scripts/download_data.py --dataset metropt3)")
    raw = pd.read_csv(path, usecols=["timestamp", signal])
    ts = pd.to_datetime(raw["timestamp"], utc=True)
    out = (
        ts.astype("datetime64[ns, UTC]").astype("int64").to_numpy(),
        raw[signal].to_numpy(dtype=np.float64),
    )
    state.mp3_cache[signal] = out
    return out


def read_series(
    state: FleetState,
    train_id: str,
    component_id: str,
    signal: str,
    *,
    t_from: Any = None,
    t_to: Any = None,
    car: int | None = None,
    max_points: int = 2000,
) -> dict[str, Any]:
    """``{signal, unit, points}`` for one channel of one component.

    Raises ``FileNotFoundError`` when the train/component has no telemetry partition and
    ``ValueError`` when the channel is not one of that subsystem's signals.
    """
    lo = None if t_from in (None, "") else parse_ts(t_from, what="from")
    hi = None if t_to in (None, "") else parse_ts(t_to, what="to")
    subsystem = _subsystem_of(component_id) or "train"
    if train_id == MP3_TRAIN_ID:
        ts_ns, vals = read_mp3_channel(state, signal)
        a = 0 if lo is None else int(np.searchsorted(ts_ns, int(lo.value), side="left"))
        b = len(ts_ns) if hi is None else int(np.searchsorted(ts_ns, int(hi.value), side="right"))
        return {
            "signal": signal,
            "unit": signal_unit("pneumatic", signal),
            "train_id": train_id,
            "component_id": component_id,
            "source_component": component_id,
            "car": 0,
            "n_raw": int(b - a),
            "points": downsample_minmax(ts_ns[a:b], vals[a:b], max_points),
        }
    if (subsystem, signal) not in S.SIGNAL_SPECS and signal not in S.SIGNALS.get("train", ()):
        raise ValueError(
            f"series: signal {signal!r} is not a {subsystem} channel; expected one of "
            f"{list(S.SIGNALS.get(subsystem, ()))}"
        )

    # Train-context channels live on their own rows (``component_id="train"``, ``car=0``), so
    # a request for one of them against a door leaf or an axle box reads the run's train rows.
    is_train_signal = signal in S.SIGNALS.get("train", ()) and component_id != S.TRAIN_COMPONENT_ID
    read_component = S.TRAIN_COMPONENT_ID if is_train_signal else component_id
    read_car = None if is_train_signal else car

    df = _overlay_long(state, train_id, read_component, read_car)
    if df is not None:
        sub = df[df["signal"].astype(str) == signal]
        if lo is not None:
            sub = sub[sub["timestamp"] >= lo]
        if hi is not None:
            sub = sub[sub["timestamp"] <= hi]
        sub = sub.sort_values("timestamp", kind="stable")
        ts_ns = sub["timestamp"].dt.tz_convert("UTC").astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        vals = sub["value"].to_numpy(dtype=np.float64)
    else:
        run = state.run_for(train_id, component_id, car)
        if run is None:
            raise FileNotFoundError(
                f"series: no telemetry run for train {train_id!r} component {component_id!r}"
                + ("" if car is None else f" car {car}")
            )
        path = run_dir(state, run) / "telemetry.parquet"
        if not path.exists():
            raise FileNotFoundError(f"series: {path} does not exist")
        raw = _read_long(path, read_component, [signal], t_from=lo, t_to=hi, car=read_car)
        ts_ns = (
            raw["timestamp"].dt.tz_convert("UTC").astype("datetime64[ns, UTC]").astype("int64").to_numpy()
            if len(raw)
            else np.empty(0, dtype=np.int64)
        )
        vals = raw["value"].to_numpy(dtype=np.float64) if len(raw) else np.empty(0, dtype=np.float64)

    return {
        "signal": signal,
        "unit": signal_unit("train" if is_train_signal else subsystem, signal),
        "train_id": train_id,
        "component_id": component_id,
        "source_component": read_component,
        "car": 0 if is_train_signal else car,
        "n_raw": int(len(ts_ns)),
        "points": downsample_minmax(ts_ns, vals, max_points),
    }


def _subsystem_of(component_id: str) -> str | None:
    for sub, cids in S.COMPONENT_IDS.items():
        if component_id in cids:
            return sub
    return None


def _split_cycles(ts_ns: np.ndarray) -> list[tuple[int, int]]:
    """Index spans of contiguous samples, cut wherever the gap exceeds :data:`CYCLE_GAP_S`."""
    if len(ts_ns) == 0:
        return []
    gaps = np.flatnonzero(np.diff(ts_ns) > CYCLE_GAP_S * 1e9)
    starts = np.concatenate(([0], gaps + 1))
    ends = np.concatenate((gaps + 1, [len(ts_ns)]))
    return list(zip(starts.tolist(), ends.tolist()))


def read_cycle(
    state: FleetState,
    train_id: str,
    component_id: str,
    ts: Any,
    *,
    car: int | None = None,
) -> dict[str, Any]:
    """The stored door cycle nearest ``ts``: ``{t, pos, pos_ref, current, pwm, vel}``.

    ``t`` is relative seconds from the cycle's first sample. Raises ``ValueError`` for a
    non-door component and ``FileNotFoundError`` when no stored cycle can be found at all.
    """
    if component_id not in S.DOOR_COMPONENT_IDS:
        raise ValueError(f"cycle: {component_id!r} is not a door leaf; door cycles exist only for doors")
    at = parse_ts(ts, what="ts")

    overlay = _overlay_long(state, train_id, component_id, car)
    path: Path | None = None
    if overlay is None:
        run = state.run_for(train_id, component_id, car)
        if run is None:
            raise FileNotFoundError(f"cycle: no telemetry run for train {train_id!r} component {component_id!r}")
        path = run_dir(state, run) / "telemetry.parquet"
        if not path.exists():
            raise FileNotFoundError(f"cycle: {path} does not exist")

    wide: pd.DataFrame | None = None
    lo = hi = at
    for half in _CYCLE_WINDOWS_S:
        lo, hi = at - pd.Timedelta(seconds=half), at + pd.Timedelta(seconds=half)
        if overlay is not None:
            df = overlay[
                overlay["signal"].astype(str).isin(DOOR_CYCLE_SIGNALS)
                & (overlay["timestamp"] >= lo)
                & (overlay["timestamp"] <= hi)
            ]
        else:
            assert path is not None
            df = _read_long(path, component_id, DOOR_CYCLE_SIGNALS, t_from=lo, t_to=hi, car=car)
        if len(df):
            wide = df.pivot_table(index="timestamp", columns="signal", values="value", aggfunc="last")
            wide = wide.sort_index()
            break
    if wide is None or not len(wide):
        raise FileNotFoundError(
            f"cycle: no stored waveform for {train_id}/{component_id} anywhere near {iso(at)}"
        )

    ts_ns = wide.index.tz_convert("UTC").astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    spans = _split_cycles(ts_ns)
    # A span that runs right up to an edge of the read window may be a cycle the window cut in
    # half, so prefer a whole one whenever the read returned any. A span that merely happens to
    # be first or last, with clear air between it and the window edge, is whole.
    edge_ns = int(CYCLE_GAP_S * 1e9)
    cut_low = int(ts_ns[0]) - int(lo.value) < edge_ns
    cut_high = int(hi.value) - int(ts_ns[-1]) < edge_ns
    whole = [
        s
        for n, s in enumerate(spans)
        if not (n == 0 and cut_low) and not (n == len(spans) - 1 and cut_high)
    ]
    candidates = whole or spans
    at_ns = int(at.value)
    a, b = min(candidates, key=lambda s: abs(int((ts_ns[s[0]] + ts_ns[s[1] - 1]) // 2) - at_ns))
    span_ts = ts_ns[a:b]
    out: dict[str, Any] = {
        "train_id": train_id,
        "component_id": component_id,
        "car": car,
        "t_start": iso(pd.Timestamp(int(span_ts[0]), unit="ns", tz="UTC")),
        "t_end": iso(pd.Timestamp(int(span_ts[-1]), unit="ns", tz="UTC")),
        "n": int(b - a),
        "t": [round((int(v) - int(span_ts[0])) / 1e9, 4) for v in span_ts],
    }
    for sig in DOOR_CYCLE_SIGNALS:
        if sig in wide.columns:
            col = wide[sig].to_numpy(dtype=np.float64)[a:b]
            out[sig] = [None if not np.isfinite(v) else float(v) for v in col]
        else:
            out[sig] = []
    return out


def read_events(
    state: FleetState,
    train_id: str,
    component_id: str,
    *,
    t_from: Any = None,
    t_to: Any = None,
    car: int | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Event-log rows of one component in a time window (for the advisory context).

    Returns ``[]`` rather than raising when the run or the file is missing: an advisory with
    no recent events is still a useful advisory.
    """
    run = state.run_for(train_id, component_id, car)
    if run is None:
        return []
    path = run_dir(state, run) / "events.parquet"
    if not path.exists():
        return []
    import pyarrow.parquet as pq

    filters: list[tuple[str, str, Any]] = [("component_id", "==", component_id)]
    if t_from not in (None, ""):
        filters.append(("timestamp", ">=", parse_ts(t_from).to_pydatetime()))
    if t_to not in (None, ""):
        filters.append(("timestamp", "<=", parse_ts(t_to).to_pydatetime()))
    try:
        df = pq.read_table(
            path, columns=["timestamp", "event", "detail_json", "component_id"], filters=filters
        ).to_pandas()
    except Exception:  # pragma: no cover - a malformed partition must not break the advisory
        return []
    if not len(df):
        return []
    df = df.sort_values("timestamp", kind="stable").tail(int(limit))
    return [
        {"timestamp": iso(row["timestamp"]), "event": str(row["event"]), "detail_json": row.get("detail_json")}
        for row in df.to_dict("records")
    ]
