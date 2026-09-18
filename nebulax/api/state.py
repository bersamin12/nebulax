"""Fleet state: everything the API serves that is small enough to hold in memory.

``FleetState`` is built once at startup from ``data/scores/*``, ``data/sim/index.json`` and
``data/sim/fault_log.parquet``. The scores frame is exploded into one small struct of numpy
arrays per *component* (``ComponentSeries``), so answering "what is this component doing at
``ts``" is a ``searchsorted`` rather than a pandas filter - the replay WebSocket asks that
question for ~190 components ten times a second.

Nothing here reads telemetry: the 64 MB-per-run parquet files are touched only by
:mod:`nebulax.api.series`, always through pyarrow filters.

If ``data/scores`` does not exist the state still builds (``scores_loaded is False``); the app
boots, ``/api/health`` says so and frames come back with an empty ``trains`` list.
"""

from __future__ import annotations

import json
import os
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from nebulax import schema as S

__all__ = [
    "Settings",
    "ComponentSeries",
    "EpisodeRec",
    "Overlay",
    "FleetState",
    "ComponentKey",
    "parse_ts",
    "iso",
]

#: ``(train_id, car, subsystem, component_id)`` - the identity of one scored component.
ComponentKey = tuple[str, int, str, str]

#: Default sim clock, contract section 1.
DEFAULT_CLOCK_START = pd.Timestamp("2026-09-01T00:00:00Z")
DEFAULT_CLOCK_END = pd.Timestamp("2026-09-30T23:59:00Z")

#: The sim replay clock never runs past this, however far the last scored row reaches (the
#: bearing windows close a moment after midnight on the last day).
SIM_CLOCK_HARD_END = pd.Timestamp("2026-09-30T23:59:59Z")

#: The MetroPT-3 unit is one extra "train" with its own clock; it never enters the sim replay.
MP3_TRAIN_ID = "MP3"

#: Episode ids the injector mints carry this prefix, so the overlay's episodes are always
#: distinguishable from the scored ones (and their cached advisories droppable on DELETE).
INJECT_EPISODE_PREFIX = "inj-"

#: Short model labels for the ticker (contract section 5 example line).
MODEL_LABELS: dict[str, str] = {
    "cusum_cycle_scalar": "CUSUM",
    "sparse_autoencoder": "SAE",
    "lgbm_residual": "LGBM",
}

#: A scored row older than this at ``ts`` makes the component ``stale`` (contract section 5).
#: One window for every subsystem: doors are out of service for up to 5.1 h every night, so a
#: shorter door window went stale mid-episode each night (contract change of 17 Sep).
DEFAULT_STALE_S = 6 * 3600.0
STALE_S: dict[str, float] = {"door": DEFAULT_STALE_S, "pneumatic": DEFAULT_STALE_S, "bearing": DEFAULT_STALE_S}

#: Detection horizon used by the sim benchmark, for the episode/fault join.
DETECTION_HORIZON_H = 72.0

_NS = 1_000_000_000


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Settings:
    """Root paths and clock. Tests point ``data_dir``/``results_dir`` at a ``tmp_path``."""

    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    web_dist: Path = Path("web/dist")
    clock_start: pd.Timestamp = DEFAULT_CLOCK_START
    clock_end: pd.Timestamp = DEFAULT_CLOCK_END

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        """``NEBULAX_DATA_DIR`` (default ``data``), ``NEBULAX_RESULTS_DIR`` (default ``results``),
        ``NEBULAX_WEB_DIST`` (default ``web/dist``)."""
        e = os.environ if env is None else env
        data_dir = Path(e["NEBULAX_DATA_DIR"]) if "NEBULAX_DATA_DIR" in e else Path("data")
        if "NEBULAX_DATA_DIR" not in e and not (data_dir / "sim" / "index.json").is_file():
            demo_dir = Path("demo_data")
            if (demo_dir / "sim" / "index.json").is_file():
                data_dir = demo_dir
        clock_start = DEFAULT_CLOCK_START
        if (data_dir / "sim" / "index.json").is_file():
            index = json.loads((data_dir / "sim" / "index.json").read_text(encoding="utf-8"))
            if index.get("sample_window", {}).get("start"):
                clock_start = parse_ts(index["sample_window"]["start"])
        return cls(
            data_dir=data_dir,
            results_dir=Path(e.get("NEBULAX_RESULTS_DIR", "results")),
            web_dist=Path(e.get("NEBULAX_WEB_DIST", "web/dist")),
            clock_start=clock_start,
        )

    @property
    def scores_dir(self) -> Path:
        return self.data_dir / "scores"

    @property
    def sim_dir(self) -> Path:
        return self.data_dir / "sim"


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def parse_ts(value: Any, *, what: str = "ts") -> pd.Timestamp:
    """Parse an ISO-8601 string / datetime into a tz-aware UTC ``pd.Timestamp``.

    Naive input is *assumed* UTC (the whole project is UTC); anything unparseable raises
    ``ValueError`` with the offending value, which the routes turn into a 400.
    """
    try:
        ts = pd.Timestamp(value)
    except Exception as exc:  # pragma: no cover - pandas raises several types
        raise ValueError(f"{what}: {value!r} is not a timestamp ({exc})") from exc
    if ts is pd.NaT or pd.isna(ts):
        raise ValueError(f"{what}: {value!r} is not a timestamp")
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts


def iso(ts: Any) -> str | None:
    """ISO-8601 UTC string with a ``Z`` suffix, or ``None`` for NaT/None."""
    if ts is None:
        return None
    t = pd.Timestamp(ts)
    if t is pd.NaT or pd.isna(t):
        return None
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.isoformat().replace("+00:00", "Z")


def _ns(ts: Any) -> int:
    """Epoch nanoseconds of one timestamp."""
    return int(parse_ts(ts).value)


def _ns_array(series: pd.Series) -> np.ndarray:
    """Epoch-nanosecond int64 array from a tz-aware datetime series."""
    if len(series) == 0:
        return np.empty(0, dtype=np.int64)
    return series.dt.tz_convert("UTC").astype("datetime64[ns, UTC]").astype("int64").to_numpy()


def _f(value: Any) -> float | None:
    """JSON-safe float (NaN/inf -> ``None``)."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _s(value: Any) -> str | None:
    """JSON-safe string (None/NaN -> ``None``)."""
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def model_label(name: str) -> str:
    """Short ticker label of a model name (``cusum_cycle_scalar`` -> ``CUSUM``)."""
    if not name:
        return ""
    if name in MODEL_LABELS:
        return MODEL_LABELS[name]
    return "".join(part[0] for part in str(name).split("_") if part).upper()


def _g(value: float | None) -> str:
    """Three significant figures, the ticker's number format (``3.9e10`` -> ``3.91e+10``)."""
    return "?" if value is None else f"{float(value):.3g}"


def _subsystem_of(component_id: str) -> str | None:
    for sub, cids in S.COMPONENT_IDS.items():
        if component_id in cids:
            return sub
    return None


def _in_scope(train_id: str, scope: str | None) -> bool:
    """``scope=None`` means *the sim fleet* - MetroPT-3 is a unit of its own, on its own clock,
    and never counts towards the 2026 fleet (contract section 1)."""
    return train_id != MP3_TRAIN_ID if scope is None else train_id == scope


# --------------------------------------------------------------------------------------
# Per-component score arrays
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class ComponentSeries:
    """The scored rows of one component, as parallel arrays sorted by time."""

    train_id: str
    car: int
    subsystem: str
    component_id: str
    model: str
    ts: np.ndarray  # int64 epoch ns, ascending
    score: np.ndarray  # float64
    threshold: np.ndarray  # float64
    alert: np.ndarray  # bool
    top_raw: np.ndarray  # object array of the raw top_signals_json strings

    @property
    def key(self) -> ComponentKey:
        return (self.train_id, int(self.car), self.subsystem, self.component_id)

    def index_at(self, ts_ns: int) -> int:
        """Index of the last row with ``timestamp <= ts``, or ``-1``."""
        return int(np.searchsorted(self.ts, ts_ns, side="right")) - 1

    def slice_between(self, lo_ns: int | None, hi_ns: int | None) -> slice:
        a = 0 if lo_ns is None else int(np.searchsorted(self.ts, lo_ns, side="left"))
        b = len(self.ts) if hi_ns is None else int(np.searchsorted(self.ts, hi_ns, side="right"))
        return slice(a, b)

    def top_signals(self, i: int) -> list[dict[str, Any]]:
        """Parsed ``top_signals_json`` of row ``i`` (parsed on demand, never for every row)."""
        if i < 0 or i >= len(self.top_raw):
            return []
        raw = self.top_raw[i]
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return []
        if isinstance(raw, (list, tuple)):
            return list(raw)
        try:
            out = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        return out if isinstance(out, list) else []


@dataclass(slots=True)
class EpisodeRec:
    """One alarm episode, as stored in ``data/scores/episodes.parquet``."""

    episode_id: str
    train_id: str
    car: int
    subsystem: str
    component_id: str
    model: str
    t_start_ns: int
    t_end_ns: int | None
    peak_score: float | None
    n_rows: int
    fault_type: str | None
    t_onset_ns: int | None
    t_failure_ns: int | None
    lead_to_failure_h: float | None
    matched: bool

    @property
    def key(self) -> ComponentKey:
        return (self.train_id, int(self.car), self.subsystem, self.component_id)

    def open_at(self, ts_ns: int) -> bool:
        return self.t_start_ns <= ts_ns and (self.t_end_ns is None or self.t_end_ns > ts_ns)

    def started_by(self, ts_ns: int) -> bool:
        return self.t_start_ns <= ts_ns


@dataclass(slots=True)
class Overlay:
    """What ``POST /api/sim/inject`` put on top of the loaded data.

    Every GET route reads the overlay first: ``components`` shadow the scored arrays,
    ``episodes`` replace the base episodes of those same components, and ``long``/``features``
    shadow the on-disk telemetry for ``/series`` and ``/cycle``.
    """

    components: dict[ComponentKey, ComponentSeries] = field(default_factory=dict)
    episodes: dict[ComponentKey, list[EpisodeRec]] = field(default_factory=dict)
    long: dict[ComponentKey, pd.DataFrame] = field(default_factory=dict)
    features: dict[ComponentKey, pd.DataFrame] = field(default_factory=dict)
    fault_log: list[dict[str, Any]] = field(default_factory=list)
    info: list[dict[str, Any]] = field(default_factory=list)

    def clear(self) -> None:
        self.components.clear()
        self.episodes.clear()
        self.long.clear()
        self.features.clear()
        self.fault_log.clear()
        self.info.clear()

    def __bool__(self) -> bool:
        return bool(self.components or self.long or self.info)


# --------------------------------------------------------------------------------------
# FleetState
# --------------------------------------------------------------------------------------


class FleetState:
    """Everything the API serves, loaded once."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.components: dict[ComponentKey, ComponentSeries] = {}
        self.episodes: list[EpisodeRec] = []
        self.episodes_by_key: dict[ComponentKey, list[EpisodeRec]] = {}
        self.episodes_by_id: dict[str, EpisodeRec] = {}
        self.manifest: dict[str, Any] = {}
        self.index: dict[str, Any] = {}
        self.runs: list[dict[str, Any]] = []
        self.fault_log: pd.DataFrame = S.empty_fault_log()
        self.mp3_clock: dict[str, str | None] | None = None
        self.scores_loaded = False
        self.overlay = Overlay()
        self.advisories: dict[str, Any] = {}
        self._ticker: list[tuple[int, str, EpisodeRec]] = []
        self._events: list[tuple[int, tuple[Any, ...]]] = []
        self._mp3_events: list[tuple[int, tuple[Any, ...]]] = []
        self._clock_start = self.settings.clock_start
        self._clock_end = self.settings.clock_end
        self.mp3_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.load()

    # -- loading -----------------------------------------------------------------------

    def load(self) -> None:
        """(Re)read every file. Missing files are tolerated; missing scores set
        ``scores_loaded = False`` and leave the fleet empty."""
        self._load_index()
        self._load_fault_log()
        self._load_scores()
        self._load_episodes()
        self._load_manifest()
        self._build_events()
        self._build_ticker()

    def _load_index(self) -> None:
        path = self.settings.sim_dir / "index.json"
        if not path.exists():
            self.index, self.runs = {}, []
            return
        self.index = json.loads(path.read_text(encoding="utf-8"))
        self.runs = list(self.index.get("runs", []))

    def _load_fault_log(self) -> None:
        path = self.settings.sim_dir / "fault_log.parquet"
        if not path.exists():
            self.fault_log = S.empty_fault_log()
            return
        self.fault_log = S.coerce_fault_log(pd.read_parquet(path, engine="pyarrow"))

    def _load_scores(self) -> None:
        self.components = {}
        frames: list[pd.DataFrame] = []
        for name in ("scores.parquet", "metropt3.parquet"):
            path = self.settings.scores_dir / name
            if path.exists():
                frames.append(pd.read_parquet(path, engine="pyarrow"))
        if not frames:
            self.scores_loaded = False
            return
        scores = S.coerce_scores(pd.concat(frames, ignore_index=True))
        self.scores_loaded = len(scores) > 0
        if not len(scores):
            return
        scores = scores.sort_values(
            ["train_id", "car", "subsystem", "component_id", "timestamp"], kind="stable"
        )
        for key, grp in scores.groupby(
            ["train_id", "car", "subsystem", "component_id"], observed=True, sort=False
        ):
            train_id, car, subsystem, component_id = key
            models = grp["model"].astype(str)
            self.components[(str(train_id), int(car), str(subsystem), str(component_id))] = ComponentSeries(
                train_id=str(train_id),
                car=int(car),
                subsystem=str(subsystem),
                component_id=str(component_id),
                model=str(models.iloc[-1]) if len(models) else "",
                ts=_ns_array(grp["timestamp"]),
                score=grp["score"].to_numpy(dtype=np.float64),
                threshold=grp["threshold"].to_numpy(dtype=np.float64),
                alert=grp["alert"].to_numpy(dtype=bool),
                top_raw=grp["top_signals_json"].to_numpy(dtype=object),
            )
        sim = [c for k, c in self.components.items() if k[0] != MP3_TRAIN_ID]
        if sim:
            self._clock_start = min(self.settings.clock_start, pd.Timestamp(min(c.ts[0] for c in sim), unit="ns", tz="UTC"))
            self._clock_end = min(
                pd.Timestamp(max(c.ts[-1] for c in sim), unit="ns", tz="UTC"), SIM_CLOCK_HARD_END
            )
        mp3 = [c for k, c in self.components.items() if k[0] == MP3_TRAIN_ID]
        if mp3:
            self.mp3_clock = {
                "start": iso(pd.Timestamp(min(c.ts[0] for c in mp3), unit="ns", tz="UTC")),
                "end": iso(pd.Timestamp(max(c.ts[-1] for c in mp3), unit="ns", tz="UTC")),
            }

    def _load_episodes(self) -> None:
        self.episodes, self.episodes_by_key, self.episodes_by_id = [], {}, {}
        path = self.settings.scores_dir / "episodes.parquet"
        if not path.exists():
            return
        df = pd.read_parquet(path, engine="pyarrow")
        for i, row in enumerate(df.to_dict("records")):
            rec = self._episode_from_row(row, fallback_id=f"ep-{i}")
            self.episodes.append(rec)
        self.episodes.sort(key=lambda e: e.t_start_ns)
        self._index_episodes()

    def _index_episodes(self) -> None:
        self.episodes_by_key = {}
        self.episodes_by_id = {}
        for rec in self.episodes:
            self.episodes_by_key.setdefault(rec.key, []).append(rec)
            self.episodes_by_id[rec.episode_id] = rec

    @staticmethod
    def _episode_from_row(row: dict[str, Any], *, fallback_id: str) -> EpisodeRec:
        def _t(col: str) -> int | None:
            v = row.get(col)
            return None if v is None or pd.isna(v) else _ns(v)

        def _s(col: str) -> str | None:
            v = row.get(col)
            return None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)

        return EpisodeRec(
            episode_id=str(row.get("episode_id") or fallback_id),
            train_id=str(row.get("train_id", "")),
            car=int(row.get("car", 0) or 0),
            subsystem=str(row.get("subsystem", "")),
            component_id=str(row.get("component_id", "")),
            model=str(row.get("model", "") or ""),
            t_start_ns=_t("t_start") or 0,
            t_end_ns=_t("t_end"),
            peak_score=_f(row.get("peak_score")),
            n_rows=int(row.get("n_rows", 0) or 0),
            fault_type=_s("fault_type"),
            t_onset_ns=_t("t_onset"),
            t_failure_ns=_t("t_failure"),
            lead_to_failure_h=_f(row.get("lead_to_failure_h")),
            matched=bool(row.get("matched", False)),
        )

    def _load_manifest(self) -> None:
        path = self.settings.scores_dir / "manifest.json"
        self.manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    # -- derived tables ----------------------------------------------------------------

    def _build_events(self) -> None:
        """The fault-log events the KPIs count, as ``(detection window opens, key)`` sorted by
        time, so ``kpis_at`` bisects instead of re-walking the fault log ten times a second.

        Two lists, never one: the sim fleet's events come from ``data/sim/fault_log.parquet``
        and MetroPT-3's from its four failure-report rows, on a clock six years earlier. Mixing
        them would let the 2020 unit count towards the 2026 fleet's KPIs.
        """
        rows = self.fault_log.to_dict("records") if len(self.fault_log) else []
        self._events = self._events_from(r for r in rows if str(r.get("train_id", "")) != MP3_TRAIN_ID)
        self._mp3_events = self._events_from(_metropt3_faults()) if self._has_mp3() else []

    @staticmethod
    def _events_from(rows: Any) -> list[tuple[int, tuple[Any, ...]]]:
        horizon_ns = int(DETECTION_HORIZON_H * 3600 * _NS)
        out: list[tuple[int, tuple[Any, ...]]] = []
        seen: set[tuple[Any, ...]] = set()
        for row in rows:
            if str(row.get("fault_type", "")) in ("healthy", "nff"):
                continue
            onset = row.get("t_onset")
            if onset is None or pd.isna(onset):
                continue
            onset_ns = _ns(onset)
            key = (
                str(row.get("train_id", "")),
                int(row.get("car", 0)),
                str(row.get("subsystem", "")),
                str(row.get("component_id", "")),
                onset_ns,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append((onset_ns - horizon_ns, key))
        out.sort(key=lambda p: p[0])
        return out

    def _build_ticker(self) -> None:
        """One entry per episode transition (open / close), sorted by time.

        Only the transition is precomputed, not its text: an open line carries *the age of the
        episode at the frame's ``ts``*, so the line is rendered in :meth:`ticker_at`.
        """
        lines: list[tuple[int, str, EpisodeRec]] = []
        for rec in self.all_episodes():
            lines.append((rec.t_start_ns, "open", rec))
            if rec.t_end_ns is not None:
                lines.append((rec.t_end_ns, "closed", rec))
        lines.sort(key=lambda p: p[0])
        self._ticker = lines

    def _score_at(self, rec: "EpisodeRec", ts_ns: int) -> tuple[float | None, float | None]:
        """``(score, threshold)`` of one component at a time, for the ticker."""
        cs = self.series_for(rec.key)
        if cs is None or not len(cs.ts):
            return rec.peak_score, None
        i = cs.index_at(ts_ns)
        if i < 0:
            return rec.peak_score, float(cs.threshold[0])
        return float(cs.score[i]), float(cs.threshold[i])

    def ticker_line(self, rec: "EpisodeRec", kind: str, ts_ns: int) -> str:
        """One ticker line, contract section 5:
        ``"08:29 T01 car 1 door L1 · CUSUM 3.2 > 2.4 · episode open 6 h"``."""
        at_ns = rec.t_start_ns if kind == "open" else (rec.t_end_ns or rec.t_start_ns)
        hhmm = pd.Timestamp(at_ns, unit="ns", tz="UTC").strftime("%H:%M")
        where = f"{hhmm} {rec.train_id} car {int(rec.car)} {rec.component_id.replace('_', ' ')}"
        score, thr = self._score_at(rec, at_ns)
        label = model_label(rec.model)
        what = f"{label} {_g(score)} > {_g(thr)}".strip()
        if kind == "open":
            age_h = max(ts_ns - rec.t_start_ns, 0) / 3.6e12
            when = f"episode open {_g(age_h)} h"
        else:
            when = f"episode closed after {_g((at_ns - rec.t_start_ns) / 3.6e12)} h"
        return f"{where} · {what} · {when}"

    # -- accessors ---------------------------------------------------------------------

    @property
    def clock(self) -> dict[str, str | None]:
        return {"start": iso(self._clock_start), "end": iso(self._clock_end)}

    @property
    def clock_start(self) -> pd.Timestamp:
        return self._clock_start

    @property
    def clock_end(self) -> pd.Timestamp:
        return self._clock_end

    def mp3_clock_start(self) -> pd.Timestamp:
        """Start of the MetroPT-3 clock: its first scored timestamp."""
        if self.mp3_clock and self.mp3_clock.get("start"):
            return parse_ts(self.mp3_clock["start"])
        return self._clock_start

    def mp3_clock_end(self) -> pd.Timestamp:
        """End of the MetroPT-3 clock (its own, never the sim fleet's)."""
        if self.mp3_clock and self.mp3_clock.get("end"):
            return parse_ts(self.mp3_clock["end"])
        return self._clock_end

    def series_for(self, key: ComponentKey) -> ComponentSeries | None:
        """The overlay's arrays if this component was injected into, else the loaded ones."""
        return self.overlay.components.get(key) or self.components.get(key)

    def component_keys(self, train_id: str | None = None) -> list[ComponentKey]:
        keys = set(self.components) | set(self.overlay.components)
        if train_id is not None:
            keys = {k for k in keys if k[0] == train_id}
        return sorted(keys)

    def find_component(self, train_id: str, component_id: str, car: int | None = None) -> ComponentSeries | None:
        """Look one component up by id, optionally pinned to a car."""
        cands = [
            k
            for k in self.component_keys(train_id)
            if k[3] == component_id and (car is None or k[1] == int(car))
        ]
        return self.series_for(cands[0]) if cands else None

    def train_ids(self, *, include_mp3: bool = True) -> list[str]:
        """Sim trains from ``index.json`` plus whatever is scored; ``MP3`` whenever the
        MetroPT-3 unit is present at all (its own scores, or the raw dataset on disk)."""
        ids = {k[0] for k in self.component_keys()} | {r["train_id"] for r in self.runs if "train_id" in r}
        if self._has_mp3():
            ids.add(MP3_TRAIN_ID)
        if not include_mp3:
            ids.discard(MP3_TRAIN_ID)
        return sorted(ids)

    def _has_mp3(self) -> bool:
        return (
            self.mp3_clock is not None
            or (self.settings.scores_dir / "metropt3.parquet").exists()
            or (self.settings.data_dir / "raw" / "metropt3").is_dir()
        )

    def sim_train_ids(self) -> list[str]:
        return self.train_ids(include_mp3=False)

    def all_episodes(self) -> list[EpisodeRec]:
        """Loaded episodes with the overlay's replacing them component by component."""
        if not self.overlay.episodes:
            return self.episodes
        shadowed = set(self.overlay.episodes)
        out = [e for e in self.episodes if e.key not in shadowed]
        for recs in self.overlay.episodes.values():
            out.extend(recs)
        out.sort(key=lambda e: e.t_start_ns)
        return out

    def episodes_for(self, key: ComponentKey) -> list[EpisodeRec]:
        if key in self.overlay.episodes:
            return self.overlay.episodes[key]
        return self.episodes_by_key.get(key, [])

    def episode(self, episode_id: str) -> EpisodeRec | None:
        for recs in self.overlay.episodes.values():
            for rec in recs:
                if rec.episode_id == episode_id:
                    return rec
        return self.episodes_by_id.get(episode_id)

    # -- the fleet layout --------------------------------------------------------------

    def instrumented(self, train_id: str) -> list[dict[str, Any]]:
        """Every component of ``train_id`` that has telemetry, from ``index.json``.

        A bearing run instruments all eight axle boxes of its car; a door run one leaf; a
        pneumatic run the unit-level APU. Everything else on the 6-car schematic is ``nodata``.
        """
        if train_id == MP3_TRAIN_ID:
            return [
                {"car": 0, "subsystem": "pneumatic", "component_id": "apu_1", "run_id": "metropt3"}
            ]
        out: list[dict[str, Any]] = []
        for run in self.runs:
            if run.get("train_id") != train_id:
                continue
            sub = str(run.get("subsystem", ""))
            car = int(run.get("car", 0))
            cids = S.AXLEBOX_COMPONENT_IDS if sub == "bearing" else (str(run.get("component_id", "")),)
            for cid in cids:
                out.append({"car": car, "subsystem": sub, "component_id": cid, "run_id": run.get("run_id")})
        for key in self.overlay.components:
            if key[0] != train_id:
                continue
            if not any(o["component_id"] == key[3] and o["car"] == key[1] for o in out):
                out.append({"car": key[1], "subsystem": key[2], "component_id": key[3], "run_id": "overlay"})
        out.sort(key=lambda d: (d["car"], d["subsystem"], d["component_id"]))
        return out

    def run_for(self, train_id: str, component_id: str, car: int | None = None) -> dict[str, Any] | None:
        """The ``index.json`` run whose telemetry carries ``component_id`` of ``train_id``."""
        sub = _subsystem_of(component_id)
        for run in self.runs:
            if run.get("train_id") != train_id:
                continue
            if car is not None and int(run.get("car", 0)) != int(car):
                continue
            rsub = str(run.get("subsystem", ""))
            if component_id == S.TRAIN_COMPONENT_ID:
                return run
            if sub is not None and rsub != sub:
                continue
            if rsub == "bearing":
                if component_id in S.AXLEBOX_COMPONENT_IDS:
                    return run
            elif str(run.get("component_id", "")) == component_id:
                return run
        return None

    def faults_for(self, train_id: str) -> list[dict[str, Any]]:
        """Fault-log rows of one train as JSON-ready dicts (overlay injections included)."""
        rows: list[dict[str, Any]] = []
        df = self.fault_log
        if len(df):
            sub = df[df["train_id"].astype(str) == train_id]
            for row in sub.to_dict("records"):
                rows.append(
                    {
                        "run_id": str(row.get("run_id", "")),
                        "train_id": str(row.get("train_id", "")),
                        "car": int(row.get("car", 0)),
                        "subsystem": str(row.get("subsystem", "")),
                        "component_id": str(row.get("component_id", "")),
                        "fault_type": str(row.get("fault_type", "")),
                        "t_onset": iso(row.get("t_onset")),
                        "t_failure": iso(row.get("t_failure")),
                        "t_functional_failure": iso(row.get("t_functional_failure")),
                        "gamma": _f(row.get("gamma")),
                        "shape": _s(row.get("shape")),
                        "injected": False,
                    }
                )
        if train_id == MP3_TRAIN_ID and not rows:
            rows.extend(_metropt3_faults())
        rows.extend(r for r in self.overlay.fault_log if r.get("train_id") == train_id)
        rows.sort(key=lambda r: (r["t_onset"] or ""))
        return rows

    def trains(self) -> list[dict[str, Any]]:
        """``GET /api/trains`` payload."""
        out = []
        for train_id in self.train_ids():
            is_mp3 = train_id == MP3_TRAIN_ID
            clock = self.mp3_clock if is_mp3 else self.clock
            out.append(
                {
                    "train_id": train_id,
                    "line": "Porto metro (MetroPT-3)" if is_mp3 else "NSL",
                    "cars": 1 if is_mp3 else S.MAX_CAR,
                    "instrumented": self.instrumented(train_id),
                    "faults": self.faults_for(train_id),
                    "clock": clock,
                }
            )
        return out

    # -- frames ------------------------------------------------------------------------

    def component_state(self, key: ComponentKey, ts_ns: int) -> dict[str, Any] | None:
        """One ``components[]`` entry of a frame, or ``None`` when nothing is scored yet.

        ``health``: ``crit`` inside an alarm episode at ``ts``; ``warn`` above threshold with no
        episode, or above 0.8 x threshold; ``ok`` otherwise. A latest row older than the
        subsystem's staleness window sets ``stale`` and falls back to ``ok``.
        """
        cs = self.series_for(key)
        if cs is None or len(cs.ts) == 0:
            return None
        i = cs.index_at(ts_ns)
        if i < 0:
            return None
        score = float(cs.score[i])
        thr = float(cs.threshold[i])
        age_s = (ts_ns - int(cs.ts[i])) / _NS
        stale = age_s > STALE_S.get(cs.subsystem, DEFAULT_STALE_S)
        if stale:
            # Contract section 5: an old row falls back to ``ok`` whatever the episode says -
            # the component is not being watched any more, so the demo must not claim it is.
            health = "ok"
        elif any(e.open_at(ts_ns) for e in self.episodes_for(key)):
            health = "crit"
        elif score > 0.8 * thr:
            # contract: "> threshold but no episode yet, or > 0.8 x threshold" - the first
            # case is contained in the second, so one comparison covers both.
            health = "warn"
        else:
            health = "ok"
        return {
            "car": int(cs.car),
            "subsystem": cs.subsystem,
            "component_id": cs.component_id,
            "ts": iso(pd.Timestamp(int(cs.ts[i]), unit="ns", tz="UTC")),
            "score": _f(score),
            "threshold": _f(thr),
            "alert": bool(cs.alert[i]),
            "health": health,
            "stale": bool(stale),
            "model": cs.model,
            "top_signals": [] if stale else cs.top_signals(i),
        }

    def alert_payload(self, rec: EpisodeRec, ts_ns: int) -> dict[str, Any]:
        """One ``alerts[]`` entry. ``t_end`` and ``peak_score`` are truncated at ``ts`` so a
        replay frame never shows the future of an episode that is still open."""
        closed = rec.t_end_ns is not None and rec.t_end_ns <= ts_ns
        peak = rec.peak_score
        cs = self.series_for(rec.key)
        if cs is not None and len(cs.ts):
            sl = cs.slice_between(rec.t_start_ns, min(ts_ns, rec.t_end_ns if rec.t_end_ns is not None else ts_ns))
            if sl.stop > sl.start:
                peak = float(np.max(cs.score[sl]))
        return {
            "episode_id": rec.episode_id,
            "train_id": rec.train_id,
            "car": int(rec.car),
            "subsystem": rec.subsystem,
            "component_id": rec.component_id,
            "model": rec.model,
            "t_start": iso(pd.Timestamp(rec.t_start_ns, unit="ns", tz="UTC")),
            "t_end": iso(pd.Timestamp(rec.t_end_ns, unit="ns", tz="UTC")) if closed else None,
            "open": not closed,
            "final": bool(closed),
            "peak_score": _f(peak),
            # Whole-episode figures, and therefore future knowledge while the episode is open:
            # they are published only once the episode has closed at or before ``ts``.
            "n_rows": int(rec.n_rows),
            "fault_type": rec.fault_type,
            "lead_to_failure_h": _f(rec.lead_to_failure_h) if closed else None,
            "matched": bool(rec.matched) if closed else None,
            "advisory": self.advisories.get(rec.episode_id),
        }

    def alerts_at(self, ts: Any, *, train_id: str | None = None, limit: int | None = 25) -> list[dict[str, Any]]:
        """Episodes open or closed at ``ts``, newest first.

        ``train_id=None`` is the sim fleet: MetroPT-3's episodes are served only when ``MP3``
        is asked for by name.
        """
        ts_ns = _ns(ts)
        recs = [
            e for e in self.all_episodes() if e.started_by(ts_ns) and _in_scope(e.train_id, train_id)
        ]
        recs.sort(key=lambda e: e.t_start_ns, reverse=True)
        if limit is not None:
            recs = recs[:limit]
        return [self.alert_payload(e, ts_ns) for e in recs]

    def kpis_at(self, ts: Any, *, train_id: str | None = None) -> dict[str, Any]:
        """``open_alerts``, ``median_lead_h``, ``fa_per_train_day``, ``events_detected/total``.

        Two scopes, never mixed (contract section 1):

        * ``train_id=None`` - the **sim fleet**: sim episodes only, ``n_trains`` = the sim
          trains, and the elapsed days counted from the sim clock's start;
        * ``train_id="MP3"`` - the MetroPT-3 unit: its own episodes, its own clock start (its
          first scored row), ``n_trains`` = 1 and its four failure-report rows as the events.

        ``events_total`` counts fault-log entries whose detection window ``[onset - H, ...]``
        has opened by ``ts``; ``events_detected`` those a matched episode that has already
        started points at. ``median_lead_h`` uses **closed, matched** episodes only - the lead
        of an episode still running is not known yet.
        """
        ts_ns = _ns(ts)
        is_mp3 = train_id == MP3_TRAIN_ID
        started = [
            e for e in self.all_episodes() if e.started_by(ts_ns) and _in_scope(e.train_id, train_id)
        ]
        open_alerts = sum(1 for e in started if e.open_at(ts_ns))
        closed = [e for e in started if e.t_end_ns is not None and e.t_end_ns <= ts_ns]
        leads = [e.lead_to_failure_h for e in closed if e.matched and e.lead_to_failure_h is not None]
        false_alarms = sum(1 for e in started if not e.matched)
        n_trains = 1 if train_id is not None else max(len(self.sim_train_ids()), 1)
        t0 = self.mp3_clock_start() if is_mp3 else self._clock_start
        elapsed_days = max((ts_ns - int(t0.value)) / (86400 * _NS), 1e-9)
        detected = {(e.train_id, e.car, e.subsystem, e.component_id, e.t_onset_ns) for e in started if e.matched}
        events = self._mp3_events if is_mp3 else self._events
        if train_id is not None and not is_mp3:
            events = [p for p in events if p[1][0] == train_id]
        total, done = self._events_upto(ts_ns, events)
        n_detected = len(detected & done) if done else len(detected)
        return {
            "open_alerts": int(open_alerts),
            "median_lead_h": _f(float(np.median(leads))) if leads else None,
            "fa_per_train_day": _f(false_alarms / (n_trains * elapsed_days)),
            "events_detected": int(n_detected),
            "events_total": int(max(total, n_detected)),
            "n_trains": int(n_trains),
            "elapsed_days": round(float(elapsed_days), 4),
        }

    @staticmethod
    def _events_upto(
        ts_ns: int, events: Sequence[tuple[int, tuple[Any, ...]]]
    ) -> tuple[int, set[tuple[Any, ...]]]:
        """Fault-log events whose detection window has opened by ``ts`` (keys + count)."""
        cut = bisect_right(events, ts_ns, key=lambda p: p[0])
        keys = {key for _, key in events[:cut]}
        return len(keys), keys

    def ticker_at(self, ts: Any, n: int = 8, *, train_id: str | None = None) -> list[str]:
        """The last ``n`` episode transitions at or before ``ts``, newest first.

        ``train_id=None`` is the **sim fleet**: MetroPT-3's 2020 episodes never appear in it
        (contract section 1 - the replay never mixes the two clocks).
        """
        ts_ns = _ns(ts)
        cut = [
            (kind, rec)
            for t, kind, rec in self._ticker
            if t <= ts_ns and _in_scope(rec.train_id, train_id)
        ]
        return [self.ticker_line(rec, kind, ts_ns) for kind, rec in reversed(cut[-n:])]

    def frame_at(
        self,
        ts: Any,
        *,
        train_ids: Sequence[str] | None = None,
        speed: float = 8640.0,
        playing: bool = False,
    ) -> dict[str, Any]:
        """One replay frame, contract section 5."""
        ts_v = parse_ts(ts)
        ts_ns = int(ts_v.value)
        # Default: the sim fleet only. ``MP3`` has its own clock and is served by name.
        wanted: list[str] = list(train_ids) if train_ids is not None else self.sim_train_ids()
        trains = []
        for train_id in wanted:
            comps = []
            for key in self.component_keys(train_id):
                state = self.component_state(key, ts_ns)
                if state is not None:
                    comps.append(state)
            comps.sort(key=lambda d: (d["car"], d["subsystem"], d["component_id"]))
            trains.append({"train_id": train_id, "components": comps})
        one_train = wanted[0] if (train_ids is not None and len(wanted) == 1) else None
        # KPIs and the ticker are fleet-wide; only ``MP3`` swaps them for its own unit, because
        # its clock is six years earlier and mixing the two would corrupt both (contract s. 1).
        scope = MP3_TRAIN_ID if one_train == MP3_TRAIN_ID else None
        return {
            "ts": iso(ts_v),
            "speed": float(speed),
            "playing": bool(playing),
            "trains": trains,
            "alerts": self.alerts_at(ts_v, train_id=one_train),
            "kpis": self.kpis_at(ts_v, train_id=scope),
            "ticker": self.ticker_at(ts_v, train_id=scope),
        }

    # -- overlay -----------------------------------------------------------------------

    def _patched_series(self, key: ComponentKey, grp: pd.DataFrame) -> ComponentSeries:
        """The injected rows of one component, *patched onto* whatever was scored before them.

        The injection covers a window, not the whole clock, so the rows the winner already
        scored before that window stay exactly where they were and the charts keep their
        history; everything from the first injected timestamp on is the injected run.
        """
        ts = _ns_array(grp["timestamp"]) if len(grp) else np.empty(0, dtype=np.int64)
        score = grp["score"].to_numpy(dtype=np.float64) if len(grp) else np.empty(0)
        threshold = grp["threshold"].to_numpy(dtype=np.float64) if len(grp) else np.empty(0)
        alert = grp["alert"].to_numpy(dtype=bool) if len(grp) else np.empty(0, dtype=bool)
        top = grp["top_signals_json"].to_numpy(dtype=object) if len(grp) else np.empty(0, dtype=object)
        model = str(grp["model"].astype(str).iloc[-1]) if len(grp) else ""
        base = self.components.get(key)
        if base is not None and len(base.ts):
            cut = int(ts[0]) if len(ts) else int(base.ts[-1]) + 1
            j = int(np.searchsorted(base.ts, cut, side="left"))
            if j > 0:
                ts = np.concatenate((base.ts[:j], ts))
                score = np.concatenate((base.score[:j], score))
                threshold = np.concatenate((base.threshold[:j], threshold))
                alert = np.concatenate((base.alert[:j], alert))
                top = np.concatenate((base.top_raw[:j], top))
            model = model or base.model
        return ComponentSeries(
            train_id=key[0],
            car=int(key[1]),
            subsystem=key[2],
            component_id=key[3],
            model=model,
            ts=ts,
            score=score,
            threshold=threshold,
            alert=alert,
            top_raw=top,
        )

    def apply_overlay(
        self,
        *,
        key: ComponentKey,
        scores: pd.DataFrame,
        episodes: pd.DataFrame | None,
        long: pd.DataFrame | None,
        features: pd.DataFrame | None,
        fault_rows: Sequence[dict[str, Any]] = (),
        info: dict[str, Any] | None = None,
    ) -> list[EpisodeRec]:
        """Install an injection's re-scored rows as the overlay.

        One injection is **not** necessarily one series: a bearing run scores all eight axle
        boxes of its car, so ``scores`` comes back with eight ``component_id`` values and each
        one becomes its own overlay entry keyed by ``(train_id, car, subsystem, component_id)``.
        ``key`` is only the component the request named, used when the scorer returns rows
        without key columns.

        Returns the **injected** episodes (not the earlier ones kept alongside them).
        """
        sc = S.coerce_scores(scores)
        for col, default in (
            ("train_id", key[0]),
            ("car", key[1]),
            ("subsystem", key[2]),
            ("component_id", key[3]),
        ):
            if col not in sc.columns:
                sc[col] = default
        sc = sc.sort_values(
            ["train_id", "car", "subsystem", "component_id", "timestamp"], kind="stable"
        )
        keys: list[ComponentKey] = []
        groups: dict[ComponentKey, pd.DataFrame] = {}
        for gkey, grp in sc.groupby(
            ["train_id", "car", "subsystem", "component_id"], observed=True, sort=False
        ):
            k = (str(gkey[0]), int(gkey[1]), str(gkey[2]), str(gkey[3]))
            keys.append(k)
            groups[k] = grp
        if not keys:  # the scorer produced nothing: still shadow the component that was asked for
            keys, groups = [key], {key: sc}
        for k in keys:
            self.overlay.components[k] = self._patched_series(k, groups[k])
            if long is not None:
                self.overlay.long[k] = long
            if features is not None:
                self.overlay.features[k] = features

        def _car_of(row: dict[str, Any]) -> int:
            value = row.get("car")
            try:
                return int(key[1]) if value is None or pd.isna(value) else int(value)
            except (TypeError, ValueError):
                return int(key[1])

        ep_rows: dict[ComponentKey, list[dict[str, Any]]] = {k: [] for k in keys}
        for row in (episodes.to_dict("records") if episodes is not None and len(episodes) else []):
            k = (
                str(row.get("train_id") or key[0]),
                _car_of(row),
                str(row.get("subsystem") or key[2]),
                str(row.get("component_id") or key[3]),
            )
            ep_rows.setdefault(k, []).append(row)
        injected: list[EpisodeRec] = []
        for k, rows in ep_rows.items():
            first_ns = int(groups[k]["timestamp"].iloc[0].value) if k in groups and len(groups[k]) else None
            recs = []
            for i, row in enumerate(rows):
                row = dict(row)
                row["train_id"], row["car"] = k[0], k[1]
                row["subsystem"], row["component_id"] = k[2], k[3]
                # Injected ids are always ``inj-<train>-<component>-<n>``: the demo has to be
                # able to tell an injected episode from a scored one (and drop its advisory).
                row["episode_id"] = f"{INJECT_EPISODE_PREFIX}{k[0]}-{k[3]}-{i}"
                recs.append(self._episode_from_row(row, fallback_id=row["episode_id"]))
            kept = [
                e
                for e in self.episodes_by_key.get(k, [])
                if first_ns is not None and e.t_end_ns is not None and e.t_end_ns <= first_ns
            ]
            injected.extend(recs)
            self.overlay.episodes[k] = sorted(kept + recs, key=lambda e: e.t_start_ns)
        self.overlay.fault_log.extend(dict(r) for r in fault_rows)
        if info is not None:
            self.overlay.info.append(info)
        self._build_ticker()
        return injected

    def clear_overlay(self) -> None:
        """Drop the injection - and the advisories cached for the episodes it invented."""
        for episode_id in [k for k in self.advisories if k.startswith(INJECT_EPISODE_PREFIX)]:
            self.advisories.pop(episode_id, None)
        self.overlay.clear()
        self._build_ticker()


def _metropt3_faults() -> list[dict[str, Any]]:
    """The four MetroPT-3 failure-report rows, from the adapter's hard-coded table."""
    try:
        from nebulax.adapters import metropt3 as adapter

        df = adapter._build_fault_log()  # noqa: SLF001 - the table is a module constant
    except Exception:  # pragma: no cover - adapter missing / renamed
        return []
    rows = []
    for row in df.to_dict("records"):
        rows.append(
            {
                "run_id": str(row.get("run_id", "metropt3")),
                "train_id": MP3_TRAIN_ID,
                "car": int(row.get("car", 0)),
                "subsystem": str(row.get("subsystem", "pneumatic")),
                "component_id": str(row.get("component_id", "apu_1")),
                "fault_type": str(row.get("fault_type", "")),
                "t_onset": iso(row.get("t_onset")),
                "t_failure": iso(row.get("t_failure")),
                "t_functional_failure": iso(row.get("t_functional_failure")),
                "gamma": _f(row.get("gamma")),
                "shape": _s(row.get("shape")),
                "injected": False,
            }
        )
    return rows
