"""The FastAPI app: ``uvicorn nebulax.api.main:app --port 8000`` from the repo root.

Every route of ``docs/app_contract.md`` section 4 lives here; the work lives in
:mod:`nebulax.api.state` (frames, alerts, KPIs), :mod:`nebulax.api.series` (telemetry),
:mod:`nebulax.api.inject` (fault injection) and :mod:`nebulax.api.replay` (the WebSocket).

``create_app(settings=...)`` is the factory tests use to point the whole app at a ``tmp_path``;
the module-level ``app`` is the same thing built from the environment
(``NEBULAX_DATA_DIR``, ``NEBULAX_RESULTS_DIR``, ``NEBULAX_WEB_DIST``).

Two imports are deliberately *lazy*, inside the routes that need them:
``nebulax.demo.scoring`` and ``nebulax.advisory``. Both are separate deliverables; the API
boots, serves and tests without either, and a demo laptop missing one still shows the fleet.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query, WebSocket
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from nebulax import schema as S
from nebulax.api import series as series_mod
from nebulax.api.inject import InjectError, InjectNotFound, inject
from nebulax.api.ps3 import router as ps3_router
from nebulax.api.replay import replay_socket
from nebulax.api.state import MP3_TRAIN_ID, FleetState, Settings, iso, parse_ts

__all__ = ["create_app", "app", "InjectBody", "selection_rows"]

#: Column headers of the leaderboard's "Selected per subsystem" table -> JSON keys.
_SELECTION_KEYS: dict[str, str] = {
    "subsystem": "subsystem",
    "dataset": "dataset",
    "model": "model",
    "split": "split",
    "input_kind": "input_kind",
    "window": "window",
    "selected on": "selected_on",
    "val lift": "val_lift",
    "vus-pr (test)": "vus_pr_test",
    "test lift": "test_lift",
    "episodes (n)": "episodes",
    "recall@budget": "recall",
    "precision": "precision",
    "fa/scored-day": "fa_per_scored_day",
    "fit (s)": "fit_s",
}

#: How the numbers in the context were produced. It rides on ``fleet_peer_summary`` (free
#: text) rather than on ``glossary``: the advisory module keeps its own per-subsystem glossary
#: there and the API must not overwrite it (contract section 6).
_METHODOLOGY = (
    "Method: the threshold is the false-alarm-budget threshold calibrated on the validation "
    "slice of the training trains (at most one alarm episode per 7 days of scored operation); "
    "an alarm episode is score > threshold for at least 3 consecutive rows on one component, "
    "merged under a 1 h gap."
)


class InjectBody(BaseModel):
    """``POST /api/sim/inject`` body."""

    train_id: str
    car: int = 0
    component_id: str
    fault_type: str
    severity_ramp_days: float = Field(gt=0.0)
    seed: int | None = None
    t_onset: str | None = None
    ts: str | None = Field(default=None, description="current replay ts; the onset defaults to it")


# --------------------------------------------------------------------------------------
# results/leaderboard.md - "Selected per subsystem"
# --------------------------------------------------------------------------------------


def _clean_cell(text: str) -> str:
    """Strip the markdown the leaderboard uses for emphasis, keep the words."""
    out = re.sub(r"\*\*|`|^_|_$", "", text.strip())
    return out.strip()


def selection_rows(results_dir: str | Path) -> list[dict[str, Any]]:
    """The "Selected per subsystem" rows of ``results/leaderboard.md`` as JSON.

    Continuation rows (the chance-floor notes, whose first cell is empty) are dropped, so what
    comes back is exactly the one-pick-per-(subsystem, dataset) table. Returns ``[]`` when the
    leaderboard has not been written yet.
    """
    path = Path(results_dir) / "leaderboard.md"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip().startswith("## Selected per subsystem"))
    except StopIteration:
        return []
    header: list[str] | None = None
    rows: list[dict[str, Any]] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("## "):
            break
        if not stripped.startswith("|"):
            if header is not None and rows:
                break
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if header is None:
            header = [c.lower().strip() for c in cells]
            continue
        if set("".join(cells)) <= {"-", ":", " "}:
            continue
        if not _clean_cell(cells[0]):  # a note row belonging to the pick above it
            continue
        row: dict[str, Any] = {}
        for name, cell in zip(header, cells):
            row[_SELECTION_KEYS.get(name, re.sub(r"[^a-z0-9]+", "_", name).strip("_"))] = _clean_cell(cell)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------
# Advisory context
# --------------------------------------------------------------------------------------


def _peer_summary(state: FleetState, rec, ts_ns: int) -> str:
    """One line comparing this component with its fleet peers of the same subsystem."""
    peers: list[tuple[str, float]] = []
    for key in state.component_keys():
        if key[2] != rec.subsystem:
            continue
        cs = state.series_for(key)
        if cs is None:
            continue
        i = cs.index_at(ts_ns)
        if i < 0:
            continue
        peers.append((f"{key[0]}/{key[3]}", float(cs.score[i])))
    if not peers:
        return f"no fleet peers scored at this time for subsystem {rec.subsystem}"
    peers.sort(key=lambda p: p[1], reverse=True)
    me = f"{rec.train_id}/{rec.component_id}"
    rank = next((n for n, (name, _) in enumerate(peers, start=1) if name == me), None)
    values = pd.Series([v for _, v in peers])
    mine = next((v for name, v in peers if name == me), None)
    return (
        f"{len(peers)} {rec.subsystem} components scored across the fleet at this time: "
        f"median {values.median():.3f}, 90th pct {values.quantile(0.9):.3f}, max {values.max():.3f} "
        f"({peers[0][0]}); this component "
        + (f"{mine:.3f}, rank {rank}/{len(peers)}" if mine is not None and rank else "not scored")
    )


def build_alert_context(state: FleetState, rec) -> dict[str, Any]:
    """The ``AlertContext`` payload for one episode (kept as a dict; the advisory module owns
    the pydantic model and we do not want to import it just to build the arguments)."""
    end_ns = rec.t_end_ns if rec.t_end_ns is not None else int(state.clock_end.value)
    cs = state.series_for(rec.key)
    score = rec.peak_score
    threshold = None  # AlertContext requires floats, so both fall back to 0.0 below
    top_signals: list[dict[str, Any]] = []
    model_name = rec.model
    if cs is not None and len(cs.ts):
        i = cs.index_at(end_ns)
        if i >= 0:
            score = float(cs.score[i]) if score is None else score
            threshold = float(cs.threshold[i])
            top_signals = cs.top_signals(i)
            model_name = model_name or cs.model
    events = series_mod.read_events(
        state,
        rec.train_id,
        rec.component_id,
        t_from=pd.Timestamp(rec.t_start_ns, unit="ns", tz="UTC"),
        t_to=pd.Timestamp(end_ns, unit="ns", tz="UTC"),
        car=rec.car,
        limit=12,
    )
    duration_h = (end_ns - rec.t_start_ns) / 3.6e12
    return {
        "episode_id": rec.episode_id,
        "train_id": rec.train_id,
        "car": int(rec.car),
        "subsystem": rec.subsystem,
        "component_id": rec.component_id,
        "model_name": model_name,
        "score": float(score) if score is not None else 0.0,
        "threshold": float(threshold) if threshold is not None else 0.0,
        "t_start": iso(pd.Timestamp(rec.t_start_ns, unit="ns", tz="UTC")),
        "t_end": iso(pd.Timestamp(rec.t_end_ns, unit="ns", tz="UTC")) if rec.t_end_ns is not None else None,
        "duration_h": round(float(duration_h), 3),
        "top_signals": top_signals,
        "recent_events": [
            f"{e['timestamp']} {e['event']}" + (f" {e['detail_json']}" if e.get("detail_json") else "")
            for e in events
        ],
        "fleet_peer_summary": f"{_peer_summary(state, rec, end_ns)}. {_METHODOLOGY}",
        "glossary": "",
        "fault_candidates": list(S.FAULT_TYPES.get(rec.subsystem, ())),
    }


def _advise_sync(state: FleetState, rec) -> dict[str, Any]:
    """Blocking advisory call - the route runs it in a worker thread so the demo never waits
    on the network in the event loop."""
    from nebulax.advisory import AlertContext, advise  # lazy: parallel deliverable

    payload = build_alert_context(state, rec)
    ctx = AlertContext(**payload)
    result = advise(ctx)
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return dict(result)  # pragma: no cover - a non-pydantic stand-in


# --------------------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------------------


def create_app(settings: Settings | None = None, *, state: FleetState | None = None) -> FastAPI:
    """Build the app. ``settings`` (or a ready-made ``state``) points it at a data root."""
    app = FastAPI(
        title="NEBULA X demo API",
        version="0.1.0",
        description="Scored rolling-stock fleet, replay frames and fault injection.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    fleet = state if state is not None else FleetState(settings)
    app.state.fleet = fleet

    def st() -> FleetState:
        return app.state.fleet

    def _ts(value: Any) -> pd.Timestamp:
        if value in (None, ""):
            return st().clock_end
        try:
            return parse_ts(value, what="ts")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # -- health / fleet ---------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        s = st()
        return {
            "status": "ok",
            "scores_loaded": bool(s.scores_loaded),
            "n_trains": len(s.sim_train_ids()),
            "clock": s.clock,
            "overlay": bool(s.overlay),
        }

    @app.get("/api/trains")
    def trains() -> list[dict[str, Any]]:
        return st().trains()

    @app.get("/api/train/{train_id}/state")
    def train_state(train_id: str, ts: str | None = Query(default=None)) -> dict[str, Any]:
        s = st()
        if train_id not in s.train_ids():
            raise HTTPException(status_code=404, detail=f"unknown train {train_id!r}")
        at = _ts(ts) if ts else (s.mp3_clock_end() if train_id == MP3_TRAIN_ID else s.clock_end)
        return s.frame_at(at, train_ids=[train_id])

    # -- charts -----------------------------------------------------------------------

    @app.get("/api/train/{train_id}/component/{component_id}/series")
    def component_series(
        train_id: str,
        component_id: str,
        signal: str = Query(...),
        from_: str | None = Query(default=None, alias="from"),
        to: str | None = Query(default=None),
        car: int | None = Query(default=None),
        max_points: int = Query(default=2000, ge=2, le=50_000),
    ) -> dict[str, Any]:
        try:
            return series_mod.read_series(
                st(),
                train_id,
                component_id,
                signal,
                t_from=from_,
                t_to=to,
                car=car,
                max_points=max_points,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/train/{train_id}/component/{component_id}/scores")
    def component_scores(
        train_id: str,
        component_id: str,
        from_: str | None = Query(default=None, alias="from"),
        to: str | None = Query(default=None),
        car: int | None = Query(default=None),
    ) -> dict[str, Any]:
        s = st()
        cs = s.find_component(train_id, component_id, car)
        if cs is None:
            raise HTTPException(
                status_code=404,
                detail=f"no scores for train {train_id!r} component {component_id!r}",
            )
        lo = None if from_ in (None, "") else int(_ts(from_).value)
        hi = None if to in (None, "") else int(_ts(to).value)
        sl = cs.slice_between(lo, hi)
        points = [
            [iso(pd.Timestamp(int(t), unit="ns", tz="UTC")), float(v), bool(a)]
            for t, v, a in zip(cs.ts[sl], cs.score[sl], cs.alert[sl])
        ]
        last = sl.stop - 1
        return {
            "train_id": train_id,
            "car": int(cs.car),
            "subsystem": cs.subsystem,
            "component_id": cs.component_id,
            "model": cs.model,
            "threshold": float(cs.threshold[last]) if sl.stop > sl.start else None,
            "points": points,
            "top_signals": cs.top_signals(last) if sl.stop > sl.start else [],
        }

    @app.get("/api/train/{train_id}/component/{component_id}/cycle")
    def component_cycle(
        train_id: str,
        component_id: str,
        ts: str | None = Query(default=None),
        car: int | None = Query(default=None),
    ) -> dict[str, Any]:
        try:
            return series_mod.read_cycle(st(), train_id, component_id, _ts(ts), car=car)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # -- alerts / kpis ----------------------------------------------------------------

    @app.get("/api/alerts")
    def alerts(
        ts: str | None = Query(default=None),
        train: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        s = st()
        at = _ts(ts) if ts else (s.mp3_clock_end() if train == MP3_TRAIN_ID else s.clock_end)
        return s.alerts_at(at, train_id=train, limit=limit)

    @app.get("/api/kpis")
    def kpis(
        ts: str | None = Query(default=None),
        train: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Sim-fleet KPIs by default; ``?train=MP3`` switches to the MetroPT-3 unit, which has
        its own clock, its own episodes and one unit (contract section 1)."""
        s = st()
        if train is not None and train not in s.train_ids():
            raise HTTPException(status_code=404, detail=f"unknown train {train!r}")
        at = _ts(ts) if ts else (s.mp3_clock_end() if train == MP3_TRAIN_ID else s.clock_end)
        return s.kpis_at(at, train_id=train)

    @app.get("/api/bench/selection")
    def bench_selection() -> dict[str, Any]:
        s = st()
        return {
            "manifest": s.manifest,
            "selection": selection_rows(s.settings.results_dir),
            "leaderboard": str(Path(s.settings.results_dir) / "leaderboard.md"),
        }

    # -- replay -----------------------------------------------------------------------

    @app.websocket("/api/replay")
    async def replay(ws: WebSocket, ts: str | None = Query(default=None)) -> None:
        await replay_socket(ws, st(), ts=ts)

    # -- injection --------------------------------------------------------------------

    @app.post("/api/sim/inject")
    async def sim_inject(body: InjectBody = Body(...)) -> dict[str, Any]:
        s = st()
        try:
            return await run_in_threadpool(
                inject,
                s,
                train_id=body.train_id,
                car=body.car,
                component_id=body.component_id,
                fault_type=body.fault_type,
                severity_ramp_days=body.severity_ramp_days,
                seed=body.seed,
                t_onset=body.t_onset,
                now=body.ts,
            )
        except InjectNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InjectError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (ModuleNotFoundError, FileNotFoundError) as exc:
            # nebulax.demo.scoring not written yet, or data/scores/models/... not produced yet
            raise HTTPException(
                status_code=503,
                detail=f"inject: the fitted winner is not available yet ({exc})",
            ) from exc

    @app.delete("/api/sim/inject")
    def sim_inject_clear() -> dict[str, Any]:
        st().clear_overlay()
        return {"cleared": True}

    # -- advisory ---------------------------------------------------------------------

    @app.post("/api/advisory/{episode_id}")
    async def advisory(episode_id: str, refresh: bool = Query(default=False)) -> dict[str, Any]:
        s = st()
        if not refresh and episode_id in s.advisories:
            return s.advisories[episode_id]
        rec = s.episode(episode_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"unknown episode {episode_id!r}")
        try:
            out = await run_in_threadpool(_advise_sync, s, rec)
        except ModuleNotFoundError as exc:
            raise HTTPException(
                status_code=503, detail=f"advisory: the advisory module is not available yet ({exc})"
            ) from exc
        s.advisories[episode_id] = out
        return out

    @app.get("/api/advisory/{episode_id}")
    def advisory_cached(episode_id: str) -> dict[str, Any]:
        out = st().advisories.get(episode_id)
        if out is None:
            raise HTTPException(status_code=404, detail=f"no cached advisory for {episode_id!r}")
        return out

    # -- PS3 predict routes (/api/ps3/...) --------------------------------------------

    app.include_router(ps3_router)

    # -- static (mounted last so it never shadows /api) -------------------------------

    dist = Path(fleet.settings.web_dist)
    if dist.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(dist), html=True), name="web")

    return app


#: The module-level app ``uvicorn nebulax.api.main:app`` serves, built on first *access* and
#: not at import: importing this module (as ``tests/test_api.py`` does, to get ``create_app``)
#: must not read the real ``data/`` tree.
_app: FastAPI | None = None


def __getattr__(name: str) -> Any:
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
