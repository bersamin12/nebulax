"""``WS /api/replay``: one asyncio session per client, pushing frames at 10 Hz.

The client sends ``{"cmd": "play" | "pause" | "seek" | "speed", "ts"?, "speed"?}``; the server
pushes the frame of :meth:`FleetState.frame_at` every tick while playing, plus one frame
immediately after every accepted command so a pause or a seek is visible without waiting.

``speed`` is *sim seconds per wall second* - the default 8640 replays one simulated day in ten
seconds, so the 30-day fleet runs in five minutes.

Receiving and ticking are separate tasks on purpose: a single loop would have to cancel a
pending ``receive()`` on every tick, and a cancelled receive can swallow the message it was
about to deliver.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from nebulax.api.state import FleetState, parse_ts

__all__ = ["DEFAULT_SPEED", "TICK_S", "ReplaySession", "replay_socket"]

#: Sim seconds per wall-clock second (1 day / 10 s), contract section 4.
DEFAULT_SPEED = 8640.0

#: Wall-clock seconds between pushed frames (10 Hz).
TICK_S = 0.1

#: Speeds outside this are refused - a runaway speed would walk the whole clock in one tick.
SPEED_LIMITS = (0.0, 30 * 86400.0)

_COMMANDS = ("play", "pause", "seek", "speed")


class _Nothing:
    """Sentinel for "the tick fired, no command arrived"."""

    __slots__ = ()


_NOTHING = _Nothing()


@dataclass(slots=True)
class _Undecodable:
    """A text frame that was not JSON. It is answered, not fatal: the socket stays open."""

    detail: str


@dataclass(slots=True)
class ReplaySession:
    """The replay position of one client. Pure logic - no I/O, so tests drive it directly."""

    state: FleetState
    ts: pd.Timestamp = field(default=None)  # type: ignore[assignment]
    speed: float = DEFAULT_SPEED
    playing: bool = False

    def __post_init__(self) -> None:
        if self.ts is None:
            self.ts = self.state.clock_start

    def clamp(self, ts: pd.Timestamp) -> pd.Timestamp:
        lo, hi = self.state.clock_start, self.state.clock_end
        return min(max(ts, lo), hi)

    def advance(self, dt_wall_s: float) -> None:
        """Move the sim clock on by ``speed x dt``; stop at the end of the clock."""
        if not self.playing or dt_wall_s <= 0:
            return
        self.ts = self.ts + pd.Timedelta(seconds=self.speed * dt_wall_s)
        if self.ts >= self.state.clock_end:
            self.ts = self.state.clock_end
            self.playing = False

    def handle(self, msg: Any) -> dict[str, Any] | None:
        """Apply one command. Returns an ``{"error": ...}`` payload for a bad one, else ``None``."""
        if not isinstance(msg, dict):
            return {"error": f"replay: expected an object, got {type(msg).__name__}"}
        cmd = str(msg.get("cmd", "")).lower()
        if cmd not in _COMMANDS:
            return {"error": f"replay: unknown cmd {msg.get('cmd')!r}; expected one of {list(_COMMANDS)}"}
        if cmd == "play":
            if self.ts >= self.state.clock_end:
                self.ts = self.state.clock_start
            self.playing = True
        elif cmd == "pause":
            self.playing = False
        elif cmd == "seek":
            if msg.get("ts") is None:
                return {"error": "replay: seek needs a ts"}
            try:
                self.ts = self.clamp(parse_ts(msg["ts"], what="seek ts"))
            except ValueError as exc:
                return {"error": str(exc)}
        if msg.get("speed") is not None or cmd == "speed":
            try:
                speed = float(msg["speed"])
            except (KeyError, TypeError, ValueError):
                return {"error": f"replay: speed {msg.get('speed')!r} is not a number"}
            if not SPEED_LIMITS[0] < speed <= SPEED_LIMITS[1]:
                return {"error": f"replay: speed must lie in {SPEED_LIMITS}, got {speed}"}
            self.speed = speed
        return None

    def display_ts(self) -> pd.Timestamp:
        """The position to publish. Wall-clock ticks land on fractional nanoseconds, which is
        noise at demo speeds, so the published stamp is floored - to the second once one tick
        is worth ten sim seconds or more, to the millisecond below that (flooring harder there
        would stop the clock advancing at all)."""
        return self.ts.floor("s" if self.speed >= 10.0 else "ms")

    def frame(self) -> dict[str, Any]:
        return self.state.frame_at(self.display_ts(), speed=self.speed, playing=self.playing)


async def _pump(ws, queue: asyncio.Queue) -> None:
    """Receive JSON forever into ``queue``.

    A frame that is not JSON (or not text at all) is one bad *message*, not a broken session:
    it goes into the queue as :class:`_Undecodable`, the session answers ``{"error": ...}`` and
    the socket stays open. Only a real disconnect puts ``None`` and ends the task.
    """
    while True:
        try:
            msg = await ws.receive_json()
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, ValueError) as exc:
            await queue.put(_Undecodable(f"replay: {type(exc).__name__}: {exc}"))
            continue
        except Exception:  # the client went away
            await queue.put(None)
            return
        await queue.put(msg)


async def replay_socket(
    ws,
    state: FleetState,
    *,
    tick_s: float = TICK_S,
    ts: Any = None,
    max_frames: int | None = None,
) -> ReplaySession:
    """Serve one WebSocket client until it disconnects (``max_frames`` bounds it in tests)."""
    session = ReplaySession(state=state, ts=None if ts is None else parse_ts(ts))
    await ws.accept()
    await ws.send_json(session.frame())
    sent = 1
    queue: asyncio.Queue = asyncio.Queue()
    pump = asyncio.create_task(_pump(ws, queue))
    last = time.monotonic()
    try:
        while max_frames is None or sent < max_frames:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=tick_s)
            except asyncio.TimeoutError:
                msg = _NOTHING
            if msg is None:  # the pump saw the socket close
                break
            now = time.monotonic()
            if isinstance(msg, _Undecodable):
                await ws.send_json({"error": f"{msg.detail}; expected a JSON object"})
                continue
            if msg is not _NOTHING:
                err = session.handle(msg)
                if err is not None:
                    await ws.send_json(err)
                    continue
                session.advance(now - last)
                last = now
                await ws.send_json(session.frame())
                sent += 1
                continue
            session.advance(now - last)
            last = now
            if session.playing:
                await ws.send_json(session.frame())
                sent += 1
    finally:
        pump.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await pump
    return session
