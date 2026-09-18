#!/usr/bin/env python
"""Run the benchmark winners over the whole demo fleet and write ``data/scores/``.

    python scripts/score_for_demo.py                       # everything that is not up to date
    python scripts/score_for_demo.py --force                # re-score regardless
    python scripts/score_for_demo.py --subsystems door,bearing
    python scripts/score_for_demo.py --skip-metropt3        # sim fleet only (no data/raw needed)
    python scripts/score_for_demo.py --episodes-only        # re-derive episodes.parquet only

``docs/app_contract.md`` section 3 is the specification. The winners and every knob they were
run with come out of ``results/runs.parquet``; the protocol for the synthetic fleet is
leave-one-train-out (``sim_loo_unit`` over all 10 trains), MetroPT-3 is its own temporal
split. Output:

    data/scores/scores.parquet          all three sim subsystems, all runs
    data/scores/metropt3.parquet        the MP3 unit, the benchmark's test slice
    data/scores/episodes.parquet        one row per alarm episode, joined to the fault log
    data/scores/manifest.json           per subsystem: what was run and what came out
    data/scores/models/<sub>/<train>.pkl   the fitted winner + its threshold, for /sim/inject

Idempotent: a subsystem whose rows are already in ``scores.parquet`` and whose manifest entry
names the same winner ``config_hash`` (and the same git revision of the scoring code) is
skipped; ``--force`` re-scores it. Skipped subsystems keep their existing rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nebulax import schema as S  # noqa: E402
from nebulax.demo import scoring as sc  # noqa: E402

LOGGER = logging.getLogger("score_for_demo")

SIM_SUBSYSTEMS = ("pneumatic", "bearing")
ALL_SUBSYSTEMS = (*SIM_SUBSYSTEMS, "metropt3")
EXPLICIT_LEGACY_SUBSYSTEMS = ("door",)

#: How ``nebulax.bench.data`` blanks rows from the measured population, per winner. Episodes
#: are detected, and scored train-days counted, on the rows that survive - exactly the
#: population the benchmark measures (see ``demo.scoring.scoreable_from_fault_log``).
SCOREABLE_RULE = {"door": "post_failure", "pneumatic": "post_failure", "bearing": "post_failure",
                  "metropt3": "post_repair"}


#: What the manifest says the numbers mean; both the scoring pass and ``--episodes-only``
#: write it, so a rebuilt manifest can never describe a rule the code no longer applies.
PROTOCOL: dict[str, str] = {
    "sim": "leave-one-train-out (sim_loo_unit over all 10 trains); fit on the other "
    "nine trains' normal-only rows, threshold calibrated on their validation carve, "
    "then every row of the held-out train is scored",
    "metropt3": "metropt_temporal (train Feb-Mar 2020, val 1-10 Apr, test 11 Apr onward)",
    "alert": "row belongs to an alarm episode: score > threshold for >= k_consecutive "
    "rows consecutive in time on one component (a run is cut wherever the step exceeds "
    "episode_max_step_s), qualifying runs merged under merge_gap_s",
    "episodes_measured_on": "the scoreable rows (the sim blanks every row after a "
    "component's t_failure, MetroPT-3 the 24 h after a failure, exactly as the benchmark "
    "does); blanked rows are still written with their score and alert=False",
    "episodes_built_by": "nebulax.bench.metrics.episodes itself, on the scoreable rows, "
    "with the fold's k_consecutive / merge_gap_s / episode_max_step_s - never re-derived "
    "from runs of array-adjacent alert rows",
    "scored_train_days": "metrics.train_days over the SCOREABLE rows, per train, so "
    "false_episodes_per_scored_train_day is comparable with the benchmark's "
    "budget_false_alarms_per_train_day in results/runs.parquet",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(sc.SCORES_DIR), help="output directory (default data/scores)")
    p.add_argument("--sim-root", default=str(sc.SIM_ROOT), help="synthetic fleet root (default data/sim)")
    p.add_argument("--runs", default=str(sc.RUNS_PARQUET), help="benchmark results frame")
    p.add_argument("--cache-dir", default=str(sc.CACHE_DIR), help="loader feature cache ('' disables it)")
    p.add_argument(
        "--subsystems",
        default=",".join(ALL_SUBSYSTEMS),
        help=f"comma-separated subset of {','.join(ALL_SUBSYSTEMS)} (legacy door only when explicit)",
    )
    p.add_argument("--skip-metropt3", action="store_true", help="sim fleet only (no data/raw/metropt3 needed)")
    p.add_argument("--force", action="store_true", help="re-score even when the manifest is up to date")
    p.add_argument(
        "--episodes-only",
        action="store_true",
        help="do not fit or score anything: rebuild episodes.parquet and the manifest's "
        "episode / event / train-day numbers from the existing scores.parquet, metropt3.parquet "
        "and manifest.json (thresholds are per row, the episode definition per subsystem entry)",
    )
    p.add_argument("--no-models", action="store_true", help="do not pickle the fitted winners")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


# --------------------------------------------------------------------------------------


def _read_manifest(out: Path) -> dict[str, Any]:
    path = out / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        LOGGER.warning("manifest.json is unreadable; re-scoring everything")
        return {}


def _up_to_date(manifest: dict[str, Any], key: str, spec: sc.WinnerSpec, out: Path) -> bool:
    entry = (manifest.get("subsystems") or {}).get(key)
    if not entry or str(entry.get("config_hash")) != spec.config_hash:
        return False
    parquet = out / ("metropt3.parquet" if key == "metropt3" else "scores.parquet")
    if not parquet.exists():
        return False
    if key != "metropt3":
        try:
            have = pd.read_parquet(parquet, columns=["subsystem"])["subsystem"].astype(str).unique()
        except Exception:
            return False
        if spec.dataset_subsystem not in set(have):
            return False
    return True


def _percentiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"min": float("nan"), "median": float("nan"), "max": float("nan")}
    return {"min": float(arr.min()), "median": float(np.median(arr)), "max": float(arr.max())}


def score_subsystem(
    key: str,
    *,
    out: Path,
    sim_root: Path,
    runs_path: Path,
    cache_dir: Path | None,
    write_models: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Score one winner over its whole population. Returns ``(scores, episodes, manifest entry)``."""
    t_block = time.perf_counter()
    root = sim_root if key != "metropt3" else None
    fleet = sc.load_fleet(key, root=root, runs_path=runs_path, cache_dir=cache_dir)
    spec = fleet.spec
    LOGGER.info(
        "%s: %s %s (%s w=%s peer_norm=%s) - %d rows, %d fold(s)",
        key,
        spec.model,
        spec.config_hash,
        spec.input_kind,
        spec.window,
        spec.peer_norm,
        len(fleet.data),
        len(fleet.splits),
    )

    held_out = fleet.held_out_trains if key != "metropt3" else [sc.METROPT_TRAIN_ID]
    frames: list[pd.DataFrame] = []
    thresholds: dict[str, float] = {}
    fit_seconds: dict[str, float] = {}
    calibration: dict[str, Any] = {}
    attribution = ""
    max_step_s = float("nan")
    window_seconds = float("nan")

    for train in held_out:
        t0 = time.perf_counter()
        fw = sc.fit_winner(
            key,
            exclude_train=(train if key != "metropt3" else None),
            index=fleet,
        )
        frame = sc.score_held_out(fw, fleet)
        thresholds[train] = float(fw.threshold)
        fit_seconds[train] = float(fw.fit_seconds)
        calibration[train] = fw.calibration
        attribution = fw.attribution
        max_step_s, window_seconds = float(fw.max_step_s), float(fw.window_seconds)
        if write_models:
            sc.save_fitted(fw, scores_dir=out)
        frames.append(frame)
        LOGGER.info(
            "  %-4s threshold=%.6g  rows=%-7d alert=%-6d  fit %.1fs  total %.1fs",
            train,
            fw.threshold,
            len(frame),
            int(frame["alert"].sum()),
            fw.fit_seconds,
            time.perf_counter() - t0,
        )

    frames = [f for f in frames if len(f)]
    scores = S.coerce_scores(pd.concat(frames, ignore_index=True)) if frames else S.empty_scores()

    if key == "metropt3":
        fault_log = _metropt_fault_log(fleet)
    else:
        fault_log = sc.fleet_fault_log(sim_root, subsystem=spec.dataset_subsystem)
    scoreable = sc.scoreable_from_fault_log(scores, fault_log, rule=SCOREABLE_RULE[key])
    episodes = sc.episodes_from_scores(
        scores,
        fault_log,
        H=spec.H,
        k=spec.k_consecutive,
        merge_gap_s=spec.merge_gap_s,
        max_step_s=max_step_s,
        scoreable=scoreable,
    )
    summary = sc.event_summary(episodes, fault_log, scores, H=spec.H, scoreable=scoreable)
    # The mask is re-derived from the fault log so that --episodes-only can reproduce it from
    # a published scores.parquet; cross-check it against the loader's own mask on the rows
    # that were actually scored. MetroPT-3 also blanks `is_transition` windows, which are a
    # property of the raw digital channels and cannot be recovered from a scores frame.
    n_blanked_loader = int(
        sum(int((~fleet.data.masks["scoreable"][np.asarray(sp.test, dtype=np.int64)]).sum()) for sp in fleet.splits)
        if key != "metropt3"
        else (~fleet.data.masks["scoreable"][np.asarray(fleet.splits[0].test, dtype=np.int64)]).sum()
    )

    entry: dict[str, Any] = {
        **spec.as_json(),
        "attribution": attribution,
        "n_folds": len(held_out),
        "held_out_trains": list(held_out),
        "thresholds": thresholds,
        "threshold_range": _percentiles(list(thresholds.values())),
        "fit_seconds": fit_seconds,
        "fit_seconds_total": float(sum(fit_seconds.values())),
        "calibration": calibration,
        "n_rows_scored": int(len(scores)),
        "n_rows_alert": int(scores["alert"].sum()) if len(scores) else 0,
        "n_rows_in_table": int(len(fleet.data)),
        # Rows written with their score but excluded from episode detection, exactly as the
        # benchmark excludes them from its test measurement (see protocol.episodes_measured_on).
        "n_rows_not_scoreable": int((~fleet.data.masks["scoreable"]).sum()),
        # Of the rows written to scores.parquet (the test folds), how many the loader blanked
        # and how many the fault-log rule re-derives. They must agree for the sim; on
        # MetroPT-3 the difference is the `is_transition` windows.
        "n_rows_blanked_by_loader": n_blanked_loader,
        "n_rows_blanked_by_rule": int((~scoreable).sum()),
        "scoreable_rule": SCOREABLE_RULE[key],
        # `alert` and episodes.parquet are two views of one metrics.episodes call; 0 says so.
        "n_alert_rows_outside_episodes": _alert_rows_outside_episodes(scores, episodes),
        # The contiguity budget and row duration MEASURED on this table (20 runs), beside the
        # frozen values the 8-run sweep published - see demo.scoring.fit_winner.
        "episode_max_step_s": max_step_s,
        "row_window_seconds": window_seconds,
        **summary,
        "elapsed_s": float(time.perf_counter() - t_block),
    }
    LOGGER.info(
        "%s: %d rows, %d episode(s) (%d matched / %d false), events %d/%d, FA/scored-day %.4f, %.1fs",
        key,
        entry["n_rows_scored"],
        entry["episodes_total"],
        entry["episodes_matched"],
        entry["episodes_false"],
        entry["events_detected"],
        entry["events_total"],
        entry["false_episodes_per_scored_train_day"],
        entry["elapsed_s"],
    )
    return scores, episodes, entry


def _metropt_fault_log(fleet: sc.FleetIndex | None) -> pd.DataFrame:
    """MetroPT-3's four air-leak episodes as a fault-log frame keyed like the scores.

    The loader carries them as ``BenchData.events`` (epoch seconds), so they are turned back
    into rows on ``(MP3, car 0, apu_1)`` rather than re-read from the adapter. ``fleet=None``
    (the ``--episodes-only`` path, which loads no recording) falls back to the adapter's own
    transcription of the UCI failure-report table, which is where the loader's events come
    from in the first place.
    """
    if fleet is None:
        from nebulax.adapters import metropt3 as A  # noqa: PLC0415 - only the fallback needs it

        fl = A._build_fault_log().copy()
        fl["train_id"] = sc.METROPT_TRAIN_ID
        fl["car"] = sc.METROPT_CAR
        fl["component_id"] = sc.METROPT_COMPONENT
        fl["t_onset"] = pd.to_datetime(fl["t_onset"], utc=True)
        fl["t_failure"] = pd.to_datetime(fl["t_failure"], utc=True)
        return fl.reset_index(drop=True)
    rows = []
    for ev in fleet.data.events:
        rows.append(
            {
                "train_id": sc.METROPT_TRAIN_ID,
                "car": sc.METROPT_CAR,
                "subsystem": fleet.spec.dataset_subsystem,
                "component_id": sc.METROPT_COMPONENT,
                "fault_type": ev.fault_type,
                "t_onset": pd.Timestamp(ev.t_onset, unit="s", tz="UTC"),
                "t_failure": pd.Timestamp(ev.t_failure, unit="s", tz="UTC")
                if np.isfinite(ev.t_failure)
                else pd.NaT,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# --episodes-only: re-derive episodes.parquet from what is already on disk
# --------------------------------------------------------------------------------------


def _published_scores(key: str, entry: dict[str, Any], out: Path, sim_scores: pd.DataFrame | None) -> pd.DataFrame:
    """The rows on disk for one winner: ``metropt3.parquet``, or this subsystem's sim rows."""
    if key == "metropt3":
        frame = _read_parquet(out / "metropt3.parquet")
        return S.empty_scores() if frame is None else frame
    if sim_scores is None or not len(sim_scores):
        return S.empty_scores()
    sub = str(entry["dataset_subsystem"])
    # MetroPT-3's rows also say subsystem="pneumatic": they are told apart by train_id.
    sel = (sim_scores["subsystem"].astype(str) == sub) & (
        sim_scores["train_id"].astype(str) != sc.METROPT_TRAIN_ID
    )
    return sim_scores[sel].reset_index(drop=True)


def _alert_rows_outside_episodes(scores: pd.DataFrame, episodes: pd.DataFrame) -> int:
    """How many ``alert=True`` rows no episode row covers.

    ``alert`` and ``episodes.parquet`` are two views of one ``metrics.episodes`` call, so this
    is 0 whenever the episode definition the entry carries is the one the column was written
    with. A non-zero count means the two would disagree and is recorded rather than hidden.
    """
    if not len(scores) or not len(episodes):
        return int(scores["alert"].astype(bool).sum()) if len(scores) else 0
    ts = pd.to_datetime(scores["timestamp"], utc=True).to_numpy()
    covered = np.zeros(len(scores), dtype=bool)
    key = sc._series_key(scores)
    e_key = sc._series_key(episodes)
    e_start = pd.to_datetime(episodes["t_start"], utc=True).to_numpy()
    e_end = pd.to_datetime(episodes["t_end"], utc=True).to_numpy()
    for i in range(len(episodes)):
        covered |= (key == e_key[i]) & (ts >= e_start[i]) & (ts <= e_end[i])
    return int((scores["alert"].to_numpy(dtype=bool) & ~covered).sum())


def rebuild_episodes(
    key: str, *, entry: dict[str, Any], out: Path, sim_root: Path, sim_scores: pd.DataFrame | None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One winner's episodes and event numbers, re-derived from the published scores.

    Nothing is fitted and nothing is loaded from ``data/sim`` beyond the fault log: the
    per-row thresholds are in ``scores.parquet`` and the episode definition
    (``k_consecutive`` / ``merge_gap_s`` / ``episode_max_step_s``) is in the manifest entry
    the scoring run wrote.
    """
    t0 = time.perf_counter()
    scores = _published_scores(key, entry, out, sim_scores)
    fault_log = (
        _metropt_fault_log(None)
        if key == "metropt3"
        else sc.fleet_fault_log(sim_root, subsystem=str(entry["dataset_subsystem"]))
    )
    H = float(entry["H"])
    max_step_s = float(entry.get("episode_max_step_s") or entry["max_step_s"])
    scoreable = sc.scoreable_from_fault_log(scores, fault_log, rule=SCOREABLE_RULE[key])
    episodes = sc.episodes_from_scores(
        scores,
        fault_log,
        H=H,
        k=int(entry["k_consecutive"]),
        merge_gap_s=float(entry["merge_gap_s"]),
        max_step_s=max_step_s,
        scoreable=scoreable,
    )
    summary = sc.event_summary(episodes, fault_log, scores, H=H, scoreable=scoreable)
    new_entry = {
        **entry,
        **summary,
        "n_rows_scored": int(len(scores)),
        "n_rows_alert": int(scores["alert"].sum()) if len(scores) else 0,
        "n_rows_blanked_by_rule": int((~scoreable).sum()),
        "scoreable_rule": SCOREABLE_RULE[key],
        "n_alert_rows_outside_episodes": _alert_rows_outside_episodes(scores, episodes),
        "episodes_rebuilt_at": pd.Timestamp.utcnow().isoformat(),
        "episodes_rebuild_s": float(time.perf_counter() - t0),
    }
    LOGGER.info(
        "%s: %d rows, %d episode(s) (%d matched / %d false), events %d/%d, "
        "%.2f scored train-days, FA/scored-day %.4f, %.1fs",
        key,
        new_entry["n_rows_scored"],
        new_entry["episodes_total"],
        new_entry["episodes_matched"],
        new_entry["episodes_false"],
        new_entry["events_detected"],
        new_entry["events_total"],
        new_entry["scored_train_days"],
        new_entry["false_episodes_per_scored_train_day"],
        new_entry["episodes_rebuild_s"],
    )
    return episodes, new_entry


def episodes_only(*, out: Path, sim_root: Path, wanted: list[str], manifest: dict[str, Any]) -> int:
    entries: dict[str, Any] = dict(manifest.get("subsystems") or {})
    todo = [k for k in wanted if k in entries]
    missing = [k for k in wanted if k not in entries]
    if missing:
        LOGGER.warning("--episodes-only: no manifest entry for %s; leaving those rows alone", ", ".join(missing))
    if not todo:
        raise SystemExit(
            f"score_for_demo --episodes-only: {out / 'manifest.json'} has no entry for any of "
            f"{wanted}; run the scoring pass first"
        )
    sim_scores = _read_parquet(out / "scores.parquet") if any(k != "metropt3" for k in todo) else None

    eps_parts: list[pd.DataFrame] = []
    old_episodes = _read_parquet(out / "episodes.parquet")
    rebuilt_subs = {str(entries[k]["dataset_subsystem"]) for k in todo if k != "metropt3"}
    if old_episodes is not None and len(old_episodes):
        is_mp3 = old_episodes["train_id"].astype(str) == sc.METROPT_TRAIN_ID
        drop = old_episodes["subsystem"].astype(str).isin(rebuilt_subs) & ~is_mp3
        if "metropt3" in todo:
            drop = drop | is_mp3
        kept = old_episodes[~drop]
        if len(kept):
            eps_parts.append(kept)

    for key in todo:
        episodes, entries[key] = rebuild_episodes(
            key, entry=entries[key], out=out, sim_root=sim_root, sim_scores=sim_scores
        )
        if len(episodes):
            eps_parts.append(episodes)

    eps_all = pd.concat(eps_parts, ignore_index=True) if eps_parts else sc.empty_episodes()
    _write(eps_all.reset_index(drop=True), out / "episodes.parquet")
    manifest_out = {**manifest, "protocol": dict(PROTOCOL), "subsystems": entries}
    manifest_out["episodes_regenerated_at"] = pd.Timestamp.utcnow().isoformat()
    manifest_out["episodes_regenerated_from"] = "scores.parquet + metropt3.parquet (--episodes-only)"
    (out / "manifest.json").write_text(json.dumps(manifest_out, indent=2, default=str), encoding="utf-8")
    print(_summary_table(entries, wanted))
    return 0


# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sim_root = Path(args.sim_root)
    runs_path = Path(args.runs)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    wanted = [s.strip() for s in args.subsystems.split(",") if s.strip()]
    accepted = (*ALL_SUBSYSTEMS, *EXPLICIT_LEGACY_SUBSYSTEMS)
    unknown = [s for s in wanted if s not in accepted]
    if unknown:
        raise SystemExit(f"score_for_demo: unknown subsystem(s) {unknown}; expected {list(accepted)}")
    if args.skip_metropt3:
        wanted = [s for s in wanted if s != "metropt3"]

    manifest = _read_manifest(out)
    if args.episodes_only:
        return episodes_only(out=out, sim_root=sim_root, wanted=wanted, manifest=manifest)

    winners = sc.load_winners(runs_path, keys=wanted)
    entries: dict[str, Any] = dict(manifest.get("subsystems") or {})

    todo = [k for k in wanted if args.force or not _up_to_date(manifest, k, winners[k], out)]
    skipped = [k for k in wanted if k not in todo]
    if skipped:
        LOGGER.info("up to date, skipping: %s (use --force to re-score)", ", ".join(skipped))
    if not todo:
        print(_summary_table(entries, wanted))
        return 0

    sim_scores: list[pd.DataFrame] = []
    sim_episodes: list[pd.DataFrame] = []
    metro_scores: pd.DataFrame | None = None
    metro_episodes: pd.DataFrame | None = None

    for key in todo:
        scores, episodes, entry = score_subsystem(
            key,
            out=out,
            sim_root=sim_root,
            runs_path=runs_path,
            cache_dir=cache_dir,
            write_models=not args.no_models,
        )
        entries[key] = entry
        if key == "metropt3":
            metro_scores, metro_episodes = scores, episodes
        else:
            sim_scores.append(scores)
            sim_episodes.append(episodes)

    # --- merge with whatever is on disk for the subsystems we did not re-score ---
    # A re-scored sim subsystem replaces its own rows and nothing else. MetroPT-3's rows also
    # say subsystem="pneumatic", so they are told apart by train_id, never by subsystem.
    rescored = {winners[k].dataset_subsystem for k in todo if k != "metropt3"}
    old_scores = _read_parquet(out / "scores.parquet")
    old_episodes = _read_parquet(out / "episodes.parquet")

    if sim_scores:
        parts: list[pd.DataFrame] = []
        if old_scores is not None and len(old_scores):
            parts.append(old_scores[~old_scores["subsystem"].astype(str).isin(rescored)])
        parts.extend(sim_scores)
        # Empty frames are dropped before the concat, not by it: pandas' own handling of
        # empty / all-NA entries is deprecated and the project turns FutureWarning into an error.
        parts = [f for f in parts if len(f)]
        merged = S.coerce_scores(pd.concat(parts, ignore_index=True)) if parts else S.empty_scores()
        _write(merged, out / "scores.parquet")

    if metro_scores is not None:
        _write(metro_scores, out / "metropt3.parquet")

    eps_parts: list[pd.DataFrame] = []
    if old_episodes is not None and len(old_episodes):
        is_mp3 = old_episodes["train_id"].astype(str) == sc.METROPT_TRAIN_ID
        drop = old_episodes["subsystem"].astype(str).isin(rescored) & ~is_mp3
        if metro_episodes is not None:
            drop = drop | is_mp3
        eps_parts.append(old_episodes[~drop])
    eps_parts.extend(e for e in sim_episodes if len(e))
    if metro_episodes is not None and len(metro_episodes):
        eps_parts.append(metro_episodes)
    eps_parts = [e for e in eps_parts if len(e)]
    eps_all = pd.concat(eps_parts, ignore_index=True) if eps_parts else sc.empty_episodes()
    _write(eps_all.reset_index(drop=True), out / "episodes.parquet")

    manifest_out = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "git_rev": sc.git_rev(),
        "runs_parquet": str(runs_path),
        "sim_root": str(sim_root),
        "protocol": dict(PROTOCOL),
        "files": {
            "scores": "scores.parquet",
            "metropt3": "metropt3.parquet",
            "episodes": "episodes.parquet",
            "models": "models/<subsystem>/<held_out_train>.pkl",
        },
        "subsystems": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest_out, indent=2, default=str), encoding="utf-8")
    print(_summary_table(entries, wanted))
    return 0


def _read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("could not read %s (%s); it will be rewritten", path, exc)
        return None


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    LOGGER.info("wrote %s (%d rows)", path, len(frame))


def _summary_table(entries: dict[str, Any], keys: list[str]) -> str:
    head = (
        f"{'subsystem':<10} {'model':<20} {'rows':>9} {'thr min':>12} {'thr max':>12} "
        f"{'episodes':>9} {'matched':>8} {'false':>6} {'events':>8} {'train-days':>11} "
        f"{'FA/day':>8} {'fit s':>7}"
    )
    lines = [head, "-" * len(head)]
    for key in keys:
        e = entries.get(key)
        if not e:
            continue
        rng = e.get("threshold_range", {})
        lines.append(
            f"{key:<10} {str(e['model']):<20} {e['n_rows_scored']:>9,} "
            f"{rng.get('min', float('nan')):>12.5g} {rng.get('max', float('nan')):>12.5g} "
            f"{e['episodes_total']:>9} {e['episodes_matched']:>8} {e['episodes_false']:>6} "
            f"{str(e['events_detected']) + '/' + str(e['events_total']):>8} "
            f"{e['scored_train_days']:>11.2f} "
            f"{e['false_episodes_per_scored_train_day']:>8.4f} {e['fit_seconds_total']:>7.1f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
