"""Verifier-added regression tests for ``nebulax.demo.scoring.episodes_from_scores``.

Added by the W3 scoring verifier, and kept green by fix round 1: the episode rows are now
built by :func:`nebulax.bench.metrics.episodes` itself, so a gap wider than ``max_step_s``
splits a run and a gap of ``merge_gap_s`` or more is never bridged. Synthetic frames only:
no ``data/`` access, no network, no model fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nebulax import schema as S
from nebulax.bench import metrics as M
from nebulax.demo import scoring as sc


def _scores(times: pd.DatetimeIndex, values: np.ndarray, threshold: float, alert: np.ndarray) -> pd.DataFrame:
    return S.coerce_scores(
        pd.DataFrame(
            {
                "timestamp": times,
                "train_id": "T01",
                "car": np.int16(1),
                "subsystem": "door",
                "component_id": "door_L1",
                "model": "cusum_cycle_scalar",
                "score": values.astype(np.float64),
                "threshold": float(threshold),
                "alert": alert,
                "top_signals_json": ["[]"] * len(values),
            }
        )
    )


def test_episodes_do_not_span_a_gap_wider_than_merge_gap_s():
    """Two alarm blocks either side of a 5 h out-of-service gap are two episodes.

    The two blocks are adjacent ROWS - a door raises no cycle overnight - so any rule that
    reads maximal runs of ``alert=True`` off the array returns one episode where
    ``nebulax.bench.metrics.episodes`` (the definition docs/app_contract.md section 3 names)
    returns two.
    """
    step = 180.0  # a door cycle every 3 min
    day1 = pd.date_range("2026-09-11T20:00:00Z", periods=10, freq=f"{int(step)}s")
    day2 = pd.date_range("2026-09-12T05:30:00Z", periods=10, freq=f"{int(step)}s")
    times = day1.append(day2)
    values = np.full(len(times), 5.0)
    thr = 1.0
    alert = sc._alert_flags(
        sc.FittedWinner(
            subsystem="door",
            model_name="cusum_cycle_scalar",
            params={},
            input_kind="cycle_features",
            window=None,
            peer_norm=True,
            threshold=thr,
            feature_names=[],
            train_median=np.zeros(0),
            train_mad=np.zeros(0),
            held_out_train="T01",
            fitted_at="",
            k_consecutive=3,
            merge_gap_s=3600.0,
            max_step_s=240.0,
        ),
        values,
        times.to_numpy(),
        np.full(len(times), "T01/door_L1", dtype=object),
    )
    assert alert.all()

    truth = M.episodes(values, times.to_numpy(), thr, k=3, merge_gap_s=3600.0, max_step_s=240.0)
    assert len(truth) == 2, "the contract's episode definition splits the 5 h gap"

    eps = sc.episodes_from_scores(_scores(times, values, thr, alert), None)
    assert len(eps) == len(truth), (
        f"episodes.parquet would carry {len(eps)} episode(s) where "
        f"nebulax.bench.metrics.episodes finds {len(truth)}"
    )


def test_published_episode_rows_have_no_internal_gap_over_the_merge_gap():
    """Every episode's own rows must be within merge_gap_s of each other."""
    step = 180.0
    day1 = pd.date_range("2026-09-11T20:00:00Z", periods=10, freq=f"{int(step)}s")
    day2 = pd.date_range("2026-09-12T05:30:00Z", periods=10, freq=f"{int(step)}s")
    times = day1.append(day2)
    values = np.full(len(times), 5.0)
    scores = _scores(times, values, 1.0, np.ones(len(times), dtype=bool))
    eps = sc.episodes_from_scores(scores, None)
    for _, row in eps.iterrows():
        sel = scores[(scores["timestamp"] >= row["t_start"]) & (scores["timestamp"] <= row["t_end"])]
        gaps = np.diff(M.to_epoch_seconds(sel["timestamp"].to_numpy()))
        assert gaps.size == 0 or gaps.max() < 3600.0, (
            f"episode {row['episode_id']} spans an internal gap of {gaps.max():.0f} s "
            f"(merge_gap_s = 3600 s)"
        )
