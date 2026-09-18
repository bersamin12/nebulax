"""Tests for nebulax.adapters.metropt3 against a small real slice of the UCI 791 CSV, plus
synthetic-trace tests (no raw data needed) for the loaded/off feature definitions and the
transition mask.

The real-data tests skip (do not fail) when the raw file has not been downloaded -- see
``scripts/download_data.py --dataset metropt3``. Runs on a short prefix of the real file
(a few thousand rows) so the whole module stays well inside 60 s; the full-file run is
exercised manually via ``python -m nebulax.adapters.validate --source metropt3``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nebulax import schema as S
from nebulax.adapters import metropt3 as M
from nebulax.adapters import validate as V
from nebulax.sim import pneumatic as P

RAW_CSV = Path("data/raw/metropt3") / M._CSV_NAME
N_SLICE = 6000  # ~40 compressor cycles at this file's ~150-row average cycle length

# Applied per-test (not as a module-level `pytestmark`) so the synthetic-trace tests below, which
# need no raw data, always run.
_needs_raw = pytest.mark.skipif(not RAW_CSV.exists(), reason=f"raw data not present at {RAW_CSV}")


@pytest.fixture(scope="module")
def slice_dir(tmp_path_factory) -> Path:
    """A tiny prefix of the real CSV, in its own directory, standing in for raw_dir."""
    out = tmp_path_factory.mktemp("metropt3_slice")
    df = pd.read_csv(RAW_CSV, nrows=N_SLICE)
    df.to_csv(out / M._CSV_NAME, index=False)
    return out


@pytest.fixture(scope="module")
def dataset(slice_dir: Path) -> S.Dataset:
    return M.load(slice_dir)


@_needs_raw
def test_load_missing_raw_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="metropt3 adapter"):
        M.load(tmp_path / "nope")


@_needs_raw
def test_load_missing_csv_in_existing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match=M._CSV_NAME.replace("(", r"\(").replace(")", r"\)")):
        M.load(tmp_path)


@_needs_raw
def test_returns_dataset_and_validates(dataset: S.Dataset):
    assert isinstance(dataset, S.Dataset)
    dataset.validate()  # raises on any schema violation


@_needs_raw
def test_long_table_shape_and_signals(dataset: S.Dataset):
    long = dataset.long
    assert set(long["source"].astype("string")) == {"metropt3"}
    assert set(long["subsystem"].astype("string")) == {"pneumatic"}
    assert set(long["component_id"].astype("string")) == {"apu_1"}
    assert (long["car"].to_numpy() == 0).all()
    present = set(long["signal"].astype("string"))
    assert present == set(S.METROPT3_SIGNALS)  # all 15, verbatim names, no Flowmeter
    assert "Flowmeter" not in present
    assert "DV_eletric" in present  # the UCI typo, kept verbatim
    assert len(long) == N_SLICE * len(S.METROPT3_SIGNALS)
    assert long["value"].isna().sum() == 0  # this slice has no dropouts
    assert np.isfinite(long["value"].to_numpy(dtype=np.float32)).all()


@_needs_raw
def test_fault_log_hard_coded_from_paper(dataset: S.Dataset):
    fl = dataset.fault_log
    assert len(fl) == 4
    assert set(fl["fault_type"].astype("string")) == {"air_leak"}  # MetroPT-3: air leaks only
    assert (fl["t_onset"] <= fl["t_failure"]).all()
    assert fl["params_json"].apply(lambda s: "verify_dates" in s).all()


@_needs_raw
def test_cycle_feature_table(dataset: S.Dataset):
    feats = dataset.features
    assert len(feats) > 0
    for col in (
        "t_loaded",
        "t_unloaded",
        "t_off",
        "duty_ratio",
        "idle_run_ratio",
        "hour_of_day",
        "TP2_minus_TP3_mean",
        "H1_loaded_mean",
        "transition_frac",
    ):
        assert col in feats.columns
    assert (feats["cycle_id"].to_numpy() == np.arange(len(feats))).all()
    assert (feats["t_end"] >= feats["t_start"]).all()
    ratio = feats["duty_ratio"].dropna().to_numpy()
    assert ((ratio >= 0) & (ratio <= 1)).all()
    # this slice (2020-02-01, first minutes) predates every hard-coded fault window
    assert not feats["is_faulty"].any()
    assert set(feats["fault_type"].astype("string")) <= {"healthy"}
    # loaded-only mean: real MetroPT-3 loaded TP2-TP3 is ~+0.3 bar (docs/parameters.md pneumatic
    # section 5). The pre-fix whole-cycle average was ~-7.9 bar (TP2 ~0 off/unloaded dominates a
    # cycle that is mostly not loaded), so this bounds check is a regression guard for the fix.
    tp2_tp3 = feats["TP2_minus_TP3_mean"].dropna().to_numpy()
    assert len(tp2_tp3) > 0
    assert (tp2_tp3 > -1.0).all() and (tp2_tp3 < 3.0).all()
    tf = feats["transition_frac"].dropna().to_numpy()
    assert ((tf >= 0.0) & (tf <= 1.0)).all()


@_needs_raw
def test_events_use_known_vocabulary(dataset: S.Dataset):
    events = dataset.events
    assert len(events) > 0
    assert set(events["event"].astype("string")) <= set(S.EVENT_TYPES)
    assert "comp_load" in set(events["event"].astype("string"))


@_needs_raw
def test_meta_carries_provenance(dataset: S.Dataset):
    meta = dataset.meta
    assert meta["doi"] == "10.24432/C5VW3R"
    assert "CC BY 4.0" in meta["licence"]
    assert meta["n_rows_raw"] == N_SLICE


@_needs_raw
def test_validate_cli_passes_on_slice(slice_dir: Path, capsys):
    rc = V.main(["--source", "metropt3", "--raw", str(slice_dir)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK: metropt3 passes the nebulax schema." in out


@_needs_raw
def test_validate_cli_strict_passes_on_slice(slice_dir: Path, capsys):
    rc = V.main(["--source", "metropt3", "--raw", str(slice_dir), "--strict", "--quiet"])
    assert rc == 0


# ---------------------------------------------------------------------- synthetic-trace tests
#
# No raw data needed: a short, hand-built 1 Hz trace covering one full compressor cycle (off ->
# loaded -> unloaded -> off) plus one sample of the next cycle's start (so the first cycle has a
# closed boundary). Exercises the loaded/off feature definitions table in
# ``nebulax.adapters.metropt3._build_cycle_features``'s docstring and cross-checks it against
# ``nebulax.sim.pneumatic``'s own masked-mean helper on the SAME samples.


def _synthetic_trace(n: int = 81) -> pd.DataFrame:
    """One 1 Hz compressor cycle: off[0:5), loaded[5:35), unloaded[35:65), off[65:80), then one
    sample (index 80) that starts the next cycle -- so cycle 0 is exactly indices [5, 80).

    Values are constant within each phase (aside from ``Reservoirs``) so the loaded-only and
    off-only means have an exact, hand-computable answer, and so ``I_loaded_mean``-style
    "drop the first few samples" nuances (present only in the simulator's current-transient
    handling, not in TP2/H1) cannot perturb the comparison.
    """
    t = pd.date_range("2024-01-01T00:00:00Z", periods=n, freq="1s")
    motor = np.full(n, 0.5)  # off: well under _I_OFF_MAX
    motor[5:35] = 6.0  # loaded: well over _I_LOADED_MIN
    motor[35:65] = 3.8  # unloaded: between the two thresholds
    motor[80] = 6.0  # next cycle's load start

    tp2 = np.zeros(n)
    tp2[5:35] = 9.3  # discharge pressure, live only while loaded
    tp3 = np.full(n, 8.7)
    tp3[5:35] = 9.0  # panel pressure while loaded: TP2 - TP3 = 0.3 bar
    tp3[35:65] = 9.6
    tp3[80] = 9.3
    h1 = np.full(n, 8.7)  # separator tap: follows panel pressure off/unloaded
    h1[5:35] = 0.0  # ~0 while loaded (vented)
    h1[35:65] = 9.6
    h1[80] = 9.3

    towers = np.zeros(n)
    towers[20:] = 1.0  # one flip mid-loaded-phase, at index 20

    return pd.DataFrame(
        {
            "timestamp": t,
            "Motor_current": motor,
            "Reservoirs": np.linspace(8.06, 10.2, n),
            "Oil_temperature": np.full(n, 58.0),
            "TP2": tp2,
            "TP3": tp3,
            "H1": h1,
            "LPS": np.zeros(n),
            "Towers": towers,
            "Pressure_switch": np.zeros(n),
        }
    )


def _multi_cycle_trace(n: int = 161) -> pd.DataFrame:
    """Three 1 Hz compressor cycles (load starts at 5, 80, 155; the third is left incomplete by
    the trace ending at 161, exactly like a real trailing cycle), each ``loaded[0:30) ->
    unloaded[30:60) -> off[60:75)`` relative to its own start, with:

    * a 2-sample 12 A ``Motor_current`` inrush at the very start of each loaded run (so
      ``I_loaded_mean`` differs between "average every loaded sample" and "drop the first 4
      samples of the cycle", exercising the exact fix in (a));
    * a higher, 15 A ``Motor_current`` spike 20 samples into cycle 1's loaded run (so
      ``I_start_peak`` differs between "max over the first 3 samples" and "max over the whole
      cycle" -- the delta the audit found: on the real file the two disagree on 97.9 % of
      cycles because the current maximum is not always at cycle start);
    * one ``Towers`` flip and one ``Pressure_switch`` purge pulse placed well inside cycle 0's
      unloaded phase (away from every cycle boundary), PLUS a second flip and a second purge
      pulse landing exactly on cycle 1's own first sample -- the boundary case the audit found:
      the adapter used to drop a flip/pulse there while the simulator's plain reduceat sum
      counts it (7.8 % of the real file's ``Towers`` flips land exactly on a cycle start); and
    * a linear ``Reservoirs`` ramp across the whole trace, so ``dP_dt_loaded`` / ``dP_dt_off``
      have a well-defined, hand-checkable answer under both the old fencepost-prone
      ``(last - first) / duration`` formula and the fixed per-sample-rate one (for a uniform
      1 Hz cadence the two coincide once both sides use the same convention).
    """
    t = pd.date_range("2024-01-01T00:00:00Z", periods=n, freq="1s")
    motor = np.full(n, 0.5)  # off
    tp2 = np.zeros(n)
    tp3 = np.full(n, 8.7)
    h1 = np.full(n, 8.7)
    towers = np.zeros(n)
    pressure_switch = np.zeros(n)

    for c in (5, 80, 155):
        loaded = slice(c, min(c + 30, n))
        unloaded = slice(c + 30, min(c + 60, n))
        off = slice(c + 60, min(c + 75, n))
        motor[loaded] = 6.0
        motor[unloaded] = 3.8
        motor[off] = 0.5
        if c + 1 < n:
            motor[c], motor[c + 1] = 12.0, 12.0  # starting-current inrush
        tp2[loaded] = 9.3
        tp3[loaded] = 9.0  # TP2 - TP3 = 0.3 bar while loaded
        h1[loaded] = 0.0  # vented while loaded
        tp3[unloaded] = 9.6
        h1[unloaded] = 9.6
    if 100 < n:
        motor[100] = 15.0  # cycle 1 (starts at 80), pos_in_cycle 20: a later, higher spike

    towers[50:65] = 1.0  # one flip up at 50, back down at 65 -- inside cycle 0, off every boundary
    towers[80:] = 1.0  # a second flip, landing exactly on cycle 1's first sample (its boundary)
    pressure_switch[45] = 1.0  # one purge pulse, inside cycle 0's unloaded phase
    pressure_switch[80] = 1.0  # a second purge pulse, landing exactly on cycle 1's first sample

    return pd.DataFrame(
        {
            "timestamp": t,
            "Motor_current": motor,
            "Reservoirs": np.linspace(8.0, 10.5, n),
            "Oil_temperature": np.linspace(50.0, 70.0, n),
            "TP2": tp2,
            "TP3": tp3,
            "H1": h1,
            "LPS": np.zeros(n),
            "Towers": towers,
            "Pressure_switch": pressure_switch,
        }
    )


# Shared loaded/off-restricted columns the adapter emits and the simulator can also produce from
# the same arrays via P._cycle_features (see both functions' docstring tables). Excludes columns
# either side does not emit (e.g. Flowmeter_max, load_frac_mean, in_service_frac -- sim only, no
# Flowmeter/service-schedule concept in the real file; or cycle_id/t_start/t_end/run_id/... --
# adapter only, table plumbing) and hour_of_day, which is NOT claimed identical (the adapter
# truncates to whole minutes, the simulator keeps fractional seconds -- a real, harmless
# difference, not asserted here).
_SHARED_CYCLE_COLUMNS: tuple[str, ...] = (
    "t_loaded",
    "t_unloaded",
    "t_off",
    "I_loaded_mean",
    "I_start_peak",
    "T_oil_max",
    "TP2_minus_TP3_mean",
    "H1_loaded_mean",
    "dP_dt_loaded",
    "dP_dt_off",
    "tower_switches",
    "purge_count",
    "LPS_any",
    "idle_run_ratio",
    "duty_ratio",
)


def test_loaded_off_features_match_simulator():
    """The adapter's ``*_loaded_*`` / ``*_off_*`` cycle features must agree with
    ``nebulax.sim.pneumatic._cycle_features`` itself (not a hand-picked helper) on every column
    the docstring table in ``M._build_cycle_features`` claims is "kept identical": this is the
    defect (a) fix, including the ``I_loaded_mean`` starting-current exclusion and the
    ``dP_dt_loaded`` / ``dP_dt_off`` fencepost, not just ``TP2_minus_TP3_mean`` / ``H1_loaded_mean``.
    """
    df = _multi_cycle_trace()
    state = M._classify_state(df["Motor_current"].to_numpy())
    dt = M._sample_durations_s(df["timestamp"])
    is_trans = M.transition_mask(df)
    feats = M._build_cycle_features(df, state, dt, is_trans)
    # both complete cycles (the third is left incomplete by the trace ending mid-cycle)
    assert feats["cycle_id"].tolist()[:2] == [0, 1]

    # exact hand-computed loaded-only values (constant within the loaded phase, once the inrush
    # is excluded from I_loaded_mean the way the simulator excludes it)
    assert feats.iloc[0]["TP2_minus_TP3_mean"] == pytest.approx(0.3)
    assert feats.iloc[0]["H1_loaded_mean"] == pytest.approx(0.0)
    assert feats.iloc[0]["I_loaded_mean"] == pytest.approx(6.0)
    assert feats.iloc[0]["I_start_peak"] == pytest.approx(12.0)

    # regression guard for the (2) audit finding: I_start_peak must be the max over the WHOLE
    # cycle (cycle 1 has a 15 A spike 20 samples in, well past "the first 3 samples"), and
    # tower_switches / purge_count must count a flip/pulse landing on the cycle's own first
    # sample (cycle 1's boundary case) rather than dropping it.
    assert feats.iloc[1]["I_start_peak"] == pytest.approx(15.0)
    assert feats.iloc[1]["tower_switches"] == 1
    assert feats.iloc[1]["purge_count"] == 1

    # regression guard for the actual (a) bug: the pre-fix whole-cycle TP2_minus_TP3_mean would
    # have been strongly negative here (TP2 ~= 0 dominates while off/unloaded).
    tp2, tp3 = df["TP2"].to_numpy(), df["TP3"].to_numpy()
    whole_cycle_mean = np.mean((tp2 - tp3)[5:80])
    assert whole_cycle_mean < -1.0
    assert feats.iloc[0]["TP2_minus_TP3_mean"] != pytest.approx(whole_cycle_mean)

    # cross-check against the simulator's OWN _cycle_features, called on the same signals, with
    # the adapter's 3-way state remapped onto the simulator's 5-way enum (LOADED/UNLOADED/OFF
    # cover the same ground as the adapter's loaded/unloaded/off for a cycle that -- like every
    # cycle in this trace -- starts directly in the loaded state, matching the simulator's own
    # "each load command starts a cycle" convention).
    sim_state = np.select([state == 0, state == 1, state == 2], [P.OFF, P.UNLOADED, P.LOADED])
    towers = df["Towers"].to_numpy()
    pressure_switch = df["Pressure_switch"].to_numpy()
    tower_flip = np.r_[False, towers[1:] != towers[:-1]]
    purge_start = np.r_[False, (pressure_switch[1:] > 0.5) & (pressure_switch[:-1] <= 0.5)]
    t_sec = np.arange(len(df), dtype=np.float64)
    sig = {
        "Reservoirs": df["Reservoirs"].to_numpy(),
        "Motor_current": df["Motor_current"].to_numpy(),
        "Oil_temperature": df["Oil_temperature"].to_numpy(),
        "TP2": tp2,
        "TP3": tp3,
        "H1": df["H1"].to_numpy(),
        "LPS": df["LPS"].to_numpy(),
        "Flowmeter": np.zeros(len(df)),  # not a real MetroPT-3 channel; sim-only requirement
    }
    load_frac = np.full(len(df), 0.5)
    in_service = np.ones(len(df), dtype=bool)
    bounds = np.array([5, 80, 155], dtype=np.int64)
    sim_feats = P._cycle_features(
        t_sec, sig, sim_state, tower_flip, purge_start, load_frac, in_service, bounds, dt=1.0
    )

    for col in _SHARED_CYCLE_COLUMNS:
        adapter_vals = feats[col].to_numpy()[:2].astype(np.float64)
        sim_vals = np.asarray(sim_feats[col][:2], dtype=np.float64)
        assert adapter_vals == pytest.approx(sim_vals, abs=1e-9), (
            f"{col}: adapter {adapter_vals} != simulator {sim_vals} -- the docstring table in "
            f"M._build_cycle_features claims these are kept identical"
        )


def test_transition_mask_flags_state_changes_and_tower_flips():
    df = _synthetic_trace()
    is_trans = M.transition_mask(df, window_s=5.0)
    assert is_trans.dtype == np.bool_
    assert len(is_trans) == len(df)

    # right at (or within 5 s of) a state change / tower flip
    for i in (5, 20, 35, 65, 80):
        assert is_trans[i], f"index {i} should be flagged as a transition"

    # far (> 5 s) from every state change (5, 65, 80) and the tower flip (20)
    for i in (48, 52):
        assert not is_trans[i], f"index {i} should NOT be flagged as a transition"

    # default window_s matches rail_phm.md's own number ("exclude +/-5 s ...")
    assert M._TRANSITION_WINDOW_S == pytest.approx(5.0)
    assert np.array_equal(is_trans, M.transition_mask(df))

    # a wider window flags a superset
    wide = M.transition_mask(df, window_s=20.0)
    assert (wide | ~is_trans).all()  # wide is True everywhere is_trans is True
    assert wide.sum() >= is_trans.sum()


def test_transition_mask_reads_comp_edges_directly():
    """``COMP`` edges are read directly, not only inferred from ``Motor_current``/``Towers`` --
    the audit's must-fix: 703 of 33,846 (2.08 %) real-file ``COMP`` digital edges are not
    coincident with the nearest ``Motor_current``/``Towers`` event (a one-sample lag), so a mask
    built only from those two under-covers "COMP load/offload edges" as rail_phm.md names them.
    """
    df = _synthetic_trace()
    # index 50 is verified > 5 s from every Motor_current/Towers event in this trace by
    # test_transition_mask_flags_state_changes_and_tower_flips (it asserts index 48 and 52, on
    # either side of 50, are both unflagged).
    comp = np.ones(len(df))
    comp[50:] = 0.0  # a COMP edge with no accompanying Motor_current/Towers change nearby
    df_with_comp = df.assign(COMP=comp)

    # no COMP column (e.g. a caller-built wide frame that dropped it): unaffected, index 50 unflagged
    without_comp = M.transition_mask(df)
    assert not without_comp[50]

    with_comp = M.transition_mask(df_with_comp)
    assert with_comp[50], "a COMP edge with no other accompanying event must be flagged"
    # COMP only adds coverage: every sample flagged without COMP stays flagged with it
    assert (with_comp | ~without_comp).all()


def test_backward_diff_rate_first_sample_matches_simulator_prepend_convention():
    """``rate[0]`` must be ``0.0``, matching ``nebulax.sim.pneumatic._cycle_features``'s own
    ``np.diff(P, prepend=P[0])`` convention (``dP[0] = P[0] - P[0] = 0``), not ``NaN`` -- the
    audit's latent edge case (never triggered by the real file, which starts OFF at 0.04 A, but
    would silently disagree with the simulator for a record whose first sample is already loaded).
    """
    ts = pd.to_datetime(pd.date_range("2024-01-01T00:00:00Z", periods=5, freq="1s"))
    values = np.array([8.0, 8.2, 8.5, 8.6, 8.9])
    rate = M._backward_diff_rate(ts, values)
    assert not np.isnan(rate[0])
    dP0 = np.diff(values, prepend=values[0])[0]
    assert rate[0] == pytest.approx(dP0 / 1.0)  # == 0.0, 1 s cadence


def test_loaded_off_features_match_simulator_when_record_starts_mid_cycle():
    """Regression for the audit's latent ``dP_dt_loaded`` mismatch (0.0200 vs 0.0180 on the
    audit's own minimal trace): a record whose very first sample is already loaded used to give
    the adapter's ``_backward_diff_rate`` a ``NaN`` at index 0 (excluded from the cycle mean)
    while the simulator's ``prepend=P[0]`` convention includes a ``0`` there instead -- fixed by
    making the adapter's ``rate[0]`` ``0.0`` too, so the two now agree exactly even in this case.
    """
    n = 10
    t = pd.date_range("2024-01-01T00:00:00Z", periods=n, freq="1s")
    df = pd.DataFrame(
        {
            "timestamp": t,
            "Motor_current": np.full(n, 6.0),  # loaded from the record's very first sample
            "Reservoirs": np.linspace(8.0, 8.9, n),
            "Oil_temperature": np.full(n, 55.0),
            "TP2": np.full(n, 9.3),
            "TP3": np.full(n, 9.0),
            "H1": np.zeros(n),
            "LPS": np.zeros(n),
            "Towers": np.zeros(n),
            "Pressure_switch": np.zeros(n),
        }
    )
    state = M._classify_state(df["Motor_current"].to_numpy())
    dt = M._sample_durations_s(df["timestamp"])
    is_trans = M.transition_mask(df)
    feats = M._build_cycle_features(df, state, dt, is_trans)
    assert feats["cycle_id"].tolist() == [0]

    sig = {
        "Reservoirs": df["Reservoirs"].to_numpy(),
        "Motor_current": df["Motor_current"].to_numpy(),
        "Oil_temperature": df["Oil_temperature"].to_numpy(),
        "TP2": df["TP2"].to_numpy(),
        "TP3": df["TP3"].to_numpy(),
        "H1": df["H1"].to_numpy(),
        "LPS": df["LPS"].to_numpy(),
        "Flowmeter": np.zeros(n),
    }
    sim_state = np.full(n, P.LOADED, dtype=np.int8)
    tower_flip = np.zeros(n, dtype=bool)
    purge_start = np.zeros(n, dtype=bool)
    load_frac = np.full(n, 1.0)
    in_service = np.ones(n, dtype=bool)
    bounds = np.array([0], dtype=np.int64)
    t_sec = np.arange(n, dtype=np.float64)
    sim_feats = P._cycle_features(
        t_sec, sig, sim_state, tower_flip, purge_start, load_frac, in_service, bounds, dt=1.0
    )

    assert feats.iloc[0]["dP_dt_loaded"] == pytest.approx(float(sim_feats["dP_dt_loaded"][0]), abs=1e-9)


def test_transition_frac_matches_manual_mean():
    df = _synthetic_trace()
    state = M._classify_state(df["Motor_current"].to_numpy())
    dt = M._sample_durations_s(df["timestamp"])
    is_trans = M.transition_mask(df)
    feats = M._build_cycle_features(df, state, dt, is_trans)
    row0 = feats.iloc[0]
    expected = is_trans[5:80].mean()
    assert row0["transition_frac"] == pytest.approx(expected)
    assert 0.0 <= row0["transition_frac"] <= 1.0
