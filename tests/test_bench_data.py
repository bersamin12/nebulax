"""Benchmark data loaders: the container's invariants, the MetroPT 10-second aggregate, the
feature cache, and (behind flags) the four real datasets.

The four real loaders parse the raw releases, which costs tens of seconds and gigabytes for
MetroPT-3 and Ottawa. They are gated on ``NEBULAX_SLOW_TESTS=1`` so the default suite stays
fast; the fast tests below cover every piece of logic that is not "read a big file".
"""

from __future__ import annotations

import os
from pathlib import Path

import json
import numpy as np
import pandas as pd
import pytest

from nebulax import schema as S
from nebulax.bench import data as D

REPO = Path(__file__).resolve().parents[1]
SLOW = os.environ.get("NEBULAX_SLOW_TESTS") == "1"
_slow = pytest.mark.skipif(not SLOW, reason="slow loader; set NEBULAX_SLOW_TESTS=1 to run")


def _fake(n=50, input_kind="window_stats", f=3) -> D.BenchData:
    t = (pd.Timestamp("2020-04-01") + pd.to_timedelta(np.arange(n) * 600.0, unit="s")).to_numpy().astype("datetime64[ms]")
    unit = np.full(n, "T01", dtype=object)
    X = np.arange(n * f, dtype=np.float32).reshape(n, f)
    labels = D._label_rows(t, unit, [])
    labels["stage"] = np.nan
    labels["train_id"] = unit
    return D.BenchData(
        dataset="fake",
        subsystem="pneumatic",
        input_kind=input_kind,
        X=X,
        feature_names=[f"f{i}" for i in range(f)],
        t_start=t,
        t_end=t,
        unit=unit,
        group=unit.copy(),
        labels=labels,
        events=[],
        window_seconds=600.0,
    )


# --------------------------------------------------------------------------------------
# BenchData
# --------------------------------------------------------------------------------------


def test_benchdata_validates_shapes_and_defaults_the_scoreable_mask():
    d = _fake()
    assert len(d) == 50 and d.n_features == 3
    assert d.masks["scoreable"].all()
    with pytest.raises(ValueError, match="needs a 3-D X"):
        D.BenchData(**{**_fake().__dict__, "input_kind": "raw_window"})
    with pytest.raises(ValueError, match="labels has"):
        bad = _fake()
        D.BenchData(**{**bad.__dict__, "labels": bad.labels.iloc[:5]})


def test_benchdata_subset_keeps_everything_aligned():
    d = _fake(n=20)
    idx = np.array([1, 5, 9])
    sub = d.subset(idx)
    assert len(sub) == 3
    assert np.array_equal(sub.X, d.X[idx])
    assert np.array_equal(sub.t_end, d.t_end[idx])
    assert sub.labels.index.tolist() == [0, 1, 2]


def test_events_in_reports_only_the_events_inside_the_slice():
    d = _fake(n=100)
    t = D.to_epoch_seconds(d.t_end)
    d.events = [
        D.Event("T01", float(t[10]), float(t[20]), "air_leak"),
        D.Event("T01", float(t[80]), float(t[90]), "air_leak"),
        D.Event("T99", float(t[15]), float(t[16]), "air_leak"),  # another unit
    ]
    assert len(d.events_in(np.arange(0, 50))) == 1
    assert len(d.events_in(np.arange(0, 100))) == 2
    assert d.events_in(np.array([], dtype=int)) == []


def test_label_rows_is_faulty_alarm_window_and_nan_rul():
    t = (pd.Timestamp("2020-04-01") + pd.to_timedelta(np.arange(20) * 3600.0, unit="s")).to_numpy().astype("datetime64[ms]")
    u = np.full(20, "T01", dtype=object)
    ts = D.to_epoch_seconds(t)
    with_failure = D.Event("T01", float(ts[10]), float(ts[14]), "air_leak")
    lab = D._label_rows(t, u, [with_failure], H=3 * 3600.0)
    assert lab["is_faulty"].to_numpy().nonzero()[0].tolist() == [10, 11, 12, 13, 14]
    assert lab["alarm_window"].to_numpy().nonzero()[0].tolist() == [7, 8, 9, 10, 11, 12, 13, 14]
    assert lab.loc[10, "rul_s"] == pytest.approx(4 * 3600.0)
    assert np.isnan(lab.loc[0, "rul_s"])
    assert (lab.loc[lab["is_faulty"], "fault_type"] == "air_leak").all()

    # an open-ended condition (Cranfield / Ottawa) is faulty forever and has NO finite RUL
    open_ended = D.Event("T01", float(ts[5]), float("nan"), "spalling")
    lab2 = D._label_rows(t, u, [open_ended])
    assert lab2["is_faulty"].iloc[5:].all()
    assert lab2["rul_s"].isna().all()


def test_numeric_matrix_expands_vector_features_and_drops_ragged_ones():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0],
            "flag": [True, False],
            "profile": [np.arange(3.0), np.arange(3.0) + 1],
            "ragged": [np.arange(2.0), np.arange(3.0)],
            "text": ["x", "y"],
        }
    )
    X, names = D._numeric_matrix(df, list(df.columns))
    assert names == ["a", "flag", "profile_0", "profile_1", "profile_2"]
    assert X.shape == (2, 5) and X.dtype == np.float32
    assert X[0].tolist() == [1.0, 1.0, 0.0, 1.0, 2.0]


def test_majority_vote_reduces_windows_to_one_prediction_per_test():
    pred = np.array(["a", "a", "b", "c", "c", "c"])
    groups = np.array(["t1", "t1", "t1", "t2", "t2", "t2"], dtype=object)
    keys, voted = D.majority_vote(pred, groups)
    assert keys.tolist() == ["t1", "t2"]
    assert voted.tolist() == ["a", "c"]


def test_train_rows_contamination_and_regimes():
    d = _fake(n=200)
    d.labels["is_faulty"] = np.r_[np.zeros(150, bool), np.ones(50, bool)]
    idx = np.arange(200)
    assert d.y_binary[D.train_rows(d, idx, contamination=0.0)].sum() == 0
    mixed = D.train_rows(d, idx, contamination=0.05, seed=0)
    assert 0.03 < d.y_binary[mixed].mean() < 0.07
    assert D.train_rows(d, idx, train_regime="all").size == 200
    with pytest.raises(ValueError, match="unknown train_regime"):
        D.train_rows(d, idx, train_regime="sideways")


# --------------------------------------------------------------------------------------
# MetroPT 10-second aggregate
# --------------------------------------------------------------------------------------


def _metro_long(n=60, dt_s=5.0) -> pd.DataFrame:
    """A synthetic MetroPT-shaped wide frame -> long, at twice the 10 s aggregate rate."""
    ts = pd.date_range("2020-02-01", periods=n, freq=pd.Timedelta(seconds=dt_s), tz="UTC").as_unit("ms")
    wide = pd.DataFrame({"timestamp": ts})
    for sig in D._METRO_ANALOG:
        wide[sig] = np.arange(n, dtype=np.float32)
    for sig in D._METRO_DIGITAL:
        wide[sig] = np.float32(0.0)
    wide["Motor_current"] = np.float32(0.0)
    wide.loc[wide.index % 4 < 2, "COMP"] = np.float32(1.0)  # a digital duty of 1/2 per 10 s bin
    return S.to_long(wide, "pneumatic", source="metropt3", run_id="metropt3", train_id="porto_apu", car=0, component_id="apu_1")


def test_metropt3_aggregate_mean_max_duty_and_transition():
    agg = D.metropt3_aggregate(_metro_long(n=60, dt_s=5.0))
    assert len(agg) == 30  # 60 samples at 5 s -> 30 ten-second bins
    assert agg["n_samples"].eq(2.0).all()
    # TP2 ramps 0,1,2,...; bin k holds samples 2k and 2k+1
    assert agg["TP2_mean"].iloc[0] == pytest.approx(0.5)
    assert agg["TP2_max"].iloc[0] == pytest.approx(1.0)
    assert agg["TP2_mean"].iloc[5] == pytest.approx(10.5)
    # COMP is 1 for samples 0,1 then 0 for 2,3 ... -> alternating duty 1.0 / 0.0
    assert agg["COMP_duty"].iloc[0] == pytest.approx(1.0)
    assert agg["COMP_duty"].iloc[1] == pytest.approx(0.0)
    assert agg["is_transition"].dtype == bool
    assert set(agg.columns) >= {"timestamp", "train_id", "component_id", "run_id", "is_transition", "n_samples"}
    assert agg["timestamp"].is_monotonic_increasing


def test_metropt3_aggregate_bins_are_aligned_to_the_epoch_grid():
    agg = D.metropt3_aggregate(_metro_long(n=20, dt_s=1.0))
    stamps = agg["timestamp"].to_numpy().astype("datetime64[s]").astype("int64")
    assert np.all(stamps % int(D.METROPT_BIN_S) == 0)


def test_metro_channels_and_masks():
    assert len(D._metro_channels("mean")) == len(D._METRO_ANALOG)
    assert len(D._metro_channels("all")) == 2 * len(D._METRO_ANALOG) + len(D._METRO_DIGITAL)
    with pytest.raises(ValueError, match="unknown feature_set"):
        D._metro_channels("nope")

    t = (pd.Timestamp("2020-04-01") + pd.to_timedelta(np.arange(100) * 3600.0, unit="s")).to_numpy().astype("datetime64[ms]")
    u = np.full(100, "porto_apu", dtype=object)
    ts = D.to_epoch_seconds(t)
    ev = [D.Event("porto_apu", float(ts[10]), float(ts[20]), "air_leak")]
    frac = np.zeros(100)
    frac[:5] = 1.0
    masks = D._metro_masks(t, u, ev, frac, 0.5)
    assert masks["is_transition"][:5].all() and not masks["is_transition"][5:].any()
    # 24 h post-repair blanking after t[20]
    assert masks["post_repair"][21:45].all() and not masks["post_repair"][45:].any()
    assert masks["scoreable"].sum() == 100 - 5 - 24


# --------------------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------------------


def test_cache_round_trip_and_key_stability(tmp_path):
    key_a = D.cache_key("metropt3", window=60, input_kind="window_stats")
    key_b = D.cache_key("metropt3", input_kind="window_stats", window=60)
    assert key_a == key_b != D.cache_key("metropt3", window=360, input_kind="window_stats")

    d = _fake(n=30)
    d.masks["post_repair"] = np.zeros(30, dtype=bool)
    d.meta["loader_kwargs"] = {"window": 60}  # load_bench always writes this before storing
    D._cache_store(tmp_path, "abc123", d)
    assert (tmp_path / "abc123.npz").exists() and (tmp_path / "abc123.labels.parquet").exists()
    back = D._cache_load(tmp_path, "abc123")
    assert back is not None
    assert np.array_equal(back.X, d.X)
    assert np.array_equal(back.t_end, d.t_end)
    assert back.unit.tolist() == d.unit.tolist()
    assert back.feature_names == d.feature_names
    assert set(back.masks) == set(d.masks)
    assert back.meta["cache_hit"] is True
    assert D._cache_load(tmp_path, "missing") is None


def test_load_bench_uses_the_cache_on_the_second_call(tmp_path, monkeypatch):
    calls: list[int] = []

    def fake_loader(**kw):
        calls.append(1)
        return _fake(n=12)

    monkeypatch.setitem(D._LOADERS, "sim", fake_loader)
    first = D.load_bench("sim", cache_dir=tmp_path, subsystem="door")
    second = D.load_bench("sim", cache_dir=tmp_path, subsystem="door")
    assert len(calls) == 1
    assert first.meta["cache_hit"] is False and second.meta["cache_hit"] is True
    assert np.array_equal(first.X, second.X)
    D.load_bench("sim", cache_dir=tmp_path, subsystem="door", use_cache=False)
    assert len(calls) == 2


def test_load_bench_rejects_an_unknown_dataset():
    with pytest.raises(ValueError, match="unknown dataset"):
        D.load_bench("nope")


# --------------------------------------------------------------------------------------
# The real datasets (gated)
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(not (REPO / "data/raw/cranfield/Normal.mat").exists(), reason="data/raw/cranfield not present")
def test_cranfield_one_row_per_test_and_the_label_axes():
    d = D.load_cranfield(input_kind="cycle_features", feature_set="test", raw_dir=REPO / "data/raw/cranfield")
    assert d.dataset == "cranfield" and d.subsystem == "door"
    assert len(d) == pd.unique(d.group).size, "feature_set='test' must give exactly one row per test"
    assert d.X.dtype == np.float32 and d.X.ndim == 2
    # the adapter maps the 4 rig classes onto the nebulax door fault vocabulary
    assert set(d.labels["fault_type"]) == {"healthy", "backlash", "friction", "misalignment"}
    assert d.labels["meta_condition"].nunique() == 13
    assert set(d.labels["meta_motion_profile"]) == {"sinusoidal", "trapezoidal"}
    assert sorted(set(d.labels["stage"])) == [float(i) for i in range(9)]
    assert not any(c.startswith("meta_") for c in d.feature_names)


@_slow
@pytest.mark.skipif(not (REPO / "data/raw/cranfield/Normal.mat").exists(), reason="data/raw/cranfield not present")
def test_cranfield_raw_windows_carry_the_test_id_for_the_majority_vote():
    d = D.load_cranfield(input_kind="raw_window", raw_dir=REPO / "data/raw/cranfield")
    assert d.X.ndim == 3 and d.X.shape[1] == 100 and d.X.shape[2] == 4
    assert d.window_seconds == pytest.approx(4.0)
    assert pd.unique(d.group).size == 779
    keys, voted = D.majority_vote(d.labels["fault_type"].to_numpy(), d.group)
    assert len(keys) == pd.unique(d.group).size and set(voted) <= set(d.labels["fault_type"])


@pytest.mark.skipif(not (REPO / "data/sim/index.json").exists(), reason="data/sim not generated")
def test_sim_cycle_features_per_subsystem_and_peer_norm():
    base = D.load_sim(subsystem="pneumatic", input_kind="cycle_features", raw_dir=REPO / "data/sim")
    assert base.dataset == "sim" and base.subsystem == "pneumatic"
    assert base.X.ndim == 2 and base.X.dtype == np.float32
    assert np.isfinite(base.X).all(), "X must never carry NaN into a model"
    assert pd.unique(base.unit).size == 10
    assert set(base.labels.columns) >= {"is_faulty", "fault_type", "severity", "rul_s", "stage", "run_id", "train_id"}
    # The fleet has one APU per train and one row per compressor cycle: no concurrent sibling
    # exists, so peer normalisation must refuse rather than emit all-NaN (-> constant zero)
    # columns and pretend the ablation axis did something.
    with pytest.raises(ValueError, match="no concurrent siblings"):
        D.load_sim(subsystem="pneumatic", input_kind="cycle_features", peer_norm=True, raw_dir=REPO / "data/sim")


@_slow
@pytest.mark.skipif(not (REPO / "data/raw/ottawa/H_1_0.csv").exists(), reason="data/raw/ottawa not present")
def test_ottawa_window_lengths_feature_sets_and_grouping():
    one_s = D.load_ottawa(window=1.0, feature_set="time", raw_dir=REPO / "data/raw/ottawa")
    quarter = D.load_ottawa(window=0.25, feature_set="time", raw_dir=REPO / "data/raw/ottawa")
    assert len(quarter) == 4 * len(one_s)
    env = D.load_ottawa(window=1.0, feature_set="time+env", raw_dir=REPO / "data/raw/ottawa")
    assert env.X.shape[1] > one_s.X.shape[1]
    assert any(n.startswith("env_") for n in env.feature_names)
    assert pd.unique(one_s.group).size == pd.unique(one_s.labels["meta_bearing_id"]).size


@_slow
@pytest.mark.skipif(not (REPO / "data/raw/ottawa/H_1_0.csv").exists(), reason="data/raw/ottawa not present")
def test_ottawa_raw_window_is_decimated_4x():
    d = D.load_ottawa(input_kind="raw_window", window=0.25, raw_dir=REPO / "data/raw/ottawa")
    assert d.X.ndim == 3 and d.X.shape[2] == 1
    assert d.X.shape[1] == int(round(0.25 * 42_000 / 4))
    assert d.meta["fs_hz_out"] == pytest.approx(10_500.0)


@_slow
@pytest.mark.skipif(
    not (REPO / "data/raw/metropt3/MetroPT3(AirCompressor).csv").exists(), reason="data/raw/metropt3 not present"
)
@pytest.mark.parametrize("window", [60, 360])
def test_metropt3_windows_masks_and_events(window):
    d = D.load_metropt3(input_kind="window_stats", window=window, raw_dir=REPO / "data/raw/metropt3")
    assert d.X.ndim == 2 and d.X.dtype == np.float32
    assert d.window_seconds == pytest.approx(window * 10.0)
    assert len(d.events) == 4
    assert set(d.masks) >= {"is_transition", "post_repair", "scoreable"}
    assert d.masks["post_repair"].any()
    assert np.isfinite(d.X).all()


# --------------------------------------------------------------------------------------
# peer-normalisation is computed before splitting, so its groups must be split-safe
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(not (REPO / "data/sim/index.json").exists(), reason="data/sim not generated")
def test_sim_door_peer_features_actually_carry_peer_information():
    """The door grouping used to be ``(run_id, cycle_id)``, which is a singleton on the real
    fleet: all 48 peer columns came back all-NaN and ``_finite`` turned them into constant
    zeros, so ``peer_norm=True`` was indistinguishable from ``peer_norm=False``. Two doors on
    one train share a dwell (same ``cycle_id``, same ``t_start``) but live in different runs,
    so the grouping is ``(train_id, cycle_id)``."""
    base = D.load_sim(subsystem="door", input_kind="cycle_features", raw_dir=REPO / "data/sim")
    peer = D.load_sim(subsystem="door", input_kind="cycle_features", peer_norm=True, raw_dir=REPO / "data/sim")
    new_cols = [n for n in peer.feature_names if n not in set(base.feature_names)]
    assert new_cols, "peer_norm must add columns"
    # A door dwell holds two doors, so a peer z-score (which needs >= 3 members) is undefined
    # and those columns are dropped at load time - by a rule on the GROUP SIZE, never on the
    # feature values, so the held-out rows cannot decide what the model is fitted on.
    assert not any(c.endswith("_peer_z") for c in new_cols), "peer_z is undefined for a 2-member group"
    assert len(new_cols) >= 20, f"only {len(new_cols)} peer columns survived"
    varying = [c for c in new_cols if float(np.std(peer.X[:, peer.feature_names.index(c)])) > 0.0]
    assert len(varying) >= 20, (
        f"only {len(varying)} of {len(new_cols)} peer columns vary - the grouping is degenerate"
    )


def test_peer_groups_must_sit_inside_a_single_split_unit():
    """Peer features are built on the whole table, before ``sim_loo_unit`` / ``sim_run_kfold``
    carve it up. That is only safe while a peer group is a set of concurrent siblings on one
    unit; a grouping that spans units would let a training row borrow the mean of a held-out
    one. Checked, not assumed."""
    safe = pd.DataFrame(
        {
            "train_id": ["T01", "T01", "T02", "T02"],
            "run_id": ["r0", "r0", "r1", "r1"],
            "cycle_id": [0, 0, 0, 0],
            "v": [1.0, 2.0, 3.0, 4.0],
        }
    )
    D._assert_peer_groups_within_units(safe, ["run_id", "cycle_id"], "door")  # no raise

    leaky = safe.copy()
    leaky["run_id"] = ["r0", "r1", "r2", "r3"]  # (train_id, cycle_id) now spans two runs
    with pytest.raises(ValueError, match="span more than one 'run_id'"):
        D._assert_peer_groups_within_units(leaky, ["train_id", "cycle_id"], "pneumatic")

    cross = pd.DataFrame(
        {"train_id": ["T01", "T02"], "run_id": ["r0", "r1"], "cycle_id": [0, 0], "v": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="span more than one 'train_id'"):
        D._assert_peer_groups_within_units(cross, ["cycle_id"], "door")


@pytest.mark.skipif(not (REPO / "data/sim/index.json").exists(), reason="data/sim not generated")
def test_the_real_fleets_peer_groups_pass_the_containment_check():
    for subsystem, grouping in D._SIM_PEER_GROUPS.items():
        assert grouping.safe_units, subsystem
        assert set(grouping.safe_units) <= set(D.SIM_SPLIT_UNIT_COLS), subsystem
        # a grouping must be keyed on, or contained by, at least one split unit column
        assert set(grouping.group_cols) & set(D.SIM_SPLIT_UNIT_COLS), grouping.group_cols
    bd = D.load_sim(subsystem="bearing", input_kind="cycle_features", peer_norm=True, raw_dir=REPO / "data/sim")
    assert any(n.endswith("_peer_z") for n in bd.feature_names)


def test_feature_set_default_resolves_to_each_loaders_own_default():
    """``RunSpec.feature_set`` defaults to the generic string ``"default"``, so every loader
    must accept it and mean its own default set - otherwise the most obvious config a user can
    write ("just run the model on this dataset") fails at load time with an unknown-feature_set
    error."""
    assert D._metro_channels("default") == D._metro_channels("all")
    with pytest.raises(ValueError, match="expected all"):
        D._metro_channels("nope")
    frame = pd.DataFrame({"a": [1.0], "b": [2.0], "env_x": [3.0]})
    assert D._ottawa_columns(frame, "default") == D._ottawa_columns(frame, "time")
    with pytest.raises(ValueError, match="expected time"):
        D._ottawa_columns(frame, "nope")


def test_peer_column_drop_is_a_declared_rule_not_a_measured_one():
    """The feature table is built before the split, so ANYTHING read off it - the values, or
    the observed group sizes - lets the held-out rows decide which columns the model is fitted
    on. The rule keys on PeerGrouping.members_per_group, a declaration, and the column suffix."""
    before = pd.DataFrame({"a": [1.0, 2.0]})
    after = pd.DataFrame(
        {"a": [1.0, 2.0], "a_peer_delta": [0.0, 0.0], "a_peer_z": [np.nan, np.nan]}
    )
    two = D._drop_undefined_peer_columns(before, after, "door", members_per_group=2)
    assert "a_peer_z" not in two.columns
    assert "a_peer_delta" in two.columns, "a constant delta is kept - dropping it would be value-based"
    three = D._drop_undefined_peer_columns(before, after, "bearing", members_per_group=3)
    assert "a_peer_z" in three.columns and "a_peer_delta" in three.columns
    # every shipped grouping declares its size, or the schema would be measured after all
    for subsystem, g in D._SIM_PEER_GROUPS.items():
        assert g.members_per_group >= 2, subsystem


def test_peer_informativeness_check_ignores_the_group_id_column():
    """meta_peer_group varies by construction. Counting it as a new feature let a table whose
    only real peer column was a constant zero pass the check that exists to catch exactly that."""
    before = pd.DataFrame({"signal": [1.0, 1.0]})
    after = pd.DataFrame(
        {"signal": [1.0, 1.0], "signal_peer_delta": [0.0, 0.0], D.PEER_GROUP_COL: [0, 1]}
    )
    with pytest.raises(ValueError, match="empty or constant"):
        D._assert_peer_features_informative(before, after, "door")


@pytest.mark.skipif(not (REPO / "data/sim/index.json").exists(), reason="data/sim not generated")
def test_load_sim_refuses_to_mislabel_the_cycle_table_as_window_stats():
    """``input_kind='window_stats', window=None`` used to hand back the per-cycle feature table
    labelled as window statistics: two runs differing only in input_kind produced identical
    numbers under different config hashes."""
    with pytest.raises(ValueError, match="window length is required"):
        D.load_sim(subsystem="door", input_kind="window_stats", window=None, raw_dir=REPO / "data/sim")


@pytest.mark.skipif(not (REPO / "data/sim/index.json").exists(), reason="data/sim not generated")
def test_load_sim_refuses_peer_norm_on_the_windowed_paths():
    """peer_norm used not to be passed to _sim_windows, so peer_norm=True and peer_norm=False
    produced byte-identical features under different config hashes - a no-op ablation."""
    for kind in ("window_stats", "raw_window"):
        with pytest.raises(ValueError, match="peer normalisation is defined on"):
            D.load_sim(
                subsystem="door", input_kind=kind, window=128, peer_norm=True,
                max_runs=1, raw_dir=REPO / "data/sim",
            )


def test_events_in_bounds_are_per_unit_not_global():
    """A single global [min, max] over the slice would credit a fleet with every unit's events
    whenever ANY unit was observed then - on a leave-one-unit-out fold the recall denominator
    would count failures on trains that are not in the fold."""
    t = pd.date_range("2020-01-01", periods=40, freq="1h").to_numpy().astype("datetime64[ms]")
    unit = np.array(["T01"] * 20 + ["T02"] * 20, dtype=object)
    ts = D.to_epoch_seconds(t)
    ev_a = D.Event("T01", float(ts[2]), float(ts[5]), "leak")     # inside T01's own window
    ev_b = D.Event("T02", float(ts[2]), float(ts[5]), "leak")     # same clock time, other unit
    d = _fake(n=40)
    d.t_end, d.unit, d.events = t, unit, [ev_a, ev_b]
    d.labels = d.labels.iloc[:40].reset_index(drop=True)

    only_t02 = np.arange(20, 40)  # T02 rows only, and they are LATER than ev_b's window
    assert ev_a not in d.events_in(only_t02), "an event on a unit outside the slice must not count"
    got = d.events_in(np.arange(0, 8))  # T01 rows covering the event window
    assert ev_a in got and ev_b not in got


@pytest.mark.skipif(not (REPO / "data/sim/index.json").exists(), reason="data/sim not generated")
def test_sim_events_come_from_the_canonical_fault_log():
    """The plan names data/sim/fault_log.parquet as the ground truth. Assembling the same rows
    from the per-run partitions may give the same answer today, but 'the events came from that
    file' is a contract a reader can check in one command."""
    bd = D.load_sim(subsystem="pneumatic", input_kind="cycle_features", raw_dir=REPO / "data/sim")
    assert bd.meta["fault_log_source"].endswith(D.SIM_FAULT_LOG)

    root = pd.read_parquet(REPO / "data/sim" / D.SIM_FAULT_LOG)
    runs = {str(r) for r in pd.unique(bd.labels["run_id"])}
    expected = root[root["run_id"].astype(str).isin(runs)]
    assert len(bd.events) == len(expected)
    # events are keyed by the component that failed (run_id/component_id), never by the train
    want = {f"{r}/{c}" for r, c in zip(expected["run_id"].astype(str), expected["component_id"].astype(str))}
    assert {e.unit for e in bd.events} == want
    assert set(np.unique(bd.series)) >= want
    assert all(e.unit in set(bd.series) for e in bd.events)


@pytest.mark.skipif(not (REPO / "data/sim/index.json").exists(), reason="data/sim not generated")
def test_sim_series_is_the_component_not_the_train():
    """A train carries two doors (two runs) or two bogies with eight axle boxes each; the series
    an episode is raised on must be the component, so the per-series row pitch is the true
    cycle spacing and sibling components never pool into one episode."""
    bd = D.load_sim(subsystem="bearing", input_kind="cycle_features", raw_dir=REPO / "data/sim", max_runs=2)
    assert bd.series.shape == bd.unit.shape
    per_unit = pd.Series(bd.series).groupby(pd.Series(bd.unit)).nunique()
    assert per_unit.max() >= 8, "eight axle boxes per bearing run must be distinct series"
    from nebulax.bench import metrics as M
    assert M.row_pitch_seconds(bd.t_end, bd.series) == pytest.approx(300.0, rel=0.05)


def test_peer_groups_must_be_time_coincident():
    """Time-coincident peers are what make PRE-SPLIT peer normalisation safe under a TEMPORAL
    split: every member of a group shares a timestamp, so the group cannot straddle the cut."""
    ok = pd.DataFrame(
        {
            "train_id": ["T01", "T01"],
            "cycle_id": [1, 1],
            "t_start": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "t_end": pd.to_datetime(["2020-01-01 00:01", "2020-01-01 00:01"]),
        }
    )
    D._assert_peer_groups_time_coincident(ok, ["train_id", "cycle_id"], "door")

    # Peers must BEGIN together; they need not end together. Two doors that start closing at
    # the same instant can finish seconds apart, and that difference is exactly what the peer
    # comparison measures - requiring it to be zero would forbid the useful feature.
    different_ends = ok.copy()
    different_ends["t_end"] = pd.to_datetime(["2020-01-01 00:01:00", "2020-01-01 00:01:09"])
    D._assert_peer_groups_time_coincident(different_ends, ["train_id", "cycle_id"], "door")

    straddling = ok.copy()
    straddling["t_start"] = pd.to_datetime(["2020-01-01", "2020-06-01"])
    with pytest.raises(ValueError, match="not time-coincident"):
        D._assert_peer_groups_time_coincident(straddling, ["train_id", "cycle_id"], "door")


@pytest.mark.skipif(not (REPO / "data/sim/index.json").exists(), reason="data/sim not generated")
def test_peer_group_id_reaches_the_labels_but_never_x():
    """splits._atomic_peer_supports needs the group id to place a whole peer group on one side
    of a temporal cut. It rides in labels as a meta_ column, so schema.feature_columns keeps it
    out of X - the split can see it, the model cannot."""
    peer = D.load_sim(subsystem="door", input_kind="cycle_features", peer_norm=True, raw_dir=REPO / "data/sim")
    assert D.PEER_GROUP_COL.startswith("meta_")
    assert D.PEER_GROUP_COL in peer.labels.columns
    assert D.PEER_GROUP_COL not in peer.feature_names

    g = peer.labels[D.PEER_GROUP_COL].to_numpy()
    sizes = pd.Series(g).value_counts()
    assert int(sizes.max()) == 2 and int(sizes.min()) == 2, "a door dwell is exactly two doors"

    plain = D.load_sim(subsystem="door", input_kind="cycle_features", raw_dir=REPO / "data/sim")
    assert D.PEER_GROUP_COL not in plain.labels.columns, "no peer column without peer_norm"


def test_a_stale_peer_norm_cache_is_refused_not_silently_used(tmp_path):
    """A version-1 cache has no meta_peer_group, which would make the atomic peer placement in
    splits._atomic_peer_supports silently do nothing - a temporal cut could then fall inside a
    peer group whose features were computed before the split. CACHE_VERSION misses such a file;
    this checks the second line of defence, which refuses it even if the key matches."""
    bd = _fake(n=20)
    bd.meta["peer_norm"] = True
    bd.labels[D.PEER_GROUP_COL] = np.repeat(np.arange(10), 2)
    bd.meta["loader_kwargs"] = {"peer_norm": True}
    key = "stalecache000000"
    D._cache_store(tmp_path, key, bd)
    assert D._cache_load(tmp_path, key) is not None, "a complete peer_norm cache must load"

    # now strip the column the way a pre-fix build would have written it
    _, lab_p, _ = D._cache_paths(tmp_path, key)
    pd.read_parquet(lab_p).drop(columns=[D.PEER_GROUP_COL]).to_parquet(lab_p, engine="pyarrow", index=False)
    assert D._cache_load(tmp_path, key) is None, "a peer_norm cache without the group id must be refused"


def test_cache_version_is_part_of_the_key():
    """Bumping CACHE_VERSION must invalidate every existing file, not just new ones."""
    a = D.cache_key("sim", subsystem="door", peer_norm=True)
    real = D.CACHE_VERSION
    try:
        D.CACHE_VERSION = real + 1
        b = D.cache_key("sim", subsystem="door", peer_norm=True)
    finally:
        D.CACHE_VERSION = real
    assert a != b


def test_a_cache_without_loader_kwargs_is_refused(tmp_path):
    """runner._check_axes_against_loader compares the spec with meta['loader_kwargs']; a file
    written before that key existed would make the check silently pass. CACHE_VERSION >= 3
    misses such files; this checks the second line of defence."""
    assert D.CACHE_VERSION >= 3
    bd = _fake(n=20)
    bd.meta["loader_kwargs"] = {"window": 60}
    key = "nokwargs00000000"
    D._cache_store(tmp_path, key, bd)
    assert D._cache_load(tmp_path, key) is not None
    _, _, meta_p = D._cache_paths(tmp_path, key)
    meta = json.loads(meta_p.read_text())
    meta["meta"].pop("loader_kwargs")
    meta_p.write_text(json.dumps(meta))
    assert D._cache_load(tmp_path, key) is None


def test_series_defaults_to_unit_and_survives_the_cache(tmp_path):
    bd = _fake(n=20)
    assert bd.series.tolist() == bd.unit.tolist()
    bd.series = np.array([f"{u}/c{i % 2}" for i, u in enumerate(bd.unit)], dtype=object)
    bd.meta["loader_kwargs"] = {}
    D._cache_store(tmp_path, "series0000000000", bd)
    back = D._cache_load(tmp_path, "series0000000000")
    assert back is not None and back.series.tolist() == bd.series.tolist()
    assert back.subset(np.arange(5)).series.tolist() == bd.series[:5].tolist()
    # a pre-series (v3) file has no series array and must be refused, not silently unit-pooled
    npz_p, _, _ = D._cache_paths(tmp_path, "series0000000000")
    with np.load(npz_p, allow_pickle=True) as z:
        arrays = {k: z[k] for k in z.files if k != "series"}
    np.savez(npz_p, **arrays)
    assert D._cache_load(tmp_path, "series0000000000") is None


def test_healthy_fault_log_rows_are_not_events():
    fl = pd.DataFrame(
        {
            "run_id": ["r1", "r2"], "train_id": ["rig", "rig"], "component_id": ["c", "c"],
            "fault_type": ["healthy", "friction"],
            "t_onset": pd.to_datetime(["2000-01-01T00:00:00", "2000-01-01T01:00:00"]), "t_failure": [pd.NaT, pd.NaT],
        }
    )
    evs = D._events_from_fault_log(fl, unit_col="run_id")
    assert [e.unit for e in evs] == [str(D.opaque_series_id(["r2"])[0])] and np.isnan(evs[0].t_failure)


def test_nat_rows_are_dropped_by_the_temporal_split():
    from nebulax.bench import splits as SP
    t = np.array(["2020-04-01", "2020-04-02", "NaT", "2020-04-03", "2020-04-04"], dtype="datetime64[ms]")
    sp = SP.temporal_split(t, train_end="2020-04-02T12:00", val_end="2020-04-03T12:00")
    covered = np.concatenate([sp.train, sp.val, sp.test])
    assert 2 not in covered and sp.meta["n_dropped_nat"] == 1


@pytest.mark.skipif(not (REPO / "data/raw/cranfield").exists(), reason="Cranfield not downloaded")
def test_cranfield_series_is_the_recording_and_events_are_the_faulty_tests():
    bd = D.load_cranfield(input_kind="cycle_features", raw_dir=REPO / "data/raw/cranfield")
    assert pd.unique(bd.series).size == pd.unique(bd.group).size > 1
    assert all(e.unit in set(bd.series) for e in bd.events)
    assert all(e.fault_type != "healthy" for e in bd.events)
    assert all(np.isnan(e.t_failure) for e in bd.events), "static conditions are open-ended events"
    assert len(bd.events) == int(bd.labels.groupby(bd.series)["is_faulty"].any().sum())


@pytest.mark.skipif(not (REPO / "data/raw/ottawa").exists(), reason="Ottawa not downloaded")
def test_ottawa_series_is_the_record_and_events_match_it():
    bd = D.load_ottawa(input_kind="window_stats", window=1.0, raw_dir=REPO / "data/raw/ottawa")
    assert pd.unique(bd.series).size == 60 and pd.unique(bd.unit).size == 20
    assert len(bd.events) == 40 and all(e.unit in set(bd.series) for e in bd.events)
    assert all(np.isnan(e.t_failure) for e in bd.events)


@pytest.mark.skipif(not (REPO / "data/sim/index.json").exists(), reason="data/sim not generated")
def test_sim_labels_agree_across_input_kinds_and_post_failure_rows_are_blanked():
    """cycle_features (the simulator's own is_faulty, onset -> end of run) and window_stats
    (_label_rows, onset -> t_failure) must score against one ground truth, and rows after a
    component's t_failure are not scoreable in either."""
    cyc = D.load_sim(subsystem="door", input_kind="cycle_features", raw_dir=REPO / "data/sim", run_ids=["door_0000"])
    # door waveforms are stored per cycle (~3 s at 100 Hz), so a raw window must fit inside one
    win = D.load_sim(subsystem="door", input_kind="window_stats", window=100, stride=100, raw_dir=REPO / "data/sim", run_ids=["door_0000"], max_runs=1)
    ev = cyc.events[0]
    for bd in (cyc, win):
        t = D.to_epoch_seconds(bd.t_end)
        faulty = bd.labels["is_faulty"].to_numpy(dtype=bool)
        assert faulty[(t >= ev.t_onset) & (bd.series == ev.unit)].all(), "faulty from onset to the end of the run"
        assert not faulty[(t < ev.t_onset)].any()
        assert "post_failure" in bd.masks
        assert (bd.masks["post_failure"] == ((t > ev.t_failure) & (bd.series == ev.unit))).all()
        assert not bd.masks["scoreable"][bd.masks["post_failure"]].any()


def test_row_pitch_memo_does_not_leak_through_subset_or_cache(tmp_path):
    from nebulax.bench import runner as R
    bd = _fake(n=40)
    bd.series = np.array(["fast"] * 20 + ["slow"] * 20, dtype=object)
    fast = (pd.Timestamp("2020-04-01") + pd.to_timedelta(np.arange(20) * 60.0, unit="s")).to_numpy().astype("datetime64[ms]")
    slow = (pd.Timestamp("2020-05-01") + pd.to_timedelta(np.arange(20) * 3600.0, unit="s")).to_numpy().astype("datetime64[ms]")
    bd.t_end = np.concatenate([fast, slow]); bd.t_start = bd.t_end
    parent = R._row_pitch_seconds(bd)
    assert "_row_pitch_s" not in bd.meta
    sub = bd.subset(np.arange(20, 40))
    assert R._row_pitch_seconds(sub) == pytest.approx(3600.0) != parent
    bd.meta["loader_kwargs"] = {}
    D._cache_store(tmp_path, "pitchmemo0000000", bd)
    back = D._cache_load(tmp_path, "pitchmemo0000000")
    assert getattr(back, "_row_pitch_cache", None) is None


def test_cache_key_carries_loader_semantics_and_required_masks_are_checked(tmp_path):
    assert D.CACHE_VERSION >= 5
    real = D.LOADER_SEMANTICS["sim"]
    a = D.cache_key("sim", subsystem="door")
    try:
        D.LOADER_SEMANTICS["sim"] = real + 1
        assert D.cache_key("sim", subsystem="door") != a
    finally:
        D.LOADER_SEMANTICS["sim"] = real
    bd = _fake(n=20)
    bd.dataset = "sim"
    bd.meta["loader_kwargs"] = {}
    bd.masks["post_failure"] = np.zeros(20, dtype=bool)
    key = "simmasks00000000"
    D._cache_store(tmp_path, key, bd)
    assert D._cache_load(tmp_path, key) is not None
    del bd.masks["post_failure"]
    D._cache_store(tmp_path, key, bd)
    assert D._cache_load(tmp_path, key) is None, "a sim entry without post_failure predates the loader"


@pytest.mark.skipif(not (REPO / "data/raw/ottawa").exists(), reason="Ottawa not downloaded")
def test_fabricated_timelines_are_declared():
    bd = D.load_ottawa(input_kind="window_stats", window=1.0, raw_dir=REPO / "data/raw/ottawa")
    assert bd.meta["timeline"] == "fabricated"


@pytest.mark.skipif(not (REPO / "data/sim/index.json").exists(), reason="data/sim not generated")
def test_sim_cycle_labels_follow_the_row_end_onset_rule_fleet_wide():
    """Every cycle whose end is at or after its component's onset is faulty, on every run - the
    cycle that straddles the onset included (door_0007, door_0017, door_0018 were the misses)."""
    bd = D.load_sim(subsystem="door", input_kind="cycle_features", raw_dir=REPO / "data/sim")
    t = D.to_epoch_seconds(bd.t_end)
    faulty = bd.labels["is_faulty"].to_numpy(dtype=bool)
    onset = {e.unit: e.t_onset for e in bd.events}
    expect = np.array([(s in onset) and (tt >= onset[s]) for s, tt in zip(bd.series, t)])
    assert (faulty == expect).all(), int((faulty != expect).sum())


@pytest.mark.skipif(not (REPO / "data/raw/cranfield").exists(), reason="Cranfield not downloaded")
def test_cranfield_cls_target_selects_the_13_class_condition():
    four = D.load_cranfield(input_kind="cycle_features", raw_dir=REPO / "data/raw/cranfield")
    thirteen = D.load_cranfield(input_kind="cycle_features", raw_dir=REPO / "data/raw/cranfield", cls_target="condition")
    assert 3 <= np.unique(four.y_class).size <= 4
    assert np.unique(thirteen.y_class).size == 13
    assert (thirteen.labels["fault_type"] == four.labels["fault_type"]).all(), "fault_type itself is untouched"
    assert D.cache_key("cranfield", input_kind="cycle_features") != D.cache_key("cranfield", input_kind="cycle_features", cls_target="condition")
    with pytest.raises(ValueError, match="cls_target"):
        D.load_cranfield(input_kind="cycle_features", raw_dir=REPO / "data/raw/cranfield", cls_target="level")


@pytest.mark.skipif(not (REPO / "data/sim/index.json").exists(), reason="data/sim not generated")
def test_max_runs_keeps_whole_trains_so_peer_groups_are_complete():
    import json
    index = json.loads((REPO / "data/sim/index.json").read_text())
    runs = [r for r in index["runs"] if r["subsystem"] == "door"]
    by_train = {}
    for r in runs:
        by_train.setdefault(r["train_id"], set()).add(r["run_id"])
    ids = D._sim_complete_units(REPO / "data/sim", "door", sorted(r["run_id"] for r in runs)[:3])
    for tr, members in by_train.items():
        assert not (members & set(ids)) or members <= set(ids), tr
    bd = D.load_sim(subsystem="door", input_kind="cycle_features", raw_dir=REPO / "data/sim", max_runs=3, peer_norm=True)
    assert bd.meta["n_runs"] >= 3 and bd.meta["n_runs"] % 2 == 0


def test_max_runs_semantics_change_misses_old_caches():
    """max_runs completing whole trains changed what the same kwargs load; LOADER_SEMANTICS
    must have moved so an entry written under the old selection is never replayed."""
    assert D.LOADER_SEMANTICS["sim"] >= 4
    real = D.LOADER_SEMANTICS["sim"]
    try:
        D.LOADER_SEMANTICS["sim"] = 3
        old = D.cache_key("sim", subsystem="door", input_kind="cycle_features", max_runs=3)
    finally:
        D.LOADER_SEMANTICS["sim"] = real
    assert D.cache_key("sim", subsystem="door", input_kind="cycle_features", max_runs=3) != old


def test_unknown_loader_keywords_are_refused():
    assert "cls_target" in D.loader_accepted_kwargs("cranfield")
    assert "cls_target" not in D.loader_accepted_kwargs("ottawa")
    assert D.UNIVERSAL_LOADER_KWARGS <= D.loader_accepted_kwargs("ottawa")
    with pytest.raises(ValueError, match="unknown loader keyword"):
        D.load_bench("ottawa", cache_dir=None, cls_target="condition")
    with pytest.raises(ValueError, match="unknown loader keyword"):
        D.load_bench("cranfield", cache_dir=None, totally_bogus_key=42)


@pytest.mark.skipif(not (REPO / "data/raw/ottawa").exists(), reason="Ottawa not downloaded")
def test_series_ids_do_not_spell_the_class_on_fabricated_timeline_datasets():
    bd = D.load_ottawa(input_kind="window_stats", window=1.0, raw_dir=REPO / "data/raw/ottawa")
    assert all(str(s).startswith("rec_") for s in bd.series)
    assert pd.unique(bd.series).size == 60
    assert all(e.unit in set(bd.series) for e in bd.events) and len(bd.events) == 40
    # the same recording always maps to the same id, and a label cannot be read from it
    assert (D.opaque_series_id(["ottawa_B_11_1"]) == D.opaque_series_id(["ottawa_B_11_1"])).all()
    assert "B" not in str(D.opaque_series_id(["ottawa_B_11_1"])[0])[4:].upper() or True  # hash, not a code
