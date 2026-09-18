"""Contract tests for nebulax.schema: round trips, validation errors, parquet IO."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nebulax import schema as S


# ---------------------------------------------------------------- registry invariants


def test_signal_registry_covers_every_subsystem():
    assert set(S.SIGNALS) == set(S.SUBSYSTEMS)
    for sub, sigs in S.SIGNALS.items():
        assert len(sigs) == len(set(sigs)), f"duplicate signal in {sub}"
        for sig in sigs:
            assert (sub, sig) in S.SIGNAL_SPECS


def test_metropt3_names_are_verbatim():
    # The MetroPT-3 column names must survive exactly, typo included.
    assert "DV_eletric" in S.SIGNALS["pneumatic"]
    assert set(S.METROPT3_SIGNALS) <= set(S.SIGNALS["pneumatic"])
    assert len(S.METROPT3_SIGNALS) == 15
    # Flowmeter is our addition (MetroPT-1/2 have it, MetroPT-3 does not).
    assert "Flowmeter" in S.SIGNALS["pneumatic"]
    assert "Flowmeter" not in S.METROPT3_SIGNALS


def test_component_id_conventions():
    assert S.DOOR_COMPONENT_IDS == (
        "door_L1", "door_L2", "door_L3", "door_L4", "door_R1", "door_R2", "door_R3", "door_R4",
    )
    assert S.AXLEBOX_COMPONENT_IDS[0] == "axlebox_1L"
    assert S.AXLEBOX_COMPONENT_IDS[-1] == "axlebox_4R"
    assert len(S.AXLEBOX_COMPONENT_IDS) == 8
    assert S.APU_COMPONENT_IDS == ("apu_1",)
    assert S.is_valid_component_id("door", "door_R4")
    assert S.is_valid_component_id("bearing", "train")  # context is legal everywhere
    assert not S.is_valid_component_id("door", "axlebox_1L")
    with pytest.raises(ValueError, match="unknown subsystem"):
        S.component_ids("brakes")


def test_empty_frames_have_declared_dtypes():
    for frame, cols in (
        (S.empty_long(), S.LONG_COLUMNS),
        (S.empty_fault_log(), S.FAULT_LOG_COLUMNS),
        (S.empty_events(), S.EVENT_LOG_COLUMNS),
        (S.empty_scores(), S.SCORES_COLUMNS),
    ):
        assert list(frame.columns) == list(cols)
        assert len(frame) == 0
    S.validate_long(S.empty_long())
    S.validate_fault_log(S.empty_fault_log())
    S.validate_events(S.empty_events())


# ---------------------------------------------------------------- long <-> wide


def test_long_to_wide_to_long_round_trip(door_long):
    door = door_long[door_long["subsystem"].astype("string") == "door"]
    wide = S.to_wide(door_long, "door")
    assert list(wide.columns[:6]) == ["timestamp", "source", "run_id", "train_id", "car", "component_id"]
    assert set(wide.columns) - set(wide.columns[:6]) == {"pos", "current", "ls_closed"}
    assert len(wide) == len(door) // 3

    back = S.to_long(wide, "door")
    assert list(back.columns) == list(S.LONG_COLUMNS)
    assert {c: str(d) for c, d in back.dtypes.items()} == {c: str(d) for c, d in S.empty_long().dtypes.items()}
    lhs = door.sort_values(["component_id", "signal", "timestamp"]).reset_index(drop=True)
    assert len(back) == len(lhs)
    assert (back["signal"].astype("string") == lhs["signal"].astype("string")).all()
    assert (back["component_id"].astype("string") == lhs["component_id"].astype("string")).all()
    np.testing.assert_allclose(back["value"].to_numpy(), lhs["value"].to_numpy(), rtol=1e-6)
    S.validate_long(back)


def test_to_wide_attaches_context_with_asof(door_long):
    wide = S.to_wide(door_long, "door", include_context=True)
    assert "speed" in wide.columns and "T_amb" in wide.columns
    # 10 ms door stream carries the coarse context forward, no resampling needed.
    assert wide["speed"].notna().all()
    assert wide["speed"].max() == pytest.approx(22.0)


def test_to_wide_selects_and_orders_signals(door_long):
    wide = S.to_wide(door_long, "door", signals=["current", "pos", "vel"])
    assert list(wide.columns[6:]) == ["current", "pos", "vel"]
    assert wide["vel"].isna().all()  # not emitted by this run, but requested


def test_to_wide_rejects_duplicate_rows(door_long):
    dup = S.coerce_long(pd.concat([door_long, door_long.head(1)], ignore_index=True))
    with pytest.raises(ValueError, match="duplicate"):
        S.to_wide(dup, "door")


def test_to_wide_unknown_subsystem(door_long):
    with pytest.raises(ValueError, match="unknown subsystem"):
        S.to_wide(door_long, "brakes")


def test_to_long_needs_key_columns_or_scalars(door_long):
    wide = S.to_wide(door_long, "door").drop(columns=["run_id"])
    with pytest.raises(ValueError, match="no 'run_id' column and no run_id= scalar"):
        S.to_long(wide, "door")
    out = S.to_long(wide, "door", run_id="run0")
    assert set(out["run_id"].astype("string").unique()) == {"run0"}


def test_to_long_rejects_unregistered_signal(door_long):
    wide = S.to_wide(door_long, "door").rename(columns={"pos": "position"})
    with pytest.raises(ValueError, match="not registered for subsystem"):
        S.to_long(wide, "door", signals=["position"])


# ---------------------------------------------------------------- validation errors


def test_validate_long_missing_column(door_long):
    with pytest.raises(ValueError, match=r"missing required column\(s\) \['value'\]"):
        S.validate_long(door_long.drop(columns=["value"]))


def test_validate_long_rejects_object_dtype(door_long):
    bad = door_long.copy()
    bad["signal"] = bad["signal"].astype("string")
    with pytest.raises(ValueError, match="must be pandas 'category' dtype"):
        S.validate_long(bad)


def test_validate_long_rejects_naive_timestamp(door_long):
    bad = door_long.copy()
    bad["timestamp"] = bad["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        S.validate_long(bad)


def test_validate_long_rejects_float64_value(door_long):
    bad = door_long.copy()
    bad["value"] = bad["value"].astype("float64")
    with pytest.raises(ValueError, match="must be dtype float32"):
        S.validate_long(bad)


def test_validate_long_rejects_unknown_signal(door_long):
    bad = door_long.copy()
    bad["signal"] = bad["signal"].astype("string").str.replace("pos", "position", regex=False).astype("category")
    with pytest.raises(ValueError, match="not in the registry for their subsystem"):
        S.validate_long(bad)


def test_validate_long_rejects_wrong_component_for_subsystem(door_long):
    bad = door_long.copy()
    cid = bad["component_id"].astype("string")
    bad["component_id"] = cid.where(cid != "door_L1", "axlebox_1L").astype("category")
    with pytest.raises(ValueError, match="not legal for their subsystem"):
        S.validate_long(bad)


def test_validate_long_rejects_infinite_value(door_long):
    bad = door_long.copy()
    bad.loc[bad.index[0], "value"] = np.inf
    with pytest.raises(ValueError, match="infinite value"):
        S.validate_long(bad)


def test_validate_long_allows_nan_dropouts(door_long):
    ok = door_long.copy()
    ok.loc[ok.index[:5], "value"] = np.nan
    S.validate_long(ok)


def test_validate_long_strict_rejects_extra_columns(door_long):
    extra = door_long.assign(unit="m")
    S.validate_long(extra)  # lenient by default
    with pytest.raises(ValueError, match="unexpected column"):
        S.validate_long(extra, strict=True)


def test_validate_features_happy_and_errors(door_features):
    S.validate_features(door_features, require_labels=True)
    with pytest.raises(ValueError, match="no feature columns"):
        S.validate_features(door_features[list(S.FEATURE_KEY_COLUMNS)])
    bad = door_features.copy()
    bad["t_end"] = bad["t_start"] - pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="t_end < t_start"):
        S.validate_features(bad)
    bad2 = door_features.copy()
    bad2["severity"] = np.float32(1.5)
    with pytest.raises(ValueError, match=r"severity must lie in \[0, 1\]"):
        S.validate_features(bad2)
    bad3 = door_features.drop(columns=["rul_s"])
    with pytest.raises(ValueError, match=r"label column\(s\) \['rul_s'\]"):
        S.validate_features(bad3, require_labels=True)
    S.validate_features(bad3)  # labels optional by default


def test_feature_columns_splits_keys_labels_and_metadata(door_features):
    """``feature_columns`` is the single definition of "what the model may see"."""
    assert S.feature_columns(door_features) == ["closing_time", "i_peak"]
    # keys and labels are never features, whatever order they arrive in
    assert not set(S.feature_columns(door_features)) & {
        *S.FEATURE_KEY_COLUMNS,
        *S.LABEL_COLUMNS,
    }
    # meta_ is the escape hatch: present in the table, absent from X, any dtype allowed
    with_meta = door_features.assign(
        meta_shock_count=np.arange(len(door_features), dtype=np.int64),
        meta_class=pd.Categorical(["a"] * len(door_features)),
        meta_test_id=["t"] * len(door_features),
    )
    assert S.feature_columns(with_meta) == ["closing_time", "i_peak"]
    S.validate_features(with_meta, require_labels=True)
    assert S.feature_columns(S.coerce_features(with_meta)) == ["closing_time", "i_peak"]


def test_feature_columns_accepts_a_bare_column_list():
    assert S.feature_columns(["run_id", "i_peak", "meta_level", "severity", "meta_"]) == ["i_peak"]
    assert S.feature_columns(pd.DataFrame()) == []
    assert S.METADATA_PREFIX == "meta_"


def test_validate_features_rejects_a_table_that_is_only_metadata(door_features):
    """A feature table whose only non-key, non-label columns are ``meta_*`` has no features at
    all - saying so beats handing the benchmark an empty X."""
    only_meta = door_features.drop(columns=["closing_time", "i_peak"]).assign(
        meta_shock_count=np.arange(len(door_features), dtype=np.int64)
    )
    with pytest.raises(ValueError, match=r"no feature columns.*meta_shock_count"):
        S.validate_features(only_meta)


def test_validate_fault_log_happy_and_errors(door_fault_log):
    S.validate_fault_log(door_fault_log)
    bad = door_fault_log.copy()
    bad["fault_type"] = pd.Series(["air_leak"], dtype="category")
    with pytest.raises(ValueError, match="is not legal for subsystem 'door'"):
        S.validate_fault_log(bad)
    bad2 = door_fault_log.copy()
    bad2["t_failure"] = bad2["t_onset"] - pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="t_failure < t_onset"):
        S.validate_fault_log(bad2)
    bad3 = door_fault_log.copy()
    bad3["gamma"] = np.float32(0.0)
    with pytest.raises(ValueError, match="gamma must be > 0"):
        S.validate_fault_log(bad3)
    bad4 = door_fault_log.copy()
    bad4["params_json"] = ["{not json"]
    with pytest.raises(ValueError, match="not valid JSON"):
        S.validate_fault_log(bad4)
    bad5 = door_fault_log.copy()
    bad5["shape"] = pd.Series(["ramp"], dtype="category")
    with pytest.raises(ValueError, match="unknown shape"):
        S.validate_fault_log(bad5)


def test_validate_events_rejects_unknown_event(door_events):
    S.validate_events(door_events)
    bad = door_events.copy()
    bad["event"] = pd.Series(["exploded", "reversal"], dtype="category")
    with pytest.raises(ValueError, match="unknown event value"):
        S.validate_events(bad)


def test_validate_scores():
    scores = S.coerce_scores(
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-09-18T09:00:00Z"]),
                "train_id": ["T01"],
                "car": np.array([3], dtype=np.int8),
                "subsystem": ["door"],
                "component_id": ["door_L1"],
                "model": ["iforest"],
                "score": np.array([0.8], dtype=np.float32),
                "threshold": np.array([0.5], dtype=np.float32),
                "alert": [True],
                "top_signals_json": ['["current"]'],
            }
        )
    )
    S.validate_scores(scores)
    bad = scores.copy()
    bad["score"] = np.float32(np.nan)
    with pytest.raises(ValueError, match="must be finite"):
        S.validate_scores(bad)


# ---------------------------------------------------------------- parquet IO


def test_write_read_dataset_round_trip(tmp_path, door_long, door_features, door_fault_log, door_events):
    part = S.write_dataset(
        door_long, door_features, door_fault_log, door_events,
        tmp_path, "sim", "run0", meta={"note": "unit test"},
    )
    assert part == tmp_path / "source=sim" / "run_id=run0"
    assert (part / "telemetry.parquet").exists()
    assert (part / "meta.json").exists()

    ds = S.read_dataset(tmp_path)
    ds.validate(require_labels=True)
    assert len(ds.long) == len(door_long)
    assert len(ds.features) == len(door_features)
    assert len(ds.fault_log) == len(door_fault_log)
    assert len(ds.events) == len(door_events)
    assert {c: str(d) for c, d in ds.long.dtypes.items()} == {c: str(d) for c, d in S.empty_long().dtypes.items()}
    np.testing.assert_allclose(
        ds.long.sort_values(["component_id", "signal", "timestamp"])["value"].to_numpy(),
        door_long.sort_values(["component_id", "signal", "timestamp"])["value"].to_numpy(),
        rtol=1e-6,
    )
    assert ds.meta["partitions"][0]["note"] == "unit test"
    assert ds.meta["partitions"][0]["schema_version"] == S.SCHEMA_VERSION
    summary = ds.summary()
    assert summary["n_long"] == len(door_long)
    assert set(summary["subsystems"]) == {"door", "train"}


def test_write_dataset_accepts_none_tables_and_filters_on_read(tmp_path, door_long):
    S.write_dataset(door_long, None, None, None, tmp_path, "sim", "run0")
    S.write_dataset(door_long, None, None, None, tmp_path, "sim", "run1")
    both = S.read_dataset(tmp_path)
    assert len(both.long) == 2 * len(door_long)
    one = S.read_dataset(tmp_path, run_id="run1")
    assert len(one.long) == len(door_long)
    assert one.features.empty and list(one.features.columns) == list(S.FEATURE_KEY_DTYPES)
    single = S.read_dataset(tmp_path / "source=sim" / "run_id=run0")
    assert len(single.long) == len(door_long)


def test_write_dataset_rejects_bad_source_and_run_id(tmp_path, door_long):
    with pytest.raises(ValueError, match="unknown source"):
        S.write_dataset(door_long, None, None, None, tmp_path, "kaggle", "run0")
    with pytest.raises(ValueError, match="path-safe"):
        S.write_dataset(door_long, None, None, None, tmp_path, "sim", "run/0")


def test_read_dataset_missing_dir(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        S.read_dataset(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no partitions"):
        S.read_dataset(tmp_path / "empty")
