"""Tests for nebulax.adapters.ottawa.

Runs on a small (2-3 file) slice of the real University of Ottawa UORED-VAFCLS raw data so it
stays well under the 60 s budget; skipped outright when the raw files are not present (e.g. a
CI checkout without ``data/raw/ottawa``).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nebulax import schema as S
from nebulax.adapters import ottawa as O

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "ottawa"

# One healthy file and both fault-state files of the same physical bearing (id 1, inner race) -
# enough to exercise every code path (healthy + developing + faulty) in one small load().
_SLICE_FILES = ("H_1_0.csv", "I_1_1.csv", "I_1_2.csv")

pytestmark = pytest.mark.skipif(
    not RAW_DIR.exists() or not all((RAW_DIR / f).exists() for f in _SLICE_FILES),
    reason="data/raw/ottawa raw files not present; run scripts/download_data.py --dataset ottawa",
)


@pytest.fixture(scope="module")
def slice_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("ottawa_slice")
    for f in _SLICE_FILES:
        shutil.copy(RAW_DIR / f, d / f)
    return d


@pytest.fixture(scope="module")
def ds(slice_dir: Path) -> S.Dataset:
    return O.load(slice_dir)


# ---------------------------------------------------------------------------- contract


def test_load_returns_valid_dataset(ds: S.Dataset):
    assert isinstance(ds, S.Dataset)
    ds.validate()  # raises on the first schema violation


def test_pure_and_offline_does_not_touch_raw_dir(slice_dir: Path):
    before = {p.name: p.stat().st_mtime for p in slice_dir.iterdir()}
    O.load(slice_dir)
    after = {p.name: p.stat().st_mtime for p in slice_dir.iterdir()}
    assert before == after
    assert sorted(before) == sorted(_SLICE_FILES)  # load() created no new files either


def test_missing_raw_dir_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        O.load(tmp_path / "nope")


def test_empty_raw_dir_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no \\*.csv files"):
        O.load(tmp_path)


def test_bad_file_name_raises_value_error(tmp_path: Path):
    (tmp_path / "not_a_valid_name.csv").write_text("Accelerometer,Speed\n1,0\n")
    with pytest.raises(ValueError, match="unrecognised file name"):
        O.load(tmp_path)


# ---------------------------------------------------------------------------- long table


def test_long_table_shape_and_signals(ds: S.Dataset):
    # 3 files * 10 one-second windows * 5 engineered signals.
    assert len(ds.long) == 3 * 10 * 5
    S.validate_long(ds.long)
    assert set(ds.long["subsystem"].astype("string")) == {"bearing"}
    assert set(ds.long["source"].astype("string")) == {"ottawa"}
    signals = set(ds.long["signal"].astype("string"))
    assert signals == {"T_box", "vib_rms", "vib_kurt", "vib_crest", "vib_bpfo"}
    values = ds.long["value"].to_numpy(dtype=np.float32)
    assert np.isfinite(values).all()


def test_long_table_component_and_train_ids(ds: S.Dataset):
    assert set(ds.long["train_id"].astype("string")) == {"bearing_01"}
    assert set(ds.long["component_id"].astype("string")) <= set(S.AXLEBOX_COMPONENT_IDS)
    assert (ds.long["car"].to_numpy() == 0).all()


def test_healthy_vs_faulty_kurtosis_differs(ds: S.Dataset):
    """Sanity: the healthy and the faulty recording should not be numerically identical."""
    wide = ds.long[ds.long["signal"].astype("string") == "vib_kurt"]
    healthy = wide.loc[wide["run_id"].astype("string") == "ottawa_H_1_0", "value"]
    faulty = wide.loc[wide["run_id"].astype("string") == "ottawa_I_1_2", "value"]
    assert len(healthy) and len(faulty)
    assert not np.allclose(healthy.to_numpy(), faulty.to_numpy())


# ---------------------------------------------------------------------------- feature table


def test_feature_table_shape_and_labels(ds: S.Dataset):
    # 3 files * (10 one-second + 40 quarter-second) windows.
    assert len(ds.features) == 3 * (10 + 40)
    S.validate_features(ds.features, require_labels=True)
    labels = ds.features[["run_id", "fault_type", "severity", "is_faulty"]].drop_duplicates()
    labels = labels.set_index(labels["run_id"].astype("string"))
    assert labels.loc["ottawa_H_1_0", "fault_type"] == "healthy"
    assert bool(labels.loc["ottawa_H_1_0", "is_faulty"]) is False
    assert float(labels.loc["ottawa_H_1_0", "severity"]) == 0.0
    assert labels.loc["ottawa_I_1_1", "fault_type"] == "inner_race"
    assert float(labels.loc["ottawa_I_1_1", "severity"]) == pytest.approx(0.5)
    assert labels.loc["ottawa_I_1_2", "fault_type"] == "inner_race"
    assert float(labels.loc["ottawa_I_1_2", "severity"]) == pytest.approx(1.0)
    assert bool(labels.loc["ottawa_I_1_2", "is_faulty"]) is True


def test_feature_table_has_envelope_and_time_domain_columns(ds: S.Dataset):
    expected = {
        "rms", "kurtosis", "crest", "skew", "p2p", "shape_factor", "impulse_factor",
        "env_bpfo_energy", "env_bpfi_energy", "env_bsf_energy", "env_ftf_energy",
        "sk_band_fc", "sk_band_bw", "sk_max", "shaft_hz", "window_s",
    }
    assert expected <= set(ds.features.columns)
    assert set(ds.features["window_s"].unique().tolist()) == {0.25, 1.0}


def test_feature_table_bearing_wise_group_column(ds: S.Dataset):
    """groups = bearing id: every row of one physical bearing shares one train_id, and the same
    id is on the feature table as ``meta_bearing_id`` so a split never reads it out of X."""
    assert set(ds.features["train_id"].astype("string")) == {"bearing_01"}
    assert set(ds.features["meta_bearing_id"].to_numpy()) == {1}
    assert "meta_bearing_id" not in S.feature_columns(ds.features)


def test_no_feature_column_encodes_the_label(ds: S.Dataset):
    """The leakage guard: the file name IS the ground truth, so every column that repeats it
    (``meta_class``, ``meta_state``, ``meta_state_label``) must be metadata, and nothing
    :func:`nebulax.schema.feature_columns` hands the model may be a bijection of the label."""
    feat = ds.features
    for col in ("meta_class", "meta_state", "meta_state_label", "meta_bearing_id"):
        assert col in feat.columns and feat[col].notna().all()
    x_cols = S.feature_columns(feat)
    assert x_cols
    assert not any(c.startswith(S.METADATA_PREFIX) for c in x_cols)
    assert not {"ottawa_class", "ottawa_state", "ottawa_state_label", "ottawa_bearing_id"} & set(
        feat.columns
    )

    sev = pd.factorize(feat["severity"].to_numpy(dtype=float))[0]
    ftype = pd.factorize(feat["fault_type"].astype("string").to_numpy())[0]
    for col in x_cols:
        vals = feat[col]
        codes = pd.factorize(
            vals.to_numpy(dtype=float) if pd.api.types.is_numeric_dtype(vals)
            else vals.astype("string").to_numpy()
        )[0]
        for label in (sev, ftype):
            same_groups = pd.Series(codes).nunique() == pd.Series(label).nunique()
            refines = pd.crosstab(codes, label).astype(bool).sum(axis=1).max() == 1
            assert not (same_groups and refines), f"feature {col!r} is a bijection of the label"
    # the metadata really is what the guard is protecting against: meta_state IS the label
    meta_state = pd.factorize(feat["meta_state"].astype("string").to_numpy())[0]
    assert pd.crosstab(meta_state, sev).astype(bool).sum(axis=1).max() == 1


# ---------------------------------------------------------------------------- fault log


def test_fault_log_only_for_non_healthy_recordings(ds: S.Dataset):
    S.validate_fault_log(ds.fault_log)
    assert len(ds.fault_log) == 2  # I_1_1 (developing) + I_1_2 (faulty), H_1_0 has no row
    assert set(ds.fault_log["fault_type"].astype("string")) == {"inner_race"}
    assert set(ds.fault_log["shape"].astype("string")) == {"step"}
    assert ds.fault_log["t_onset"].notna().all()
    assert ds.fault_log["t_failure"].isna().all()  # no run-to-failure trace in this dataset
    for raw in ds.fault_log["params_json"]:
        json.loads(raw)  # parseable


# ---------------------------------------------------------------------------- meta


def test_meta_has_provenance(ds: S.Dataset):
    assert ds.meta["doi"] == "10.17632/y2px5tg92h.5"
    assert ds.meta["licence"] == "CC BY 4.0"
    assert ds.meta["fs_hz"] == O.FS_HZ
    assert "corrections_to_rail_phm" in ds.meta and ds.meta["corrections_to_rail_phm"]
    assert ds.meta["sample_count_anomalies"] == []


# ---------------------------------------------------------------------------- DC coupling


def test_feature_table_carries_both_couplings_and_the_dc_offset(ds: S.Dataset):
    """The DC bias stays visible: DC-coupled rms/crest, AC-coupled rms_ac/crest_ac and the
    offset itself all sit side by side (docs/parameters.md, bearing section 4)."""
    f = ds.features
    assert {"rms_ac", "crest_ac", "dc_offset_ms2"} <= set(f.columns)
    rms = f["rms"].to_numpy(dtype=np.float64)
    rms_ac = f["rms_ac"].to_numpy(dtype=np.float64)
    dc = f["dc_offset_ms2"].to_numpy(dtype=np.float64)
    assert np.isfinite(rms_ac).all() and np.isfinite(dc).all()
    # dc_offset_ms2 IS the window mean, and rms_ac = sqrt(rms^2 - mean^2) <= rms
    np.testing.assert_allclose(dc, f["mean"].to_numpy(dtype=np.float64), rtol=1e-6)
    np.testing.assert_allclose(rms_ac, np.sqrt(np.maximum(rms**2 - dc**2, 0.0)), rtol=1e-3)
    assert (rms_ac <= rms + 1e-6).all()
    # this slice really does contain the pathology: windows whose DC bias exceeds their own AC
    # RMS, where the DC-coupled crest collapses towards 1 and the AC-coupled one does not
    crest = f["crest"].to_numpy(dtype=np.float64)
    crest_ac = f["crest_ac"].to_numpy(dtype=np.float64)
    polluted = np.abs(dc) > rms_ac
    assert polluted.any()
    assert (crest_ac[polluted] > crest[polluted]).all()
    assert crest[polluted].min() < 2.0 < crest_ac[polluted].min()


def test_long_vib_rms_and_crest_are_the_ac_coupled_pair(ds: S.Dataset):
    """vib_rms / vib_crest must be the AC-coupled statistics, vib_kurt the (central) kurtosis."""
    w1 = ds.features[ds.features["window_s"].to_numpy(dtype=np.float64) == 1.0]
    key = w1["run_id"].astype("string") + "|" + w1["t_start"].astype("int64").astype("string")
    long = ds.long
    long_key = long["run_id"].astype("string") + "|" + long["timestamp"].astype("int64").astype("string")
    for signal, column in (("vib_rms", "rms_ac"), ("vib_crest", "crest_ac"), ("vib_kurt", "kurtosis")):
        sel = long["signal"].astype("string") == signal
        got = pd.Series(long.loc[sel, "value"].to_numpy(), index=long_key[sel])
        want = pd.Series(w1[column].to_numpy(), index=key)
        assert len(got) == len(want) == 3 * 10
        np.testing.assert_allclose(got.to_numpy(), want.reindex(got.index).to_numpy(), rtol=1e-6)


def test_meta_records_the_per_record_dc_offset(ds: S.Dataset):
    assert "ac_coupling" in ds.meta and "ac_couple=" in ds.meta["ac_coupling"].replace("=\n", "=")
    offsets = ds.meta["dc_offset_ms2"]
    assert set(offsets) == {"ottawa_H_1_0", "ottawa_I_1_1", "ottawa_I_1_2"}
    assert all(np.isfinite(v) for v in offsets.values())
