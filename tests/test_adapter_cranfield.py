"""Tests for nebulax.adapters.cranfield against the REAL CORD release layout.

The fixtures here are tiny ``.mat`` files written in exactly the layout the delivered files
use - ``Normal.mat`` / ``LackLubrication{1,2}.mat`` / ``Backlash{1,2}.mat`` /
``Spalling{1..8}.mat``, each holding ``(n, 3)`` matrices named
``<class><profile><level><load><rep>`` with columns ``[set point (mm), error (mm),
current (A)]`` - so the parser is exercised on the real shape without carrying 25 MB of data
into the test suite.  Two tests touch ``data/raw/cranfield/`` itself and assert the real
counts; they skip when the release is not in the checkout.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nebulax import schema as S
from nebulax.adapters import cranfield as C
from nebulax.adapters import validate as V

scipy_io = pytest.importorskip("scipy.io")


# --------------------------------------------------------------------------------------
# fixture builders - the real layout, in miniature
# --------------------------------------------------------------------------------------

STROKE_MM = 120.0
FS = C.FS_HZ


def _profile_mm(profile: str, sequences: int = 1) -> np.ndarray:
    """The rig's own commanded set point: 120 mm travel with a wait at each end, out and back.

    Trapezoidal = 5 s travel / 3 s wait, sinusoidal = 6 s / 2 s (PDF section 3); either way one
    out-and-back sequence is 16 s, which is why 5 of them fill an 80 s recording.
    """
    travel_s, wait_s = C.PROFILE_TIMING_S[profile]
    n_tr, n_wt = int(round(travel_s * FS)), int(round(wait_s * FS))
    u = np.arange(n_tr) / n_tr
    ramp = STROKE_MM * (u if profile == "trap" else 0.5 * (1.0 - np.cos(np.pi * u)))
    one = np.concatenate(
        [np.full(n_wt, 0.0), ramp, np.full(n_wt, STROKE_MM), STROKE_MM - ramp]
    )
    return np.tile(one, sequences) + 20.0  # the release's set point runs 20..140 mm


def _matrix(profile: str, *, err_mm: float = 2.0, i_hold: float = 0.40, i_move: float = 0.85,
            sequences: int = 1) -> np.ndarray:
    """One ``(n, 3)`` test matrix: set point, tracking error, drive current."""
    ref = _profile_mm(profile, sequences)
    moving = np.abs(np.diff(ref, prepend=ref[0])) > 1e-9
    err = np.where(moving, err_mm, 0.0)
    cur = np.where(moving, i_move, i_hold)
    return np.column_stack([ref, err, cur]).astype(np.float64)


def _write_file(raw: Path, name: str, cls: str, stage: str, **kw: float) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    variables = {}
    for profile in ("sin", "trap"):
        for load in ("20kg", "40kg", "neg40kg"):
            for rep in (1, 2):
                variables[f"{cls}{profile}{stage}{load}{rep}"] = _matrix(profile, **kw)
    scipy_io.savemat(raw / name, variables)


@pytest.fixture
def cranfield_raw(tmp_path: Path) -> Path:
    """Four of the thirteen release files; the other nine are deliberately absent so that the
    missing-file WARNING path is exercised by every test that uses this fixture."""
    raw = tmp_path / "cranfield"
    _write_file(raw, "Normal.mat", "train", "")
    _write_file(raw, "LackLubrication2.mat", "lub", "2nd", i_move=1.05)
    _write_file(raw, "Backlash1.mat", "back", "1st", err_mm=3.0)
    _write_file(raw, "Spalling8.mat", "point", "8th", i_move=0.78)
    return raw


# --------------------------------------------------------------------------------------
# variable-name parsing (PDF Fig. 8)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,cls,fault,level,profile,load,rep",
    [
        ("trainsin20kg3", "normal", "healthy", 0, "sinusoidal", 20.0, 3),
        ("traintrapneg40kg10", "normal", "healthy", 0, "trapezoidal", -40.0, 10),
        ("lubsin1st20kg1", "lack_of_lubrication", "friction", 1, "sinusoidal", 20.0, 1),
        ("lubtrap2ndneg40kg7", "lack_of_lubrication", "friction", 2, "trapezoidal", -40.0, 7),
        ("backtrap1st40kg9", "backlash", "backlash", 1, "trapezoidal", 40.0, 9),
        ("pointsin3rd20kg10", "spalling", "misalignment", 3, "sinusoidal", 20.0, 10),
        ("pointtrap8thneg40kg10", "spalling", "misalignment", 8, "trapezoidal", -40.0, 10),
    ],
)
def test_parse_variable_reads_the_release_naming_scheme(name, cls, fault, level, profile, load, rep):
    t = C.parse_variable(name, "Whatever")
    assert t is not None
    assert (t.cranfield_class, t.fault_type, t.level) == (cls, fault, level)
    assert (t.motion_profile, t.load_kg, t.rep) == (profile, load, rep)


@pytest.mark.parametrize("name", ["", "__header__", "trainsin20kg", "train20kg1", "wobblesin1st20kg1", "trainsin9th20kg1"])
def test_parse_variable_rejects_anything_else(name):
    assert C.parse_variable(name, "Normal") is None


def test_negative_load_is_the_opposing_40_kgf_case():
    """`neg40kg` is 40 kgf *against* the motion, and must not be confused with `40kg`."""
    assert C.parse_variable("trainsinneg40kg1", "Normal").load_kg == -40.0
    assert C.parse_variable("trainsin40kg1", "Normal").load_kg == 40.0


# --------------------------------------------------------------------------------------
# severity: the ML label, including the spalling saturation
# --------------------------------------------------------------------------------------


def test_severity_label_is_the_ordinal_scale_capped_at_one():
    assert C.severity_from_level(0, "healthy") == 0.0
    assert C.severity_from_level(1, "friction") == pytest.approx(0.2)
    assert C.severity_from_level(2, "backlash") == pytest.approx(0.4)
    # spalling has 8 stages, so the ordinal label saturates from stage 5 on - documented, not a bug
    assert C.N_SPALLING_STAGES == 8
    assert C.severity_from_level(5, "misalignment") == pytest.approx(1.0)
    assert C.severity_from_level(8, "misalignment") == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# channel extraction
# --------------------------------------------------------------------------------------


def test_extract_channels_converts_mm_and_reconstructs_the_measurement():
    """Column 1 is the *error*, so the measured position has to be rebuilt from it."""
    mat = np.column_stack([np.array([20.0, 80.0, 140.0]), np.array([2.0, -1.0, 0.5]), np.array([0.4, 0.9, 0.4])])
    ch = C.extract_channels(mat)
    assert ch["pos_ref"] == pytest.approx([0.020, 0.080, 0.140])
    assert ch["pos_err"] == pytest.approx([0.002, -0.001, 0.0005])
    assert ch["pos"] == pytest.approx([0.018, 0.081, 0.1395])
    assert ch["current"] == pytest.approx([0.4, 0.9, 0.4])


def test_extract_channels_rejects_a_matrix_that_is_not_three_columns():
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        C.extract_channels(np.zeros((10, 4)))


def test_segment_strokes_finds_two_strokes_per_sequence_and_drops_the_waits():
    ref = _profile_mm("trap", sequences=5) * 1e-3
    segs = C.segment_strokes(ref)
    assert len(segs) == 2 * 5  # 5 out-and-back sequences per 80 s recording
    # 5 s of travel at 25 Hz, and no 3 s wait welded onto either end
    assert all(abs((b - a) - 125) <= 2 for a, b in segs)


# --------------------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------------------


def test_discover_reports_present_and_missing_release_files(cranfield_raw: Path):
    found, missing = C.discover(cranfield_raw)
    assert [p.name for p in found] == ["Normal.mat", "Backlash1.mat", "LackLubrication2.mat", "Spalling8.mat"]
    assert len(missing) == 9
    assert "Spalling1.mat" in missing and "Normal.mat" not in missing


def test_discover_ignores_a_mat_file_that_is_not_part_of_the_release(cranfield_raw: Path):
    """An unknown filename carries no condition or stage, so it is skipped, never guessed at."""
    scipy_io.savemat(cranfield_raw / "SomeoneElsesExperiment.mat", {"trainsin20kg1": _matrix("trap")})
    found, _ = C.discover(cranfield_raw)
    assert all(p.name != "SomeoneElsesExperiment.mat" for p in found)


def test_missing_files_are_a_warning_not_an_error(cranfield_raw: Path, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="nebulax.adapters.cranfield"):
        ds = C.load(cranfield_raw)
    ds.validate()
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 9
    assert any("Spalling1.mat" in w for w in warned)
    assert sorted(ds.meta["missing_files"]) == sorted(
        n for n in C.EXPECTED_FILES if n not in {"Normal.mat", "LackLubrication2.mat", "Backlash1.mat", "Spalling8.mat"}
    )
    assert any("4 of the 13" not in u for u in ds.meta["unverified"])


# --------------------------------------------------------------------------------------
# missing / empty raw_dir
# --------------------------------------------------------------------------------------


def test_missing_raw_dir_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        C.load(tmp_path / "nope")


def test_empty_dir_without_manifest_raises_file_not_found(tmp_path: Path):
    empty = tmp_path / "empty_cranfield"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="contains none of the 13"):
        C.load(empty)


def test_manifest_only_dir_returns_empty_valid_dataset(tmp_path: Path):
    """A directory holding only a MANIFEST recording a failed download must not crash the CLI."""
    raw = tmp_path / "cranfield"
    raw.mkdir()
    (raw / "MANIFEST.json").write_text(
        json.dumps(
            {
                "dataset": "cranfield",
                "status": "partial",
                "licence": "unknown",
                "files": [],
                "notes": ["manual download required"],
                "errors": ["DOI resolves to a Cloudflare bot-challenge page"],
            }
        )
    )
    ds = C.load(raw)
    assert isinstance(ds, S.Dataset)
    ds.validate()
    assert len(ds.long) == 0 and len(ds.features) == 0 and len(ds.fault_log) == 0
    assert ds.meta["download_status"] == "partial"
    assert ds.meta["n_files"] == 0 and ds.meta["n_runs"] == 0
    assert len(ds.meta["missing_files"]) == 13


# --------------------------------------------------------------------------------------
# the main parsing path
# --------------------------------------------------------------------------------------


def test_load_parses_the_real_layout(cranfield_raw: Path):
    ds = C.load(cranfield_raw)
    ds.validate()
    S.validate_long(ds.long, strict=False)
    S.validate_features(ds.features, require_labels=False)
    S.validate_fault_log(ds.fault_log, strict=False)
    S.validate_events(ds.events, strict=False)

    assert ds.meta["n_files"] == 4
    assert ds.meta["n_runs"] == 4 * 12  # 2 profiles x 3 loads x 2 reps per file
    assert ds.meta["fs_hz"] == 25.0
    assert set(ds.meta["fault_class_counts"]) == {"healthy", "friction", "backlash", "misalignment"}

    signals = set(ds.long["signal"].astype("string"))
    assert signals == {"pos_ref", "pos", "current", "vel"}
    assert set(ds.long["subsystem"].astype("string")) == {"door"}
    assert set(ds.long["component_id"].astype("string")) == {C.COMPONENT_ID}

    # one run per (file, variable), and the run_id says which
    runs = set(ds.features["run_id"].astype("string"))
    assert "cranfield_Normal_trainsin20kg1" in runs
    assert "cranfield_Spalling8_pointtrap8thneg40kg2" in runs
    assert len(runs) == 48

    # one fault-log row per run, all static bench fixtures
    assert len(ds.fault_log) == 48
    assert set(ds.fault_log["shape"].astype("string")) == {"step"}
    assert ds.fault_log["t_failure"].isna().all()
    params = json.loads(ds.fault_log["params_json"].iloc[0])
    assert {"source_file", "variable", "motion_profile", "load_kg", "rep", "level"} <= set(params)


def test_feature_table_carries_the_test_metadata_columns(cranfield_raw: Path):
    """The test matrix rides along under ``meta_`` - present, never a model input."""
    feat = C.load(cranfield_raw).features
    for col in ("meta_motion_profile", "load_kg", "meta_rep", "meta_level", "meta_class",
                "meta_test_id"):
        assert col in feat.columns, col
        assert feat[col].notna().all()
    assert set(feat["meta_motion_profile"].astype("string")) == {"sinusoidal", "trapezoidal"}
    assert sorted(set(feat["load_kg"].to_numpy())) == [-40.0, 20.0, 40.0]
    assert sorted(set(feat["meta_rep"].to_numpy())) == [1, 2]
    # one group id per recording, and it groups exactly as run_id does - so a
    # leave-one-test-out split never has to re-parse the run_id string
    assert feat.groupby("meta_test_id", observed=True)["run_id"].nunique().max() == 1
    assert feat["meta_test_id"].nunique() == feat["run_id"].nunique() == 48
    assert "Normal:trainsin20kg1" in set(feat["meta_test_id"].astype("string"))


def test_no_feature_column_leaks_the_label(cranfield_raw: Path):
    """The leakage guard: nothing :func:`nebulax.schema.feature_columns` hands the model may be
    the target or a relabelling of it. ``meta_level`` is a bijection of ``severity`` and
    ``meta_class`` of ``fault_type`` - which is exactly why both wear the prefix."""
    feat = C.load(cranfield_raw).features
    x_cols = S.feature_columns(feat)
    assert x_cols
    assert not any(c.startswith(S.METADATA_PREFIX) for c in x_cols)
    assert not {"cranfield_class", "level", "motion_profile", "rep"} & set(feat.columns)
    assert "load_kg" in x_cols  # an operating condition the controller knows: a real feature

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
    # ... and the columns that DO encode it are exactly the ones held back
    meta_level = pd.factorize(feat["meta_level"].to_numpy())[0]
    assert pd.crosstab(meta_level, sev).astype(bool).sum(axis=1).max() == 1


def test_fault_type_and_severity_label_mapping(cranfield_raw: Path):
    feat = C.load(cranfield_raw).features
    by_class = feat.groupby(feat["meta_class"].astype("string"), observed=True)

    healthy = by_class.get_group("normal")
    assert set(healthy["fault_type"].astype("string")) == {"healthy"}
    assert (healthy["severity"].to_numpy() == 0.0).all() and not healthy["is_faulty"].any()

    lube = by_class.get_group("lack_of_lubrication")
    assert set(lube["fault_type"].astype("string")) == {"friction"}
    assert np.allclose(lube["severity"].to_numpy(), 0.4)  # stage 2 -> 0.2 + 0.2*(2-1)

    back = by_class.get_group("backlash")
    assert set(back["fault_type"].astype("string")) == {"backlash"}
    assert np.allclose(back["severity"].to_numpy(), 0.2)

    spall = by_class.get_group("spalling")
    assert set(spall["fault_type"].astype("string")) == {"misalignment"}
    assert np.allclose(spall["severity"].to_numpy(), 1.0)  # stage 8 -> capped
    assert spall["is_faulty"].all()


def test_one_feature_row_per_stroke(cranfield_raw: Path):
    ds = C.load(cranfield_raw)
    per_run = ds.features.groupby("run_id", observed=True).size()
    assert (per_run == 2).all()  # one out-and-back sequence per fixture matrix
    assert (ds.features["t_end"] >= ds.features["t_start"]).all()
    for col in ("duration_s", "i_mean", "i_rms", "i_peak", "pos_ref_range", "pos_err_max", "direction"):
        assert ds.features[col].notna().all(), col
    # the fixture's 120 mm stroke comes back in metres
    assert ds.features["pos_ref_range"].median() == pytest.approx(0.120, rel=0.05)


def test_non_nominal_matrix_lengths_are_recorded_rather_than_rejected(cranfield_raw: Path):
    """The release is 2000 samples per matrix; anything else is worth saying out loud."""
    meta = C.load(cranfield_raw).meta
    assert meta["non_nominal_lengths"]
    assert all(v == 400 for v in meta["non_nominal_lengths"].values())


def test_meta_records_the_stepper_caveat_and_the_unverified_licence(cranfield_raw: Path):
    meta = C.load(cranfield_raw).meta
    assert "STEPPER" in meta["stepper_caveat"]
    assert "PROXY" in meta["stepper_caveat"]
    assert meta["licence"].startswith("UNVERIFIED")
    assert meta["rig"]["screw_lead_mm"] == 5.0
    assert meta["channels"]["voltage"].startswith("ABSENT")


def test_validate_cli_passes_on_the_synthetic_release(cranfield_raw: Path, capsys: pytest.CaptureFixture[str]):
    rc = V.main(["--source", "cranfield", "--raw", str(cranfield_raw), "--quiet"])
    assert rc == 0
    assert "OK: cranfield passes the nebulax schema." in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# the real release, if it is in this checkout
# --------------------------------------------------------------------------------------

REAL_RAW = REPO_ROOT / "data" / "raw" / "cranfield"


def _skip_without_real_data() -> None:
    found, _ = C.discover(REAL_RAW) if REAL_RAW.exists() else ([], [])
    if len(found) != 13:
        pytest.skip(f"the 13-file Cranfield release is not present under {REAL_RAW}")


def test_validate_cli_on_the_real_raw_dir():
    _skip_without_real_data()
    assert V.main(["--source", "cranfield", "--raw", str(REAL_RAW), "--quiet"]) == 0


def test_real_release_counts_match_the_data_description():
    """60 matrices x 13 files, minus the one repetition Backlash1 is short of (PDF section 4)."""
    _skip_without_real_data()
    ds = C.load(REAL_RAW)
    ds.validate()
    assert ds.meta["missing_files"] == []
    assert ds.meta["n_files"] == 13
    assert ds.meta["n_runs"] == 13 * 60 - 1 == 779
    assert ds.meta["unexpected_variables"] == []
    assert ds.meta["non_nominal_lengths"] == {}  # every matrix is the nominal 2000 samples
    assert ds.meta["fault_class_counts"] == {
        "healthy": 60, "backlash": 119, "friction": 120, "misalignment": 480
    }
    # 5 out-and-back sequences per 80 s recording -> 10 strokes
    assert len(ds.features) == 7790
    assert (ds.features.groupby("run_id", observed=True).size() == 10).all()
    assert sorted(set(ds.features["meta_level"].to_numpy())) == list(range(0, 9))
    assert sorted(set(ds.features["load_kg"].to_numpy())) == [-40.0, 20.0, 40.0]
    # the 120 mm stroke and 25 Hz rate the PDF states
    assert float(ds.features["pos_ref_range"].median()) == pytest.approx(0.120, abs=0.002)
    # 5 s of trapezoidal travel and 6 s of sinusoidal, minus the samples at each end of the
    # sinusoid that sit inside the dead band (its speed starts and ends at zero)
    assert float(ds.features["duration_s"].min()) == pytest.approx(5.04, abs=0.1)
    assert float(ds.features["duration_s"].max()) == pytest.approx(5.88, abs=0.15)
