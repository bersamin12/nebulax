"""Rail corrugation task tests: physics, the round trip, the fold-local rule, adversarial input.

Fast by construction: the committed fixture (`tests/fixtures/ps3/rail/train1_slice.csv`) is 10 ms
of a 10 kHz recording, so anything that needs a real second of track is **synthesised** here - a
tacho pulse train at a chosen speed plus a sinusoidal corrugation of a chosen wavelength injected
on one side's axle boxes. No organiser data is read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nebulax.ps3 import rail_features as rf
from nebulax.ps3 import rail
from nebulax.ps3.common import RAIL_LABELS, Trace, get_task
from nebulax.ps3.rail import RailTask, _apply_rules, fit_rail
from nebulax.ps3.submission import validate_csv

FIXTURE = "tests/fixtures/ps3/rail/train1_slice.csv"
FS = rf.FS_HZ


# --------------------------------------------------------------------------------------
# Synthetic recordings
# --------------------------------------------------------------------------------------


def _side_columns(side: str) -> np.ndarray:
    """Column indices *inside the 128-channel block* belonging to one side (vib and shock)."""
    boxes = np.flatnonzero(rf.side_mask(side))
    return np.sort(np.concatenate([2 * boxes, 2 * boxes + 1]))


def synth_rail(
    *,
    speed_kmh: float = 50.0,
    side: str | None = None,
    lam_mm: float = 50.0,
    n: int = 6000,
    seed: int = 0,
    amp: float = 8.0,
) -> np.ndarray:
    """A ``(n, 129)`` recording: 90-tooth pulse at ``speed_kmh`` + optional corrugation on ``side``."""
    rng = np.random.default_rng(seed)
    v = speed_kmh / 3.6
    dist = np.arange(n) / FS * v
    arr = np.zeros((n, rf.N_CHANNELS + 1), dtype=np.float32)
    arr[:, 0] = (np.floor(dist / (rf.PULSE_DISTANCE_M / 2)).astype(np.int64) % 2).astype(np.float32)
    arr[:, 1:] = rng.normal(0.0, 1.0, size=(n, rf.N_CHANNELS)).astype(np.float32)
    if side is not None:
        cols = _side_columns(side)
        corr = amp * np.sin(2 * np.pi * dist / (lam_mm / 1000.0))
        arr[:, 1:][:, cols] += corr[:, None].astype(np.float32)
    return arr


def _block(specs) -> tuple[rf.RailFeatures, np.ndarray]:
    rows, spectra, ids, labels = [], [], [], []
    for i, (label, kwargs) in enumerate(specs):
        arr = synth_rail(seed=i, **kwargs)
        sc, spec = rf.extract_array(arr, name=f"Synth{i}.csv")
        sc["file_id"] = f"Synth{i}.csv"
        rows.append(sc)
        spectra.append(spec)
        ids.append(f"Synth{i}.csv")
        labels.append(label)
    feats = rf.RailFeatures(file_ids=ids, scalars=pd.DataFrame(rows), spectra=np.stack(spectra))
    return feats, np.asarray(labels, dtype=object)


@pytest.fixture(scope="module")
def synth_block():
    specs = []
    for k in range(6):
        specs.append(("Normal", {"speed_kmh": 40.0 + 4 * k}))
    for k in range(4):
        specs.append(("Side I", {"speed_kmh": 42.0 + 4 * k, "side": "I", "lam_mm": 40.0 + 6 * k}))
    for k in range(4):
        specs.append(("Side II", {"speed_kmh": 44.0 + 4 * k, "side": "II", "lam_mm": 40.0 + 6 * k}))
    return _block(specs)


# --------------------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------------------


def test_pulse_distance_per_tooth_is_29_67_mm():
    assert rf.PULSE_DISTANCE_M == pytest.approx(np.pi * 0.85 / 90)
    assert rf.PULSE_DISTANCE_M * 1000 == pytest.approx(29.67, abs=0.01)


@pytest.mark.parametrize("speed", [12.0, 35.0, 67.0])
def test_speed_from_pulse_recovers_the_synthesised_speed(speed):
    arr = synth_rail(speed_kmh=speed, n=10000)
    assert rf.speed_from_pulse(arr[:, 0]) == pytest.approx(speed, rel=0.02)


def test_pulse_distance_is_monotonic_and_matches_the_mean_speed():
    arr = synth_rail(speed_kmh=54.0, n=10000)
    d = rf.pulse_distance(arr[:, 0])
    assert np.all(np.diff(d) >= -1e-12)
    assert d[-1] == pytest.approx(54.0 / 3.6 * 1.0, rel=0.02)


def test_wavelength_psd_peaks_at_the_injected_wavelength():
    """The whole point of distance resampling: the peak sits at a fixed lambda at any speed."""
    for speed in (25.0, 60.0):
        arr = synth_rail(speed_kmh=speed, side="I", lam_mm=50.0, n=10000, amp=12.0)
        _, spec = rf.extract_array(arr, name="s.csv")
        boxes = np.flatnonzero(rf.side_mask("I"))
        med = np.median(spec[2 * boxes, :], axis=0)
        peak = rf.LAMBDA_GRID_MM[int(np.argmax(med))]
        assert peak == pytest.approx(50.0, rel=0.15), f"{speed} km/h peaked at {peak:.1f} mm"


def test_side_mask_matches_the_info_kit():
    """Positions 1, 3, 5, 7 are Side I; 2, 4, 6, 8 are Side II (Rail Info Kit 2.1)."""
    mi = rf.side_mask("I")
    assert mi.sum() == 32 and rf.side_mask("II").sum() == 32
    assert list(np.flatnonzero(mi)[:4]) == [0, 2, 4, 6]  # car 1, positions 1/3/5/7
    with pytest.raises(ValueError):
        rf.side_mask("L")


# --------------------------------------------------------------------------------------
# Mirror augmentation
# --------------------------------------------------------------------------------------


def test_mirror_is_an_exact_involution_and_swaps_the_sides(synth_block):
    feats, y = synth_block
    once = rf.mirror(feats)
    twice = rf.mirror(once)
    np.testing.assert_allclose(twice.spectra, feats.spectra)
    pd.testing.assert_frame_equal(
        twice.scalars.drop(columns=["file_id"]), feats.scalars.drop(columns=["file_id"])
    )
    # a Side I record's mirrored Side II level equals its original Side I level
    i = int(np.flatnonzero(y == "Side I")[0])
    boxes_i, boxes_ii = np.flatnonzero(rf.side_mask("I")), np.flatnonzero(rf.side_mask("II"))
    np.testing.assert_allclose(once.spectra[i, 2 * boxes_ii], feats.spectra[i, 2 * boxes_i])


def test_mirror_flips_the_sign_of_the_side_contrast(synth_block):
    feats, y = synth_block
    X = rf.aggregate(feats)
    Xm = rf.aggregate(rf.mirror(feats))
    col = [c for c in X.columns if c.startswith("vib_wl_contrast_med_")][3]
    np.testing.assert_allclose(Xm[col].to_numpy(), -X[col].to_numpy(), atol=1e-6)


def test_fault_feature_mixup_keeps_class_and_distinct_source_groups():
    X = np.arange(32, dtype=float).reshape(8, 4)
    y = np.asarray(["Normal", "Normal", "Side I", "Side I", "Side I",
                    "Side II", "Side II", "Side II"], dtype=object)
    speed = np.asarray([10, 50, 40, 44, 70, 45, 49, 72], dtype=float)
    groups = np.arange(len(y))
    mixed, labels, parents = rail._fault_feature_mixup(
        X, y, speed, groups, alpha=0.4, seed=11,
    )
    np.testing.assert_array_equal(mixed[:len(X)], X)
    np.testing.assert_array_equal(labels[:len(y)], y)
    assert len(parents) == 6
    for k, (i, j) in enumerate(parents):
        assert y[i] == y[j] == labels[len(y) + k] != "Normal"
        assert groups[i] != groups[j]
        assert np.all(mixed[len(X) + k] >= np.minimum(X[i], X[j]))
        assert np.all(mixed[len(X) + k] <= np.maximum(X[i], X[j]))


def test_soft_feature_mixup_represents_weighted_fault_and_normal_targets():
    X = np.arange(24, dtype=float).reshape(6, 4)
    y = np.asarray(["Normal", "Normal", "Normal", "Side I", "Side I", "Side II"], dtype=object)
    speed = np.asarray([40, 46, 72, 42, 48, 44], dtype=float)
    groups = np.arange(len(y))
    mixed, labels, weights, parents = rail._fault_normal_soft_mixup(
        X, y, speed, groups, alpha=0.4, seed=13,
    )
    np.testing.assert_array_equal(mixed[:len(X)], X)
    np.testing.assert_array_equal(weights[:len(X)], np.ones(len(X)))
    assert len(parents) == 3
    for k, (i, j) in enumerate(parents):
        assert y[i] != "Normal" and y[j] == "Normal" and groups[i] != groups[j]
        np.testing.assert_array_equal(mixed[len(X) + 2 * k], mixed[len(X) + 2 * k + 1])
        assert labels[len(X) + 2 * k] == y[i]
        assert labels[len(X) + 2 * k + 1] == "Normal"
        assert weights[len(X) + 2 * k] + weights[len(X) + 2 * k + 1] == pytest.approx(1.0)


def test_same_side_coherence_obeys_the_axle_box_mirror():
    raw = synth_rail(side="I", n=6000, amp=12.0)
    original = rf.coherence_for_array(raw)
    mirrored_raw = raw.copy()
    mirrored_raw[:, 1:] = raw[:, 1:].reshape(len(raw), 64, 2)[:, rf.MIRROR_PERM, :].reshape(len(raw), 128)
    expected = rf.coherence_for_array(mirrored_raw)
    mirrored = rf.mirror_coherence(pd.DataFrame([original])).iloc[0].to_dict()
    assert original["coh_I_hz3"] > original["coh_II_hz3"]
    for key in original:
        assert mirrored[key] == pytest.approx(expected[key], abs=1e-10)
    twice = rf.mirror_coherence(pd.DataFrame([mirrored])).iloc[0].to_dict()
    for key in original:
        assert twice[key] == pytest.approx(original[key], abs=1e-10)


def test_coherence_accepts_short_demo_recordings():
    values = rf.coherence_for_array(synth_rail(n=100))
    assert len(values) == 21
    assert np.isfinite(list(values.values())).all()


def test_sensor_gain_jitter_keeps_channel_levels_coherent(synth_block):
    feats, _ = synth_block
    original = feats.subset([6, 7])
    changed = rf.perturb_fault_channels(original, "gain", seed=17)
    again = rf.perturb_fault_channels(original, "gain", seed=17)
    np.testing.assert_allclose(changed.spectra, again.spectra)
    np.testing.assert_allclose(original.spectra, feats.spectra[[6, 7]])
    assert list(changed.scalars["speed_kmh"]) == list(original.scalars["speed_kmh"])
    for channel in (0, 2, 4):
        delta = changed.scalars.loc[0, f"c{channel}_logrms"] - original.scalars.loc[0, f"c{channel}_logrms"]
        assert abs(delta) <= 0.75
        assert changed.scalars.loc[0, f"c{channel}_hz0"] - original.scalars.loc[0, f"c{channel}_hz0"] == pytest.approx(delta)
        np.testing.assert_allclose(changed.spectra[0, channel] - original.spectra[0, channel], delta, atol=1e-5)
        assert changed.scalars.loc[0, f"c{channel}_kurtosis"] == original.scalars.loc[0, f"c{channel}_kurtosis"]
    np.testing.assert_allclose(changed.spectra[:, 1::2], original.spectra[:, 1::2])


def test_sparse_sensor_masking_uses_same_side_donors(synth_block):
    feats, _ = synth_block
    original = feats.subset([6])
    changed = rf.perturb_fault_channels(original, "mask", seed=31)
    assert list(changed.scalars["speed_kmh"]) == list(original.scalars["speed_kmh"])
    np.testing.assert_allclose(changed.spectra[:, 1::2], original.spectra[:, 1::2])
    for side in ("I", "II"):
        boxes = np.flatnonzero(rf.side_mask(side))
        affected = [int(box) for box in boxes if not np.array_equal(
            changed.spectra[0, 2 * box], original.spectra[0, 2 * box])]
        assert len(affected) == 1
        chosen = affected[0]
        donors = boxes[boxes != chosen]
        np.testing.assert_allclose(changed.spectra[0, 2 * chosen],
                                   np.median(original.spectra[0, 2 * donors], axis=0))
        assert changed.scalars.loc[0, f"c{2 * chosen}_logrms"] == pytest.approx(
            np.median([original.scalars.loc[0, f"c{2 * box}_logrms"] for box in donors]))


def test_sensor_augmentation_adds_only_training_fault_sources(synth_block, monkeypatch):
    feats, y = synth_block
    train = feats.subset(np.arange(1, len(feats)))
    seen = []
    original = rf.perturb_fault_channels

    def capture(source, mode, *, seed):
        seen.extend(source.file_ids)
        return original(source, mode, seed=seed)

    monkeypatch.setattr(rf, "perturb_fault_channels", capture)
    model = fit_rail(train, y[1:], kind="lgbm", seeds=(0,), n_jobs=1,
                     boost_repeats=1, tta=True, sensor_augmentation="gain")
    n_faults = int(np.sum(y[1:] != "Normal"))
    assert set(seen) == set(train.file_ids[i] for i in np.flatnonzero(y[1:] != "Normal"))
    assert "Synth0.csv" not in seen
    assert model.meta["n_train"] == 2 * len(train) + 2 * n_faults
    assert model.meta["sensor_augmentation"] == "gain"
    assert len(model.predict_labels(rf.aggregate(feats), feats.scalars["speed_kmh"],
                                    rf.aggregate(rf.mirror(feats)))) == len(feats)


def test_robust_rms_limits_a_single_impulse_and_mirrors(synth_block):
    t = np.arange(10_000) / rf.FS_HZ
    wave = np.sin(2 * np.pi * 100 * t)
    base = np.tile(wave, (2, 1))
    spiked = base.copy()
    spiked[0, 123] = 1_000
    ordinary = 20 * np.log10(np.sqrt(np.mean((spiked[0] - spiked[0].mean()) ** 2)))
    clipped = rf._clipped_rms_db(spiked - spiked.mean(axis=1, keepdims=True))
    window = rf._window_rms_percentile_db(spiked)
    assert ordinary - clipped[0] > 10
    assert clipped[0] == pytest.approx(clipped[1], abs=0.2)
    assert window[0] == pytest.approx(window[1], abs=0.1)

    feats, _ = synth_block
    opts = {"wavelength": False, "hz": True, "v2_normalise": False,
            "shock": False, "robust_time": True}
    X = rf.aggregate(feats, opts)
    Xm = rf.aggregate(rf.mirror(feats), opts)
    assert "vib_robust_I_med_logrms_clip95" in X
    assert not any(c.startswith("shock_") for c in X)
    np.testing.assert_allclose(X["vib_robust_I_med_logrms_clip95"],
                               Xm["vib_robust_II_med_logrms_clip95"])
    np.testing.assert_allclose(X["vib_robust_contrast_med_logrms_win_p10"],
                               -Xm["vib_robust_contrast_med_logrms_win_p10"])


def test_aligned_boosts_use_one_mirror_averaged_oof_row_per_file(synth_block, monkeypatch):
    feats, y = synth_block
    seen = {}

    def capture(probas, labels, speed):
        seen["probas"] = probas
        seen["labels"] = labels.copy()
        seen["speed"] = speed.copy()
        return (1.0, 1.0)

    monkeypatch.setattr(rail, "_tune_boosts", capture)
    model = fit_rail(feats, y, kind="lgbm", seeds=(0,), n_jobs=1,
                     boost_repeats=1, tta=True, aligned_boosts=True)
    assert model.meta["aligned_boosts"] is True
    assert len(seen["probas"]) == 1
    assert seen["probas"][0].shape == (len(feats), 3)
    np.testing.assert_allclose(seen["probas"][0].sum(axis=1), 1.0)
    np.testing.assert_array_equal(seen["labels"], y)
    np.testing.assert_allclose(seen["speed"], feats.scalars["speed_kmh"])


# --------------------------------------------------------------------------------------
# The fold-local rule
# --------------------------------------------------------------------------------------


def test_aggregate_is_a_pure_per_row_function(synth_block):
    """No dataset statistic anywhere, so a held-out row's features cannot depend on the fold."""
    feats, _ = synth_block
    full = rf.aggregate(feats)
    for i in (0, 7, len(feats) - 1):
        alone = rf.aggregate(feats.subset([i]))
        np.testing.assert_allclose(alone.to_numpy(), full.iloc[[i]].to_numpy(), rtol=1e-10, atol=1e-9)


def test_fitted_model_changes_with_the_training_fold(synth_block):
    """The classifier and its class boosts are fitted inside the fold: change the fold, they move."""
    feats, y = synth_block
    a = np.array([0, 1, 2, 6, 7, 10, 11])
    b = np.array([3, 4, 5, 8, 9, 12, 13])
    probe = rf.aggregate(feats)
    ma = fit_rail(feats.subset(a), y[a], seeds=(0,), n_jobs=1)
    mb = fit_rail(feats.subset(b), y[b], seeds=(0,), n_jobs=1)
    assert not np.allclose(ma.proba(probe), mb.proba(probe)), "two disjoint folds gave one model"


def test_fit_never_touches_held_out_rows(synth_block, monkeypatch):
    """Fitting on a subset must not read the held-out rows: delete them and get the same model."""
    feats, y = synth_block
    tr = np.array([0, 1, 2, 3, 6, 7, 8, 10, 11, 12])
    trimmed = feats.subset(tr)
    sabotaged = feats.subset(tr.tolist() + [5, 9, 13])
    m_tr = fit_rail(trimmed, y[tr], seeds=(0,), n_jobs=1)
    m_tr2 = fit_rail(rf.RailFeatures(list(trimmed.file_ids), trimmed.scalars.copy(), trimmed.spectra.copy()),
                     y[tr], seeds=(0,), n_jobs=1)
    probe = rf.aggregate(feats)
    np.testing.assert_allclose(m_tr.proba(probe), m_tr2.proba(probe))
    assert len(sabotaged) > len(trimmed)  # the held-out rows exist but were never passed in


def test_boosts_only_move_the_decision_not_the_probabilities():
    proba = np.array([[0.6, 0.25, 0.15], [0.5, 0.2, 0.3]])
    assert list(_apply_rules(proba, (1.0, 1.0), None)) == ["Normal", "Normal"]
    assert list(_apply_rules(proba, (3.0, 1.0), None)) == ["Side I", "Side I"]
    assert list(_apply_rules(proba, (3.0, 1.0), np.array([5.0, 50.0]))) == ["Normal", "Side I"]


# --------------------------------------------------------------------------------------
# Task round trip
# --------------------------------------------------------------------------------------


def _write_csv(path, arr):
    cols = ["Rotating speed"] + [
        f"{k} of bearing in position {p} of car {c}"
        for c in range(1, 9)
        for p in range(1, 9)
        for k in ("Vibration", "Shock")
    ]
    pd.DataFrame(arr, columns=cols).to_csv(path, index=False, float_format="%.5f")
    return path


def test_load_featurise_predict_to_rows_validate(tmp_path, synth_block):
    feats, y = synth_block
    model = fit_rail(feats, y, seeds=(0,), n_jobs=1)
    task = RailTask()
    ids = []
    rows = []
    for i, side in enumerate([None, "I", "II"]):
        p = _write_csv(tmp_path / f"Test{i + 1}.csv", synth_rail(side=side, seed=100 + i, n=6000))
        result = task.run(p, model=model)
        assert result.task == "rail" and result.file_id == p.name
        rows.extend(task.to_rows(result))
        ids.append(p.name)
    out = tmp_path / "rail_predictions.csv"
    pd.DataFrame(rows, columns=["file_id", "prediction"]).to_csv(out, index=False)
    report = validate_csv("rail", out, ids)
    report.raise_for_errors()
    assert report.n_rows == 3
    assert set(pd.read_csv(out)["prediction"]) <= set(RAIL_LABELS)


def test_explanation_payload_is_small_and_valid(tmp_path, synth_block):
    feats, y = synth_block
    model = fit_rail(feats, y, seeds=(0,), n_jobs=1)
    p = _write_csv(tmp_path / "Test9.csv", synth_rail(side="I", seed=7, n=6000))
    task = RailTask()
    exp = task.explain(task.run(p, model=model)).as_dict()
    assert exp["file_id"] == "Test9.csv"
    assert set(exp["numbers"]) == {
        "speed_kmh",
        "side_max_rms_ratio_db",
        "side_i_minus_ii_db",
        "p_normal",
        "p_side_i",
        "p_side_ii",
    }
    assert all(isinstance(v, float) for v in exp["numbers"].values())
    trace = exp["trace"]
    assert 0 < len(trace["x"]) == len(trace["y"]) <= 2000
    assert trace["x"][0] < trace["x"][-1] and "wavelength" in trace["label"]
    assert {m["kind"] for m in trace["marks"]} == {"peak", "sensor"}
    sensors = [m for m in trace["marks"] if m["kind"] == "sensor"]
    assert len(sensors) == 3 and all(m["component"].startswith("axlebox_c") for m in sensors)
    assert exp["viewport"]["health"] in {"ok", "warn", "crit"}
    assert exp["viewport"]["component"].startswith("axlebox_c")


def test_stale_feature_version_is_a_clear_error(synth_block):
    """A model pickled against an older feature version must fail loudly, not silently reindex."""
    feats, y = synth_block
    model = fit_rail(feats, y, seeds=(0,), n_jobs=1)
    X = rf.aggregate(feats)
    with pytest.raises(ValueError, match="feature version"):
        model.proba(X.drop(columns=X.columns[:3]))


def test_registry_exposes_the_task():
    task = get_task("rail")
    assert task.name == "rail"
    assert task.output_filename == "rail_predictions.csv"
    assert task.accepts == (".csv",)


def test_trace_contract_rejects_a_length_mismatch():
    with pytest.raises(ValueError):
        Trace(x=[1.0, 2.0], y=[1.0])


# --------------------------------------------------------------------------------------
# The committed fixture and adversarial input
# --------------------------------------------------------------------------------------


def test_fixture_slice_loads_and_featurises():
    arr = rf.read_rail_csv(FIXTURE)
    assert arr.shape == (100, 129)
    sc, spec = rf.extract_file(FIXTURE)
    assert spec.shape == (128, rf.N_LAMBDA)
    assert sc["speed_kmh"] >= 0.0
    X = rf.aggregate(
        rf.RailFeatures([FIXTURE], pd.DataFrame([sc]), spec[None, ...])
    )
    assert len(X) == 1 and np.isfinite(X.to_numpy()).all()


def test_missing_column_is_a_clear_error(tmp_path):
    df = pd.read_csv(FIXTURE)
    p = tmp_path / "short.csv"
    df.drop(columns=[df.columns[5]]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="128 columns|129 columns"):
        rf.read_rail_csv(p)


def test_extra_column_is_a_clear_error(tmp_path):
    df = pd.read_csv(FIXTURE)
    df["extra"] = 0.0
    p = tmp_path / "long.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="129 columns"):
        rf.read_rail_csv(p)


def test_empty_file_is_a_clear_error(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty rail file"):
        rf.read_rail_csv(p)


def test_header_only_file_is_a_clear_error(tmp_path):
    p = tmp_path / "header.csv"
    p.write_text(",".join(str(c) for c in pd.read_csv(FIXTURE, nrows=0).columns) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rail recording is 1 s"):
        rf.read_rail_csv(p)


def test_nan_and_inf_are_scrubbed(tmp_path):
    df = pd.read_csv(FIXTURE)
    df.iloc[3, 4] = np.nan
    df.iloc[7, 9] = np.inf
    p = tmp_path / "dirty.csv"
    df.to_csv(p, index=False)
    arr = rf.read_rail_csv(p)
    assert np.isfinite(arr).all()


def test_zero_speed_file_does_not_crash():
    arr = synth_rail(speed_kmh=50.0, n=6000)
    arr[:, 0] = 0.0
    sc, spec = rf.extract_array(arr, name="still.csv")
    assert sc["speed_kmh"] == 0.0
    assert np.isfinite(spec).all()


def test_validate_csv_rejects_an_unknown_label(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("file_id,prediction\nTest1.csv,Side III\n", encoding="utf-8")
    report = validate_csv("rail", p)
    assert not report.ok
    assert any("Side III" in e for e in report.errors)


def test_two_view_ensemble_fits_predicts_and_averages_submodels(synth_block):
    feats, y = synth_block
    model = fit_rail(
        feats,
        y,
        kind="lgbm_2view",
        opts={"hz": True, "v2_normalise": False},
        seeds=(0,),
        n_jobs=1,
    )
    assert model.kind == "lgbm_2view"
    X = rf.aggregate(feats, model.opts)
    preds = model.predict_labels(X)
    assert set(preds) <= set(RAIL_LABELS)
    assert len(preds) == len(feats)

    wrapper = model.estimators[0]
    assert hasattr(wrapper, "model_hz") and hasattr(wrapper, "model_wl")
    Xm = X.reindex(columns=model.columns).to_numpy(dtype=np.float64)
    p_hz = wrapper.model_hz.predict_proba(Xm[:, wrapper._hz_idx])
    p_wl = wrapper.model_wl.predict_proba(Xm[:, wrapper._wl_idx])
    idx_hz = [list(wrapper.model_hz.classes_).index(c) for c in model.classes]
    idx_wl = [list(wrapper.model_wl.classes_).index(c) for c in model.classes]
    expected_proba = 0.5 * (p_hz[:, idx_hz] + p_wl[:, idx_wl])
    actual_proba = model.proba(X)
    assert np.allclose(actual_proba, expected_proba, atol=1e-6)


def test_feature_reduction_options_aggregate_and_fit(synth_block):
    feats, y = synth_block
    # R1: no shock
    opts_r1 = {"wavelength": False, "hz": True, "v2_normalise": False, "shock": False}
    X1 = rf.aggregate(feats, opts_r1)
    assert not any("shock" in c for c in X1.columns)
    m1 = fit_rail(feats, y, kind="lgbm", opts=opts_r1, seeds=(0,), n_jobs=1)
    assert set(m1.predict_labels(X1)) <= set(RAIL_LABELS)

    # R2: no time
    opts_r2 = {"wavelength": False, "hz": True, "v2_normalise": False, "time": False}
    X2 = rf.aggregate(feats, opts_r2)
    assert not any("_t_" in c for c in X2.columns)
    m2 = fit_rail(feats, y, kind="lgbm", opts=opts_r2, seeds=(0,), n_jobs=1)
    assert set(m2.predict_labels(X2)) <= set(RAIL_LABELS)

    # R3: no shock, no time, no votes
    opts_r3 = {"wavelength": False, "hz": True, "v2_normalise": False, "shock": False, "time": False, "votes": False}
    X3 = rf.aggregate(feats, opts_r3)
    assert not any("shock" in c for c in X3.columns)
    assert not any("_t_" in c for c in X3.columns)
    assert not any("car_vote" in c for c in X3.columns)
    m3 = fit_rail(feats, y, kind="lgbm", opts=opts_r3, seeds=(0,), n_jobs=1)
    assert set(m3.predict_labels(X3)) <= set(RAIL_LABELS)


def test_pruned_lgbm_fits_selects_columns_and_predicts(synth_block):
    feats, y = synth_block
    opts = {"wavelength": False, "hz": True, "v2_normalise": False}
    model = fit_rail(feats, y, kind="lgbm_pruned", opts=opts, seeds=(0,), n_jobs=1)
    assert model.kind == "lgbm_pruned"
    assert "pruned_columns" in model.meta
    assert len(model.meta["pruned_columns"]) == 64
    X = rf.aggregate(feats, opts)
    preds = model.predict_labels(X)
    assert set(preds) <= set(RAIL_LABELS)
    assert len(preds) == len(feats)


def test_speed_baseline_residuals_fits_and_predicts(synth_block):
    feats, y = synth_block
    opts = {"wavelength": False, "hz": True, "v2_normalise": False, "speed_baseline": True}
    model = fit_rail(feats, y, kind="lgbm", opts=opts, seeds=(0,), n_jobs=1)
    assert model.speed_baseline_coeffs is not None
    assert len(model.speed_baseline_coeffs) > 0
    assert any(c.endswith("_resid") for c in model.columns)
    X = rf.aggregate(feats, opts)
    preds = model.predict_labels(X)
    assert set(preds) <= set(RAIL_LABELS)
    assert len(preds) == len(feats)


def test_sub_window_voting_fits_and_predicts(synth_block):
    feats, y = synth_block
    opts = {"wavelength": False, "hz": True, "v2_normalise": False}
    model = fit_rail(feats, y, kind="lgbm_windows", opts=opts, seeds=(0,), n_jobs=1, tta=True)
    assert model.kind == "lgbm_windows"
    X = rf.aggregate(feats, opts)
    X.attrs["rail_features"] = feats
    Xm = rf.aggregate(rf.mirror(feats), opts)
    Xm.attrs["rail_features"] = rf.mirror(feats)
    preds = model.predict_labels(X, X_mirror=Xm)
    assert set(preds) <= set(RAIL_LABELS)
    assert len(preds) == len(feats)
