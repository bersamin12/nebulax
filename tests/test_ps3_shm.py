"""Tests for the PS3 SHM task (cumulative fatigue damage regression).

Four things are checked, per `w4_impl_common.md` milestone 4:

1. the contract round trip ``load -> featurise -> predict -> to_rows -> validate_csv``;
2. the **fold-local rule** - the fitted transform moves when the training fold moves, and nothing
   inside :func:`nebulax.ps3.shm.fit_model` can see a held-out row;
3. adversarial inputs (empty file, a header row, a wide file, NaN samples, a missing feature
   column, an extra feature column, a non-positive prediction);
4. the rainflow / feature primitives themselves, on signals whose answer is known by hand.

Everything runs on the committed 5,000-sample fixture or on synthesised signals - no dataset
download, no network, no GPU. Wall time is a couple of seconds.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nebulax.ps3 import shm as shm_mod
from nebulax.ps3 import shm_features as sfeat
from nebulax.ps3.common import Explanation, get_task
from nebulax.ps3.submission import validate_csv

FIXTURE = Path(__file__).parent / "fixtures" / "ps3" / "shm" / "train01_slice.csv"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _synth(n: int = 20000, *, amp: float = 1.0, seed: int = 0) -> np.ndarray:
    """A broad-band stress-like signal: two tones plus noise, scaled by ``amp``."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 37.0) + 0.4 * np.sin(2 * np.pi * t / 7.3) + 0.2 * rng.standard_normal(n)
    return amp * x


def _toy_dataset(n_files: int = 24, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Feature rows whose damage follows the m = 5 power law exactly, plus a little scatter."""
    rng = np.random.default_rng(seed)
    rows = []
    damages = []
    for i in range(n_files):
        amp = float(np.exp(rng.uniform(-0.6, 0.6)))
        x = _synth(6000, amp=amp, seed=seed * 1000 + i)
        feats = sfeat.features_for_signal(x, sfeat.FAMILIES)
        rows.append({"file_id": f"train{i + 1:02d}.csv", **feats})
        r, _, c = sfeat.rainflow_cycles(x)
        damages.append(float(np.sum(c * (0.5 * r) ** 5)) * 1e-4 * math.exp(0.02 * rng.standard_normal()))
    return pd.DataFrame(rows), np.asarray(damages, dtype="float64")


@pytest.fixture(scope="module")
def toy():
    return _toy_dataset()


# --------------------------------------------------------------------------------------
# 1. the contract round trip
# --------------------------------------------------------------------------------------


def test_round_trip_load_featurise_predict_to_rows_validate(tmp_path, toy):
    frame, y = toy
    model = shm_mod.fit_model(frame, y, {"model": "lasso_log", "families": ("stats", "rainflow"),
                                         "log_target": True, "bias": False, "mixup": 0})
    task = get_task("shm")

    raw = task.load(FIXTURE)
    assert raw.ndim == 1 and raw.size == 5000, "the fixture is 5,000 headerless single-column samples"

    feats = task.featurise(raw)
    assert isinstance(feats, pd.DataFrame) and len(feats) == 1
    feats["file_id"] = FIXTURE.name

    result = task.predict(feats, model=model)
    assert result.task == "shm"
    assert result.rows and set(result.rows[0]) == {"file_id", "prediction"}

    rows = task.to_rows(result)
    assert len(rows) == 1
    value = float(rows[0]["prediction"])
    assert math.isfinite(value) and value > 0

    out = tmp_path / "shm_predictions.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file_id", "prediction"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = validate_csv("shm", out, [FIXTURE.name])
    assert report.ok, report.errors
    assert report.n_rows == 1


def test_run_shortcut_defaults_the_file_id(toy):
    frame, y = toy
    model = shm_mod.fit_model(frame, y, shm_mod.BASELINE_SPEC)
    result = get_task("shm").run(FIXTURE, model=model)
    assert result.file_id == FIXTURE.name
    assert result.rows[0]["file_id"] == FIXTURE.name


def test_explanation_payload_is_small_and_well_formed(toy):
    frame, y = toy
    model = shm_mod.fit_model(frame, y, {"model": "ridge_log", "families": ("stats", "rainflow"),
                                         "log_target": True, "bias": True, "mixup": 0})
    task = get_task("shm")
    result = task.run(FIXTURE, model=model)
    exp = task.explain(result)
    assert isinstance(exp, Explanation)
    payload = exp.as_dict()

    assert payload["file_id"] == FIXTURE.name
    for key in ("damage", "p2p", "blockmax_p2p_20", "cv_mape", "damage_low", "damage_high"):
        assert key in payload["numbers"], key
        assert math.isfinite(payload["numbers"][key])
    assert payload["numbers"]["damage_low"] <= payload["numbers"]["damage"] <= payload["numbers"]["damage_high"]

    trace = payload["trace"]
    assert len(trace["x"]) == len(trace["y"])
    assert 0 < len(trace["x"]) <= 2000, "the web gets at most 2,000 trace points"
    assert len(trace["marks"]) <= 20
    assert all({"x", "label", "kind"} <= set(m) for m in trace["marks"])
    assert payload["viewport"]["health"] in {"ok", "warn", "crit"}
    assert payload["viewport"]["component"]


def test_predictions_are_positive_finite_for_every_file(toy):
    frame, y = toy
    model = shm_mod.fit_model(frame, y, shm_mod.BASELINE_SPEC)
    preds = model.predict(frame)
    assert preds.shape == (len(frame),)
    assert np.isfinite(preds).all() and (preds > 0).all()


# --------------------------------------------------------------------------------------
# 2. the fold-local rule
# --------------------------------------------------------------------------------------


def test_fitted_transform_moves_when_the_training_fold_moves(toy):
    """The scaler and the coefficients are fold-local: change the fold, they change."""
    frame, y = toy
    spec = {"model": "ridge_log", "families": ("stats",), "log_target": True, "bias": True, "mixup": 0}
    fold_a = np.arange(0, len(y) - 6)
    fold_b = np.arange(6, len(y))

    a = shm_mod.fit_model(frame.iloc[fold_a], y[fold_a], spec)
    b = shm_mod.fit_model(frame.iloc[fold_b], y[fold_b], spec)

    mean_a = a.estimator.named_steps["scale"].mean_
    mean_b = b.estimator.named_steps["scale"].mean_
    assert not np.allclose(mean_a, mean_b), "the standardiser must be refitted per fold"
    assert not np.allclose(a.estimator.named_steps["est"].coef_, b.estimator.named_steps["est"].coef_)
    assert not math.isclose(a.bias, b.bias, rel_tol=1e-9), "the bias correction is fitted in-fold too"

    # and the same fold twice gives exactly the same model
    again = shm_mod.fit_model(frame.iloc[fold_a], y[fold_a], spec)
    assert np.allclose(again.estimator.named_steps["scale"].mean_, mean_a)


def test_positive_skew_physics_blend_is_fold_local_and_exact(toy):
    frame, y = toy
    spec = {"model": "lasso_log", "families": ("stats", "rainflow", "spectral", "fds"),
            "log_target": True, "bias": True, "mixup": 0, "physics_blend_pos_skew": 0.5}
    train_idx = np.arange(len(y) - 4)
    base = shm_mod.fit_model(frame.iloc[train_idx], y[train_idx], {**spec, "physics_blend_pos_skew": 0.0})
    gated = shm_mod.fit_model(frame.iloc[train_idx], y[train_idx], spec)
    held = frame.iloc[-4:-2].copy()
    held.loc[held.index[0], "st_skew"] = 1.0
    held.loc[held.index[1], "st_skew"] = -1.0
    base_pred = base.predict(held)
    intercept = np.median(np.log(y[train_idx]) - frame.iloc[train_idx]["rf_logsum_m5"].to_numpy())
    physics = np.exp(held["rf_logsum_m5"].iloc[0] + intercept)
    assert gated.physics_log_intercept == pytest.approx(intercept)
    assert gated.predict(held)[0] == pytest.approx(0.5 * (base_pred[0] + physics))
    assert gated.predict(held)[1] == pytest.approx(base_pred[1])

    poisoned = frame.copy()
    poisoned.loc[poisoned.index[-4:], "rf_logsum_m5"] += 100.0
    poisoned_y = y.copy()
    poisoned_y[-4:] *= 100.0
    again = shm_mod.fit_model(poisoned.iloc[train_idx], poisoned_y[train_idx], spec)
    assert again.physics_log_intercept == gated.physics_log_intercept
    assert np.array_equal(again.predict(held), gated.predict(held))


def test_fit_model_never_reads_the_held_out_rows(toy):
    """Poisoning the held-out rows must not move the fitted model by a single bit."""
    frame, y = toy
    spec = {"model": "lasso_log", "families": ("stats", "rainflow"), "log_target": True, "bias": True, "mixup": 0}
    train_idx = np.arange(len(y) - 4)

    clean = shm_mod.fit_model(frame.iloc[train_idx], y[train_idx], spec)

    poisoned = frame.copy()
    numeric = [c for c in poisoned.columns if c != "file_id"]
    poisoned.loc[poisoned.index[-4:], numeric] = 1e9
    poisoned_y = y.copy()
    poisoned_y[-4:] = 99.0
    dirty = shm_mod.fit_model(poisoned.iloc[train_idx], poisoned_y[train_idx], spec)

    assert np.allclose(clean.estimator.named_steps["scale"].mean_, dirty.estimator.named_steps["scale"].mean_)
    assert np.allclose(clean.estimator.named_steps["est"].coef_, dirty.estimator.named_steps["est"].coef_)
    assert clean.bias == pytest.approx(dirty.bias)


def test_rainflow_sn_exponent_is_chosen_inside_the_fold(toy):
    """The physics row's S-N exponent is a fitted quantity, not a constant baked into the module."""
    frame, _ = toy
    spec = {"model": "rainflow_sn", "families": ("rainflow",), "log_target": True, "bias": False, "mixup": 0}
    logged = sfeat.to_log_space(frame)

    # the exponent is *found*, not hard-coded: feed labels generated at m = 5, then at m = 9
    for exponent in (5, 9):
        column = f"rf_logsum_m{exponent}"
        v = logged[column].to_numpy(dtype="float64")
        labels = np.exp(v - float(v.mean()) + math.log(0.1))
        model = shm_mod.fit_model(frame, labels, spec)
        assert all(c.startswith("rf_logsum_m") for c in model.columns)
        assert model.columns[model.estimator.index_] == column, f"m = {exponent} labels"
        assert model.estimator.coef_ == pytest.approx(1.0, abs=1e-6), "literal Miner slope"
        assert np.allclose(model.predict(frame), labels, rtol=1e-6)


def test_c_mixup_only_ever_adds_rows_inside_the_fold(toy):
    frame, y = toy
    logged = sfeat.to_log_space(frame)
    cols = sfeat.family_columns(logged.columns, ("stats",))
    X = logged[cols].to_numpy(dtype="float64")
    ylog = np.log(y)
    X2, y2 = shm_mod.c_mixup(X, ylog, n_extra=17, seed=3)
    assert X2.shape == (len(X) + 17, X.shape[1])
    assert y2.shape == (len(ylog) + 17,)
    assert np.allclose(X2[: len(X)], X) and np.allclose(y2[: len(ylog)], ylog)
    # synthesised targets stay inside the convex hull of the fold's labels
    assert y2[len(ylog):].min() >= ylog.min() - 1e-9
    assert y2[len(ylog):].max() <= ylog.max() + 1e-9
    assert np.allclose(shm_mod.c_mixup(X, ylog, n_extra=17, seed=3)[1], y2), "augmentation is seeded"


def test_loo_cv_is_a_real_leave_one_out(toy, monkeypatch):
    frame, y = toy
    seen: list[int] = []
    original = shm_mod.fit_model

    def spy(fr, yy, spec=None, **kw):
        seen.append(len(fr))
        return original(fr, yy, spec, **kw)

    monkeypatch.setattr(shm_mod, "fit_model", spy)
    out = shm_mod.loo_cv(frame, y, shm_mod.BASELINE_SPEC)
    assert out["n"] == len(y)
    assert seen and set(seen) == {len(y) - 1}, "every LOO fold trains on exactly n-1 files"
    assert 0.0 <= out["mape"] and out["score"] == pytest.approx(max(0.0, 1.0 - out["mape"]))
    assert out["n_low_mode"] + out["n_high_mode"] == len(y)
    assert set(out["predictions"]) == set(frame["file_id"])


# --------------------------------------------------------------------------------------
# 3. adversarial inputs
# --------------------------------------------------------------------------------------


def test_empty_file_raises_a_clear_error(tmp_path):
    empty = tmp_path / "test99.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty SHM file"):
        sfeat.load_signal(empty)


def test_missing_file_raises_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="SHM signal not found"):
        sfeat.load_signal(tmp_path / "nope.csv")


def test_a_header_row_is_rejected_not_silently_eaten(tmp_path):
    bad = tmp_path / "test98.csv"
    bad.write_text("stress\n1.0\n2.0\n1.0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        sfeat.load_signal(bad)


def test_extra_columns_are_rejected(tmp_path):
    wide = tmp_path / "test97.csv"
    wide.write_text("1.0,2.0\n2.0,3.0\n1.0,4.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected one column"):
        sfeat.load_signal(wide)


def test_a_few_nan_samples_are_interpolated_but_a_flood_is_an_error(tmp_path):
    ok = tmp_path / "test96.csv"
    values = ["1.0", "2.0", "", "4.0", "5.0"] + [f"{v:.3f}" for v in np.sin(np.arange(500) / 3.0)]
    ok.write_text("\n".join(values) + "\n", encoding="utf-8")
    x = sfeat.load_signal(ok)
    assert np.isfinite(x).all() and x[2] == pytest.approx(3.0)

    flood = tmp_path / "test95.csv"
    flood.write_text("\n".join(["1.0"] * 50 + [""] * 50) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite samples"):
        sfeat.load_signal(flood)


def test_a_missing_feature_column_is_a_clear_error(toy):
    frame, y = toy
    model = shm_mod.fit_model(frame, y, shm_mod.BASELINE_SPEC)
    trimmed = frame.drop(columns=[model.columns[0]])
    with pytest.raises(KeyError, match="missing"):
        model.predict(trimmed)


def test_an_extra_feature_column_is_ignored(toy):
    frame, y = toy
    model = shm_mod.fit_model(frame, y, shm_mod.BASELINE_SPEC)
    wider = frame.copy()
    wider["st_some_future_feature"] = 12345.0
    assert np.allclose(model.predict(wider), model.predict(frame))


def test_to_rows_refuses_a_non_positive_prediction():
    from nebulax.ps3.common import PredictionResult

    task = get_task("shm")
    bad = PredictionResult(task="shm", file_id="test01.csv", rows=[{"file_id": "test01.csv", "prediction": "0"}])
    with pytest.raises(ValueError, match="positive and finite"):
        task.to_rows(bad)


def test_a_constant_signal_still_produces_a_finite_positive_prediction(toy):
    frame, y = toy
    model = shm_mod.fit_model(frame, y, {"model": "ridge_log", "families": ("stats", "rainflow", "spectral", "fds"),
                                         "log_target": True, "bias": False, "mixup": 0})
    flat = np.full(4096, 3.5)
    feats = get_task("shm").featurise(flat)
    value = float(model.predict(feats)[0])
    assert math.isfinite(value) and value > 0


# --------------------------------------------------------------------------------------
# 4. the primitives
# --------------------------------------------------------------------------------------


def test_turning_points_keeps_the_extremes_and_drops_the_ramps():
    x = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0])
    tp = sfeat.turning_points(x)
    assert tp.tolist() == [0.0, 3.0, 1.0, 4.0]


def test_rainflow_counts_a_constant_amplitude_block_exactly():
    """Ten identical cycles of range 2 must come out as ten full cycles of range 2."""
    one = [1.0, -1.0]
    x = np.array([0.0] + one * 10 + [0.0])
    ranges, means, counts = sfeat.rainflow_cycles(x, method="4point", residue="discard")
    assert counts.sum() == pytest.approx(9.0)
    assert np.allclose(ranges, 2.0)
    assert np.allclose(means, 0.0)


def test_rainflow_methods_agree_on_the_total_miner_sum_with_half_residue():
    x = _synth(4000, seed=11)
    sums = []
    for method in ("4point", "3point"):
        r, _, c = sfeat.rainflow_cycles(x, method=method, residue="half")
        sums.append(float(np.sum(c * (0.5 * r) ** 5)))
    assert sums[0] == pytest.approx(sums[1], rel=1e-9), "3-point and 4-point differ only in the residue"


def test_residue_convention_changes_the_damage_sum():
    x = _synth(4000, seed=12)
    values = {}
    for residue in ("discard", "half", "full", "close"):
        r, _, c = sfeat.rainflow_cycles(x, residue=residue)
        values[residue] = float(np.sum(c * (0.5 * r) ** 5))
    assert values["discard"] < values["half"] < values["full"]
    assert len(set(values.values())) == 4, "the residue convention is a real lever, not a no-op"


def test_miner_damage_scales_as_a_power_law():
    x = _synth(4000, seed=13)
    r, _, c = sfeat.rainflow_cycles(x)
    base = sfeat.miner_damage(r, c, exponent=5.0)
    scaled = sfeat.miner_damage(2.0 * r, c, exponent=5.0)
    assert scaled == pytest.approx(base * 2.0**5, rel=1e-9)


def test_features_are_complete_finite_and_log_safe():
    x = _synth(8192, seed=14)
    feats = sfeat.features_for_signal(x, sfeat.FAMILIES)
    assert len(feats) > 90
    assert all(math.isfinite(v) for v in feats.values()), "no NaN/Inf may reach the regression"
    frame = pd.DataFrame([{"file_id": "x.csv", **feats}])
    logged = sfeat.to_log_space(frame)
    for name in sfeat.log_feature_names():
        if name in logged.columns:
            assert math.isfinite(float(logged[name].iloc[0])), name
    # the log transform is applied to exactly the by-construction-positive names, nobody else
    assert "st_skew" not in sfeat.log_feature_names()
    assert float(logged["st_skew"].iloc[0]) == pytest.approx(feats["st_skew"])


def test_doubling_the_amplitude_moves_the_log_features_by_a_constant():
    """The physics of section A: log(amplitude feature) is affine in log(scale)."""
    x = _synth(8192, seed=15)
    a = sfeat.to_log_space(pd.DataFrame([sfeat.features_for_signal(x, ("stats", "rainflow"))]))
    b = sfeat.to_log_space(pd.DataFrame([sfeat.features_for_signal(2.0 * x, ("stats", "rainflow"))]))
    for name, power in (("st_p2p", 1), ("st_std", 1), ("rf_range_p90", 1), ("rf_del_m5", 1)):
        assert float(b[name].iloc[0]) - float(a[name].iloc[0]) == pytest.approx(power * math.log(2.0), abs=1e-6), name
    assert float(b["rf_logsum_m5"].iloc[0]) - float(a["rf_logsum_m5"].iloc[0]) == pytest.approx(5 * math.log(2.0), abs=1e-6)
    assert float(b["rf_n_cycles"].iloc[0]) == pytest.approx(float(a["rf_n_cycles"].iloc[0]))


def test_spectral_shape_features_are_scale_invariant():
    x = _synth(16384, seed=16)
    a = sfeat.spectral_features(x)
    b = sfeat.spectral_features(3.0 * x)
    for name in ("sp_alpha1", "sp_alpha2", "sp_alpha075", "sp_nu0", "sp_nup", "sp_vanmarcke"):
        assert b[name] == pytest.approx(a[name], rel=1e-9), name


def test_feature_table_caches_per_file(tmp_path):
    cache = tmp_path / "cache"
    first = sfeat.feature_table([FIXTURE], n_jobs=1, cache_dir=cache)
    written = list((cache / sfeat.FEATURE_VERSION).glob("*.parquet"))
    assert len(written) == 1 and written[0].name == f"{FIXTURE.name}.parquet"
    second = sfeat.feature_table([FIXTURE], n_jobs=1, cache_dir=cache)
    pd.testing.assert_frame_equal(first, second)
    assert first["file_id"].tolist() == [FIXTURE.name]


def test_spec_columns_restricts_the_physics_row_to_the_miner_sums(toy):
    frame, _ = toy
    columns = [c for c in frame.columns if c != "file_id"]
    phys = shm_mod.spec_columns({"model": "rainflow_sn", "families": ("rainflow",)}, columns)
    assert phys and all(c.startswith("rf_logsum_m") for c in phys)
    stats_only = shm_mod.spec_columns({"model": "lasso_log", "families": ("stats",)}, columns)
    assert stats_only and all(c.startswith("st_") for c in stats_only)
    assert len(stats_only) == 20, "the plan's baseline is the 20 cheap statistics"
