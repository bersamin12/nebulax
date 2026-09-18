"""Runner: config expansion, an end-to-end run of a dummy registered detector on a 500-row
fake dataset, skip-existing, failure capture, and the protocol test that no test-slice index
ever reaches ``thresholds.calibrate`` or ``model.fit``."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nebulax.bench import data as D
from nebulax.bench import runner as R
from nebulax.bench import metrics as M
from nebulax.bench import splits as SP
from nebulax.bench import thresholds as TH
from nebulax.bench.base import AnomalyDetector, Classifier
from nebulax.bench.registry import is_registered, register, unregister

N_ROWS = 500
FAULT_SLICE = slice(430, 470)


# --------------------------------------------------------------------------------------
# A dummy registered detector + classifier, live only for this module
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _dummy_models():
    @register("t_dummy_ad", input_kind="window_stats", family="control", task="ad")
    class DummyAD(AnomalyDetector):
        """Distance from the training mean - enough to find a step change, nothing more."""

        def _fit(self, X, t=None, **kw):
            self.mu_ = X.mean(axis=0)
            self.sd_ = X.std(axis=0) + 1e-9

        def _score(self, X, t=None):
            return np.abs((X - self.mu_) / self.sd_).mean(axis=1)

    @register("t_dummy_cls", input_kind="window_stats", family="control", task="cls")
    class DummyCLS(Classifier):
        def _fit(self, X, y, **kw):
            self.major_ = pd.Series(y).mode().iloc[0]

        def _predict(self, X):
            return np.full(X.shape[0], self.major_)

    @register("t_boom", input_kind="window_stats", family="control", task="ad")
    class Boom(AnomalyDetector):
        def _fit(self, X, t=None, **kw):
            raise RuntimeError("deliberate explosion")

        def _score(self, X, t=None):  # pragma: no cover - never reached
            return np.zeros(len(X))

    yield
    for name in ("t_dummy_ad", "t_dummy_cls", "t_boom"):
        unregister(name)


@pytest.fixture
def fake_data() -> D.BenchData:
    """500 windows on one unit, 10 min apart, with a 40-window step fault inside the test slice."""
    rng = np.random.default_rng(7)
    X = rng.normal(0.0, 1.0, (N_ROWS, 4)).astype(np.float32)
    X[FAULT_SLICE] += 6.0
    t_end = (
        pd.Timestamp("2020-04-01") + pd.to_timedelta(np.arange(N_ROWS) * 600.0, unit="s")
    ).to_numpy().astype("datetime64[ms]")
    unit = np.full(N_ROWS, "T01", dtype=object)
    t_s = D.to_epoch_seconds(t_end)
    events = [D.Event("T01", float(t_s[FAULT_SLICE.start]), float(t_s[FAULT_SLICE.stop - 1]), "air_leak")]
    labels = D._label_rows(t_end, unit, events, H=6 * 3600.0)
    labels["stage"] = labels["is_faulty"].astype(float)
    labels["train_id"] = unit
    labels["run_id"] = np.full(N_ROWS, "fake", dtype=object)
    return D.BenchData(
        dataset="fake",
        subsystem="pneumatic",
        input_kind="window_stats",
        X=X,
        feature_names=[f"f{i}" for i in range(4)],
        t_start=t_end - np.timedelta64(600_000, "ms"),
        t_end=t_end,
        unit=unit,
        group=np.full(N_ROWS, "fake", dtype=object),
        labels=labels,
        events=events,
        window_seconds=600.0,
    )


def _spec(**kw: Any) -> R.RunSpec:
    base = dict(dataset="fake", model="t_dummy_ad", task="ad", input_kind="window_stats", split="temporal_fracs", H=6 * 3600.0)
    base.update(kw)
    return R.RunSpec(**base)


# --------------------------------------------------------------------------------------
# expand
# --------------------------------------------------------------------------------------


def test_expand_makes_the_cartesian_product_of_ablation_lists():
    cfg = {
        "defaults": {"seed": 0, "H": 259200},
        "runs": [
            {
                "name": "b1",
                "dataset": "metropt3",
                "subsystem": "pneumatic",
                "task": "ad",
                "model": ["t_dummy_ad"],
                "split": ["temporal_fracs"],
                "window": [60, 360],
                "contamination": [0.0, 0.05],
                "params": {"k": [1, 2]},
            }
        ],
    }
    specs = R.expand(cfg)
    assert len(specs) == 2 * 2 * 2
    assert {s.window for s in specs} == {60, 360}
    assert {s.contamination for s in specs} == {0.0, 0.05}
    assert {json.dumps(s.params, sort_keys=True) for s in specs} == {'{"k": 1}', '{"k": 2}'}
    assert all(s.H == 259200 and s.block == "b1" for s in specs)
    assert len({s.config_hash for s in specs}) == len(specs)


def test_expand_resolves_input_kind_from_the_registry_by_default():
    (spec,) = R.expand({"runs": [{"dataset": "fake", "model": "t_dummy_ad"}]})
    assert spec.input_kind == "window_stats"


def test_expand_rejects_unknown_keys_and_missing_dataset():
    with pytest.raises(ValueError, match="unknown key"):
        R.expand({"runs": [{"dataset": "fake", "model": "t_dummy_ad", "wobble": 3}]})
    with pytest.raises(ValueError, match="no 'dataset'"):
        R.expand({"runs": [{"model": "t_dummy_ad"}]})


def test_expand_deduplicates_identical_specs():
    block = {"dataset": "fake", "model": ["t_dummy_ad", "t_dummy_ad"]}
    assert len(R.expand({"runs": [block]})) == 1


def test_config_hash_ignores_only_the_presentation_field():
    """`block` is a label for the config file and touches nothing. Everything else does -
    `max_minutes` included, since a cap decides success versus timeout."""
    assert _spec(block="x").config_hash == _spec(block="y").config_hash
    assert _spec(seed=1).config_hash != _spec().config_hash
    assert _spec(max_minutes=5).config_hash != _spec(max_minutes=99).config_hash


# --------------------------------------------------------------------------------------
# run_one
# --------------------------------------------------------------------------------------


def test_run_one_end_to_end_on_a_fake_dataset(fake_data, tmp_path):
    row = R.run_one(_spec(), cache_dir=None, splits_dir=tmp_path / "splits", data=fake_data)
    assert row["status"] == "ok", row["traceback"]
    assert row["n_rows"] == N_ROWS and row["n_input_features"] == 4
    for col in R.ROW_COLUMNS:
        assert col in row, col
    for col in ("vus_pr", "auprc", "auroc", "budget_threshold", "budget_event_recall", "budget_n_events"):
        assert col in row, col
    # the step fault sits in the test slice and is large -> a mean-distance detector finds it
    assert row["budget_n_events"] == 1
    assert row["budget_event_recall"] == 1.0
    assert row["auprc"] > 0.5
    assert row["fit_seconds"] >= 0.0 and np.isfinite(row["score_seconds_per_10k"])
    assert row["peak_rss_mb"] > 0.0
    assert row["git_rev"]
    # the split audit parquet exists and is joinable on config_hash
    audit = pd.read_parquet(tmp_path / "splits" / f"{_spec().config_hash}.parquet")
    assert set(audit["part"]) == {"train", "val", "test"}
    assert audit["config_hash"].iloc[0] == _spec().config_hash


def test_run_one_records_a_failure_instead_of_raising(fake_data):
    row = R.run_one(_spec(model="t_boom"), cache_dir=None, data=fake_data)
    assert row["status"] == "failed"
    assert "deliberate explosion" in row["traceback"]
    assert row["config_hash"] == _spec(model="t_boom").config_hash


def test_run_one_cls_task(fake_data):
    row = R.run_one(_spec(model="t_dummy_cls", task="cls"), cache_dir=None, data=fake_data)
    assert row["status"] == "ok", row["traceback"]
    assert 0.0 <= row["macro_f1"] <= 1.0
    assert "confusion_matrix" in row


def test_run_one_refuses_a_mismatched_input_kind(fake_data):
    row = R.run_one(_spec(input_kind="raw_window"), cache_dir=None, data=fake_data)
    assert row["status"] == "failed" and "input_kind" in row["traceback"]


# --------------------------------------------------------------------------------------
# THE protocol test
# --------------------------------------------------------------------------------------


def test_no_test_index_reaches_calibrate_or_fit(fake_data, monkeypatch):
    """`test_data_in_threshold_calibration` is on the ladder's forbidden list. This asserts it
    structurally: every index array handed to ``_fit_model`` or to ``thresholds.calibrate`` is
    checked against the split's own test indices, and the test scores must not exist yet when
    calibration runs."""
    spec = _spec()
    (split,) = SP.make_split(spec.split, fake_data, seed=spec.seed)
    test_set = set(split.test.tolist())
    assert test_set, "the fixture must actually have a test slice"

    seen: dict[str, list[np.ndarray]] = {"fit": [], "calibrate": []}
    order: list[str] = []

    real_fit = R._fit_model
    real_calibrate = TH.calibrate
    real_score = R._score

    def spy_fit(model, data, sp, train_idx, **kw):
        seen["fit"].append(np.asarray(train_idx))
        order.append("fit")
        return real_fit(model, data, sp, train_idx, **kw)

    def spy_calibrate(validation, **kw):
        seen["calibrate"].append(np.asarray(validation.index))
        assert validation.part == "val"
        order.append("calibrate")
        return real_calibrate(validation, **kw)

    def spy_score(model, data, idx):
        idx = np.asarray(idx)
        if set(idx.tolist()) & test_set:
            order.append("score_test")
        return real_score(model, data, idx)

    monkeypatch.setattr(R, "_fit_model", spy_fit)
    monkeypatch.setattr(R.T, "calibrate", spy_calibrate)
    monkeypatch.setattr(R, "_score", spy_score)

    row = R.run_one(spec, cache_dir=None, data=fake_data)
    assert row["status"] == "ok", row["traceback"]

    assert seen["fit"] and seen["calibrate"], "both gates must actually have been exercised"
    for where, arrays in seen.items():
        for idx in arrays:
            leaked = sorted(set(idx.tolist()) & test_set)
            assert not leaked, f"{where} was handed {len(leaked)} test-slice row(s), first={leaked[:5]}"
    # and the ordering: nothing touches the test slice until after calibration
    assert order.index("fit") < order.index("calibrate") < order.index("score_test")


def test_contamination_controls_what_the_model_fits_on(fake_data):
    split = SP.Split(
        name="x",
        train=np.arange(N_ROWS),
        val=np.array([], dtype=np.int64),
        test=np.array([], dtype=np.int64),
        n_rows=N_ROWS,
    )
    clean = D.train_rows(fake_data, split.train, train_regime="normal_only", contamination=0.0)
    assert fake_data.y_binary[clean].sum() == 0

    dirty = D.train_rows(fake_data, split.train, train_regime="normal_only", contamination=0.05, seed=0)
    frac = fake_data.y_binary[dirty].mean()
    assert 0.03 < frac < 0.07, frac

    everything = D.train_rows(fake_data, split.train, train_regime="all")
    assert everything.size == N_ROWS


# --------------------------------------------------------------------------------------
# run(): skip-existing and the combined parquet
# --------------------------------------------------------------------------------------


def test_run_skips_existing_hashes_and_writes_a_combined_parquet(fake_data, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "load_bench", lambda dataset, **kw: fake_data)
    cfg = {"runs": [{"name": "b", "dataset": "fake", "model": ["t_dummy_ad", "t_boom"], "split": ["temporal_fracs"], "H": 21600}]}
    out = tmp_path / "runs"

    first = R.run(cfg, out_dir=out, max_workers=0, cache_dir=None)
    assert len(first) == 2
    assert set(first["status"]) == {"ok", "failed"}
    assert (tmp_path / "runs.parquet").exists()
    files = sorted(out.glob("*.parquet"))
    assert len(files) == 2
    stamps = {p.name: p.stat().st_mtime_ns for p in files}

    calls: list[str] = []
    real_run_one = R.run_one
    monkeypatch.setattr(R, "run_one", lambda spec, **kw: (calls.append(spec.model), real_run_one(spec, **kw))[1])

    second = R.run(cfg, out_dir=out, max_workers=0, cache_dir=None, skip_existing=True)
    assert calls == [], "skip_existing must not re-run a spec that already has a parquet"
    assert len(second) == 2
    assert {p.name: p.stat().st_mtime_ns for p in sorted(out.glob("*.parquet"))} == stamps

    R.run(cfg, out_dir=out, max_workers=0, cache_dir=None, skip_existing=False)
    assert sorted(calls) == ["t_boom", "t_dummy_ad"]


def test_run_dry_run_expands_without_executing(fake_data, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "load_bench", lambda dataset, **kw: pytest.fail("dry run must not load data"))
    cfg = {"runs": [{"dataset": "fake", "model": ["t_dummy_ad"], "window": [60, 360]}]}
    df = R.run(cfg, out_dir=tmp_path / "runs", dry_run=True, max_workers=0)
    assert len(df) == 2 and "config_hash" in df.columns
    assert not list((tmp_path / "runs").glob("*.parquet"))


def test_run_limit(fake_data, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "load_bench", lambda dataset, **kw: fake_data)
    cfg = {"runs": [{"dataset": "fake", "model": ["t_dummy_ad"], "window": [60, 360, 720]}]}
    df = R.run(cfg, out_dir=tmp_path / "runs", max_workers=0, cache_dir=None, limit=1)
    assert len(df) == 1


# --------------------------------------------------------------------------------------
# subprocess execution: real parallelism, per-run wall-clock cap
# --------------------------------------------------------------------------------------

_WORKER_MODULE = '''
"""Models for the subprocess-execution test. Imported by the spawned worker via
NEBULAX_MODEL_MODULES."""
import time

import numpy as np
import pandas as pd

from nebulax.bench import data as D
from nebulax.bench import runner as R
from nebulax.bench.base import AnomalyDetector
from nebulax.bench.registry import register

N = 200


def fake_data(*_a, **_k):
    t = (pd.Timestamp("2020-04-01") + pd.to_timedelta(np.arange(N) * 600.0, unit="s")).to_numpy().astype("datetime64[ms]")
    unit = np.full(N, "T01", dtype=object)
    X = np.zeros((N, 2), dtype=np.float32)
    X[150:170] = 5.0
    labels = D._label_rows(t, unit, [])
    labels["stage"] = np.nan
    labels["train_id"] = unit
    return D.BenchData(
        dataset="fake", subsystem="pneumatic", input_kind="window_stats", X=X,
        feature_names=["a", "b"], t_start=t, t_end=t, unit=unit, group=unit.copy(),
        labels=labels, events=[], window_seconds=600.0,
    )


R.load_bench = fake_data


@register("t_sub_ok", input_kind="window_stats", family="control", task="ad")
class Ok(AnomalyDetector):
    def _fit(self, X, t=None, **kw):
        self.mu_ = X.mean(axis=0)

    def _score(self, X, t=None):
        return np.abs(X - self.mu_).sum(axis=1)


@register("t_sub_slow", input_kind="window_stats", family="control", task="ad")
class Slow(AnomalyDetector):
    def _fit(self, X, t=None, **kw):
        time.sleep(600)

    def _score(self, X, t=None):
        return np.zeros(len(X))
'''


def test_subprocess_runs_capture_ok_and_timeout(tmp_path, monkeypatch):
    """Exercises the real execution path: spawned workers, one row each, and a run that blows
    its wall-clock cap coming back as status='timeout' rather than vanishing."""
    mod_dir = tmp_path / "pkg"
    mod_dir.mkdir()
    (mod_dir / "bench_worker_models.py").write_text(_WORKER_MODULE)
    monkeypatch.syspath_prepend(str(mod_dir))
    monkeypatch.setenv("NEBULAX_MODEL_MODULES", "bench_worker_models")
    monkeypatch.setenv("PYTHONPATH", str(mod_dir))

    cfg = {
        "runs": [
            {
                "dataset": "fake",
                "model": ["t_sub_ok", "t_sub_slow"],
                "input_kind": "window_stats",
                "split": ["temporal_fracs"],
                "max_minutes": 20 / 60.0,  # 20 s
            }
        ]
    }
    df = R.run(cfg, out_dir=tmp_path / "runs", max_workers=2, cache_dir=None)
    assert len(df) == 2
    by_model = df.set_index("model")["status"].to_dict()
    assert by_model["t_sub_ok"] == "ok"
    assert by_model["t_sub_slow"] == "timeout"
    assert "wall-clock cap" in df.set_index("model").loc["t_sub_slow", "traceback"]
    assert len(list((tmp_path / "runs").glob("*.parquet"))) == 2


# --------------------------------------------------------------------------------------
# configs that cannot be benchmarked honestly are refused at expansion time
# --------------------------------------------------------------------------------------


def test_expand_refuses_a_split_preset_that_leaks_its_datasets_groups():
    cfg = {
        "runs": [
            {
                "name": "leaky",
                "dataset": "ottawa",
                "model": ["t_dummy_ad"],
                "split": ["temporal_fracs"],
            }
        ]
    }
    with pytest.raises(ValueError, match="not defined for dataset"):
        R.expand(cfg)


def test_expand_refuses_the_degenerate_metropt_cycle_ad_target():
    cfg = {
        "runs": [
            {
                "name": "degenerate",
                "dataset": "metropt3",
                "model": ["t_dummy_ad"],
                "task": ["ad"],
                "input_kind": ["cycle_features"],
                "split": ["metropt_temporal"],
            }
        ]
    }
    with pytest.raises(ValueError, match="degenerate pointwise target"):
        R.expand(cfg)
    # the same combination is refused however the RunSpec is built
    with pytest.raises(ValueError, match="degenerate pointwise target"):
        R.RunSpec(dataset="metropt3", model="t_dummy_ad", task="ad", input_kind="cycle_features", split="metropt_temporal")
    # ... but the cycle table is still legal for the tasks it is honest for
    R.RunSpec(dataset="metropt3", model="t_dummy_cls", task="cls", input_kind="cycle_features", split="metropt_temporal")


def test_runspec_refuses_a_preset_its_dataset_is_not_declared_for():
    with pytest.raises(ValueError, match="not defined for dataset"):
        R.RunSpec(dataset="cranfield", model="t_dummy_ad", split="temporal_fracs")


# --------------------------------------------------------------------------------------
# an AD run with no operating point must say so
# --------------------------------------------------------------------------------------


def test_an_empty_validation_slice_marks_the_row_uncalibrated(fake_data, monkeypatch):
    no_val = SP.Split(
        name="novel",
        n_rows=N_ROWS,
        train=np.arange(0, 300, dtype=np.int64),
        val=np.array([], dtype=np.int64),
        test=np.arange(300, N_ROWS, dtype=np.int64),
    )
    monkeypatch.setattr(R, "make_split", lambda *a, **kw: [no_val])
    row = R.run_one(_spec(), cache_dir=None, data=fake_data)
    assert row["status"] == "ok", row["traceback"]
    assert row["uncalibrated"] is True
    assert "empty validation slice" in row["uncalibrated_reason"]
    assert "budget_false_alarms_per_train_day" not in row or not np.isfinite(
        row.get("budget_false_alarms_per_train_day", float("nan"))
    )


def test_a_calibrated_run_is_not_marked_uncalibrated_and_carries_validation_metrics(fake_data):
    row = R.run_one(_spec(), cache_dir=None, data=fake_data)
    assert row["status"] == "ok", row["traceback"]
    assert row["uncalibrated"] is False
    for col in ("val_auroc", "val_auprc", "val_vus_pr", "val_alarm_window_rows", "n_val_scoreable_rows"):
        assert col in row, col
    assert np.isfinite(row["budget_false_alarms_per_train_day"])
    assert np.isfinite(row["episode_max_step_s"]) and row["episode_max_step_s"] > 0
    assert row["vus_buffer_windows"] >= 4 and row["vus_buffer_seconds"] > 0


def test_calibration_only_ever_sees_scoreable_validation_rows(fake_data, monkeypatch):
    """The false-alarm budget must be counted against the same population it is reported on:
    the test metrics drop non-scoreable rows, so calibration must drop them too."""
    fake_data.masks["scoreable"] = np.ones(N_ROWS, dtype=bool)
    fake_data.masks["scoreable"][320:360] = False  # a blanked stretch inside the val slice
    (split,) = SP.make_split("temporal_fracs", fake_data)
    blanked = set(np.flatnonzero(~fake_data.masks["scoreable"]).tolist())
    assert blanked & set(split.val.tolist()), "the fixture must blank part of the val slice"

    seen: list[np.ndarray] = []
    real = TH.calibrate
    monkeypatch.setattr(R.T, "calibrate", lambda v, **kw: (seen.append(np.asarray(v.index)), real(v, **kw))[1])
    row = R.run_one(_spec(), cache_dir=None, data=fake_data)
    assert row["status"] == "ok", row["traceback"]
    assert seen and not (set(seen[0].tolist()) & blanked)
    assert row["n_val_excluded_rows"] > 0


# --------------------------------------------------------------------------------------
# the calibration audit trail, and peer-norm / split compatibility
# --------------------------------------------------------------------------------------


def test_the_calibrated_rows_are_persisted_and_joinable(fake_data, tmp_path):
    """thresholds.py promises "any published threshold can be checked against the split
    parquet after the fact". The index array is too big for a results cell, so the row carries
    a digest and count and the runner writes the rows themselves next to the split audit."""
    spec = _spec()
    row = R.run_one(spec, cache_dir=None, splits_dir=tmp_path, data=fake_data)
    assert row["status"] == "ok", row["traceback"]

    audit = pd.read_parquet(tmp_path / f"{spec.config_hash}.calibration.parquet")
    assert set(audit.columns) >= {"fold", "split", "part", "row_index", "config_hash"}
    assert (audit["part"] == "calibrated_on").all()
    assert audit["config_hash"].iloc[0] == spec.config_hash
    assert len(audit) == row["calibrated_on_n"] > 0

    # the digest in the row is reproducible from the persisted indices
    idx = np.sort(audit["row_index"].to_numpy(dtype=np.int64))
    ts = TH.ThresholdSet(thresholds={}, calibrated_on_index=idx, n_val_rows=idx.size,
                         window_seconds=600.0, k_consecutive=3, merge_gap_s=3600.0)
    assert ts.index_digest() == row["calibrated_on_digest"]

    # ... and those rows are validation rows, never test rows
    (split,) = SP.make_split(spec.split, fake_data, seed=spec.seed)
    assert set(idx.tolist()) <= set(split.val.tolist())
    assert not set(idx.tolist()) & set(split.test.tolist())


def test_peer_norm_is_refused_under_a_split_that_holds_out_a_spanned_unit():
    """Door peers are the two doors of one train at one dwell, and the fleet puts them in
    different runs, so the peer group spans run_id. Leave-one-unit-out (train_id) is safe;
    GroupKFold by run_id is not."""
    R.RunSpec(dataset="sim", subsystem="door", model="t_dummy_ad", split="sim_loo_unit", peer_norm=True)
    with pytest.raises(ValueError, match="would leak"):
        R.RunSpec(dataset="sim", subsystem="door", model="t_dummy_ad", split="sim_run_kfold", peer_norm=True)
    # ... and without peer_norm the same split is perfectly fine
    R.RunSpec(dataset="sim", subsystem="door", model="t_dummy_ad", split="sim_run_kfold", peer_norm=False)


def test_peer_norm_is_refused_on_a_subsystem_with_no_siblings():
    with pytest.raises(ValueError, match="not defined for sim subsystem"):
        R.RunSpec(dataset="sim", subsystem="pneumatic", model="t_dummy_ad", split="sim_loo_unit", peer_norm=True)


def test_events_in_uses_the_detection_horizon(fake_data):
    """An event whose [onset - H, failure] window reaches into the slice must be counted in
    the denominator, because event_metrics will happily detect it there."""
    t0 = D.to_epoch_seconds(fake_data.t_end)[0]
    future = D.Event("T01", float(t0 + 10 * 86400.0), float(t0 + 11 * 86400.0), "air_leak")
    fake_data.events = list(fake_data.events) + [future]
    early = np.arange(50)
    assert future not in fake_data.events_in(early, 0.0)
    assert future in fake_data.events_in(early, 11 * 86400.0)


# --------------------------------------------------------------------------------------
# supervised tasks keep their labelled rows; window-level CLS is voted to the declared level
# --------------------------------------------------------------------------------------


def test_normal_only_is_ignored_for_supervised_tasks(fake_data):
    """`normal_only` is a semi-supervised AD idea. Applied to a classifier it deletes every
    faulty row and leaves one class to learn; applied to a regressor it deletes exactly the
    run-to-failure rows the target is defined on."""
    all_train = np.arange(N_ROWS)
    ad = D.train_rows(fake_data, all_train, train_regime="normal_only", task="ad")
    assert fake_data.y_binary[ad].sum() == 0

    for task in ("cls", "rul"):
        sup = D.train_rows(fake_data, all_train, train_regime="normal_only", task=task)
        assert fake_data.y_binary[sup].sum() > 0, f"{task} lost every faulty row"
    assert len(set(np.unique(fake_data.y_class[D.train_rows(fake_data, all_train, task="cls")]))) > 1


def test_rul_training_rows_drop_non_finite_targets(fake_data):
    """No regressor can fit a NaN target; keeping those rows only pushes the failure into the
    model (or, worse, into a silent nan-to-zero somewhere)."""
    finite_before = int(np.isfinite(fake_data.rul_s).sum())
    assert 0 < finite_before < N_ROWS, "the fixture must have both finite and NaN targets"
    idx = D.train_rows(fake_data, np.arange(N_ROWS), train_regime="all", task="rul")
    assert np.isfinite(fake_data.rul_s[idx]).all()
    assert idx.size == finite_before


def test_cls_metrics_are_voted_to_the_level_the_loader_declares(fake_data):
    """Cranfield's raw_window loader declares meta["vote_group"] = "meta_test_id": the unit of
    evaluation is the test, not the window, or a long recording outvotes a short one."""
    fake_data.meta["vote_group"] = "meta_test_id"
    # 10 "tests" of 50 windows each; the truth is constant within a test
    tid = np.repeat([f"t{i}" for i in range(10)], N_ROWS // 10)
    fake_data.labels["meta_test_id"] = tid
    fake_data.labels["fault_type"] = np.where(np.isin(tid, ["t0", "t1"]), "leak", "healthy")

    test_idx = np.arange(N_ROWS)
    pred = fake_data.y_class.copy()
    # flip a minority of windows inside one test: the vote must absorb it, the window score not
    pred[:20] = "healthy"
    out = R._cls_fold_metrics(fake_data, _spec(task="cls"), test_idx, pred)
    assert out["voted"] is True and out["vote_group"] == "meta_test_id"
    assert out["n_test_groups"] == 10
    assert out["macro_f1"] == pytest.approx(1.0), "the vote must survive a minority of bad windows"
    assert out["window_macro_f1"] < 1.0, "the window-level number is kept, not discarded"


def test_cls_metrics_are_not_voted_when_no_group_is_declared(fake_data):
    out = R._cls_fold_metrics(fake_data, _spec(task="cls"), np.arange(N_ROWS), fake_data.y_class.copy())
    assert out["voted"] is False and "n_test_groups" not in out


def test_aggregate_folds_uses_the_union_of_fold_keys():
    """A metric that only exists on a later fold - the threshold columns of the first fold that
    managed to calibrate - was silently dropped when fold 0 happened not to have it."""
    folds = [
        {"auroc": 0.5, "n_test_rows": 10},
        {"auroc": 0.9, "n_test_rows": 10, "budget_event_recall": 1.0, "budget_threshold": 3.0},
    ]
    out = R._aggregate_folds(folds)
    assert out["budget_event_recall"] == pytest.approx(1.0)
    assert out["budget_threshold"] == pytest.approx(3.0)
    assert out["auroc"] == pytest.approx(0.7)
    assert out["n_test_rows"] == 20


def test_peer_norm_is_allowed_under_a_temporal_split_and_checked_not_assumed():
    """A temporal split is safe BECAUSE peer groups are time-coincident (data.py asserts that
    at load time), so all members land on the same side of the cut."""
    R.RunSpec(dataset="sim", subsystem="door", model="t_dummy_ad", split="temporal_fracs", peer_norm=True)
    R.RunSpec(dataset="sim", subsystem="bearing", model="t_dummy_ad", split="sim_run_kfold", peer_norm=True)
    with pytest.raises(ValueError, match="would leak"):
        R.RunSpec(dataset="sim", subsystem="door", model="t_dummy_ad", split="sim_run_kfold", peer_norm=True)


def test_validation_counts_are_summed_over_folds_not_averaged(fake_data, tmp_path):
    """The results row must agree with the audit trail beside it. Averaging the calibration
    counts made a 5-fold run report calibrated_on_n=100 while its calibration.parquet held
    500 rows - the same defect, in the opposite direction, as averaging a recall count."""
    folds = [
        {"calibrated_on_n": 100, "n_val_scoreable_rows": 100, "val_faulty_rows": 2,
         "n_val_excluded_rows": 1, "val_alarm_window_rows": 3, "calibrated_on_min": 10,
         "calibrated_on_max": 99, "auroc": 0.5},
        {"calibrated_on_n": 400, "n_val_scoreable_rows": 400, "val_faulty_rows": 8,
         "n_val_excluded_rows": 4, "val_alarm_window_rows": 7, "calibrated_on_min": 200,
         "calibrated_on_max": 999, "auroc": 0.9},
    ]
    out = R._aggregate_folds(folds)
    assert out["calibrated_on_n"] == 500
    assert out["n_val_scoreable_rows"] == 500
    assert out["val_faulty_rows"] == 10
    assert out["n_val_excluded_rows"] == 5
    assert out["val_alarm_window_rows"] == 10
    # index extremes are extremes, not means: the mean of two row indices is a row nothing used
    assert out["calibrated_on_min"] == 10 and out["calibrated_on_max"] == 999
    assert out["auroc"] == pytest.approx(0.7)  # a rate is still a mean


def test_the_row_count_matches_the_calibration_parquet(fake_data, tmp_path):
    spec = _spec()
    row = R.run_one(spec, cache_dir=None, splits_dir=tmp_path, data=fake_data)
    audit = pd.read_parquet(tmp_path / f"{spec.config_hash}.calibration.parquet")
    assert row["calibrated_on_n"] == len(audit)
    assert row["calibrated_on_min"] == int(audit["row_index"].min())
    assert row["calibrated_on_max"] == int(audit["row_index"].max())


def test_validation_false_alarm_rate_is_micro_averaged(fake_data):
    """Summed episodes over summed train-days, matching the test-side treatment, so the rate
    the report filters on reconciles with the episode count printed beside it."""
    folds = [
        {"threshold_budget_val_episodes": 1, "threshold_budget_val_train_days": 10.0,
         "threshold_budget_val_far_per_train_day": 0.1},
        {"threshold_budget_val_episodes": 5, "threshold_budget_val_train_days": 10.0,
         "threshold_budget_val_far_per_train_day": 0.5},
    ]
    out = R._aggregate_folds(folds)
    assert out["threshold_budget_val_episodes"] == 6
    assert out["threshold_budget_val_train_days"] == pytest.approx(20.0)
    assert out["threshold_budget_val_far_per_train_day"] == pytest.approx(6 / 20.0)


# --------------------------------------------------------------------------------------
# the results row must describe the run it actually was
# --------------------------------------------------------------------------------------


def test_a_spec_that_contradicts_the_registry_is_refused(fake_data):
    """t_dummy_ad is registered ad/window_stats. A run filed under cpd, or under a different
    input_kind, would produce real numbers under a wrong label - worse than a failure."""
    row = R.run_one(_spec(task="cpd"), cache_dir=None, data=fake_data)
    assert row["status"] == "failed" and "registered for task" in row["traceback"]

    with pytest.raises(ValueError, match="registered for task"):
        R._check_registry_contract(_spec(task="cpd"))
    with pytest.raises(ValueError, match="registered for input_kind"):
        R._check_registry_contract(_spec(input_kind="cycle_features"))
    R._check_registry_contract(_spec())  # the honest one passes
    R._check_registry_contract(_spec(model="not_registered_anywhere"))  # left to build()


def test_sim_requires_an_explicit_subsystem():
    """load_sim defaults to door; without this the results row would say subsystem='' while the
    numbers came from the door fleet."""
    with pytest.raises(ValueError, match="needs an explicit subsystem"):
        R.RunSpec(dataset="sim", model="t_dummy_ad", split="sim_loo_unit")
    R.RunSpec(dataset="sim", subsystem="door", model="t_dummy_ad", split="sim_loo_unit")


def test_a_loader_that_returns_another_subsystem_is_refused(fake_data):
    fake_data.subsystem = "bearing"
    row = R.run_one(_spec(subsystem="pneumatic"), cache_dir=None, data=fake_data)
    assert row["status"] == "failed" and "subsystem" in row["traceback"]


def test_the_training_budget_is_part_of_a_deep_runs_identity():
    """run_one hands spec.max_minutes to a deep model's budget_s, so 1 minute and 99 minutes
    train different models and must not collide on one config hash."""
    @register("t_deep_ad", input_kind="window_stats", family="autoencoder", task="ad")
    class Deep(AnomalyDetector):
        def _fit(self, X, t=None, budget_s=None, **kw):
            self.mu_ = X.mean(axis=0)

        def _score(self, X, t=None):
            return np.abs(X - self.mu_).mean(axis=1)

    try:
        d1 = _spec(model="t_deep_ad", max_minutes=1.0)
        d2 = _spec(model="t_deep_ad", max_minutes=99.0)
        assert d1.is_deep and d1.config_hash != d2.config_hash
    finally:
        unregister("t_deep_ad")


def test_max_minutes_per_run_reaches_the_model_budget_not_just_the_timeout(fake_data, tmp_path, monkeypatch):
    seen: list[float | None] = []
    real = R._fit_model
    monkeypatch.setattr(R, "_fit_model", lambda m, d, sp, tr, **kw: (seen.append(kw.get("budget_s")), real(m, d, sp, tr, **kw))[1])
    monkeypatch.setattr(R, "load_bench", lambda *a, **kw: fake_data)
    cfg = {"runs": [{"name": "b", "dataset": "fake", "model": ["t_dummy_ad"], "split": ["temporal_fracs"],
                     "H": 21600, "max_minutes": 90}]}
    R.run(cfg, out_dir=tmp_path / "runs", max_workers=0, skip_existing=False, max_minutes_per_run=2.0,
          cache_dir=None)
    assert seen and seen[0] is not None
    assert seen[0] <= 2.0 * 60.0, f"the model was given the config's budget, not the CLI's: {seen[0]}"


def test_config_hash_does_not_depend_on_registry_state():
    """is_deep reads the registry, so hashing through it made the same spec hash differently
    depending on whether the model happened to be imported yet - and a worker process, which
    imports models separately, could disagree with its parent about what was already done."""
    spec = _spec(model="t_not_registered_yet")
    before = spec.config_hash
    assert not spec.is_deep

    @register("t_not_registered_yet", input_kind="window_stats", family="autoencoder", task="ad")
    class Deep(AnomalyDetector):
        def _fit(self, X, t=None, **kw):
            self.mu_ = X.mean(axis=0)

        def _score(self, X, t=None):
            return np.abs(X - self.mu_).mean(axis=1)

    try:
        assert spec.is_deep, "the fixture must actually flip is_deep"
        assert spec.config_hash == before, "config_hash moved when the registry changed"
    finally:
        unregister("t_not_registered_yet")


def test_every_budget_is_part_of_the_config_hash():
    """A cap changes success versus timeout, and therefore both the row and what skip_existing
    considers done - for a classical model as much as a deep one."""
    assert _spec(max_minutes=1.0).config_hash != _spec(max_minutes=99.0).config_hash
    assert "max_minutes" in _spec().identity
    assert "block" not in _spec().identity  # the only field that is pure presentation


def test_dry_run_resolves_input_kind_auto_from_the_registry(tmp_path):
    """--dry-run exists to check a config before spending hours on it, so it must populate the
    registry first: `input_kind: auto` is unresolvable in a fresh process otherwise."""
    cfg = {"runs": [{"name": "b", "dataset": "metropt3", "model": ["t_dummy_ad"],
                     "split": ["metropt_temporal"], "H": 21600}]}
    df = R.run(cfg, out_dir=tmp_path / "runs", dry_run=True)
    assert len(df) == 1
    assert df["input_kind"].iloc[0] == "window_stats"  # taken from the @register row


def test_expand_rejects_a_spec_that_contradicts_the_registry():
    """--dry-run must report this in a second, rather than a sweep discovering it after
    loading a 6 GB dataset."""
    cfg = {"runs": [{"name": "mislabelled", "dataset": "fake", "model": ["t_dummy_ad"],
                     "task": ["cpd"], "split": ["temporal_fracs"]}]}
    with pytest.raises(ValueError, match="registered for task"):
        R.expand(cfg)
    kind = {"runs": [{"name": "wrongkind", "dataset": "fake", "model": ["t_dummy_ad"],
                      "input_kind": ["raw_window"], "split": ["temporal_fracs"]}]}
    with pytest.raises(ValueError, match="registered for input_kind"):
        R.expand(kind)
    # input_kind: auto takes the registered value, so the honest config expands fine
    ok = {"runs": [{"name": "ok", "dataset": "fake", "model": ["t_dummy_ad"], "split": ["temporal_fracs"]}]}
    assert R.expand(ok)[0].input_kind == "window_stats"


# --------------------------------------------------------------------------------------
# data_kwargs may not falsify an ablation axis
# --------------------------------------------------------------------------------------


def test_data_block_cannot_override_an_ablation_axis():
    """`data:` is a passthrough for loader keywords the results table is NOT indexed by.

    Letting it set an axis was a real leak, not a cosmetic one: `peer_norm: false` on the axis
    plus `data: {peer_norm: true}` underneath loaded peer-normalised features while
    `_check_peer_norm_split` (which reads `spec.peer_norm`) stayed silent, so a sim/door run
    under `sim_run_kfold` could be published as `peer_norm=False` with ~40 k two-member peer
    groups astride the train/test fence.
    """
    cfg = {
        "runs": [
            {
                "name": "sneaky",
                "dataset": "sim",
                "subsystem": "door",
                "model": ["t_dummy_ad"],
                "split": ["sim_run_kfold"],
                "peer_norm": [False],
                "data": {"peer_norm": True},
            }
        ]
    }
    with pytest.raises(ValueError, match="peer_norm"):
        R.expand(cfg)
    # ... and the dataclass refuses it too, for a spec built by any other route
    with pytest.raises(ValueError, match="ablation axes"):
        R.RunSpec(dataset="fake", model="t_dummy_ad", data_kwargs={"H": 0.0})


def test_axis_values_win_over_data_kwargs_in_loader_kwargs():
    """Belt and braces behind the guard above: even if a spec somehow carries a colliding
    ``data_kwargs``, the loader is called with the value the row publishes."""
    spec = R.RunSpec(dataset="fake", model="t_dummy_ad", H=259200.0, feature_set="all", window=60)
    object.__setattr__(spec, "data_kwargs", {"H": 0.0, "feature_set": "mean", "window": 360, "stride": 7})
    kw = spec.loader_kwargs()
    assert (kw["H"], kw["feature_set"], kw["window"]) == (259200.0, "all", 60)
    assert kw["stride"] == 7  # non-axis keywords still pass through


def test_run_one_refuses_a_table_built_with_a_different_axis(fake_data):
    """The row must not claim an axis the loaded table contradicts."""
    fake_data.meta["loader_kwargs"] = {"H": 0.0, "peer_norm": True, "feature_set": "mean"}
    row = R.run_one(_spec(), cache_dir=None, data=fake_data)
    assert row["status"] == "failed"
    assert "refusing to publish a mislabelled run" in row["traceback"]


def test_vus_buffer_is_counted_in_rows_but_published_in_seconds(fake_data):
    """`vus_pr(max_window=w)` counts w in ROWS and rows are spaced by the stride, not by the
    window length, so the buffer has to come from the measured row pitch - otherwise the
    published tolerance is wrong by the overlap factor and drifts across the `window` axis."""
    row = R.run_one(_spec(), cache_dir=None, data=fake_data)
    assert row["status"] == "ok", row["traceback"]
    assert row["vus_row_pitch_s"] == pytest.approx(600.0)
    assert row["vus_buffer_windows"] == 6  # 3600 s / 600 s
    assert row["vus_buffer_seconds"] == pytest.approx(R.VUS_BUFFER_SECONDS)
    assert row["vus_buffer_seconds"] == pytest.approx(row["vus_buffer_windows"] * row["vus_row_pitch_s"])


def test_model_ladder_gives_each_name_one_input_kind_family_and_task():
    """`@register(name, input_kind, family, task)` keys on the NAME alone, so a name that the
    ladder declares twice with different input kinds can never be satisfied by one
    registration - `runner._check_registry_contract` would refuse half its own rows."""
    import collections
    from pathlib import Path

    import yaml

    doc = yaml.safe_load(Path("configs/model_ladder.yaml").read_text())
    rows: list[dict] = []

    def walk(obj):
        if isinstance(obj, dict):
            if {"name", "input_kind", "family"} <= set(obj):
                rows.append(obj)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(doc)
    assert len(rows) > 100, "model_ladder.yaml parsed no model rows - the test is not looking at it"
    for field in ("input_kind", "family", "task"):
        seen: dict[str, set] = collections.defaultdict(set)
        for r in rows:
            seen[str(r["name"])].add(str(r.get(field, "ad" if field == "task" else "")))
        clashes = {n: sorted(v) for n, v in seen.items() if len(v) > 1}
        assert not clashes, f"model_ladder.yaml declares conflicting {field} for {clashes}"


def test_a_sparse_cycle_table_still_forms_episodes(tmp_path):
    """Synthetic door cycle_features: 36 s cycles every ~155 s. The contiguity budget must
    come from the row pitch, or a 40-cycle step fault can never be an episode."""
    rng = np.random.default_rng(3)
    X = rng.normal(0.0, 1.0, (N_ROWS, 4)).astype(np.float32)
    X[FAULT_SLICE] += 6.0
    t_end = (pd.Timestamp("2020-04-01") + pd.to_timedelta(np.arange(N_ROWS) * 155.0, unit="s")).to_numpy().astype("datetime64[ms]")
    unit = np.full(N_ROWS, "T01", dtype=object)
    t_s = D.to_epoch_seconds(t_end)
    events = [D.Event("T01", float(t_s[FAULT_SLICE.start]), float(t_s[FAULT_SLICE.stop - 1]), "friction")]
    labels = D._label_rows(t_end, unit, events, H=6 * 3600.0)
    labels["stage"] = labels["is_faulty"].astype(float)
    labels["train_id"] = unit
    labels["run_id"] = np.full(N_ROWS, "fake", dtype=object)
    data = D.BenchData(
        dataset="fake", subsystem="door", input_kind="window_stats", X=X,
        feature_names=[f"f{i}" for i in range(4)], t_start=t_end - np.timedelta64(36_100, "ms"),
        t_end=t_end, unit=unit, group=np.full(N_ROWS, "fake", dtype=object), labels=labels,
        events=events, window_seconds=36.1,
    )
    row = R.run_one(_spec(subsystem="door", H=3 * 3600.0), cache_dir=None, splits_dir=tmp_path / "splits", data=data)
    assert row["status"] == "ok", row["traceback"]
    assert row["episode_row_pitch_s"] == pytest.approx(155.0)
    assert row["episode_max_step_s"] == pytest.approx(155.0 * 1.5)
    assert row["budget_n_events"] == 1 and row["budget_event_recall"] == 1.0


def _two_component_train(pitch_val: float = 155.0, pitch_test: float | None = None) -> D.BenchData:
    """One train, two doors (run A, run B) interleaved in time; a step fault on door B only,
    inside the test slice. ``pitch_test`` != ``pitch_val`` makes the row density change over
    time, which is what used to give calibration and test different contiguity budgets."""
    rng = np.random.default_rng(11)
    n = 300
    pitch_test = pitch_val if pitch_test is None else pitch_test
    t0 = pd.Timestamp("2020-04-01")
    secs = np.concatenate([np.arange(200) * pitch_val, 200 * pitch_val + np.arange(1, n - 200 + 1) * pitch_test])
    t_a = (t0 + pd.to_timedelta(secs, unit="s")).to_numpy().astype("datetime64[ms]")
    t_b = t_a + np.timedelta64(20, "ms")  # door B cycles 20 ms after door A: fully interleaved
    t_end = np.concatenate([t_a, t_b])
    order = np.argsort(t_end, kind="stable")
    t_end = t_end[order]
    series = np.concatenate([np.full(n, "door_A/door_L1", dtype=object), np.full(n, "door_B/door_L2", dtype=object)])[order]
    group = np.array([s.split("/")[0] for s in series], dtype=object)
    unit = np.full(2 * n, "T01", dtype=object)
    X = rng.normal(0.0, 1.0, (2 * n, 4)).astype(np.float32)
    fault = (series == "door_B/door_L2") & (np.arange(2 * n) >= 2 * n - 80) & (np.arange(2 * n) < 2 * n - 20)
    X[fault] += 6.0
    t_s = D.to_epoch_seconds(t_end)
    events = [D.Event("door_B/door_L2", float(t_s[fault].min()), float(t_s[fault].max()), "friction")]
    labels = D._label_rows(t_end, series, events, H=3 * 3600.0)
    labels["stage"] = labels["is_faulty"].astype(float)
    labels["train_id"] = unit
    labels["run_id"] = group
    return D.BenchData(
        dataset="fake", subsystem="door", input_kind="window_stats", X=X, feature_names=[f"f{i}" for i in range(4)],
        t_start=t_end - np.timedelta64(36_100, "ms"), t_end=t_end, unit=unit, group=group, labels=labels,
        events=events, window_seconds=36.1, series=series,
    )


def test_row_pitch_is_per_component_and_shared_by_calibration_and_test(tmp_path):
    """Two interleaved doors on one train: the pitch is the per-door 155 s (not the pooled
    ~77 s), and the SAME contiguity budget is used for calibration and for the test folds even
    when the row density differs between the two slices."""
    data = _two_component_train(pitch_val=155.0, pitch_test=1000.0)
    assert R._row_pitch_seconds(data) == pytest.approx(M.row_pitch_seconds(data.t_end, data.series))
    row = R.run_one(_spec(subsystem="door", H=3 * 3600.0), cache_dir=None, splits_dir=tmp_path / "splits", data=data)
    assert row["status"] == "ok", row["traceback"]
    assert row["episode_max_step_s"] == pytest.approx(row["max_step_s"])
    assert row["episode_max_step_s"] == pytest.approx(M.max_step_seconds(36.1, row["episode_row_pitch_s"]))
    assert row["episode_row_pitch_s"] > 100.0  # per-door, not the interleaved half-pitch


def test_episodes_and_events_are_per_component_not_per_train(tmp_path):
    """Cycles of two sibling doors can never pool into one '>= 3 consecutive' episode, and a
    fault on door B is credited only to episodes on door B."""
    data = _two_component_train()
    row = R.run_one(_spec(subsystem="door", H=3 * 3600.0), cache_dir=None, splits_dir=tmp_path / "splits", data=data)
    assert row["status"] == "ok", row["traceback"]
    assert row["budget_n_events"] == 1 and row["budget_event_recall"] == 1.0

    # hand-built: 1 above-threshold cycle on door A + 2 on door B, adjacent in the train's timeline
    t = data.t_end[:6]
    s = np.array([0.0, 9.0, 9.0, 9.0, 0.0, 0.0])
    step = M.max_step_seconds(36.1, M.row_pitch_seconds(data.t_end, data.series))
    pooled = M.episodes(s, t, 1.0, k=3, merge_gap_s=3600.0, units=data.unit[:6], max_step_s=step)
    per_component = M.episodes(s, t, 1.0, k=3, merge_gap_s=3600.0, units=data.series[:6], max_step_s=step)
    assert len(pooled) == 1 and per_component == []


def test_val_alarm_window_rows_are_counted_per_component(tmp_path):
    """The validation alarm-window count must key events by series, like the episodes do."""
    data = _two_component_train()
    ev = data.events[0]
    t = D.to_epoch_seconds(data.t_end)
    idx = np.flatnonzero((t >= ev.t_onset - 3 * 3600.0) & (t <= ev.t_failure))
    n = R._val_alarm_window_rows(data, idx, 3 * 3600.0)
    assert n == int((data.series[idx] == ev.unit).sum()) > 0


def test_runner_passes_declared_row_context_and_scores_sequential_models_per_series(tmp_path):
    """A model that declares feature_names / series / unit on _fit or _score receives them for
    exactly the rows it is given; a SEQUENTIAL model is scored one series at a time in time
    order, so interleaved components never share a running statistic."""
    from nebulax.bench.base import AnomalyDetector
    from nebulax.bench.registry import register, unregister

    seen: dict[str, Any] = {}

    @register("t_ctx_seq", input_kind="window_stats", family="control", task="ad")
    class CtxSeq(AnomalyDetector):
        SEQUENTIAL = True

        def _fit(self, X, t=None, *, feature_names=None, series=None, **kw):
            seen["fit_names"] = feature_names
            seen["fit_series"] = np.unique(series).tolist()
            self.mu_ = X.mean(axis=0)

        def _score(self, X, t=None, *, series=None, unit=None):
            # a running cumulative sum: only sane if every call is ONE series in time order
            assert len(set(series.tolist())) == 1, series
            assert np.all(np.diff(M.to_epoch_seconds(t)) >= 0)
            seen.setdefault("score_calls", []).append((series[0], len(X), unit[0]))
            return np.cumsum(np.abs(X - self.mu_).mean(axis=1))

    try:
        data = _two_component_train()
        row = R.run_one(_spec(model="t_ctx_seq", subsystem="door", H=3 * 3600.0), cache_dir=None, splits_dir=tmp_path / "s", data=data)
        assert row["status"] == "ok", row["traceback"]
        assert seen["fit_names"] == data.feature_names
        assert seen["fit_series"] == ["door_A/door_L1", "door_B/door_L2"]
        assert {c[0] for c in seen["score_calls"]} == {"door_A/door_L1", "door_B/door_L2"}
        assert all(c[2] == "T01" for c in seen["score_calls"])
        # scores land back in the caller's row order: recompute one series by hand
        idx = np.flatnonzero(data.series == "door_A/door_L1")[:10]
        m = CtxSeq().fit(data.X[idx], data.t_end[idx], feature_names=data.feature_names, series=data.series[idx])
        direct = R._score(m, data, idx)
        assert np.all(np.diff(direct) >= 0)
    finally:
        unregister("t_ctx_seq")


def test_plain_models_are_called_exactly_as_before(fake_data, tmp_path):
    row = R.run_one(_spec(), cache_dir=None, splits_dir=tmp_path / "s", data=fake_data)
    assert row["status"] == "ok", row["traceback"]


def test_episodes_impossible_is_flagged_not_published_as_zero(tmp_path):
    """One row per series (Cranfield's per-test table): the >= 3-consecutive rule can never
    fire, so recall / FA per train-day are NaN with a flag, not 0.0 / 0.0."""
    data = _two_component_train()
    data.series = np.array([f"rec{i}" for i in range(len(data.series))], dtype=object)
    data.events = [D.Event(data.series[-30], float(D.to_epoch_seconds(data.t_end)[-30]), np.nan, "x")]
    row = R.run_one(_spec(subsystem="door", H=3600.0), cache_dir=None, splits_dir=tmp_path / "s", data=data)
    assert row["status"] == "ok", row["traceback"]
    assert row["episodes_impossible"] is True and row["max_rows_per_series"] == 1
    assert row["val_episodes_impossible"] is True
    assert np.isnan(row["budget_event_recall"]) and np.isnan(row["budget_false_alarms_per_train_day"])
    assert np.isfinite(row["auroc"])


def test_a_thin_validation_slice_does_not_blank_a_measurable_test_slice():
    """The validation-side flag must never overwrite the test-side one: a fold whose validation
    rows are singleton series but whose test slice has long series keeps its false alarms."""
    val = {"uncalibrated": False, "val_episodes_impossible": True, "threshold_budget_silent": True}
    fold = {
        "episodes_impossible": False, "max_rows_per_series": 120, "train_days": 10.0, "n_events": 1,
        "budget_n_events": 1, "budget_n_events_detected": 1, "budget_n_false_alarms": 12,
        "budget_n_episodes": 13, "budget_false_alarms_per_train_day": 1.2, "budget_event_recall": 1.0,
        **val,
    }
    out = R._aggregate_folds([fold])
    assert out["episodes_impossible"] is False and out["val_episodes_impossible"] is True
    assert out["budget_n_false_alarms"] == 12 and out["budget_false_alarms_per_train_day"] == pytest.approx(1.2)


def test_fold_aggregation_keeps_the_measurable_folds():
    good = {
        "episodes_impossible": False, "train_days": 20.0, "n_events": 2, "budget_n_events": 2,
        "budget_n_events_detected": 1, "budget_n_false_alarms": 12, "budget_n_episodes": 13,
        "budget_false_alarms_per_train_day": 0.6, "budget_event_recall": 0.5,
    }
    bad = {
        "episodes_impossible": True, "train_days": float("nan"), "n_events": 1, "budget_n_events": float("nan"),
        "budget_n_events_detected": float("nan"), "budget_n_false_alarms": float("nan"), "budget_n_episodes": float("nan"),
        "budget_false_alarms_per_train_day": float("nan"), "budget_event_recall": float("nan"),
    }
    out = R._aggregate_folds([good] * 4 + [bad])
    assert out["episodes_impossible"] is False and out["n_folds_episodes_impossible"] == 1
    assert out["budget_n_false_alarms"] == 48 and out["budget_false_alarms_per_train_day"] == pytest.approx(48 / 80.0)
    assert out["budget_event_recall"] == pytest.approx(0.5)
    allbad = R._aggregate_folds([bad] * 3)
    assert allbad["episodes_impossible"] is True and np.isnan(allbad["budget_n_false_alarms"])


def test_sequential_models_are_scored_on_fresh_copies(tmp_path):
    """A streaming model that learns inside _score gives the same test scores whether or not
    the validation slice was scored first, and the same per-series scores in any series order."""
    from nebulax.bench.base import AnomalyDetector

    class Streaming(AnomalyDetector):
        SEQUENTIAL = True

        def _fit(self, X, t=None, **kw):
            self.mu_ = X.mean(axis=0)
            self.seen_ = 0

        def _score(self, X, t=None):
            out = np.empty(len(X))
            for j, x in enumerate(X):
                self.seen_ += 1
                out[j] = np.abs(x - self.mu_).mean() * (1.0 + 0.01 * self.seen_)  # state-dependent
            return out

    data = _two_component_train()
    idx_a = np.flatnonzero(data.series == "door_A/door_L1")[:50]
    idx_b = np.flatnonzero(data.series == "door_B/door_L2")[:50]
    m = Streaming().fit(data.X[:100], data.t_end[:100])
    both = R._score(m, data, np.concatenate([idx_a, idx_b]))
    only_b = R._score(m, data, idx_b)
    assert np.allclose(both[50:], only_b), "door B must not see door A's rows"
    again = R._score(m, data, np.concatenate([idx_a, idx_b]))
    assert np.allclose(both, again), "scoring must not mutate the fitted model"
    assert m.seen_ == 0


def test_fabricated_timeline_publishes_no_false_alarm_rate(tmp_path):
    data = _two_component_train()
    data.meta["timeline"] = "fabricated"
    row = R.run_one(_spec(subsystem="door", H=3 * 3600.0), cache_dir=None, splits_dir=tmp_path / "s", data=data)
    assert row["status"] == "ok", row["traceback"]
    assert row["far_not_applicable_reason"] == "fabricated timeline"
    assert np.isnan(row["train_days"]) and np.isnan(row["budget_false_alarms_per_train_day"])
    assert np.isfinite(row["budget_n_false_alarms"]) and np.isfinite(row["budget_event_recall"])
    assert np.isnan(row["threshold_budget_val_far_per_train_day"])
    assert row["budget_applicable"] is False


def test_runner_passes_the_delivered_sampling_rate_to_models_that_declare_it():
    from nebulax.bench.base import AnomalyDetector

    seen = {}

    class NeedsFs(AnomalyDetector):
        def _fit(self, X, t=None, *, fs_hz=None, **kw):
            seen["fit"] = fs_hz
        def _score(self, X, t=None, *, fs_hz=None):
            seen["score"] = fs_hz
            return np.zeros(len(X))

    data = _two_component_train()
    data.meta["fs_hz_out"] = 10500.0
    data.meta["fs_hz"] = 42000.0  # the native rate must lose to the delivered one
    m = NeedsFs()
    idx = np.arange(10)
    ctx = R._context(m._fit, data, idx)
    assert ctx["fs_hz"] == 10500.0
    m.fit(data.X[idx], data.t_end[idx], **ctx)
    R._score(m, data, idx)
    assert seen == {"fit": 10500.0, "score": 10500.0}
    del data.meta["fs_hz_out"]; del data.meta["fs_hz"]
    assert "fs_hz" not in R._context(m._fit, data, idx)


def test_an_unseen_data_kwarg_cannot_be_published(fake_data, tmp_path):
    fake_data.meta["loader_kwargs"] = {"window": 60}
    row = R.run_one(_spec(data_kwargs={"cls_target": "condition"}), cache_dir=None, splits_dir=tmp_path / "s", data=fake_data)
    assert row["status"] == "failed" and "never seen by the loader" in row["traceback"]


def test_positive_rates_are_published(fake_data, tmp_path):
    row = R.run_one(_spec(), cache_dir=None, splits_dir=tmp_path / "s", data=fake_data)
    assert row["status"] == "ok", row["traceback"]
    assert 0.0 < row["test_positive_rate"] < 1.0
    assert row["n_test_faulty_rows"] == pytest.approx(row["test_positive_rate"] * row["n_scored_rows"])
    assert 0.0 <= row["val_positive_rate"] <= 1.0


def test_cpd_is_refused_on_the_degenerate_metropt_cycle_target():
    with pytest.raises(ValueError, match="degenerate"):
        R.RunSpec(dataset="metropt3", model="page_hinkley_cycle_scalar", task="cpd", input_kind="cycle_features", split="metropt_temporal")


def test_positive_rate_is_micro_averaged_over_folds():
    f1 = {"n_scored_rows": 7800, "n_test_faulty_rows": 7200, "test_positive_rate": 7200 / 7800,
          "n_val_scoreable_rows": 1560, "val_faulty_rows": 1420, "val_positive_rate": 1420 / 1560}
    f2 = {"n_scored_rows": 7780, "n_test_faulty_rows": 7180, "test_positive_rate": 7180 / 7780,
          "n_val_scoreable_rows": 1560, "val_faulty_rows": 1420, "val_positive_rate": 1420 / 1560}
    out = R._aggregate_folds([f1, f2])
    assert out["n_test_faulty_rows"] == 14380 and out["n_scored_rows"] == 15580
    assert out["test_positive_rate"] == pytest.approx(14380 / 15580)
    assert out["val_faulty_rows"] == 2840 and out["val_positive_rate"] == pytest.approx(2840 / 3120)


# --------------------------------------------------------------------------------------
# what a deep row says about its own training
# --------------------------------------------------------------------------------------


def test_deep_train_info_reads_a_single_train_loop_and_an_ensemble():
    """``deep.train_loop`` returns {"epochs", "best_loss", "stop_reason", "val_rows"}; the row
    carries the first three under reporting names. A model that never trained carries none."""

    class _M:
        pass

    m = _M()
    assert R._deep_train_info(m) == {}
    m.train_info_ = {"epochs": 7, "best_loss": 0.25, "stop_reason": "early_stopping", "val_rows": 30}
    assert R._deep_train_info(m) == {
        "epochs_run": 7.0,
        "best_val_loss": 0.25,
        "stop_reason": "early_stopping",
    }
    # an ensemble reports its members' mean epochs / loss and the distinct stop reasons
    m.train_info_ = {
        "n_members": 2,
        "members": [
            {"epochs": 2, "best_loss": 1.0, "stop_reason": "budget"},
            {"epochs": 4, "best_loss": 2.0, "stop_reason": "early_stopping"},
        ],
    }
    info = R._deep_train_info(m)
    assert info["epochs_run"] == pytest.approx(3.0)
    assert info["best_val_loss"] == pytest.approx(1.5)
    assert info["stop_reason"] == "budget;early_stopping"


def test_a_deep_run_records_epochs_stop_reason_and_best_val_loss(fake_data):
    """`usad` fits in 1.68 s and scores at chance; without the epoch count that is
    indistinguishable from patience firing at epoch 1. The row must say."""

    @register("t_dummy_deep", input_kind="window_stats", family="control", task="ad")
    class DummyDeep(AnomalyDetector):
        def _fit(self, X, t=None, **kw):
            self.mu_ = X.mean(axis=0)
            self.train_info_ = {"epochs": 11, "best_loss": 0.5, "stop_reason": "budget", "val_rows": 8}

        def _score(self, X, t=None):
            return np.abs(X - self.mu_).mean(axis=1)

    try:
        row = R.run_one(_spec(model="t_dummy_deep"), cache_dir=None, data=fake_data)
    finally:
        unregister("t_dummy_deep")
    assert row["status"] == "ok"
    assert row["epochs_run"] == pytest.approx(11.0)
    assert row["best_val_loss"] == pytest.approx(0.5)
    assert row["stop_reason"] == "budget"


def test_a_non_deep_run_carries_no_training_columns(fake_data):
    """The columns are deep-only: a classical row must not invent an epoch count."""
    row = R.run_one(_spec(), cache_dir=None, data=fake_data)
    assert row["status"] == "ok"
    assert not {"epochs_run", "best_val_loss", "stop_reason"} & set(row)
