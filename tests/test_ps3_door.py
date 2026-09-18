"""Door subsystem tests: segmentation, the fold-local rule, the CSV round trip, bad inputs.

Everything here runs on the committed fixture (`tests/fixtures/ps3/door/train_slice.csv`, the
first three labelled cycles of `Train.csv` verbatim) or on streams synthesised in the test, so
the suite needs neither the organisers' clone nor a fitted artefact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nebulax.ps3.common import (
    Explanation,
    format_door_timestamp,
    get_task,
    load_model,
    parse_door_timestamp,
    read_door_segments,
    save_model,
)
from nebulax.ps3.door import DoorClassifier, DoorTask, write_predictions
from nebulax.ps3.door_features import (
    BASELINE_FEATURES,
    RATIO_CLIP,
    Z_CLIP,
    DoorFeats,
    FoldBaseline,
    cycle_features,
    label_segments,
    load_stream,
    segment,
    segment_by_gaps,
    segment_by_state,
)
from nebulax.ps3.scoring import iou_f1
from nebulax.ps3.submission import validate_csv


@pytest.fixture(scope="module")
def door_dir(pytestconfig: pytest.Config):
    return pytestconfig.rootpath / "tests" / "fixtures" / "ps3" / "door"


@pytest.fixture(scope="module")
def stream(door_dir):
    return load_stream(door_dir / "train_slice.csv")


@pytest.fixture(scope="module")
def answer(door_dir):
    return read_door_segments(door_dir / "train_slice_answer.csv")


@pytest.fixture(scope="module")
def feats(stream):
    return cycle_features(stream, segment(stream))


@pytest.fixture(scope="module")
def labels(feats, answer):
    y, matched = label_segments(feats, answer)
    assert matched.all()
    return np.asarray([str(v) for v in y])


def _fitted(feats, labels, **kwargs) -> DoorClassifier:
    return DoorClassifier(features=BASELINE_FEATURES, **kwargs).fit(feats, labels)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def test_load_canonicalises_columns_and_timestamps(stream, door_dir):
    raw = pd.read_csv(door_dir / "train_slice.csv")
    assert len(stream.table) == len(raw) == 466
    assert stream.t.dtype == np.dtype("datetime64[ms]")
    assert stream.t.iloc[0] == parse_door_timestamp("2023-7-5-0-0-0-0")
    # the header says "Motor electrodynamic force"; the parameter list calls it back-EMF
    assert stream.table["emf"].iloc[0] == raw["Motor electrodynamic force"].iloc[0]
    assert stream.table["current"].iloc[0] == raw["Motor current(mA)"].iloc[0]
    assert stream.extra_columns == ()


def test_zero_padded_and_iso_timestamps_parse(door_dir):
    raw = pd.read_csv(door_dir / "train_slice.csv")
    padded = raw.copy()
    padded["Datetime"] = [
        parse_door_timestamp(v).strftime("%Y-%m-%d-%H-%M-%S-") + f"{parse_door_timestamp(v).microsecond // 1000:03d}"
        for v in raw["Datetime"]
    ]
    assert padded["Datetime"].iloc[0] == "2023-07-05-00-00-00-000"
    loaded = load_stream(padded)
    assert (loaded.t.to_numpy() == load_stream(raw).t.to_numpy()).all()


def test_missing_column_is_a_clear_error(door_dir):
    raw = pd.read_csv(door_dir / "train_slice.csv").drop(columns=["Door leaf position"])
    with pytest.raises(ValueError, match="missing required column"):
        load_stream(raw)


def test_extra_column_is_ignored_and_warned(door_dir):
    raw = pd.read_csv(door_dir / "train_slice.csv")
    raw["Operator Notes"] = "n/a"
    loaded = load_stream(raw)
    assert loaded.extra_columns == ("Operator Notes",)
    feats = cycle_features(loaded, segment(loaded))
    assert any("Operator Notes" in w for w in feats.warnings)
    assert len(feats.table) == 3


def test_empty_file_is_a_clear_error(tmp_path, door_dir):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_stream(empty)
    header_only = tmp_path / "header.csv"
    header_only.write_text(
        (door_dir / "train_slice.csv").read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no data rows"):
        load_stream(header_only)


def test_unparseable_timestamp_is_a_clear_error(door_dir):
    raw = pd.read_csv(door_dir / "train_slice.csv")
    raw.loc[3, "Datetime"] = "not-a-time"
    with pytest.raises(ValueError, match="door timestamp"):
        load_stream(raw)


# --------------------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------------------


def test_gap_rule_reproduces_the_answer_segments_exactly(stream, answer):
    spans = segment_by_gaps(stream)
    assert len(spans) == len(answer) == 3
    t_ms = stream.t.to_numpy(dtype="datetime64[ms]").astype("int64")
    for (a, b), (_, row) in zip(spans, answer.iterrows()):
        assert t_ms[a] == pd.Timestamp(row["t_start"]).value // 1_000_000
        assert t_ms[b - 1] == pd.Timestamp(row["t_end"]).value // 1_000_000
        assert b - a == int(row["n_rows"])


def test_fallback_state_machine_works_with_the_gaps_removed(stream):
    """The Info Kit's warning: the segmenter must not depend on the timestamp gaps alone."""
    gapless = load_stream(
        pd.DataFrame(
            {
                "Datetime": [
                    format_door_timestamp(pd.Timestamp("2023-07-05") + pd.Timedelta(milliseconds=20 * i))
                    for i in range(len(stream.table))
                ],
                "Motor current(mA)": stream.table["current"],
                "Motor Voltage(10mV)": stream.table["voltage"],
                "Motor electrodynamic force": stream.table["emf"],
                "Close command": stream.table["close_cmd"],
                "Open command": stream.table["open_cmd"],
                "Door leaf position": stream.table["position"],
            }
        )
    )
    assert len(segment_by_gaps(gapless)) == 1  # no gaps left to cut on
    spans = segment_by_state(gapless)
    assert len(spans) == 3
    boundaries = [a for a, _ in spans]
    assert boundaries == [a for a, _ in segment_by_gaps(stream)]
    # and `auto` falls back to it without being told
    assert segment(gapless, method="auto") == spans


def test_segment_end_is_the_last_row_never_trimmed(stream):
    spans = segment(stream)
    t_ms = stream.t.to_numpy(dtype="datetime64[ms]").astype("int64")
    feats = cycle_features(stream, spans)
    for i, (a, b) in enumerate(spans):
        assert feats.table["start_ms"].iloc[i] == t_ms[a]
        assert feats.table["end_ms"].iloc[i] == t_ms[b - 1]


def test_predicted_segments_never_overlap(stream, feats):
    starts = feats.table["start_ms"].to_numpy()
    ends = feats.table["end_ms"].to_numpy()
    assert (starts[1:] >= ends[:-1]).all()


def test_unknown_segmentation_method_raises(stream):
    with pytest.raises(ValueError, match="unknown door segmentation method"):
        segment(stream, method="magic")


# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------


def test_operation_comes_from_the_position_direction(feats, answer):
    assert list(feats.table["operation"]) == list(answer["operation"])


def test_mid_stroke_current_separates_the_fixture_labels(feats, labels):
    table = feats.table
    opens = table["operation"] == "Open"
    normal = table.loc[opens & (labels == "Normal"), "i_mean_cruise"].mean()
    abnormal = table.loc[opens & (labels != "Normal"), "i_mean_cruise"].mean()
    assert abnormal > 1.3 * normal


def test_phase_feature_names_carry_their_regime(feats):
    names = set(feats.table.columns)
    assert {"i_mean_opening", "i_mean_cruise", "i_mean_closing"} <= names
    assert "op_code" in names and "open" not in "op_code".replace("op_code", "")  # no stray marker


def test_commanded_time_register_ignores_unlatched_zero_readings(door_dir):
    """`Test.csv` carries 0 in the open/close time register on some rows; a mean would blow the
    duration ratio up by three orders of magnitude, so only valid readings count."""
    raw = pd.read_csv(door_dir / "train_slice.csv")
    zeroed = raw.copy()
    zeroed.loc[::2, "Door closing time(.1s)"] = 0  # every other reading unlatched
    zeroed.loc[::2, "Door opening time(.1s)"] = 0
    base = cycle_features(load_stream(raw), segment(load_stream(raw))).table["duration_vs_cmd"]
    hit = cycle_features(load_stream(zeroed), segment(load_stream(zeroed))).table["duration_vs_cmd"]
    assert (base.dropna() < 2.0).all()
    assert np.allclose(base.to_numpy(), hit.to_numpy(), equal_nan=True)
    # a cycle with no valid reading at all is missing, not enormous
    dead = raw.copy()
    dead["Door closing time(.1s)"] = 0
    dead["Door opening time(.1s)"] = 0
    gone = cycle_features(load_stream(dead), segment(load_stream(dead))).table["duration_vs_cmd"]
    assert gone.isna().all()


def test_missing_ratio_reads_as_the_baseline_and_extremes_are_clipped(feats, labels):
    """A missing ratio must mean "no evidence" (1.0), not "collapsed to zero", and no single
    out-of-distribution channel may swing a fitted direction by hundreds of sigma."""
    fitted = FoldBaseline().fit(feats, labels)
    broken = DoorFeats(
        table=feats.table.copy(),
        profiles=feats.profiles.copy(),
        cycles=list(feats.cycles),
        spans=list(feats.spans),
    )
    broken.table.loc[0, "duration_vs_cmd"] = np.nan
    broken.table.loc[1, "i_mean_cruise"] = 1e9
    frame = fitted.transform(broken)
    assert frame["duration_vs_cmd_rel"].iloc[0] == pytest.approx(1.0)
    assert frame["i_mean_cruise_rel"].iloc[1] == pytest.approx(RATIO_CLIP)
    assert abs(frame["i_mean_cruise_z"].iloc[1]) <= Z_CLIP
    assert np.isfinite(frame.to_numpy()).all()


# --------------------------------------------------------------------------------------
# The fold-local rule
# --------------------------------------------------------------------------------------


def _subset(feats: DoorFeats, idx) -> DoorFeats:
    idx = list(idx)
    return DoorFeats(
        table=feats.table.iloc[idx].reset_index(drop=True),
        profiles=feats.profiles[idx],
        cycles=[feats.cycles[i] for i in idx],
        spans=[feats.spans[i] for i in idx] if feats.spans else [],
    )


def test_baseline_changes_with_the_training_fold(feats, labels):
    a = FoldBaseline().fit(_subset(feats, [0, 1]), labels[[0, 1]])
    b = FoldBaseline().fit(_subset(feats, [1, 2]), labels[[1, 2]])
    ca = a.stats_["Open"]["i_mean_cruise"][0]
    cb = b.stats_["Open"]["i_mean_cruise"][0]
    assert ca != pytest.approx(cb), "the per-operation baseline must move when the fold moves"


def test_transform_of_a_held_row_never_depends_on_the_other_held_rows(feats, labels):
    """Fold-local rule: the held-out batch must not influence its own normalisation."""
    fitted = FoldBaseline().fit(_subset(feats, [0, 1]), labels[[0, 1]])
    alone = fitted.transform(_subset(feats, [2]))["i_mean_cruise_rel"].iloc[0]
    together = fitted.transform(_subset(feats, [1, 2]))["i_mean_cruise_rel"].iloc[1]
    assert alone == pytest.approx(together)


def test_batch_mode_is_the_declared_ablation_and_does_depend_on_the_batch(feats, labels):
    batch = FoldBaseline(mode="batch").fit(_subset(feats, [0, 1]), labels[[0, 1]])
    alone = batch.transform(_subset(feats, [2]))["i_mean_cruise_rel"].iloc[0]
    together = batch.transform(_subset(feats, [1, 2]))["i_mean_cruise_rel"].iloc[1]
    assert alone != pytest.approx(together)


def test_baseline_is_built_from_normal_rows_only_when_labels_are_given(feats, labels):
    # the fixture holds only two Normal cycles, below the "at least 3 normals" guard, so the
    # fold is widened by repeating them (a stand-in for a real fold's 60-odd normal cycles); the
    # abnormal cycle is repeated more often so an unlabelled median would land on it
    idx = [0, 1, 1, 2, 2, 2]
    wide, wide_y = _subset(feats, idx), labels[idx]
    with_labels = FoldBaseline().fit(wide, wide_y)
    without = FoldBaseline().fit(wide, None)
    open_rows = wide.table["operation"] == "Open"
    normal_only = wide.table.loc[open_rows & (wide_y == "Normal"), "i_mean_cruise"].median()
    assert with_labels.stats_["Open"]["i_mean_cruise"][0] == pytest.approx(normal_only)
    assert without.stats_["Open"]["i_mean_cruise"][0] != pytest.approx(normal_only)


def test_baseline_widens_to_every_row_when_a_fold_has_too_few_normals(feats, labels):
    """Documented guard: fewer than three Normal cycles and the median over the fold is used."""
    fitted = FoldBaseline().fit(feats, labels)  # two Normal cycles only
    open_rows = feats.table["operation"] == "Open"
    assert fitted.stats_["Open"]["i_mean_cruise"][0] == pytest.approx(
        feats.table.loc[open_rows, "i_mean_cruise"].median()
    )


def test_threshold_is_tuned_inside_the_training_fold_only(feats, labels):
    """The decision threshold is fitted like any other parameter: training rows only."""
    idx = [0, 1, 2] * 4  # 12 cycles, enough for the inner splits to run
    wide, wide_y = _subset(feats, idx), labels[idx]
    model = DoorClassifier(features=BASELINE_FEATURES).fit(wide, wide_y)
    assert 0.0 < model.threshold < 1.0
    before = model.threshold
    model.predict(feats)  # scoring held-out cycles must not move it
    assert model.threshold == before
    # and asking for no tuning leaves the declared default alone
    fixed = DoorClassifier(features=BASELINE_FEATURES, tune_threshold=False).fit(wide, wide_y)
    assert fixed.threshold == 0.5


def test_classifier_rejects_a_label_length_mismatch(feats, labels):
    with pytest.raises(ValueError, match="cycles but"):
        DoorClassifier().fit(feats, labels[:-1])


# --------------------------------------------------------------------------------------
# Round trip: load -> featurise -> predict -> to_rows -> validate_csv
# --------------------------------------------------------------------------------------


def test_round_trip_to_a_valid_submission_csv(door_dir, stream, feats, labels, tmp_path):
    task = get_task("door")
    assert isinstance(task, DoorTask)
    model = _fitted(feats, labels)
    raw = task.load(door_dir / "train_slice.csv")
    result = task.predict(task.featurise(raw), model)
    rows = task.to_rows(result)
    assert len(rows) == 3
    assert list(rows[0]) == ["start_time", "end_time", "prediction"]
    out = write_predictions(result, tmp_path / "door_predictions.csv")
    report = validate_csv("door", out, None)
    assert report.ok, report.errors
    assert report.n_rows == 3


def test_output_timestamps_are_byte_identical_to_the_source(door_dir, feats, labels):
    model = _fitted(feats, labels)
    task = get_task("door")
    result = task.predict(feats, model)
    source = pd.read_csv(door_dir / "train_slice.csv")["Datetime"].tolist()
    answer = read_door_segments(door_dir / "train_slice_answer.csv")
    for row, (_, truth) in zip(result.rows, answer.iterrows()):
        assert row["start_time"] in source
        assert row["end_time"] in source
        assert row["start_time"] == truth["start_time"]
        assert row["end_time"] == truth["end_time"]


def test_predictions_score_against_the_answer_file(door_dir, feats, labels):
    """Fitted and scored on the same three cycles - a smoke test of the metric wiring, not a CV."""
    model = _fitted(feats, labels)
    result = get_task("door").predict(feats, model)
    truth = read_door_segments(door_dir / "train_slice_answer.csv")
    pred = pd.DataFrame(result.rows).rename(columns={"prediction": "label"})
    score = iou_f1(truth, pred)
    assert score["score"] == pytest.approx(1.0)


def test_empty_stream_predicts_nothing_rather_than_crashing(feats, labels):
    model = _fitted(feats, labels)
    empty = DoorFeats(table=pd.DataFrame(), profiles=np.zeros((0, 1)), cycles=[], spans=[])
    result = get_task("door").predict(empty, model)
    assert result.rows == []


def test_artefact_round_trips_through_save_and_load(feats, labels, tmp_path):
    model = _fitted(feats, labels)
    save_model("door", model, {"scheme": "test"}, model_dir=tmp_path)
    reloaded = load_model("door", model_dir=tmp_path)
    assert isinstance(reloaded, DoorClassifier)
    assert (reloaded.predict(feats) == model.predict(feats)).all()


# --------------------------------------------------------------------------------------
# Explanation
# --------------------------------------------------------------------------------------


def test_explanation_payload_is_small_and_well_formed(feats, labels):
    task = get_task("door")
    model = _fitted(feats, labels)
    result = task.predict(feats, model)

    stream_level = task.explain(result)
    assert isinstance(stream_level, Explanation)
    payload = stream_level.as_dict()
    assert len(payload["trace"]["x"]) == len(payload["trace"]["y"]) <= 2000
    assert len(payload["trace"]["marks"]) == 3
    assert {"x", "x1", "label", "kind"} <= set(payload["trace"]["marks"][0])
    assert payload["viewport"]["component"] == "door_L1"

    rows = task.explain_rows(result)
    assert len(rows) == 3
    for explanation in rows:
        numbers = explanation.as_dict()["numbers"]
        assert {"i_mid_rel", "threshold", "duration_s"} <= set(numbers)
        assert numbers["duration_s"] > 0
        trace = explanation.as_dict()["trace"]
        assert 0 < len(trace["x"]) == len(trace["y"]) <= 2000
        assert any(m["kind"] == "envelope" for m in trace["marks"])
    abnormal = [e for e, y in zip(rows, labels) if y != "Normal"]
    assert abnormal and abnormal[0].as_dict()["numbers"]["i_mid_rel"] > 1.2
    assert abnormal[0].as_dict()["viewport"]["health"] == "crit"


def test_single_class_fold_falls_back_to_the_physics_rule(feats, labels):
    """A training fold with only Normal cycles must still flag a clearly abnormal one."""
    normals = [i for i, y in enumerate(labels) if y == "Normal"]
    model = _fitted(_subset(feats, normals), labels[normals])
    proba = model.predict_proba(feats)
    assert proba[labels != "Normal"].max() > proba[labels == "Normal"].max()
