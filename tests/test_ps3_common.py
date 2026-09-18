"""The frozen PS3 contract: timestamps, paths, payloads, the task registry and the validator.

These tests are the definition of "the interface other PS3 agents code against"; changing one
means changing `docs/ps3_contract.md` and telling every downstream agent.
"""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from nebulax.ps3 import common as C
from nebulax.ps3 import submission as SUB

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ps3"
EXAMPLE_DIR = (
    C.REPO_ROOT / "readingmaterials" / "problem_statement" / "PS3" / "04_Example_Submission"
)


@pytest.fixture
def clean_registry():
    """Snapshot and restore ``TASKS`` so a test's dummy task never leaks."""
    saved = dict(C.TASKS)
    try:
        yield C.TASKS
    finally:
        C.TASKS.clear()
        C.TASKS.update(saved)


# --------------------------------------------------------------------------------------
# Door timestamps
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,iso",
    [
        ("2023-7-5-0-11-17-664", "2023-07-05 00:11:17.664"),
        ("2023-7-5-0-0-0-0", "2023-07-05 00:00:00.000"),
        ("2023-7-5-0-0-0-20", "2023-07-05 00:00:00.020"),
        ("2023-7-5-0-2-59-77", "2023-07-05 00:02:59.077"),  # ms field is NOT zero padded
        ("2023-12-31-23-59-59-999", "2023-12-31 23:59:59.999"),
    ],
)
def test_native_timestamp_parses_and_round_trips_exactly(text, iso):
    ts = C.parse_door_timestamp(text)
    assert ts == pd.Timestamp(iso)
    assert C.format_door_timestamp(ts) == text


def test_round_trip_is_exact_over_a_real_stream_slice():
    raw = pd.read_csv(FIXTURES / "door" / "train_slice.csv", usecols=["Datetime"], dtype=str)["Datetime"]
    parsed = C.parse_door_timestamps(raw)
    assert parsed.dtype == "datetime64[ms]"
    assert parsed.is_monotonic_increasing
    assert C.format_door_timestamps(parsed) == raw.tolist()


def test_iso_timestamps_are_accepted_on_input():
    assert C.parse_door_timestamp("2023-07-05T00:11:17.664") == pd.Timestamp("2023-07-05 00:11:17.664")
    mixed = C.parse_door_timestamps(["2023-7-5-0-0-0-0", "2023-07-05T00:00:00.500"])
    assert mixed.tolist() == [pd.Timestamp("2023-07-05 00:00:00"), pd.Timestamp("2023-07-05 00:00:00.5")]


def test_timezone_aware_input_is_written_as_naive_utc():
    assert C.format_door_timestamp(pd.Timestamp("2023-07-05T01:00:00.250+01:00")) == "2023-7-5-0-0-0-250"


def test_datetime_like_input_passes_through():
    ts = pd.Timestamp("2023-07-05 00:00:01.100")
    assert C.parse_door_timestamp(ts) == ts
    assert C.parse_door_timestamps(pd.Series([ts])).iloc[0] == ts


def test_bad_timestamps_raise_with_the_offending_value():
    with pytest.raises(ValueError, match="not a door timestamp"):
        C.parse_door_timestamp("yesterday")
    with pytest.raises(ValueError, match="not a door timestamp"):
        C.parse_door_timestamps(["2023-7-5-0-0-0-0", "nope"])


def test_sub_millisecond_cannot_be_written_in_the_native_format():
    with pytest.raises(ValueError, match="sub-millisecond"):
        C.format_door_timestamp(pd.Timestamp("2023-07-05 00:00:00.0005"))


def test_to_ms_accepts_every_boundary_form():
    assert C.to_ms("2023-7-5-0-0-1-0") - C.to_ms("2023-7-5-0-0-0-0") == 1000
    assert C.to_ms(1234) == 1234
    assert C.to_ms(1234.4) == 1234
    with pytest.raises((TypeError, ValueError)):
        C.to_ms(True)


def test_read_door_segments_handles_answer_and_prediction_files(tmp_path):
    answer = C.read_door_segments(FIXTURES / "door" / "train_slice_answer.csv")
    assert list(answer["label"]) == ["Normal", "Normal", "Abnormal resistance"]
    assert answer["t_end"].iloc[0] == pd.Timestamp("2023-07-05 00:00:03.700")

    pred = tmp_path / "door_predictions.csv"
    pred.write_text("start_time,end_time,prediction\n2023-7-5-0-0-0-0,2023-7-5-0-0-3-700,Normal\n")
    assert C.read_door_segments(pred)["label"].tolist() == ["Normal"]

    bad = tmp_path / "bad.csv"
    bad.write_text("start_time,end_time\n2023-7-5-0-0-0-0,2023-7-5-0-0-3-700\n")
    with pytest.raises(ValueError, match="no label column"):
        C.read_door_segments(bad)


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------


def test_data_root_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("NEBULAX_PS3_DATA", str(tmp_path))
    assert C.data_root() == tmp_path
    assert C.dataset_dir("rail") == tmp_path / "Rail_Corrugation"
    assert C.train_dir("shm") == tmp_path / "SHM" / "Train"
    assert C.test_dir("acv") == tmp_path / "ACV" / "Test"
    assert C.labels_path("rail") == tmp_path / "Rail_Corrugation" / "Train_Labels.csv"
    monkeypatch.delenv("NEBULAX_PS3_DATA")
    assert C.data_root() == C.DEFAULT_DATA_ROOT


def test_door_has_no_train_test_subfolders():
    assert C.train_dir("door") == C.dataset_dir("door")
    assert C.test_dir("door") == C.dataset_dir("door")
    assert C.labels_path("door").name == "Train_Segments_Answer.csv"


def test_fixed_paths_are_where_the_plan_says():
    assert C.MODEL_DIR == C.REPO_ROOT / "models" / "ps3"
    assert C.CACHE_DIR == C.REPO_ROOT / "data" / "ps3_cache"
    assert C.RESULTS_DIR == C.REPO_ROOT / "results" / "ps3"


def test_unknown_task_name_is_rejected_everywhere():
    for fn in (C.dataset_dir, C.train_dir, C.test_dir, C.labels_path, C.model_path):
        with pytest.raises(KeyError, match="unknown PS3 task"):
            fn("brakes")


# --------------------------------------------------------------------------------------
# Explanation payload
# --------------------------------------------------------------------------------------


def test_explanation_shape_is_json_ready():
    exp = C.Explanation(
        file_id="Test7.csv",
        numbers={"side_i_score": 0.81, "speed_kmh": 52},
        trace=C.Trace(x=[1, 2, 3], y=[0.1, 0.2, 0.3], marks=[{"x": 2, "label": "peak"}], label="wavelength PSD"),
        viewport=C.Viewport(car=3, side="I", health="crit", component="axlebox_c3_p1"),
    )
    payload = exp.as_dict()
    assert sorted(payload) == ["file_id", "numbers", "trace", "viewport"]
    assert sorted(payload["trace"]) == ["label", "marks", "x", "y"]
    assert sorted(payload["viewport"]) == ["car", "component", "health", "side"]
    assert payload["numbers"] == {"side_i_score": 0.81, "speed_kmh": 52.0}
    json.dumps(payload)  # must be serialisable as-is


def test_explanation_accepts_plain_dicts_for_trace_and_viewport():
    exp = C.Explanation(file_id="x", trace={"x": [0], "y": [1]}, viewport={"health": "warn"})
    assert isinstance(exp.trace, C.Trace) and isinstance(exp.viewport, C.Viewport)
    assert exp.trace.y == [1.0] and exp.viewport.health == "warn"


def test_trace_and_viewport_reject_nonsense():
    with pytest.raises(ValueError, match="x has 2 points"):
        C.Trace(x=[1, 2], y=[1.0])
    with pytest.raises(ValueError, match="health"):
        C.Viewport(health="broken")
    with pytest.raises(ValueError, match="car must be"):
        C.Viewport(car=9)
    with pytest.raises(ValueError, match="side must be"):
        C.Viewport(side="left")


def test_prediction_result_normalises_its_payload():
    res = C.PredictionResult(task="shm", file_id="test01.csv", rows=[{"file_id": "test01.csv", "prediction": 0.3}])
    assert res.numbers == {} and res.trace is None and res.extras == {}
    with pytest.raises(KeyError, match="unknown PS3 task"):
        C.PredictionResult(task="brakes")


# --------------------------------------------------------------------------------------
# Task registry
# --------------------------------------------------------------------------------------


class _DummyDoor(C.BaseTask):
    name = "door"

    def load(self, path):
        return pd.read_csv(path)

    def featurise(self, raw):
        return raw.head(1)

    def predict(self, feats, model=None):
        return C.PredictionResult(
            task="door",
            rows=[{"start_time": "2023-7-5-0-0-0-0", "end_time": "2023-7-5-0-0-3-700", "prediction": "Normal"}],
            numbers={"i_mid": 2.4},
        )


def test_base_task_fills_the_contract_fields(clean_registry):
    task = _DummyDoor()
    assert (task.label, task.accepts, task.output_filename) == ("Door", (".csv",), "door_predictions.csv")
    assert isinstance(task, C.Task)


def test_register_and_get_task_round_trip(clean_registry):
    task = C.register_task(_DummyDoor())
    assert C.get_task("door") is task
    assert C.get_task("DOOR") is task
    assert "door" in C.available_tasks()


def test_register_task_rejects_a_broken_task(clean_registry):
    class NoPredict(C.BaseTask):
        name = "rail"

    broken = NoPredict()
    broken.output_filename = "rail.csv"
    with pytest.raises(ValueError, match="output_filename"):
        C.register_task(broken)

    class NotATask:
        name = "shm"

    with pytest.raises(TypeError, match="not a PS3 Task"):
        C.register_task(NotATask())

    bad_suffix = _DummyDoor()
    bad_suffix.accepts = ("csv",)
    with pytest.raises(ValueError, match="leading dot"):
        C.register_task(bad_suffix)


def test_base_task_defaults_produce_rows_and_an_explanation(clean_registry):
    task = _DummyDoor()
    result = task.run(FIXTURES / "door" / "train_slice.csv")
    rows = task.to_rows(result)
    assert rows[0]["prediction"] == "Normal"
    exp = task.explain(result)
    assert exp.numbers == {"i_mid": 2.4}
    assert exp.trace.x == [] and exp.viewport.health == "ok"


def test_get_task_reports_a_missing_module_clearly(clean_registry, monkeypatch):
    # the real nebulax.ps3.shm may already have registered itself (another test module imported
    # it); clean_registry restores the entry afterwards, so dropping it here is local to this test
    clean_registry.pop("shm", None)
    monkeypatch.setattr(
        C.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError("no module"))
    )
    with pytest.raises(KeyError, match="not registered"):
        C.get_task("shm")
    with pytest.raises(KeyError, match="unknown PS3 task"):
        C.get_task("brakes")


# --------------------------------------------------------------------------------------
# Model artefacts
# --------------------------------------------------------------------------------------


def test_save_and_load_model_with_a_json_sidecar(tmp_path):
    path = C.save_model("rail", {"weights": [1, 2, 3]}, {"cv": {"macro_f1": 0.81}, "scheme": "sgkf5"}, model_dir=tmp_path)
    assert path == tmp_path / "rail.pkl"
    assert C.load_model("rail", model_dir=tmp_path) == {"weights": [1, 2, 3]}
    meta = C.model_meta("rail", model_dir=tmp_path)
    assert meta["task"] == "rail" and meta["cv"] == {"macro_f1": 0.81}
    assert meta["git_rev"] and meta["saved_at"] and meta["bytes"] > 0


def test_missing_model_points_at_the_trainer(tmp_path):
    with pytest.raises(FileNotFoundError, match="ps3_train.py --task shm"):
        C.load_model("shm", model_dir=tmp_path)
    assert C.model_meta("shm", model_dir=tmp_path) == {}


def test_oversized_artefacts_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MAX_MODEL_BYTES", 128)
    with pytest.raises(ValueError, match="over the"):
        C.save_model("door", list(range(20000)), model_dir=tmp_path)
    assert not (tmp_path / "door.pkl").exists()


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def test_acv_car_ids_read_the_files_own_headers():
    narrow = C.acv_car_ids(FIXTURES / "acv" / "case01_slice.xlsx")
    wide = C.acv_car_ids(FIXTURES / "acv" / "case04_wide_slice.xlsx")
    assert narrow == ["01", "02", "03", "04", "05", "06", "07", "08"] == wide
    assert C.acv_car_ids(["Time", "Car 03 - ACV Running Mode", "Car 3 - nope"]) == ["03", "3"]


def test_acv_car_ids_accept_one_or_two_digits_and_preserve_spelling():
    headers = ["Time", "Car 1 - Indoor Temperature", "Car 02 - Indoor Temperature"]
    assert C.acv_car_ids(headers) == ["02", "1"]


def test_natural_key_orders_test_files_like_a_human():
    names = ["Test10.csv", "Test2.csv", "Test1.csv"]
    assert sorted(names, key=C.natural_key) == ["Test1.csv", "Test2.csv", "Test10.csv"]


# --------------------------------------------------------------------------------------
# Submission validator
# --------------------------------------------------------------------------------------


def write_csv(path: Path, header, rows) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


@pytest.fixture
def good(tmp_path):
    """One valid submission CSV per task, plus the ids they key on."""
    door = write_csv(
        tmp_path / "door_predictions.csv",
        ["start_time", "end_time", "prediction"],
        [
            ["2023-7-5-0-0-0-0", "2023-7-5-0-0-3-700", "Normal"],
            ["2023-7-5-0-0-23-999", "2023-7-5-0-0-26-839", "Normal"],
            ["2023-7-5-0-0-51-266", "2023-7-5-0-0-53-986", "Abnormal resistance"],
        ],
    )
    acv = write_csv(
        tmp_path / "acv_predictions.csv",
        ["file_id", "ranked_cars"],
        [["acv_test_case.xlsx", "03|01|05|02|04|06|07|08"]],
    )
    rail = write_csv(
        tmp_path / "rail_predictions.csv",
        ["file_id", "prediction"],
        [["Test1.csv", "Normal"], ["Test2.csv", "Side I"], ["Test10.csv", "Side II"]],
    )
    shm = write_csv(
        tmp_path / "shm_predictions.csv",
        ["file_id", "prediction"],
        [["test01.csv", "0.6123"], ["test02.csv", "0.0812"]],
    )
    return {"door": door, "acv": acv, "rail": rail, "shm": shm}


def test_headers_match_the_organisers_example_files():
    if not EXAMPLE_DIR.exists():
        pytest.skip("organisers' 04_Example_Submission not present in this checkout")
    for name in sorted(p.name for p in EXAMPLE_DIR.glob("*_predictions.csv")):
        with open(EXAMPLE_DIR / name, encoding="utf-8-sig", newline="") as fh:
            header = tuple(next(csv.reader(fh)))
        assert header == SUB.CSV_HEADERS[SUB.task_for_filename(name)]


def test_valid_submissions_pass(good):
    for task, path in good.items():
        rep = SUB.validate_csv(task, path)
        assert rep.ok, rep.errors
        assert rep.raise_for_errors() is rep
    assert SUB.validate_csv("rail", good["rail"], ["Test1.csv", "Test2.csv", "Test10.csv"]).ok
    assert SUB.validate_csv(
        "acv", good["acv"], ["acv_test_case.xlsx"], expected_cars={"acv_test_case.xlsx": C.acv_car_ids(FIXTURES / "acv" / "case01_slice.xlsx")}
    ).ok


def test_wrong_header_is_an_error(tmp_path):
    p = write_csv(tmp_path / "rail_predictions.csv", ["filename", "prediction"], [["Test1.csv", "Normal"]])
    rep = SUB.validate_csv("rail", p)
    assert not rep.ok and "header is" in rep.errors[0]
    with pytest.raises(SUB.SubmissionError, match="header is"):
        rep.raise_for_errors()


def test_extra_column_is_an_error_except_door_confidence(tmp_path):
    rail = write_csv(
        tmp_path / "rail_predictions.csv", ["file_id", "prediction", "score"], [["Test1.csv", "Normal", "0.9"]]
    )
    assert not SUB.validate_csv("rail", rail).ok

    door = write_csv(
        tmp_path / "door_predictions.csv",
        ["start_time", "end_time", "prediction", "confidence"],
        [["2023-7-5-0-0-0-0", "2023-7-5-0-0-3-700", "Normal", "0.99"]],
    )
    rep = SUB.validate_csv("door", door)
    assert rep.ok and any("ignored by the judge" in w for w in rep.warnings)


@pytest.mark.parametrize(
    "rows,needle",
    [
        ([["Test1.csv", "normal"]], "not in ['Normal', 'Side I', 'Side II']"),
        ([["Test1.csv", "Side 1"]], "not in"),
        ([["Test1.csv", "Normal"], ["Test1.csv", "Side I"]], "duplicate file_id"),
        ([["Test1", "Normal"]], "must keep its extension"),
        ([["Test/Test1.csv", "Normal"]], "bare basename"),
        ([["Test1.csv", ""]], "empty 'prediction'"),
        ([["Test1.csv", "NaN"]], "NaN/Inf"),
    ],
)
def test_rail_row_level_checks(tmp_path, rows, needle):
    p = write_csv(tmp_path / "rail_predictions.csv", ["file_id", "prediction"], rows)
    rep = SUB.validate_csv("rail", p)
    assert not rep.ok
    assert any(needle in e for e in rep.errors), rep.errors


def test_file_id_comparison_is_case_sensitive(tmp_path):
    p = write_csv(tmp_path / "rail_predictions.csv", ["file_id", "prediction"], [["test1.csv", "Normal"]])
    rep = SUB.validate_csv("rail", p, ["Test1.csv"])
    assert not rep.ok
    assert any("missing" in e for e in rep.errors)
    assert any("unexpected" in e for e in rep.errors)


def test_expected_ids_set_equality(tmp_path, good):
    rep = SUB.validate_csv("rail", good["rail"], ["Test1.csv", "Test2.csv", "Test10.csv", "Test11.csv"])
    assert not rep.ok and "missing 1 expected file_id(s): ['Test11.csv']" in rep.errors[0]


@pytest.mark.parametrize(
    "value,needle",
    [
        ("3|1|2", "two-digit car ids"),
        ("03|03|01", "repeats"),
        ("Car 03|01", "two-digit car ids"),
        ("03,01", "two-digit car ids"),
    ],
)
def test_acv_ranked_cars_checks(tmp_path, value, needle):
    p = write_csv(tmp_path / "acv_predictions.csv", ["file_id", "ranked_cars"], [["acv_test_case.xlsx", value]])
    rep = SUB.validate_csv("acv", p)
    assert not rep.ok and any(needle in e for e in rep.errors), rep.errors


def test_acv_must_rank_every_car_in_that_file(tmp_path):
    p = write_csv(
        tmp_path / "acv_predictions.csv", ["file_id", "ranked_cars"], [["acv_test_case.xlsx", "01|02|03|04"]]
    )
    rep = SUB.validate_csv("acv", p, expected_cars={"acv_test_case.xlsx": ["0%d" % i for i in range(1, 9)]})
    assert not rep.ok and any("exactly once" in e for e in rep.errors)
    # Without the file's headers we can only warn about the unusual count.
    rep2 = SUB.validate_csv("acv", p)
    assert rep2.ok and any("4 cars ranked" in w for w in rep2.warnings)


@pytest.mark.parametrize(
    "rows,needle",
    [
        ([["2023-7-5-0-0-3-700", "2023-7-5-0-0-0-0", "Normal"]], "not before"),
        ([["2023-7-5-0-0-0-0", "2023-7-5-0-0-0-0", "Normal"]], "not before"),
        ([["not-a-time", "2023-7-5-0-0-3-700", "Normal"]], "not a door timestamp"),
        ([["2023-7-5-0-0-0-0", "2023-7-5-0-0-3-700", "Abnormal"]], "not in ['Normal', 'Abnormal resistance']"),
        (
            [
                ["2023-7-5-0-0-0-0", "2023-7-5-0-0-3-700", "Normal"],
                ["2023-7-5-0-0-2-0", "2023-7-5-0-0-5-0", "Normal"],
            ],
            "overlap",
        ),
        (
            [
                ["2023-7-5-0-0-0-0", "2023-7-5-0-0-3-700", "Normal"],
                ["2023-7-5-0-0-0-0", "2023-7-5-0-0-3-700", "Abnormal resistance"],
            ],
            "duplicate segment",
        ),
    ],
)
def test_door_row_level_checks(tmp_path, rows, needle):
    p = write_csv(tmp_path / "door_predictions.csv", ["start_time", "end_time", "prediction"], rows)
    rep = SUB.validate_csv("door", p)
    assert not rep.ok and any(needle in e for e in rep.errors), rep.errors


def test_door_accepts_iso_timestamps_and_touching_segments(tmp_path):
    p = write_csv(
        tmp_path / "door_predictions.csv",
        ["start_time", "end_time", "prediction"],
        [
            ["2023-07-05T00:00:00.000", "2023-07-05T00:00:03.700", "Normal"],
            ["2023-07-05T00:00:03.700", "2023-07-05T00:00:06.000", "Abnormal resistance"],
        ],
    )
    assert SUB.validate_csv("door", p).ok


def test_door_rejects_expected_ids(tmp_path, good):
    rep = SUB.validate_csv("door", good["door"], ["Test.csv"])
    assert not rep.ok and any("no file_id column" in e for e in rep.errors)


@pytest.mark.parametrize("value,needle", [("-0.2", "must be positive"), ("0", "must be positive"), ("abc", "not a number"), ("inf", "NaN/Inf")])
def test_shm_predictions_must_be_positive_and_finite(tmp_path, value, needle):
    p = write_csv(tmp_path / "shm_predictions.csv", ["file_id", "prediction"], [["test01.csv", value]])
    rep = SUB.validate_csv("shm", p)
    assert not rep.ok and any(needle in e for e in rep.errors), rep.errors


def test_empty_and_missing_files_are_errors(tmp_path):
    missing = SUB.validate_csv("shm", tmp_path / "shm_predictions.csv")
    assert not missing.ok and "does not exist" in missing.errors[0]
    empty = write_csv(tmp_path / "shm_predictions.csv", ["file_id", "prediction"], [])
    rep = SUB.validate_csv("shm", empty)
    assert not rep.ok and "no prediction rows" in rep.errors[0]


def test_ragged_rows_are_caught(tmp_path):
    p = tmp_path / "shm_predictions.csv"
    p.write_text("file_id,prediction\ntest01.csv,0.3,extra\n")
    rep = SUB.validate_csv("shm", p)
    assert not rep.ok and "3 fields, expected 2" in rep.errors[0]


def test_expected_ids_for_lists_the_distributed_test_files(tmp_path, monkeypatch):
    monkeypatch.setenv("NEBULAX_PS3_DATA", str(tmp_path))
    (tmp_path / "Rail_Corrugation" / "Test").mkdir(parents=True)
    for n in (1, 2, 10):
        (tmp_path / "Rail_Corrugation" / "Test" / f"Test{n}.csv").write_text("x\n")
    (tmp_path / "ACV" / "Test").mkdir(parents=True)
    (tmp_path / "ACV" / "Test" / "acv_test_case.xlsx").write_bytes(b"PK")
    (tmp_path / "ACV" / "Test" / "~$acv_test_case.xlsx").write_bytes(b"PK")  # Excel lock file
    assert SUB.expected_ids_for("rail") == ["Test1.csv", "Test2.csv", "Test10.csv"]
    assert SUB.expected_ids_for("acv") == ["acv_test_case.xlsx"]
    assert SUB.expected_ids_for("door") is None
    with pytest.raises(FileNotFoundError):
        SUB.expected_ids_for("shm")


# --------------------------------------------------------------------------------------
# predictions.zip
# --------------------------------------------------------------------------------------


def test_pack_puts_the_csvs_at_the_archive_root(tmp_path, good):
    out = SUB.pack(good.values(), tmp_path / "out" / "predictions.zip")
    with zipfile.ZipFile(out) as zf:
        assert sorted(zf.namelist()) == sorted(C.OUTPUT_FILENAMES.values())
        assert all("/" not in n for n in zf.namelist())
    rep = SUB.validate_zip(out)
    assert rep.ok, rep.errors
    assert rep.n_rows == 3 + 1 + 3 + 2


def test_pack_is_byte_deterministic(tmp_path, good):
    a = SUB.pack(good.values(), tmp_path / "a.zip").read_bytes()
    b = SUB.pack(good.values(), tmp_path / "b.zip").read_bytes()
    assert a == b


def test_pack_refuses_unknown_and_duplicate_members(tmp_path, good):
    stray = write_csv(tmp_path / "results.csv", ["file_id", "prediction"], [["a.csv", "1"]])
    with pytest.raises(ValueError, match="not a PS3 prediction file"):
        SUB.pack([stray], tmp_path / "predictions.zip")
    copy = tmp_path / "copy" / "rail_predictions.csv"
    copy.parent.mkdir()
    copy.write_bytes(good["rail"].read_bytes())
    with pytest.raises(ValueError, match="same zip member"):
        SUB.pack([good["rail"], copy], tmp_path / "predictions.zip")
    with pytest.raises(ValueError, match="nothing to pack"):
        SUB.pack([], tmp_path / "predictions.zip")


def test_partial_submissions_are_allowed(tmp_path, good):
    out = SUB.pack([good["door"], good["shm"]], tmp_path / "predictions.zip")
    rep = SUB.validate_zip(out)
    assert rep.ok and rep.ids == ["door", "shm"]


def test_validate_zip_rejects_subfolders_and_strays(tmp_path, good):
    bad = tmp_path / "predictions.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("predictions/rail_predictions.csv", good["rail"].read_text())
        zf.writestr("__MACOSX/._rail_predictions.csv", "junk")
    rep = SUB.validate_zip(bad)
    assert not rep.ok
    assert any("archive root" in e for e in rep.errors)
    assert any("no *_predictions.csv" in e for e in rep.errors)


def test_validate_zip_re_runs_the_csv_checks(tmp_path):
    broken = write_csv(tmp_path / "rail_predictions.csv", ["file_id", "prediction"], [["Test1.csv", "normal"]])
    out = SUB.pack([broken], tmp_path / "predictions.zip")
    rep = SUB.validate_zip(out, {"rail": ["Test1.csv"]})
    assert not rep.ok and rep.errors[0].startswith("rail_predictions.csv: ")


def test_validate_zip_on_junk(tmp_path):
    junk = tmp_path / "predictions.zip"
    junk.write_text("not a zip")
    assert "not a zip archive" in SUB.validate_zip(junk).errors
    assert "does not exist" in SUB.validate_zip(tmp_path / "nope.zip").errors[0]
