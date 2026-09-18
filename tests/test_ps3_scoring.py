"""The four PS3 organiser metrics, checked against every worked example in the Info Kits.

If one of these fails, `nebulax/ps3/scoring.py` no longer agrees with `judge_leaderboard.py` and
every CV number in `results/ps3/` is measuring something else.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nebulax.ps3 import common as C
from nebulax.ps3 import scoring as S

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ps3"

N = "Normal"
A = "Abnormal resistance"


def seg(start_ms: int, end_ms: int, label: str = N) -> tuple[int, int, str]:
    """A segment in plain epoch-milliseconds (the metric's native unit)."""
    return (start_ms, end_ms, label)


# --------------------------------------------------------------------------------------
# Door: IoU-weighted F1 (Door_Subsystem_Info_Kit.md section 4)
# --------------------------------------------------------------------------------------


def test_iou_formula_matches_the_info_kit():
    assert S.iou((0, 100), (0, 100)) == 1.0
    # intersection 50, union 100 + 100 - 50 = 150
    assert S.iou((0, 100), (50, 150)) == pytest.approx(50 / 150)
    assert S.iou((0, 100), (100, 200)) == 0.0  # touching, not overlapping
    assert S.iou((0, 100), (200, 300)) == 0.0
    assert S.iou((0, 0), (0, 0)) == 0.0  # union <= 0


def test_perfect_submission_scores_one():
    truth = [seg(0, 1000), seg(2000, 3000, A)]
    out = S.iou_f1(truth, list(truth))
    assert out["score"] == 1.0
    assert out["soft_recall"] == 1.0 and out["soft_precision"] == 1.0
    assert out["n_matches"] == 2 and out["misses"] == [] and out["false_positives"] == []


def test_wrong_label_is_a_miss_plus_a_false_positive():
    out = S.iou_f1([seg(0, 1000, N)], [seg(0, 1000, A)])
    assert out["score"] == 0.0
    assert out["misses"] == [0] and out["false_positives"] == [0]
    assert out["n_matches"] == 0
    # Identical to predicting nothing at all for recall, and worse than nothing for precision.
    assert S.iou_f1([seg(0, 1000, N)], [])["soft_recall"] == 0.0


def test_partial_overlap_earns_its_iou_not_a_full_point():
    out = S.iou_f1([seg(0, 1000)], [seg(500, 1500)])
    expected = 500 / 1500
    assert out["matches"][0]["iou"] == pytest.approx(expected)
    assert out["soft_recall"] == pytest.approx(expected)
    assert out["soft_precision"] == pytest.approx(expected)
    assert out["score"] == pytest.approx(expected)


def test_two_predictions_on_one_true_segment_keeps_the_better_one():
    """One-to-one matching: the tighter overlap wins, the other is a false positive."""
    out = S.iou_f1([seg(0, 1000)], [seg(0, 900), seg(0, 200)])
    assert out["n_matches"] == 1
    assert out["matches"][0]["pred_index"] == 0
    assert out["false_positives"] == [1]
    assert out["soft_recall"] == pytest.approx(900 / 1000)
    assert out["soft_precision"] == pytest.approx(0.9 / 2)


def test_no_overlap_scores_zero():
    out = S.iou_f1([seg(0, 1000)], [seg(5000, 6000)])
    assert out["score"] == 0.0
    assert out["misses"] == [0] and out["false_positives"] == [0]


def test_greedy_takes_the_highest_iou_first_not_the_first_listed():
    """pred 0 overlaps both trues; the greedy order must give it to the true it fits best."""
    truth = [seg(0, 1000), seg(900, 2000)]
    preds = [seg(880, 2010)]
    out = S.iou_f1(truth, preds)
    assert out["n_matches"] == 1
    assert out["matches"][0]["true_index"] == 1
    assert out["misses"] == [0]


def test_equal_iou_tie_break_is_deterministic_and_documented():
    """Two true segments tie for one prediction: the earlier true index wins, every time."""
    truth = [seg(0, 100), seg(200, 300)]
    preds = [seg(0, 100)]
    forward = S.iou_f1(truth, preds)
    assert forward["matches"][0]["true_index"] == 0
    # A tie between two identical predictions goes to the earlier prediction index.
    out = S.iou_f1([seg(0, 100)], [seg(0, 100), seg(0, 100)])
    assert out["matches"][0]["pred_index"] == 0
    assert out["false_positives"] == [1]


def test_empty_inputs_score_zero_not_nan():
    assert S.iou_f1([], [])["score"] == 0.0
    assert S.iou_f1([seg(0, 1)], [])["score"] == 0.0
    assert S.iou_f1([], [seg(0, 1)])["score"] == 0.0


def test_segments_accept_timestamps_dicts_and_frames():
    native = [("2023-7-5-0-0-0-0", "2023-7-5-0-0-3-700", N)]
    iso = [{"start": "2023-07-05T00:00:00.000", "end": "2023-07-05T00:00:03.700", "label": N}]
    assert S.iou_f1(native, iso)["score"] == 1.0

    answer = C.read_door_segments(FIXTURES / "door" / "train_slice_answer.csv")
    assert S.iou_f1(answer, answer)["score"] == 1.0
    assert S.iou_f1(answer, answer)["n_matches"] == 3


def test_real_slice_scores_sensibly_when_boundaries_slip():
    """Three true cycles; predict them 100 ms late and the score drops but stays high."""
    answer = C.read_door_segments(FIXTURES / "door" / "train_slice_answer.csv")
    shifted = [
        (t + 100, e + 100, lab)
        for t, e, lab in (
            (C.to_ms(r.t_start), C.to_ms(r.t_end), r.label) for r in answer.itertuples()
        )
    ]
    out = S.iou_f1(answer, shifted)
    assert out["n_matches"] == 3
    assert 0.9 < out["score"] < 1.0


def test_backwards_segment_is_rejected():
    with pytest.raises(ValueError, match="before start"):
        S.iou_f1([seg(1000, 0)], [])


# --------------------------------------------------------------------------------------
# ACV: linear rank decay (ACV_Subsystem_Info_Kit.md section 4)
# --------------------------------------------------------------------------------------


CARS8 = ["01", "02", "03", "04", "05", "06", "07", "08"]


@pytest.mark.parametrize(
    "rank,expected",
    [(1, 1.000), (2, 0.875), (3, 0.750), (4, 0.625), (5, 0.500), (6, 0.375), (7, 0.250), (8, 0.125)],
)
def test_rank_decay_reproduces_the_worked_table(rank, expected):
    """The Info Kit's 8-car table, 1st through 8th."""
    assert S.rank_decay(CARS8, CARS8[rank - 1]) == pytest.approx(expected)


def test_true_car_not_ranked_scores_zero():
    assert S.rank_decay(CARS8, "09") == 0.0
    assert S.rank_decay([], "01") == 0.0


def test_rank_decay_uses_the_files_own_two_digit_ids():
    assert S.rank_decay(CARS8, "3") == 0.0  # "3" is not how the headers spell it
    assert S.rank_decay([3, 1, 2], 3) == 1.0  # non-strings are stringified consistently


def test_rank_decay_explicit_n():
    assert S.rank_decay(["03", "01"], "01", n=8) == pytest.approx(7 / 8)


def test_rank_decay_mean_over_cases_scores_missing_cases_zero():
    out = S.rank_decay_mean(
        {"acv_case_01.xlsx": CARS8, "acv_case_02.xlsx": ["02"] + CARS8[:1]},
        {"acv_case_01.xlsx": "01", "acv_case_02.xlsx": "02", "acv_case_03.xlsx": "03"},
    )
    assert out["per_file"] == {
        "acv_case_01.xlsx": 1.0,
        "acv_case_02.xlsx": 1.0,
        "acv_case_03.xlsx": 0.0,
    }
    assert out["score"] == pytest.approx(2 / 3)


# --------------------------------------------------------------------------------------
# Rail: macro F1 (Rail_Corrugation_Info_Kit.md section 4)
# --------------------------------------------------------------------------------------


def test_macro_f1_reproduces_the_worked_example():
    """Per-class F1 0.97 / 0.40 / 0.60 -> 0.657."""
    assert round(S.macro_f1_from_class_f1([0.97, 0.40, 0.60]), 3) == 0.657


def test_always_normal_predictor_is_punished():
    y_true = ["Normal"] * 10 + ["Side I", "Side II"]
    y_pred = ["Normal"] * 12
    rep = S.class_f1_report(y_true, y_pred)
    assert rep["per_class"]["Side I"]["f1"] == 0.0
    assert rep["per_class"]["Side II"]["f1"] == 0.0
    assert rep["macro_f1"] == pytest.approx((2 * 10 / 12 * 1.0 / (10 / 12 + 1.0)) / 3)
    assert rep["macro_f1"] < 0.34  # ~0.30, against ~83 % plain accuracy
    assert rep["accuracy"] > 0.8


def test_fixed_vocabulary_even_when_a_class_is_absent():
    rep = S.class_f1_report(["Normal", "Normal"], ["Normal", "Normal"])
    assert rep["labels"] == list(C.RAIL_LABELS)
    assert rep["per_class_f1"] == [1.0, 0.0, 0.0]
    assert rep["macro_f1"] == pytest.approx(1 / 3)


def test_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score

    y_true = ["Normal"] * 6 + ["Side I"] * 2 + ["Side II"] * 3
    y_pred = ["Normal"] * 5 + ["Side I"] + ["Side I", "Side II"] + ["Side II", "Normal", "Side II"]
    assert S.macro_f1(y_true, y_pred) == pytest.approx(
        f1_score(y_true, y_pred, labels=list(C.RAIL_LABELS), average="macro", zero_division=0)
    )


def test_unknown_labels_are_reported_and_never_credited():
    rep = S.class_f1_report(["Normal", "Side I"], ["Normal", "side i"])
    assert rep["unknown_labels"] == ["side i"]
    assert rep["per_class"]["Side I"]["recall"] == 0.0


def test_macro_f1_length_mismatch_raises():
    with pytest.raises(ValueError, match="rows"):
        S.macro_f1(["Normal"], ["Normal", "Normal"])


# --------------------------------------------------------------------------------------
# SHM: max(0, 1 - MAPE) (SHM_Info_Kit.md section 4)
# --------------------------------------------------------------------------------------


def test_mape_reproduces_the_worked_example():
    out = S.mape_score([0.10, 0.30, 0.50, 0.70, 0.90], [0.15, 0.28, 0.55, 0.68, 0.85])
    assert round(out["mape"], 3) == 0.150
    assert round(out["score"], 3) == 0.850
    assert out["ape"][0] == pytest.approx(0.50)


def test_constant_predictor_floors_at_zero():
    out = S.mape_score([0.10, 0.30, 0.50, 0.70, 0.90], [0.5] * 5)
    assert out["mape"] == pytest.approx(1.0794, abs=1e-4)  # ~108 %
    assert out["score"] == 0.0


def test_perfect_prediction_scores_one():
    assert S.mape_score([0.2, 0.9], [0.2, 0.9])["score"] == 1.0


def test_mape_rejects_zero_or_non_finite_truth():
    with pytest.raises(ValueError, match="non-zero"):
        S.mape_score([0.0, 0.5], [0.1, 0.5])
    with pytest.raises(ValueError, match="NaN/Inf"):
        S.mape_score([0.1, 0.5], [float("nan"), 0.5])


# --------------------------------------------------------------------------------------
# Specification section 5.3
# --------------------------------------------------------------------------------------


def test_combined_scores_reward_breadth_and_depth_separately():
    out = S.combined_scores({"door": 1.0, "rail": 0.8})
    assert out["overall"] == pytest.approx(0.45)  # (1.0 + 0.8) / 4
    assert out["average"] == pytest.approx(0.9)  # (1.0 + 0.8) / 2
    assert out["attempted"] == ["door", "rail"]
    assert S.combined_scores({})["overall"] == 0.0
