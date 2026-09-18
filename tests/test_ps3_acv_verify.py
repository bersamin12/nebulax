"""Adversarial verification of the ACV ranker (W4 verifier, task = acv).

These tests do **not** duplicate `tests/test_ps3_acv.py`; they pin the places where the first
committed headline (`results/ps3/acv_cv.md`, mean rank decay 1.0000) was optimistic, where a
ladder row claimed a signal its tie-break was actually producing, and where the deployed `Task`
path could silently disagree with the cross-validated rule.

Six of them were ``xfail(strict=True)`` - they stated the property that *should* hold and failed.
The implementer has since fixed all six (the headline is the pre-registered baseline 0.9792, the
fold's rule is chosen on the fold's own cases, `_is_degenerate` catches a few-valued feature, the
`selection_pool` docstring quotes the number its artefact carries, the chance floor accounts for
the empty cars, and `ACVTask` honours the model's mask), so the markers are gone and the
assertions stand unprotected: a regression fails the file.

The real-data tests reuse the parquet feature cache (`data/ps3_cache/acv`); they skip when it is
cold, because `acv_case_04.xlsx` takes ~55 s to parse and this file must stay under a minute.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nebulax.ps3 import acv
from nebulax.ps3 import acv_features as af
from nebulax.ps3.common import RESULTS_DIR, train_dir
from nebulax.ps3.scoring import rank_decay

CARS = [f"{i:02d}" for i in range(1, 9)]
BASELINE_SCORE = 0.9791666666666666  # five cases at rank 1, acv_case_04 at rank 2


# --------------------------------------------------------------------------------------
# Real six-case bank (skipped unless the feature cache is warm)
# --------------------------------------------------------------------------------------


def _cache_key(path: Path, hot_quantile: float = 0.5, cooling_only: bool = True) -> Path:
    stamp = f"{path.name}|{path.stat().st_mtime_ns}|{af.FEATURE_VERSION}|{hot_quantile}|{int(cooling_only)}"
    key = hashlib.sha1(stamp.encode("utf-8")).hexdigest()[:12]
    return af.CACHE_DIR / "acv" / f"{path.stem}__{key}.parquet"


def _warm_paths() -> list[Path] | None:
    try:
        paths = acv._case_paths(train_dir("acv"))
    except Exception:  # pragma: no cover - the organisers' clone is not present
        return None
    if len(paths) != 6 or not all(_cache_key(p).exists() for p in paths):
        return None
    return paths


@pytest.fixture(scope="module")
def real():
    """``(bank, labels, feats)`` for the six organiser cases, from the warm parquet cache."""
    paths = _warm_paths()
    if paths is None:
        pytest.skip("ACV train cases or their feature cache are not available")
    bank = acv.FeatureBank(paths)
    labels = acv._labels()
    return bank, labels, bank.for_config(0.5, True)


@pytest.fixture(scope="module")
def committed_cv():
    p = RESULTS_DIR / "acv_cv.json"
    if not p.exists():
        pytest.skip("results/ps3/acv_cv.json not written")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def committed_ladder():
    p = RESULTS_DIR / "acv_ladder.json"
    if not p.exists():
        pytest.skip("results/ps3/acv_ladder.json not written")
    return {r["row"]: r for r in json.loads(p.read_text(encoding="utf-8"))["rows"]}


# --------------------------------------------------------------------------------------
# 1. The per-fold selection never looks at the training cases
# --------------------------------------------------------------------------------------


def test_every_fold_ties_at_the_top_of_the_selection_pool(real):
    """In all six folds at least three of the five pool rules tie at the best training score.

    Rank decay over five cases cannot separate them, which is why `fit_ranker` used to fall
    through to the candidate's *position in the pool* - an ordering the implementer chose after
    reading the full ladder - and now falls through to `acv.separation`, a margin read off those
    same five training cases. On the `acv_case_04` fold, the one case that separates 1.0000 from
    0.9792, all five candidates tie on rank decay.
    """
    bank, labels, _ = real
    files = sorted(bank.files)
    tied = {}
    for held in files:
        train_files = [f for f in files if f != held]
        chosen = acv.fit_ranker(bank, labels, acv.selection_pool(), files=train_files)
        tied[held] = chosen.cv["n_tied_at_top"]
    assert min(tied.values()) >= 3, tied
    assert tied["acv_case_04.xlsx"] == len(acv.selection_pool()), tied


def test_the_fold_selection_depends_on_the_training_cases(real):
    """A nested selection is only nested if the choice can move with the fold.

    It could not: `peer_delta_hot_loo` is first in `selection_pool()` and ties at the top of every
    fold's training score, so `fit_ranker` returned it for all six folds and the nested CV was
    mathematically identical to hard-coding the rule that tops the six-case ladder. `fit_ranker`
    now breaks that tie on `acv.separation`, read off the fold's own five training cases: the
    `acv_case_04` fold - where all five pool rules tie on rank decay - picks `peer_delta_steady`,
    the other five pick `peer_delta_hot_loo`.
    """
    bank, labels, _ = real
    files = sorted(bank.files)
    chosen = {
        held: acv.fit_ranker(
            bank, labels, acv.selection_pool(), files=[f for f in files if f != held]
        ).name
        for held in files
    }
    assert len(set(chosen.values())) > 1, chosen


def test_headline_is_the_pre_registered_baseline(committed_cv):
    """The headline should be 0.9792, with 1.0000 reported as the best ladder row.

    `train()` used to promote the ladder winner whenever it beat the baseline on the same six
    cases the ladder was read on. It no longer promotes at all: the headline is the pre-registered
    baseline, and the ladder's best row is reported beside it, labelled for what it is.
    """
    assert committed_cv["score"] == pytest.approx(committed_cv["baseline_score"])
    assert committed_cv["score"] == pytest.approx(BASELINE_SCORE)
    assert committed_cv["artefact"].startswith("baseline")
    best = committed_cv["ladder_best"]
    assert best["row"] == "peer_delta_hot_loo" and best["score"] == pytest.approx(1.0)
    assert best["label"] == "selected on all six cases"


# --------------------------------------------------------------------------------------
# 2. What the 0.9792 -> 1.0000 gap is actually made of
# --------------------------------------------------------------------------------------


def test_the_whole_headline_gap_is_one_four_car_case(committed_ladder):
    """Every ladder row at or above the baseline scores 1.000 on five of the six cases.

    The only case that moves is `acv_case_04.xlsx`, which has four entirely empty cars, so the
    ladder's resolution over its top 21 rows is a single rank step on a four-car problem.
    """
    top = {k: r for k, r in committed_ladder.items() if not r.get("nested") and r["score"] >= BASELINE_SCORE - 1e-9}
    assert len(top) >= 21, len(top)
    for name, row in top.items():
        for fid, score in row["per_file"].items():
            if fid != "acv_case_04.xlsx":
                assert score == pytest.approx(1.0), (name, fid, score)


def test_case_04_pooled_gap_between_the_top_two_cars_is_milli_kelvin(real):
    """`peer_delta_hot` puts car 04 above the true car 01 by 4.2 mK on a 4-car median.

    That 0.0042 K is the entire difference between the pre-registered baseline (0.9792) and the
    committed headline (1.0000). It is far below any plausible sensor or aggregation tolerance.
    """
    _, _, feats = real
    f = feats["acv_case_04.xlsx"]
    assert int(f["enough_data"].sum()) == 4
    gap = float(f.loc["04", "peer_delta_hot"]) - float(f.loc["01", "peer_delta_hot"])
    assert 0.0 < gap < 0.01, gap


def test_chance_floor_accounts_for_the_empty_cars(committed_cv, real):
    """A signal-free ranker that puts the empty cars last scores 0.6042, not 0.5625.

    `acv_case_04.xlsx` has four all-NaN cars, which `rank_cars` sorts last for free, so its blind
    expectation is 0.8125 rather than 0.5625 and the six-case floor is 0.6042. The committed
    chance floor overstates the headroom of every row in the table.
    """
    _, _, feats = real
    blind = []
    for frame in feats.values():
        n = len(frame.index)
        k = int(frame["enough_data"].sum())
        blind.append(float(np.mean([(n - (r - 1)) / n for r in range(1, k + 1)])))
    assert committed_cv["chance_floor"] == pytest.approx(float(np.mean(blind)), abs=1e-3)
    assert committed_cv["chance_floor"] == pytest.approx(0.6042, abs=1e-3)
    assert committed_cv["chance_floor_uniform"] == pytest.approx(0.5625, abs=1e-3)


# --------------------------------------------------------------------------------------
# 3. robust_peer_z: the ladder's other 1.0000 row is its tie-break
# --------------------------------------------------------------------------------------


def test_robust_peer_z_takes_only_three_distinct_values_per_case(real):
    """The MAD-scaled median z collapses to ~3 tiers per case, so most cars tie."""
    _, _, feats = real
    for fid, frame in feats.items():
        assert frame["robust_peer_z"].nunique() <= 3, (fid, frame["robust_peer_z"].to_dict())


def test_robust_peer_z_without_its_tie_break_only_matches_the_baseline(real):
    """Strip the `peer_delta_hot_loo` tie-break and the row drops from 1.0000 to 0.9792.

    So the ladder's two 1.0000 rows are not independent evidence for the peer-delta family: the
    second one is the first one, applied inside `robust_peer_z`'s tie groups.
    """
    _, labels, feats = real
    ranker = acv.ACVRanker(features=("robust_peer_z",), tie_break=(), name="rz_no_tiebreak")
    scores = [rank_decay(acv.rank_cars(feats[f], ranker)[0], labels[f], 8) for f in sorted(feats)]
    assert float(np.mean(scores)) == pytest.approx(BASELINE_SCORE)


def test_degenerate_flag_catches_a_row_its_tie_break_is_ranking(real):
    """`_is_degenerate` marks `cooling_duty` and `load_halved_share` but not `robust_peer_z`.

    A row whose feature cannot separate the top cars, and whose reported score is produced by the
    tie-break, must carry the same note - otherwise the table reads as two independent rules
    reaching the ceiling.
    """
    bank, _, _ = real
    rz = acv.ACVRanker(features=("robust_peer_z",), tie_break=("peer_delta_hot_loo",), name="robust_peer_z")
    assert acv._is_degenerate(bank, rz) is True
    assert "3 distinct values" in (acv._degenerate_reason(bank, rz) or "")
    # a real ordering feature must not be swept up by the same rule
    pd_hot = acv.ACVRanker(features=("peer_delta_hot",), name="peer_delta_hot")
    assert acv._is_degenerate(bank, pd_hot) is False


# --------------------------------------------------------------------------------------
# 4. Documentation that contradicts the artefact it cites
# --------------------------------------------------------------------------------------


def test_selection_pool_docstring_quotes_the_committed_open_pool_score(committed_ladder):
    """The stated reason for narrowing the pool is a number the committed ladder contradicts.

    `selection_pool.__doc__` says "Letting a fold choose from all 26 rankers scores 0.854 over the
    six cases, *worse* than the fixed baseline's 0.979". `acv_ladder.json` scores that same nested
    open-pool row at 0.9792 - equal to the baseline, not worse - so the justification for the
    five-rule pool does not hold as written.
    """
    row = next(r for k, r in committed_ladder.items() if r.get("nested") and "open pool" in k)
    quoted = {float(m) for m in re.findall(r"scores (\d\.\d+) over the six cases", acv.selection_pool.__doc__ or "")}
    assert quoted, "no open-pool score quoted in the docstring"
    assert min(abs(q - row["score"]) for q in quoted) < 5e-4, (quoted, row["score"])


# --------------------------------------------------------------------------------------
# 5. The deployed Task path vs the cross-validated rule
# --------------------------------------------------------------------------------------


def _write_case(path, *, faulty="03", lift=2.0, rows=900, cars=CARS, seed=0, setpoint=True, time=None):
    rng = np.random.default_rng(seed)
    time = pd.date_range("2024-07-01", periods=rows, freq="30s") if time is None else time
    hour = time.hour.to_numpy() + time.minute.to_numpy() / 60.0
    ambient = 26.0 + 6.0 * np.sin(2 * np.pi * (hour - 9.0) / 24.0)
    hot = ambient >= np.median(ambient)
    data = {"Time": time}
    for car in cars:
        indoor = 23.0 + 0.15 * rng.standard_normal(rows)
        if car == faulty:
            indoor = indoor + lift * hot
        data[f"Car {car} - Indoor Average Temperature"] = indoor
        if setpoint:
            data[f"Car {car} - ACV Control Temperature (Cooling)"] = np.full(rows, 23.0)
        data[f"Car {car} - Outdoor Average Temperature"] = ambient
        data[f"Car {car} - ACV Running Mode"] = ["Automatic Cooling"] * rows
        data[f"Car {car} - ACV Information Valid"] = ["Valid"] * rows
    pd.DataFrame(data).to_excel(path, index=False)
    return path


def test_task_run_honours_the_models_mask_configuration(tmp_path):
    """`ACVTask.featurise` ignores `ranker.feature_config`, so a non-default rule is mis-served.

    14 of the 106 committed ladder rows use a non-default mask and two of them score 1.0000, so a
    future promotion of such a row would ship predictions computed with the wrong mask while the
    CV json reported the right one. Nothing asserts the two agree.
    """
    p = _write_case(tmp_path / "case.xlsx")
    case = af.load_case(p)
    expected = af.case_masks(case, hot_quantile=0.75)["hot_threshold"]
    ranker = acv.ACVRanker(features=("peer_delta_hot_loo",), hot_quantile=0.75, name="hq75")
    res = acv.ACVTask().run(p, ranker)
    assert float(res.extras["hot_threshold"]) == pytest.approx(float(expected))


# --------------------------------------------------------------------------------------
# 6. Adversarial inputs through load / featurise / predict
# --------------------------------------------------------------------------------------


def test_duplicate_timestamps_still_produce_a_full_ranking(tmp_path):
    time = pd.DatetimeIndex(np.repeat(pd.date_range("2024-07-01", periods=450, freq="60s"), 2))
    p = _write_case(tmp_path / "dupes.xlsx", time=time)
    res = acv.ACVTask().run(p, acv.BASELINE_RANKER)
    ranked = res.rows[0]["ranked_cars"].split("|")
    assert sorted(ranked) == CARS and len(set(ranked)) == 8


def test_a_train_with_more_than_eight_cars_is_ranked_in_full(tmp_path):
    cars = [f"{i:02d}" for i in range(1, 13)]
    p = _write_case(tmp_path / "twelve.xlsx", cars=cars, faulty="11")
    res = acv.ACVTask().run(p, acv.BASELINE_RANKER)
    ranked = res.rows[0]["ranked_cars"].split("|")
    assert sorted(ranked) == cars
    assert ranked[0] == "11"
    assert rank_decay(ranked, "11", len(cars)) == pytest.approx(1.0)


def test_a_file_with_no_setpoint_column_still_ranks(tmp_path):
    """The default tie-break `ctrl_residual` is all-NaN without a setpoint; ranking must survive."""
    p = _write_case(tmp_path / "nosp.xlsx", setpoint=False, faulty="05")
    feats = acv.ACVTask().featurise(af.load_case(p))
    assert feats["ctrl_residual"].isna().all()
    ranked, _ = acv.rank_cars(feats, acv.BASELINE_RANKER)
    assert sorted(ranked) == CARS and ranked[0] == "05"


def test_a_short_file_of_one_thousand_samples_is_not_dropped(tmp_path):
    p = _write_case(tmp_path / "short.xlsx", rows=1000, faulty="02")
    feats = acv.ACVTask().featurise(af.load_case(p))
    assert bool(feats["enough_data"].all())
    assert acv.rank_cars(feats, acv.BASELINE_RANKER)[0][0] == "02"


def test_a_constant_indoor_temperature_ranks_by_header_order_without_crashing(tmp_path):
    rows = 600
    time = pd.date_range("2024-07-01", periods=rows, freq="30s")
    data = {"Time": time}
    for car in CARS:
        data[f"Car {car} - Indoor Average Temperature"] = np.full(rows, 23.0)
        data[f"Car {car} - ACV Control Temperature (Cooling)"] = np.full(rows, 23.0)
        data[f"Car {car} - Outdoor Average Temperature"] = np.full(rows, 30.0)
        data[f"Car {car} - ACV Running Mode"] = ["Automatic Cooling"] * rows
    p = tmp_path / "flat.xlsx"
    pd.DataFrame(data).to_excel(p, index=False)
    res = acv.ACVTask().run(p, acv.BASELINE_RANKER)
    assert res.rows[0]["ranked_cars"].split("|") == CARS
    assert all(np.isfinite(v) for v in res.numbers.values())


def test_single_digit_car_headers_are_read_and_echoed_as_spelled(tmp_path):
    """`CAR_COL_RE` required two digits; a `Car 1 - ...` file was refused outright.

    The Info Kit only promises the *parameter set* varies between files, so the two-digit spelling
    was the documented contract - but it is also the one header spelling that would have made a
    held-out case unreadable rather than mis-ranked. `acv_features.CAR_COL_RE` now takes one or
    two digits and keeps the id exactly as the header spells it, because `ranked_cars` has to echo
    the file's own identifiers.
    """
    rows = 600
    time = pd.date_range("2024-07-01", periods=rows, freq="30s")
    hour = time.hour.to_numpy() + time.minute.to_numpy() / 60.0
    ambient = 26.0 + 6.0 * np.sin(2 * np.pi * (hour - 9.0) / 24.0)
    rng = np.random.default_rng(0)
    data = {"Time": time}
    for i in range(1, 9):
        indoor = 23.0 + 0.15 * rng.standard_normal(rows)
        if i == 3:
            indoor = indoor + 2.0 * (ambient >= np.median(ambient))
        data[f"Car {i} - Indoor Average Temperature"] = indoor
        data[f"Car {i} - Outdoor Average Temperature"] = ambient
        data[f"Car {i} - ACV Running Mode"] = ["Automatic Cooling"] * rows
    p = tmp_path / "onedigit.xlsx"
    pd.DataFrame(data).to_excel(p, index=False)

    case = af.load_case(p)
    assert case.cars == [str(i) for i in range(1, 9)]  # "1", not "01"
    res = acv.ACVTask().run(p, acv.BASELINE_RANKER)
    ranked = res.rows[0]["ranked_cars"].split("|")
    assert sorted(ranked, key=int) == case.cars
    assert ranked[0] == "3"


def test_two_digit_car_headers_still_read_as_two_digits(tmp_path):
    """The organisers' own spelling is untouched: `Car 01` stays `"01"` in the CSV."""
    p = _write_case(tmp_path / "twodigit.xlsx", faulty="06")
    assert af.load_case(p).cars == CARS
    res = acv.ACVTask().run(p, acv.BASELINE_RANKER)
    assert res.rows[0]["ranked_cars"].split("|")[0] == "06"
