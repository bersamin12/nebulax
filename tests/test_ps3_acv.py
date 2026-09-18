"""ACV subsystem tests: loader, ranker, fold-local rule, explanation, CSV round trip.

Fixtures are the two committed header shapes (`tests/fixtures/ps3/acv/`, see its README): the
eight-parameter file five of the six training cases and the test case use, and the 60-parameter
`acv_case_04` shape. Both are 8-30 row slices - enough to prove the loader reads a file by its own
headers, far too short to rank anything - so every behavioural test **synthesises** a full
multi-day case with :func:`make_case`, exactly as the fixture README prescribes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nebulax.ps3 import acv
from nebulax.ps3 import acv_features as af
from nebulax.ps3.common import RESULTS_DIR, Explanation, acv_car_ids, get_task
from nebulax.ps3.scoring import rank_decay
from nebulax.ps3.submission import validate_csv

FIXTURES = pytest.importorskip("pathlib").Path(__file__).parent / "fixtures" / "ps3" / "acv"
CASE_8 = FIXTURES / "case01_slice.xlsx"
CASE_60 = FIXTURES / "case04_wide_slice.xlsx"

CARS = [f"{i:02d}" for i in range(1, 9)]


# --------------------------------------------------------------------------------------
# Synthetic cases (the fixtures are too short to rank; the README says to synthesise)
# --------------------------------------------------------------------------------------


def make_case(
    path,
    *,
    faulty: str | None = "03",
    lift: float = 2.0,
    days: float = 0.5,
    cars=CARS,
    outdoor_name: str = "Outdoor Average Temperature",
    empty_cars: tuple[str, ...] = (),
    seed: int = 0,
):
    """Write a plausible eight-parameter ACV case: a diurnal ambient, eight cars, one warm one.

    The faulty car sits ``lift`` K above its siblings **only in the hot half of the record**,
    which is the signature the ranker is built for; the others share one tight noise band.
    """
    rng = np.random.default_rng(seed)
    n = int(days * 24 * 120)  # 30 s sampling
    time = pd.date_range("2024-07-01", periods=n, freq="30s")
    hour = time.hour.to_numpy() + time.minute.to_numpy() / 60.0
    ambient = 26.0 + 6.0 * np.sin(2 * np.pi * (hour - 9.0) / 24.0)
    data = {"Car model": "A", "Train number": 620, "Time": time}
    hot = ambient >= np.median(ambient)
    for car in cars:
        if car in empty_cars:
            nan = np.full(n, np.nan)
            data[f"Car {car} - Indoor Average Temperature"] = nan
            data[f"Car {car} - ACV Control Temperature (Cooling)"] = nan
            data[f"Car {car} - {outdoor_name}"] = nan
            data[f"Car {car} - ACV Running Mode"] = [None] * n
            data[f"Car {car} - ACV Information Valid"] = [None] * n
            data[f"Car {car} - Load Halved"] = [None] * n
            continue
        indoor = 23.0 + 0.15 * rng.standard_normal(n) + 0.05 * np.sin(2 * np.pi * hour / 24.0)
        if car == faulty:
            indoor = indoor + lift * hot
        data[f"Car {car} - Indoor Average Temperature"] = np.round(indoor * 2) / 2
        data[f"Car {car} - ACV Control Temperature (Cooling)"] = np.full(n, 23.0)
        data[f"Car {car} - {outdoor_name}"] = np.round(ambient * 2) / 2
        data[f"Car {car} - ACV Running Mode"] = ["Automatic Cooling"] * n
        data[f"Car {car} - ACV Information Valid"] = ["Valid"] * n
        data[f"Car {car} - Load Halved"] = ["Normal"] * n
    pd.DataFrame(data).to_excel(path, index=False)
    return path


@pytest.fixture(scope="module")
def synth_cases(tmp_path_factory):
    """Four synthetic cases with different faulty cars - a miniature leave-one-case-out set."""
    d = tmp_path_factory.mktemp("acv_synth")
    labels = {"case_a.xlsx": "03", "case_b.xlsx": "07", "case_c.xlsx": "01", "case_d.xlsx": "05"}
    for i, (name, car) in enumerate(labels.items()):
        make_case(d / name, faulty=car, seed=i, lift=2.0 + 0.2 * i)
    return d, labels


# --------------------------------------------------------------------------------------
# Loader: each file read by its own headers
# --------------------------------------------------------------------------------------


def test_loads_the_eight_parameter_shape():
    case = af.load_case(CASE_8)
    assert case.cars == CARS
    assert case.file_id == "case01_slice.xlsx"
    assert set(case.panel) >= {"indoor", "setpoint", "outdoor", "running_mode", "valid"}
    assert case.panel["indoor"].shape == (len(case.time), 8)
    assert np.nanmax(case.panel["indoor"].to_numpy(dtype=float)) > 0
    assert case.panel["running_mode"].iloc[0].str.islower().all()
    assert case.unmapped == []
    assert case.dt_seconds == pytest.approx(30.0)


def test_loads_the_sixty_parameter_shape_through_synonyms():
    """`acv_case_04` spells indoor / setpoint / outdoor differently and carries 55 extra params."""
    case = af.load_case(CASE_60)
    assert case.cars == CARS
    assert case.panel["indoor"].notna().any().any(), "Passenger Cabin Temperature must map to indoor"
    assert case.panel["setpoint"].notna().any().any(), "Target Temperature Value must map to setpoint"
    assert case.panel["outdoor"].notna().any().any(), "Fresh Air Temperature must map to outdoor"
    assert "Refrigeration System 1 High Pressure Value" in case.unmapped
    assert len(case.unmapped) > 40


def test_car_ids_match_the_contract_helper():
    assert af.load_case(CASE_8).cars == acv_car_ids(CASE_8)


@pytest.mark.parametrize("path_name", ["case01_slice.xlsx", "case04_wide_slice.xlsx"])
def test_featurise_runs_on_both_header_shapes(path_name):
    """A 30-row slice cannot rank, but it must never raise or invent numbers."""
    feats = af.car_features(af.load_case(FIXTURES / path_name))
    assert list(feats.index) == CARS
    assert set(af.RANKERS) <= set(feats.columns)
    assert not feats.loc[~feats["enough_data"], "peer_delta_hot"].notna().any()


# --------------------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------------------


def test_ranks_the_leaking_car_first(tmp_path):
    p = make_case(tmp_path / "c.xlsx", faulty="06")
    feats = af.car_features(af.load_case(p))
    ranked, scores = acv.rank_cars(feats, acv.BASELINE_RANKER)
    assert ranked[0] == "06"
    assert rank_decay(ranked, "06") == 1.0
    assert scores.loc["06", "score"] > scores.loc["01", "score"]


def test_ranking_lists_every_header_car_exactly_once(tmp_path):
    p = make_case(tmp_path / "c.xlsx", faulty="02")
    ranked, _ = acv.rank_cars(af.car_features(af.load_case(p)), acv.BASELINE_RANKER)
    assert sorted(ranked) == CARS
    assert len(set(ranked)) == 8


def test_cars_with_no_data_rank_last(tmp_path):
    """`acv_case_04` has four entirely empty cars; they must never outrank a car with data."""
    p = make_case(tmp_path / "c.xlsx", faulty="02", empty_cars=("05", "06", "07", "08"))
    ranked, _ = acv.rank_cars(af.car_features(af.load_case(p)), acv.BASELINE_RANKER)
    assert ranked[0] == "02"
    assert ranked[-4:] == ["05", "06", "07", "08"]


def test_outdoor_synonym_does_not_change_the_answer(tmp_path):
    """Two training cases spell the ambient sensor 'Outside Temperature Sensor Reading'."""
    a = make_case(tmp_path / "a.xlsx", faulty="04", seed=3)
    b = make_case(
        tmp_path / "b.xlsx", faulty="04", seed=3, outdoor_name="Outside Temperature Sensor Reading"
    )
    ra, _ = acv.rank_cars(af.car_features(af.load_case(a)), acv.BASELINE_RANKER)
    rb, _ = acv.rank_cars(af.car_features(af.load_case(b)), acv.BASELINE_RANKER)
    assert ra == rb


def test_no_car_id_prior(tmp_path):
    """The same physics on a different car must move the answer to that car, not to a favourite."""
    for car in ("01", "04", "08"):
        p = make_case(tmp_path / f"{car}.xlsx", faulty=car, seed=7)
        ranked, _ = acv.rank_cars(af.car_features(af.load_case(p)), acv.BASELINE_RANKER)
        assert ranked[0] == car


# --------------------------------------------------------------------------------------
# The fold-local rule
# --------------------------------------------------------------------------------------


def test_selection_never_reads_the_held_out_case(synth_cases):
    """Corrupting the held-out case must not change the rule chosen on the training cases."""
    d, labels = synth_cases
    bank = acv.FeatureBank(sorted(d.glob("*.xlsx")))
    held = "case_a.xlsx"
    train_files = [f for f in sorted(labels) if f != held]
    cands = acv.selection_pool()
    chosen = acv.fit_ranker(bank, labels, cands, files=train_files)

    feats = bank.for_config(0.5, True)
    original = feats[held].copy()
    try:
        for col in feats[held].columns:
            if col not in ("enough_data",):
                feats[held][col] = np.nan
        again = acv.fit_ranker(bank, labels, cands, files=train_files)
    finally:
        feats[held] = original
    assert again.as_dict() == chosen.as_dict()
    assert held not in chosen.trained_on
    assert set(chosen.trained_on) == set(train_files)


def test_the_chosen_rule_moves_with_the_training_fold(synth_cases):
    """Different training cases must be able to select a different rule (nothing is hard-wired)."""
    d, labels = synth_cases
    bank = acv.FeatureBank(sorted(d.glob("*.xlsx")))
    cands = [
        acv.ACVRanker(features=("peer_delta_hot",), name="A"),
        acv.ACVRanker(features=("ctrl_residual",), tie_break=("peer_delta_hot",), name="B"),
    ]
    feats = bank.for_config(0.5, True)
    # make rule B the winner on case_a only, by ruining A's feature there
    feats["case_a.xlsx"] = feats["case_a.xlsx"].copy()
    feats["case_a.xlsx"]["peer_delta_hot"] = np.linspace(1.0, 0.0, 8)  # points at car 01, wrong
    on_a = acv.fit_ranker(bank, labels, cands, files=["case_a.xlsx"])
    on_bcd = acv.fit_ranker(bank, labels, cands, files=["case_b.xlsx", "case_c.xlsx", "case_d.xlsx"])
    assert on_a.name == "B"
    assert on_bcd.name == "A"


def test_leave_one_case_out_scores_every_case(synth_cases):
    d, labels = synth_cases
    bank = acv.FeatureBank(sorted(d.glob("*.xlsx")))
    cv = acv.leave_one_case_out(bank, labels, None)
    assert cv["n_cases"] == 4
    assert [f["held_out"] for f in cv["folds"]] == sorted(labels)
    assert cv["score"] == pytest.approx(1.0)
    assert all(f["rank"] == 1 for f in cv["folds"])


# --------------------------------------------------------------------------------------
# Task protocol, explanation, CSV round trip
# --------------------------------------------------------------------------------------


def test_task_registered_with_the_contract_attributes():
    task = get_task("acv")
    assert task.name == "acv"
    assert task.output_filename == "acv_predictions.csv"
    assert task.accepts == (".xlsx", ".xls")


def test_run_to_rows_validate_csv_round_trip(tmp_path):
    task = get_task("acv")
    src = make_case(tmp_path / "acv_test_case.xlsx", faulty="05")
    result = task.run(src, acv.BASELINE_RANKER)
    rows = task.to_rows(result)
    assert rows == [{"file_id": "acv_test_case.xlsx", "ranked_cars": "|".join(result.extras["ranked_cars"])}]
    out = tmp_path / "acv_predictions.csv"
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    report = validate_csv(
        "acv", out, ["acv_test_case.xlsx"], expected_cars={"acv_test_case.xlsx": acv_car_ids(src)}
    )
    report.raise_for_errors()
    assert report.ok and report.n_rows == 1


def test_explanation_payload_is_small_and_valid(tmp_path):
    task = get_task("acv")
    src = make_case(tmp_path / "case.xlsx", faulty="07", days=2.0)  # > MAX_TRACE_POINTS rows
    expl = task.explain(task.run(src, acv.BASELINE_RANKER))
    assert isinstance(expl, Explanation)
    payload = expl.as_dict()
    assert payload["viewport"] == {"car": 7, "side": None, "health": "crit", "component": "car7_ac1"}
    assert payload["numbers"]["top_car"] == 7.0
    assert payload["numbers"]["rank_margin_to_2nd"] > 0
    assert set(payload["numbers"]) >= {f"car_{c}_peer_delta_hot_k" for c in CARS}
    trace = payload["trace"]
    assert 0 < len(trace["x"]) <= acv.MAX_TRACE_POINTS
    assert len(trace["x"]) == len(trace["y"])
    assert trace["marks"] and trace["marks"][0]["kind"] == "peak"
    assert all(isinstance(x, str) for x in trace["x"])


def test_predict_accepts_a_plain_mapping_as_the_model(tmp_path):
    p = make_case(tmp_path / "c.xlsx", faulty="03")
    task = get_task("acv")
    feats = task.featurise(task.load(p))
    result = task.predict(feats, {"features": ("guo_residual",), "name": "guo"})
    assert result.extras["ranked_cars"][0] == "03"


# --------------------------------------------------------------------------------------
# Adversarial inputs
# --------------------------------------------------------------------------------------


def test_missing_time_column(tmp_path):
    p = tmp_path / "bad.xlsx"
    pd.DataFrame({"Car 01 - Indoor Average Temperature": [1.0, 2.0]}).to_excel(p, index=False)
    with pytest.raises(ValueError, match="no 'Time' column"):
        af.load_case(p)


def test_no_car_columns_at_all(tmp_path):
    p = tmp_path / "bad.xlsx"
    pd.DataFrame({"Time": pd.date_range("2024-01-01", periods=3, freq="30s"), "x": [1, 2, 3]}).to_excel(
        p, index=False
    )
    with pytest.raises(ValueError, match="does not look like an ACV case file"):
        af.load_case(p)


def test_empty_sheet(tmp_path):
    p = tmp_path / "empty.xlsx"
    pd.DataFrame(columns=["Time", "Car 01 - Indoor Average Temperature"]).to_excel(p, index=False)
    with pytest.raises(ValueError, match="empty"):
        af.load_case(p)


def test_unparseable_timestamps(tmp_path):
    p = tmp_path / "bad.xlsx"
    pd.DataFrame(
        {"Time": ["not-a-date"] * 3, "Car 01 - Indoor Average Temperature": [22.0, 23.0, 24.0]}
    ).to_excel(p, index=False)
    with pytest.raises(ValueError, match="no parseable timestamp"):
        af.load_case(p)


def test_extra_columns_and_none_tokens_are_tolerated(tmp_path):
    """An unknown parameter is listed, not fatal; 'None'/'Invalid' become NaN, not strings."""
    p = make_case(tmp_path / "c.xlsx", faulty="02")
    df = pd.read_excel(p)
    df["Car 01 - Some Future Parameter"] = "whatever"
    col = df["Car 01 - Indoor Average Temperature"].astype(object)
    col.iloc[: len(df) // 2] = "Invalid"
    df["Car 01 - Indoor Average Temperature"] = col
    df.to_excel(p, index=False)
    case = af.load_case(p)
    assert "Some Future Parameter" in case.unmapped
    assert case.panel["indoor"]["01"].isna().sum() > len(df) // 3
    ranked, _ = acv.rank_cars(af.car_features(case), acv.BASELINE_RANKER)
    assert ranked[0] == "02"


def test_all_cars_empty_still_returns_a_full_ranking(tmp_path):
    p = make_case(tmp_path / "c.xlsx", faulty=None, empty_cars=tuple(CARS))
    ranked, _ = acv.rank_cars(af.car_features(af.load_case(p)), acv.BASELINE_RANKER)
    assert sorted(ranked) == CARS, "a dead file must still produce a valid, complete ranking"


def test_unknown_ranker_feature_is_refused():
    with pytest.raises(ValueError, match="unknown ACV ranker feature"):
        acv.ACVRanker(features=("no_such_feature",))


# --------------------------------------------------------------------------------------
# The deployed path and the cross-validated rule must agree (W4 verifier findings 6-9)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("hot_quantile,cooling_only", [(0.5, True), (0.75, True), (0.5, False)])
def test_cv_and_predict_use_the_same_mask(tmp_path, hot_quantile, cooling_only):
    """`ACVTask` must featurise with the ranker's own mask, not always with the default.

    `hot_quantile` / `cooling_only` are part of the rule: 14 of the 106 ladder rows use a
    non-default mask. The cross-validation builds the frame at the ranker's configuration
    (`FeatureBank.for_ranker`), so the deployed path has to as well, or a promoted row would ship
    predictions computed differently from the number reported for it.
    """
    p = make_case(tmp_path / f"c{hot_quantile}{cooling_only}.xlsx", faulty="04", days=0.4)
    ranker = acv.ACVRanker(
        features=("peer_delta_hot_loo",),
        hot_quantile=hot_quantile,
        cooling_only=cooling_only,
        name=f"hq{hot_quantile}",
    )
    # what the CV sees
    cv_feats = af.car_features(
        af.load_case(p), hot_quantile=hot_quantile, cooling_only=cooling_only
    )
    cv_ranked, _ = acv.rank_cars(cv_feats, ranker)
    # what the app and the submission packer see
    result = acv.ACVTask().run(p, ranker)

    assert result.rows[0]["ranked_cars"].split("|") == cv_ranked
    assert float(result.extras["hot_threshold"]) == pytest.approx(
        float(cv_feats.attrs["hot_threshold"])
    )
    assert tuple(result.extras["features"].attrs["feature_config"]) == pytest.approx(
        ranker.feature_config
    )


def test_predict_refuses_a_feature_frame_built_with_another_mask(tmp_path):
    """The one way the two paths could still disagree is a frame passed in by hand."""
    p = make_case(tmp_path / "c.xlsx", faulty="04", days=0.3)
    task = get_task("acv")
    feats = task.featurise(task.load(p), acv.BASELINE_RANKER)  # default mask
    with pytest.raises(ValueError, match="hot_quantile/cooling_only"):
        task.predict(feats, acv.ACVRanker(features=("peer_delta_hot",), hot_quantile=0.75))


def test_the_ladder_winner_and_the_baseline_agree_on_the_test_row(tmp_path):
    """`train()` no longer promotes a ladder winner - and it costs nothing.

    The two rules (`peer_delta_hot`, the pre-registered baseline, and `peer_delta_hot_loo`, the
    row that tops the six-case ladder) differ only in whether a car is included in the median it
    is compared against, which cannot change a clear leak. The committed CSVs are the proof on the
    organisers' own Test file: they are byte-identical, so the headline change moves no
    deliverable.
    """
    p = make_case(tmp_path / "acv_test_case.xlsx", faulty="07", days=0.3)
    task = get_task("acv")
    baseline = task.run(p, acv.BASELINE_RANKER)
    ladder_best = task.run(p, acv.ACVRanker(features=("peer_delta_hot_loo",), name="peer_delta_hot_loo"))
    assert baseline.extras["ranked_cars"][0] == ladder_best.extras["ranked_cars"][0] == "07"

    committed = [
        RESULTS_DIR / "acv_predictions_baseline.csv",
        RESULTS_DIR / "acv_predictions_ladder.csv",
    ]
    if not all(c.exists() for c in committed):
        pytest.skip("committed ACV prediction CSVs are not present")
    assert committed[0].read_bytes() == committed[1].read_bytes()


def test_chance_floors_count_the_empty_cars(tmp_path):
    """A blind ranking of a case with four dead cars scores 0.8125, not 0.5625."""
    full = af.car_features(af.load_case(make_case(tmp_path / "full.xlsx", faulty="02", days=0.3)))
    holed = af.car_features(
        af.load_case(
            make_case(
                tmp_path / "holed.xlsx", faulty="02", days=0.3, empty_cars=("05", "06", "07", "08")
            )
        )
    )
    assert acv.chance_floors({"a": full}) == pytest.approx({"uniform": 0.5625, "blind": 0.5625})
    floors = acv.chance_floors({"a": full, "b": holed})
    assert floors["uniform"] == pytest.approx(0.5625)
    assert floors["blind"] == pytest.approx((0.5625 + 0.8125) / 2)


def test_a_few_valued_feature_is_flagged_tie_break_only(tmp_path):
    """`_is_degenerate` must catch a feature that cannot order the fleet, not only a constant one.

    `robust_peer_z` is the real case: the MAD scaling quantises it to three levels per case, so
    its ladder score is produced by its `peer_delta_hot_loo` tie-break.
    """
    p = make_case(tmp_path / "c.xlsx", faulty="02")
    bank = acv.FeatureBank([p])
    feats = bank.for_config(0.5, True)
    assert feats[p.name]["robust_peer_z"].nunique() <= af.MIN_DISTINCT_LEVELS
    rz = acv.ACVRanker(features=("robust_peer_z",), tie_break=("peer_delta_hot_loo",), name="rz")
    assert acv._is_degenerate(bank, rz) is True
    assert acv._is_degenerate(bank, acv.BASELINE_RANKER) is False


def test_a_fold_that_ties_on_rank_decay_chooses_on_the_training_margin(synth_cases):
    """With rank decay tied, the pick must still come from the training cases, not the pool order.

    Two rules that both rank the true car first on every training case tie on score and on top-1;
    `acv.separation` then prefers the one that puts the labelled car further from the pack (in
    across-car MAD units), and swapping which rule that is swaps the winner - which a pool-order
    tie-break could never do. This is the mechanism that makes the per-fold selection on the
    organisers' six cases genuinely fold-dependent (the `acv_case_04` fold picks a different rule
    from the other five).
    """
    d, labels = synth_cases
    bank = acv.FeatureBank(sorted(d.glob("*.xlsx")))
    feats = bank.for_config(0.5, True)
    cands = [
        acv.ACVRanker(features=("peer_delta_hot",), tie_break=(), name="A"),
        acv.ACVRanker(features=("guo_residual",), tie_break=(), name="B"),
    ]
    files = ["case_b.xlsx", "case_c.xlsx"]

    def plant(col: str, pack_spread: float) -> None:
        """Rank the true car first in every training file, `pack_spread` apart from the pack."""
        for fid in files:
            frame = feats[fid]
            others = [c for c in frame.index if c != labels[fid]]
            frame.loc[labels[fid], col] = 1.0
            frame.loc[others, col] = np.arange(len(others)) * pack_spread

    for fid in files:
        feats[fid] = feats[fid].copy()
    plant("peer_delta_hot", 0.1)
    plant("guo_residual", 0.001)  # same order, far tighter pack -> far larger margin
    chosen = acv.fit_ranker(bank, labels, cands, files=files)
    assert chosen.name == "B", chosen.cv

    plant("peer_delta_hot", 0.001)
    plant("guo_residual", 0.1)
    chosen = acv.fit_ranker(bank, labels, cands, files=files)
    assert chosen.name == "A", chosen.cv
