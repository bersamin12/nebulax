"""Tests for the physics / exclusion / wrapper rows (:mod:`nebulax.models.physics`) and the
river drift rows (:mod:`nebulax.models.drift`).

Fast by construction: every smoke case is 1,000 fake rows/windows of the right shape, the
raw-window cases use short windows, and the whole module runs in a few seconds.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest
import yaml

from nebulax.bench.base import AnomalyDetector
from nebulax.bench.registry import build, get_spec, list_models, register, unregister
from nebulax.models import drift as Dm  # noqa: F401  (registers the drift rows)
from nebulax.models import physics as Pm

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
LADDER = REPO_ROOT / "configs" / "model_ladder.yaml"

#: Every row this module owns, with the ladder metadata it must be registered under.
ROWS: dict[str, tuple[str, str, str]] = {
    # name: (input_kind, family, task)
    "duty_cycle_ratio_cusum": ("cycle_features", "physics_feature", "ad"),
    "kurtogram_envelope_bpfo": ("raw_window", "physics_feature", "ad"),
    "peer_delta_temperature": ("window_stats", "physics_feature", "ad"),
    "thermal_residual_model": ("window_stats", "physics_residual", "ad"),
    "dwt_w8w9_startpeak": ("raw_window", "time_frequency", "ad"),
    "transition_mask": ("raw_window", "exclusion_rule", "cpd"),
    "page_hinkley_cycle_scalar": ("cycle_features", "drift_detector", "cpd"),
    "adwin_kswin_residual": ("cycle_features", "drift_detector", "cpd"),
    "page_hinkley_residual": ("raw_window", "drift_detector", "cpd"),
    "adwin_residual": ("raw_window", "drift_detector", "cpd"),
    "page_hinkley_thermal_residual": ("window_stats", "drift_detector", "cpd"),
    "peer_normalisation": ("cycle_features", "wrapper", "ad"),
    "peer_normalisation_shared": ("window_stats", "wrapper", "ad"),
    "k_of_n_corroboration": ("window_stats", "wrapper", "ad"),
}

N_SMOKE = 1000
L_SMOKE = 256
PITCH_S = 600.0

# The column / channel layouts the shipped feature tables really have (dumped from
# nebulax.bench.data.load_bench on 16 Sep 2026); the verifier's findings quote these indices.
SIM_PNEUMATIC_CYCLE_NAMES = [
    "t_loaded", "t_unloaded", "t_off", "dP_dt_off", "dP_dt_loaded", "I_loaded_mean",
    "I_start_peak", "T_oil_max", "TP2_minus_TP3_mean", "H1_loaded_mean", "tower_switches",
    "purge_count", "LPS_any", "hour_of_day", "load_frac_mean", "idle_run_ratio",
    "duty_ratio", "Flowmeter_max", "P_res_min", "in_service_frac", "dropout_frac",
]
SIM_PNEUMATIC_RAW_CHANNELS = [
    "TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current",
    "COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level",
    "Caudal_impulses", "Flowmeter",
]
SIM_DOOR_RAW_CHANNELS = [
    "pos_ref", "pos", "vel", "current", "voltage", "pwm", "ls_open", "ls_closed",
    "interlock", "door_key", "emergency_relay", "obstruction", "T_motor",
]
CRANFIELD_RAW_CHANNELS = ["pos_ref", "pos", "vel", "current"]
METROPT3_WS_NAMES = [
    f"{sig}_{agg}_{stat}"
    for sig in ("TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current")
    for agg in ("mean", "max")
    for stat in ("mean", "std", "min", "max", "slope", "rms", "kurtosis", "crest")
] + [
    f"{sig}_duty_{stat}"
    for sig in ("COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses")
    for stat in ("mean", "std", "min", "max", "slope", "rms", "kurtosis", "crest")
]
SIM_BEARING_WS_NAMES = [
    f"{sig}_{stat}"
    for sig in ("T_box", "T_box_wayside", "vib_rms", "vib_kurt", "vib_crest", "vib_bpfo")
    for stat in ("mean", "std", "min", "max", "slope", "rms", "kurtosis", "crest")
]
#: One synthetic-fleet bogie: four boxes a side, the L/R letter in the component id.
BEARING_SERIES = [f"bearing_0000/axlebox_{i}{side}" for side in ("L", "R") for i in (1, 2, 3, 4)]


def fleet_context(n_steps: int, series: list[str], units: list[str] | None = None, *, pitch_s: float = PITCH_S):
    """``(t, series, unit)`` for ``n_steps`` sweeps of ``series``, all sharing each timestamp."""
    per = len(series)
    t = fake_times(n_steps * per, peers=per, pitch_s=pitch_s)
    ser = np.array(series * n_steps, dtype=object)
    u = np.array([(units or ["T01"] * per)[i % per] for i in range(n_steps * per)], dtype=object)
    return t, ser, u


# ======================================================================================
# fake-data helpers (the "helper in tests/" the brief asks for)
# ======================================================================================


def fake_times(n: int, *, peers: int = 1, pitch_s: float = PITCH_S) -> np.ndarray:
    """``datetime64[ms]`` end-timestamps; ``peers`` rows share each timestamp."""
    steps = np.repeat(np.arange((n + peers - 1) // peers), peers)[:n]
    ms = (steps * pitch_s * 1000.0).astype("int64")
    return np.datetime64("2020-02-01T00:00:00", "ms") + ms.astype("timedelta64[ms]")


def fake_X(
    input_kind: str,
    n: int = N_SMOKE,
    *,
    seed: int = 0,
    n_features: int = 12,
    n_channels: int = 3,
    length: int = L_SMOKE,
) -> np.ndarray:
    """Fake feature matrix / window stack of the shape ``input_kind`` promises."""
    rng = np.random.default_rng(seed)
    if input_kind == "raw_window":
        return rng.normal(0.0, 1.0, size=(n, length, n_channels)).astype(np.float32)
    if input_kind == "cycle_features":
        return np.abs(rng.normal(3.0, 0.6, size=(n, n_features))).astype(np.float32)
    return rng.normal(20.0, 1.0, size=(n, n_features)).astype(np.float32)


def smoke_data(name: str, *, peers: int = 1) -> tuple[np.ndarray, np.ndarray]:
    kind = get_spec(name).input_kind
    return fake_X(kind), fake_times(N_SMOKE, peers=peers)


@register("_dummy_inner_for_tests", input_kind="window_stats", family="wrapper", task="ad")
class _DummyInner(AnomalyDetector):
    """Trivial registered detector used to prove the wrappers really forward to an inner."""

    def __init__(self, bias: float = 0.0) -> None:
        super().__init__(bias=bias)
        self.bias = float(bias)
        self.saw_shape_: tuple[int, ...] = ()

    def _fit(self, X: np.ndarray, t: np.ndarray | None = None, **kwargs: Any) -> None:
        self.saw_shape_ = X.shape

    def _score(self, X: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        flat = np.asarray(X, dtype=np.float64).reshape(X.shape[0], -1)
        return np.abs(flat).max(axis=1) + self.bias


# ======================================================================================
# registry / ladder agreement
# ======================================================================================


def _ladder_rows() -> dict[str, dict[str, Any]]:
    cfg = yaml.safe_load(LADDER.read_text())
    found: dict[str, dict[str, Any]] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            nm = node.get("name")
            if isinstance(nm, str) and nm in ROWS:
                found[nm] = node
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(cfg)
    return found


@pytest.mark.parametrize("name", sorted(ROWS))
def test_registered_exactly_as_the_ladder_row(name: str) -> None:
    row = _ladder_rows()[name]
    spec = get_spec(name)
    assert spec.input_kind == row["input_kind"] == ROWS[name][0]
    assert spec.family == row["family"] == ROWS[name][1]
    assert spec.task == row.get("task", "ad") == ROWS[name][2]


def test_every_row_is_reachable_through_build() -> None:
    registered = {s.name for s in list_models()}
    assert set(ROWS) <= registered
    for name in ROWS:
        assert isinstance(build(name), AnomalyDetector)


# ======================================================================================
# smoke fit/score on 1k fake rows
# ======================================================================================


@pytest.mark.parametrize("name", sorted(ROWS))
def test_smoke_fit_score(name: str) -> None:
    X, t = smoke_data(name, peers=4)
    model = build(name)
    t0 = time.perf_counter()
    model.fit(X, t)
    assert time.perf_counter() - t0 < 30.0
    scores = model.score(X, t)
    assert scores.shape == (X.shape[0],)
    assert scores.dtype == np.float64
    assert np.isfinite(scores).all()
    assert model.fit_seconds >= 0.0
    assert model.get_params() == model.params


@pytest.mark.parametrize("name", sorted(ROWS))
def test_scores_are_finite_without_timestamps_and_with_nans(name: str) -> None:
    X, _ = smoke_data(name)
    Xn = np.array(X, dtype=np.float32)
    Xn.reshape(Xn.shape[0], -1)[::37, 0] = np.nan  # a NaN in every 37th row
    model = build(name).fit(Xn, None)
    scores = model.score(Xn, None)
    assert np.isfinite(scores).all() and scores.size == Xn.shape[0]


@pytest.mark.parametrize("name", sorted(ROWS))
def test_score_before_fit_raises(name: str) -> None:
    X, t = smoke_data(name)
    with pytest.raises(RuntimeError):
        build(name).score(X, t)


def test_cpd_rows_emit_a_continuous_normalised_score() -> None:
    for name, (_, _, task) in ROWS.items():
        if task != "cpd":
            continue
        X, t = smoke_data(name)
        s = build(name).fit(X, t).score(X, t)
        assert s.min() >= 0.0 and s.max() <= 1.0 + 1e-9, name
        if name != "transition_mask":  # the mask can legitimately flatten a quiet stream
            assert np.unique(s).size > 2, name


# ======================================================================================
# physics behaviour
# ======================================================================================


def _duty_run(n: int, *, leak_from: int, floor: float = 0.1, seed: int = 3) -> np.ndarray:
    """``(n, 2)`` pneumatic-ish cycle rows whose idle/run ratio collapses from ``leak_from``."""
    rng = np.random.default_rng(seed)
    x = np.abs(rng.normal(5.0, 0.4, n))
    if leak_from < n:
        x[leak_from:] *= np.linspace(1.0, floor, n - leak_from)
    return np.column_stack([x, rng.normal(0.0, 1.0, n)]).astype(np.float32)


def test_duty_cycle_ratio_cusum_rises_on_a_collapsing_idle_ratio() -> None:
    n, leak = 1200, 600  # 600 s pitch -> 200 h, so the leak has far more than the 24 h hold
    X = _duty_run(n, leak_from=leak)
    t = fake_times(n)
    m = build("duty_cycle_ratio_cusum", feature_names=["idle_run_ratio", "noise"], window_h=6.0, baseline_days=1.0)
    s = m.fit(X[:500], t[:500]).score(X, t)
    assert m.feature_index_ == 0 and m.feature_name_ == "idle_run_ratio"
    assert s[-50:].mean() > s[:500].mean() + 1.0


def test_duty_cycle_ratio_cusum_needs_the_24_h_hold_and_clears_on_recovery() -> None:
    """R101: a 12 h excursion must not score, and a recovered unit must not stay elevated."""
    pitch, n = 600.0, 900  # 10 min cycles
    per_hour = int(3600.0 / pitch)
    rng = np.random.default_rng(101)
    x = np.abs(rng.normal(5.0, 0.2, n))
    short = slice(300, 300 + 12 * per_hour)  # a 12 h excursion, then full recovery
    long_start = 500
    x[short] = 0.2
    x[long_start:] = 0.2  # a 60+ h sustained crossing
    X = np.column_stack([x, rng.normal(0.0, 1.0, n)]).astype(np.float32)
    t = fake_times(n, pitch_s=pitch)
    m = build(
        "duty_cycle_ratio_cusum",
        feature_names=["idle_run_ratio", "noise"],
        window_h=6.0,
        baseline_days=1.0,
        hold_h=24.0,
    )
    s = m.fit(X[:250], t[:250]).score(X, t)
    # the whole 12 h excursion (plus its rolling-median tail) is below the 24 h hold
    assert s[300 : 300 + 18 * per_hour].max() == 0.0
    # recovered rows are back to zero, not left elevated
    assert s[300 + 18 * per_hour : long_start].max() == 0.0
    # the sustained crossing is silent for its first 24 h and then accumulates
    assert s[long_start : long_start + 24 * per_hour].max() == 0.0
    assert s[-1] > 0.0 and s[-1] == s.max()


def test_duty_cycle_ratio_cusum_finds_idle_run_ratio_in_the_shipped_pneumatic_table() -> None:
    """The shipped sim pneumatic cycle table puts idle_run_ratio at column 15, not column 0."""
    names = SIM_PNEUMATIC_CYCLE_NAMES
    n = 400
    rng = np.random.default_rng(5)
    X = np.abs(rng.normal(3.0, 0.5, size=(n, len(names)))).astype(np.float32)
    t = fake_times(n)
    m = build("duty_cycle_ratio_cusum")
    m.fit(X, t, feature_names=names)  # exactly how the runner calls it
    assert m.feature_index_ == names.index("idle_run_ratio") == 15
    assert m.feature_name_ == "idle_run_ratio"


def test_duty_cycle_ratio_cusum_falls_back_to_duty_ratio_then_refuses() -> None:
    n = 200
    X = np.abs(np.random.default_rng(7).normal(3.0, 0.5, size=(n, 3))).astype(np.float32)
    t = fake_times(n)
    # documented alias: a table with duty_ratio but no idle_run_ratio still works
    m = build("duty_cycle_ratio_cusum").fit(X, t, feature_names=["t_off", "duty_ratio", "noise"])
    assert m.feature_index_ == 1
    # a table with neither is refused by name, not silently resolved to column 0
    with pytest.raises(ValueError, match="idle_run_ratio"):
        build("duty_cycle_ratio_cusum").fit(X, t, feature_names=["t_off", "t_loaded", "noise"])


def test_duty_cycle_ratio_cusum_is_scored_one_series_at_a_time() -> None:
    assert Pm.DutyCycleRatioCusum.SEQUENTIAL is True


def test_duty_cycle_ratio_cusum_switches_off_a_zero_inflated_baseline() -> None:
    """A 6 h median whose healthy p5 is 0.000 can never be crossed: use duty_ratio instead."""
    n = 900
    rng = np.random.default_rng(139)
    idle = np.abs(rng.normal(5.0, 1.0, n))
    # zero-inflated the way MetroPT-3 and the sim fleet are: the OFF phase disappears during
    # the peak-service block of every day, so whole 6 h windows have a median of exactly 0
    idle[(np.arange(n) % 144) < 48] = 0.0
    duty = 0.07 + rng.normal(0.0, 0.002, n)
    duty[600:] = 0.2  # the leak: duty rises and stays up
    X = np.column_stack([idle, duty]).astype(np.float32)
    t = fake_times(n, pitch_s=600.0)
    names = ["idle_run_ratio", "duty_ratio"]
    # the substitution has to be ASKED for: it is a different detector from the ladder row
    m = build("duty_cycle_ratio_cusum", baseline_days=1.0, hold_h=24.0, fallback_feature="duty_ratio")
    s = m.fit(X[:400], t[:400], feature_names=names).score(X, t)
    assert m.fallback_used_ is True
    assert m.feature_name_ == "duty_ratio" and m.direction_ == "up"
    assert s[-1] > 0.0 and (np.diff(s[600:]) >= -1e-9).all()
    # asking for the bare literal rule keeps the unfireable baseline, and says so
    lit = build("duty_cycle_ratio_cusum", baseline_days=1.0, fallback_feature=None)
    lit.fit(X[:400], t[:400], feature_names=names)
    assert lit.fallback_used_ is False and lit.baseline_ == 0.0
    assert lit.score(X, t).max() == 0.0


def test_duty_cycle_ratio_cusum_never_substitutes_duty_ratio_by_default() -> None:
    """R101 identity: the DEFAULT fit is idle_run_ratio/downward, whatever the baseline does.

    The results row carries only the RunSpec fields, so a silent switch to duty_ratio/upward
    would ship a row labelled ``duty_cycle_ratio_cusum`` that is in fact a different detector.
    Default is the literal rule; what a fitted model resolved to is persisted either way.
    """
    n = 900
    rng = np.random.default_rng(1409)
    idle = np.abs(rng.normal(5.0, 1.0, n))
    idle[(np.arange(n) % 144) < 48] = 0.0  # zero-inflated: the healthy p5 lands on the floor
    duty = 0.07 + rng.normal(0.0, 0.002, n)
    X = np.column_stack([idle, duty]).astype(np.float32)
    t = fake_times(n, pitch_s=600.0)
    names = ["idle_run_ratio", "duty_ratio"]
    m = build("duty_cycle_ratio_cusum", baseline_days=1.0)
    m.fit(X[:400], t[:400], feature_names=names)
    assert build("duty_cycle_ratio_cusum").params["fallback_feature"] is None
    assert m.fallback_used_ is False
    assert m.feature_name_ == "idle_run_ratio" and m.feature_index_ == 0
    assert m.direction_ == "down"  # never flipped to the upward duty_ratio rule
    # ... and the shipped pneumatic table resolves the named column, still without a switch
    big = np.abs(rng.normal(3.0, 0.5, size=(400, len(SIM_PNEUMATIC_CYCLE_NAMES)))).astype(np.float32)
    r = build("duty_cycle_ratio_cusum").fit(big, fake_times(400), feature_names=SIM_PNEUMATIC_CYCLE_NAMES)
    assert (r.feature_name_, r.direction_, r.fallback_used_) == ("idle_run_ratio", "down", False)


def test_duty_cycle_ratio_cusum_rejects_a_bad_direction() -> None:
    with pytest.raises(ValueError):
        build("duty_cycle_ratio_cusum", direction="sideways")


def test_thermal_residual_model_flags_a_box_hotter_than_its_context() -> None:
    n = 800
    rng = np.random.default_rng(5)
    speed = rng.uniform(0.0, 30.0, n)
    ambient = rng.normal(30.0, 2.0, n)
    load = rng.uniform(0.5, 1.0, n)
    t_box = 25.0 + 0.8 * speed + 0.9 * ambient + 5.0 * load + rng.normal(0.0, 0.3, n)
    t_box[700:] += 8.0  # a bearing running hot at the same operating point
    X = np.column_stack([t_box, speed, ambient, load]).astype(np.float32)
    names = ["T_box_mean", "speed_mean", "ambient_mean", "load_mean"]
    m = build("thermal_residual_model", feature_names=names)
    m.fit(X[:600], fake_times(600))
    s = m.score(X, fake_times(n))
    assert m.target_index_ == 0 and 0 not in m.covariate_index_
    assert s[700:].mean() > 5.0 > s[:600].mean()


def test_peer_delta_temperature_flags_one_hot_box_among_its_peers() -> None:
    n_steps, peers = 200, 4
    rng = np.random.default_rng(7)
    base = np.repeat(rng.normal(40.0, 3.0, n_steps), peers)  # common-mode: duty, ambient
    val = base + rng.normal(0.0, 0.2, n_steps * peers)
    hot = np.zeros_like(val, dtype=bool)
    hot[np.arange(n_steps * peers) % peers == 2] = True
    val[hot & (np.repeat(np.arange(n_steps), peers) > 150)] += 6.0
    X = np.column_stack([val, rng.normal(0.0, 1.0, val.size)]).astype(np.float32)
    t = fake_times(val.size, peers=peers)
    m = build("peer_delta_temperature", feature_names=["T_box_mean", "other"], time_tol_s=1.0)
    fit_rows = np.repeat(np.arange(n_steps), peers) <= 150
    m.fit(X[fit_rows], t[fit_rows])
    s = m.score(X, t)
    late_hot = hot & (np.repeat(np.arange(n_steps), peers) > 150)
    assert s[late_hot].mean() > 5.0
    assert abs(s[~late_hot].mean()) < 2.0


def test_dwt_detail_energies_match_a_haar_hand_calculation() -> None:
    x = np.array([[1.0, 3.0, 5.0, 11.0]])
    e = Pm.dwt_detail_energies(x, "haar", 1)
    # Haar detail = (x[2k] - x[2k+1]) / sqrt(2)  ->  (-2, -6)/sqrt(2) -> energy 2 + 18
    assert e.shape == (1, 1)
    assert e[0, 0] == pytest.approx(20.0, rel=1e-9)


def test_dwt_w8w9_clamps_levels_to_what_the_window_supports() -> None:
    X = fake_X("raw_window", 120, length=128, seed=11)
    m = build("dwt_w8w9_startpeak").fit(X, fake_times(120))
    assert m.levels_used_ and max(m.levels_used_) <= 9
    assert np.isfinite(m.score(X, fake_times(120))).all()


def test_dwt_w8w9_startpeak_reacts_to_an_inflated_start_peak() -> None:
    n = 200
    rng = np.random.default_rng(13)
    X = rng.normal(0.0, 1.0, size=(n, 512, 1)).astype(np.float32)
    X[150:, :40, 0] += 12.0  # a breakaway current peak on a stiff door
    m = build("dwt_w8w9_startpeak").fit(X[:120], fake_times(120))
    s = m.score(X, fake_times(n))
    assert s[150:].mean() > s[:120].mean() + 1.0


def test_kurtogram_envelope_bpfo_separates_an_impulsive_record() -> None:
    fs, length, n = 20_000.0, 2048, 60
    rng = np.random.default_rng(17)
    X = rng.normal(0.0, 0.05, size=(n, length, 1)).astype(np.float32)
    idx = np.arange(length)
    carrier = np.sin(2 * np.pi * 4000.0 * idx / fs)
    impulses = np.zeros(length)
    impulses[:: int(fs / 105.0)] = 1.0  # ~BPFO repetition rate
    env = np.convolve(impulses, np.exp(-np.linspace(0, 6, 60)), mode="same")
    X[40:, :, 0] += (2.0 * env * carrier).astype(np.float32)
    m = build("kurtogram_envelope_bpfo", fs=fs, shaft_rpm=1200.0, max_level=3, max_fit_windows=40)
    s = m.fit(X[:40], fake_times(40)).score(X, fake_times(n))
    assert s[40:].mean() > s[:40].mean() + 1.0


def _comp_towers_windows(n: int = 300, length: int = 64, seed: int = 19):
    """Pneumatic-shaped raw windows with COMP edges, Towers flips and a noisy TP2."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 0.01, size=(n, length, len(SIM_PNEUMATIC_RAW_CHANNELS))).astype(np.float32)
    ci = SIM_PNEUMATIC_RAW_CHANNELS.index("COMP")
    ti = SIM_PNEUMATIC_RAW_CHANNELS.index("Towers")
    X[:, :, ci] = 1.0
    X[:, :, ti] = 0.0
    # TP2 is analog and noisy with huge steps: the "biggest training step" heuristic picks it
    X[:, :, 0] = rng.normal(8.0, 2.0, size=(n, length))
    comp_edges, tower_edges = np.array([100, 180]), np.array([240])
    for e in comp_edges:
        X[e, length // 2 :, ci] = 0.0
    for e in tower_edges:
        X[e, length // 2 :, ti] = 1.0
    return X, comp_edges, tower_edges, ci, ti


def test_transition_mask_resolves_comp_and_towers_by_name() -> None:
    """The ladder row names COMP and Towers; the old default picked one high-step channel."""
    X, comp_edges, tower_edges, ci, ti = _comp_towers_windows()
    t = fake_times(X.shape[0], pitch_s=10.0)
    m = build("transition_mask")
    m.fit(X, t, feature_names=SIM_PNEUMATIC_RAW_CHANNELS)  # exactly how the runner calls it
    assert m.channels_ == [ci, ti] == [7, 9]
    assert m.channel_names_ == ["COMP", "Towers"]
    assert m.digital_.all()  # both are digital, so any flip is an edge (no sigma tuning)
    flags = m.is_transition(X, t)
    assert flags[comp_edges].all() and flags[tower_edges].all()
    assert flags.sum() < X.shape[0] // 2  # the noisy analog TP2 is NOT what is being masked
    # a named table that carries neither is refused by name, not silently re-routed
    with pytest.raises(ValueError, match="COMP"):
        build("transition_mask").fit(X[:, :, :3], t, feature_names=["TP2", "TP3", "H1"])


def test_transition_mask_scores_masked_regions_with_a_change_point_statistic() -> None:
    """The ladder excludes the masked rows from POINT scoring, it does not zero them."""
    X, comp_edges, tower_edges, _, _ = _comp_towers_windows()
    t = fake_times(X.shape[0], pitch_s=10.0)
    m = build("transition_mask").fit(X, t, feature_names=SIM_PNEUMATIC_RAW_CHANNELS)
    flags = m.is_transition(X, t)
    s = m.score(X, t)
    assert flags.any() and 0.0 <= s.min() and s.max() <= 1.0 + 1e-9
    assert np.unique(s[flags]).size > 1  # a change-point statistic, not one constant
    assert not np.allclose(s[flags], 0.0)
    # the legacy constant behaviour is still reachable, but only by asking for it
    z = build("transition_mask", masked_score="constant").fit(
        X, t, feature_names=SIM_PNEUMATIC_RAW_CHANNELS
    )
    assert np.all(z.score(X, t)[z.is_transition(X, t)] == 0.0)


def test_transition_mask_zeroes_the_flagged_rows_and_exposes_the_flag() -> None:
    n, length = 300, 64
    rng = np.random.default_rng(19)
    X = rng.normal(0.0, 0.01, size=(n, length, 2)).astype(np.float32)
    X[:, :, 1] = 1.0
    edges = np.array([100, 180, 240])
    for e in edges:  # a COMP load/offload edge inside the window
        X[e, length // 2 :, 1] = 0.0
    t = fake_times(n, pitch_s=10.0)
    m = build("transition_mask", channel=1, mask_width_s=10.0, step_sigma=3.0, masked_score="constant")
    m.fit(X, t)
    flags = m.is_transition(X, t)
    assert flags[edges].all()
    assert flags.sum() >= edges.size * 3  # +/- 10 s at a 10 s pitch dilates each edge
    assert flags.sum() < n // 2
    s = m.score(X, t)
    assert np.all(s[flags] == 0.0)


def test_transition_mask_wraps_a_registered_inner_detector() -> None:
    n, length = 200, 32
    X = fake_X("raw_window", n, length=length, n_channels=2, seed=23)
    t = fake_times(n, pitch_s=10.0)
    m = build("transition_mask", inner="_dummy_inner_for_tests", inner_params={"bias": 0.25}, channel=0)
    m.fit(X, t)
    assert isinstance(m.inner_, _DummyInner)
    s = m.score(X, t)
    keep = ~m.is_transition(X, t)
    # the un-masked rows are the INNER detector's ranking (monotonically remapped to [0, 1])
    raw = m.inner_.score(X[keep], t[keep])
    assert np.corrcoef(s[keep], raw)[0, 1] > 0.99


# ======================================================================================
# wrappers
# ======================================================================================


def test_peer_normalisation_cancels_a_fleet_wide_common_mode() -> None:
    n_steps, peers = 150, 4
    rng = np.random.default_rng(29)
    step = np.repeat(np.arange(n_steps), peers)
    common = np.repeat(rng.normal(0.0, 5.0, n_steps), peers)  # every sibling moves together
    pos = np.tile(np.arange(peers), n_steps)
    val = common + rng.normal(0.0, 0.1, n_steps * peers)
    sick = (pos == 2) & (step >= 120)
    val = val + np.where(sick, 3.0, 0.0)  # one component drifts away from its peers
    X = np.column_stack([val, common + rng.normal(0.0, 0.1, val.size)]).astype(np.float32)
    t = fake_times(val.size, peers=peers)
    m = build("peer_normalisation", time_tol_s=1.0, standardise=False)
    healthy = step < 120
    s = m.fit(X[healthy], t[healthy]).score(X, t)
    delta = m.transform(X, t)[:, 0]
    # the 5-sigma common mode is gone from the peer delta, the 3-unit divergence is not
    assert np.abs(delta[healthy]).mean() < 0.5
    assert np.abs(delta[sick]).mean() > 2.0
    assert s[sick].mean() > s[healthy].mean() + 3.0


def test_peer_normalisation_shared_takes_the_side_from_the_component_id() -> None:
    """R151 same-side rule: the L/R letter of the series id, never the row position."""
    n_steps = 120
    rng = np.random.default_rng(31)
    t, ser, unit = fleet_context(n_steps, BEARING_SERIES)
    side_is_r = np.array([s.endswith("R") for s in ser])
    val = rng.normal(0.0, 0.05, ser.size) + np.where(side_is_r, 6.0, 0.0)  # R runs hotter
    X = np.column_stack([val, rng.normal(0.0, 0.05, val.size)]).astype(np.float32)
    shared = build("peer_normalisation_shared", time_tol_s=1.0, standardise=False)
    pooled = build("peer_normalisation_shared", same_side=False, time_tol_s=1.0, standardise=False)
    d_shared = np.abs(shared.fit(X, t, series=ser, unit=unit).transform(X, t, ser, unit)[:, 0])
    d_pooled = np.abs(pooled.fit(X, t, series=ser, unit=unit).transform(X, t, ser, unit)[:, 0])
    assert d_shared.mean() < 0.5  # like-for-like: the side offset cancels
    assert d_pooled.mean() > 2.0  # pooled across the bogie: every box looks 3 K off


def test_same_side_transform_is_invariant_to_row_order_within_a_timestamp() -> None:
    """The parity heuristic changed 376/400 transforms under a shuffle; identity cannot."""
    n_steps = 50
    rng = np.random.default_rng(33)
    t, ser, unit = fleet_context(n_steps, BEARING_SERIES)
    X = rng.normal(40.0, 1.0, size=(ser.size, 3)).astype(np.float32)
    m = build("peer_normalisation_shared", time_tol_s=1.0, standardise=False)
    m.fit(X, t, series=ser, unit=unit)
    base = m.transform(X, t, ser, unit)
    perm = np.concatenate(
        [rng.permutation(np.arange(i * 8, i * 8 + 8)) for i in range(n_steps)]
    )  # shuffle the 8 boxes inside every timestamp
    shuffled = m.transform(X[perm], t[perm], ser[perm], unit[perm])
    np.testing.assert_allclose(shuffled, base[perm], atol=1e-6)


def test_peer_normalisation_forwards_params_to_the_inner_model() -> None:
    X, t = smoke_data("peer_normalisation_shared", peers=4)
    m = build("peer_normalisation_shared", inner="_dummy_inner_for_tests", inner_params={"bias": 7.0})
    s = m.fit(X, t).score(X, t)
    assert isinstance(m.inner_, _DummyInner)
    assert m.inner_.saw_shape_ == (X.shape[0], X.shape[1])
    assert s.min() >= 7.0


def test_k_of_n_corroboration_is_the_kth_largest_member_score() -> None:
    n = 400
    rng = np.random.default_rng(37)
    X = rng.normal(0.0, 1.0, size=(n, 5)).astype(np.float32)
    X[300:, :2] += 8.0  # only two of five channels move
    X[350:, 2] += 8.0  # a third joins: this is where a 3-of-5 rule may fire
    t = fake_times(n)
    m = build("k_of_n_corroboration", k=3, feature_indices=[0, 1, 2, 3, 4])
    s = m.fit(X[:250], t[:250]).score(X, t)
    assert s[300:350].max() < s[350:].min()  # two votes never outrank three
    z = np.abs((m._member_matrix(X, t) - m.center_) / m.scale_)
    # score is the k-th largest member: exactly k members reach it, fewer than k beat it
    np.testing.assert_allclose(s, np.sort(z, axis=1)[:, -3])
    assert ((z >= s[:, None]).sum(axis=1) >= 3).all()
    assert ((z > s[:, None]).sum(axis=1) < 3).all()


def test_k_of_n_corroboration_rejects_k_below_one() -> None:
    with pytest.raises(ValueError):
        build("k_of_n_corroboration", k=0)


# ======================================================================================
# drift
# ======================================================================================


def test_page_hinkley_statistic_grows_after_a_mean_shift() -> None:
    rng = np.random.default_rng(41)
    x = np.r_[rng.normal(0.0, 1.0, 400), rng.normal(4.0, 1.0, 200)]
    stat, flags = Dm.page_hinkley_statistic(x, min_instances=30)
    assert stat[-1] > stat[:400].max()
    assert flags.any()


def test_adwin_statistic_reports_a_magnitude_and_kswin_a_pvalue() -> None:
    rng = np.random.default_rng(43)
    x = np.r_[rng.normal(0.0, 1.0, 300), rng.normal(5.0, 1.0, 300)]
    a_stat, a_flags = Dm.adwin_statistic(x, center=0.0, scale=1.0)
    k_stat, k_flags = Dm.kswin_statistic(x, seed=0)
    assert a_stat[-1] > a_stat[:300].mean() + 1.0 and a_flags.any()
    assert k_stat[300:].max() > k_stat[:300].max()
    assert (k_stat >= 0).all() and (a_stat >= 0).all()


@pytest.mark.parametrize(
    "name",
    ["page_hinkley_cycle_scalar", "adwin_kswin_residual", "page_hinkley_residual", "adwin_residual"],
)
def test_drift_rows_score_higher_after_the_change_point(name: str) -> None:
    n, cut = 600, 400
    rng = np.random.default_rng(47)
    kind = get_spec(name).input_kind
    if kind == "raw_window":
        X = rng.normal(0.0, 1.0, size=(n, 32, 3)).astype(np.float32)
        X[cut:, :, 0] += 6.0
    else:
        X = rng.normal(0.0, 1.0, size=(n, 4)).astype(np.float32)
        X[cut:, 0] += 6.0
    t = fake_times(n)
    # pin the carrier column/channel: with no feature_names the auto-pick is variance-based
    # and the training slice is entirely pre-change, so it cannot know where to look.
    m = build(name, channels=[0]) if kind == "raw_window" else build(name, feature_index=0)
    s = m.fit(X[:300], t[:300]).score(X, t)
    assert s[cut:].mean() > s[:cut].mean()
    assert m.n_drifts_ >= 1


def test_page_hinkley_thermal_residual_streams_the_ridge_residual() -> None:
    n, cut = 600, 450
    rng = np.random.default_rng(53)
    speed = rng.uniform(0.0, 30.0, n)
    ambient = rng.normal(30.0, 2.0, n)
    t_box = 25.0 + 0.8 * speed + 0.9 * ambient + rng.normal(0.0, 0.3, n)
    t_box[cut:] += np.linspace(0.0, 10.0, n - cut)  # slow thermal degradation
    X = np.column_stack([t_box, speed, ambient]).astype(np.float32)
    t = fake_times(n)
    m = build("page_hinkley_thermal_residual", feature_names=["T_box_mean", "speed_mean", "ambient_mean"])
    s = m.fit(X[:400], t[:400]).score(X, t)
    assert m.residual_model_ is not None
    assert s[cut:].mean() > s[:cut].mean() + 0.1
    assert 0.0 <= s.min() and s.max() <= 1.0


def test_drift_rows_are_order_invariant_because_they_sort_by_time() -> None:
    n = 400
    rng = np.random.default_rng(59)
    X = rng.normal(0.0, 1.0, size=(n, 4)).astype(np.float32)
    X[250:, 0] += 5.0
    t = fake_times(n)
    perm = rng.permutation(n)
    m = build("page_hinkley_cycle_scalar", feature_index=0).fit(X, t)
    s_ordered = m.score(X, t)
    s_shuffled = m.score(X[perm], t[perm])
    np.testing.assert_allclose(s_shuffled, s_ordered[perm], atol=1e-9)


# ======================================================================================
# row context: named columns / channels, peer identity  (the verifier's must-fix list)
# ======================================================================================


def test_peer_groups_never_span_units_and_hold_one_row_per_series() -> None:
    """Timestamp-only grouping pooled 160 rows over 20 runs and 10 trains per timestamp."""
    n_steps = 40
    series = [f"bearing_{r:04d}/axlebox_{i}{sd}" for r in (0, 1) for sd in ("L", "R") for i in (1, 2)]
    units = ["T01", "T01", "T01", "T01", "T02", "T02", "T02", "T02"]
    t, ser, unit = fleet_context(n_steps, series, units)
    t_s = Pm.times_seconds(t, ser.size)
    gid = Pm.peer_group_ids(t_s, 1.0, unit=unit, series=ser)
    for g in np.unique(gid):
        sel = gid == g
        assert len(set(unit[sel])) == 1, "a peer group spanned two trains"
        assert len(set(ser[sel])) == int(sel.sum()), "a series appeared twice in one group"
    assert np.unique(gid).size == n_steps * 2  # one group per train per timestamp
    # and the old behaviour (no context) is what it always was: one group per timestamp
    assert np.unique(Pm.peer_group_ids(t_s, 1.0)).size == n_steps


def test_peer_group_ids_do_not_chain_across_cycles() -> None:
    """A tolerance chain used to walk a dense stream into one 10-row group."""
    t_s = np.arange(12, dtype=float) * 0.5  # every row within 1 s of the previous one
    ser = np.array([f"door_L{i % 4 + 1}" for i in range(12)], dtype=object)
    gid = Pm.peer_group_ids(t_s, 1.0, unit=np.full(12, "T01", dtype=object), series=ser)
    sizes = np.bincount(gid)
    assert sizes.max() <= 4  # bounded by the number of distinct components, not by the chain


def test_peer_rows_refuse_a_table_with_no_peers_at_all() -> None:
    """A slice holding ONE series has no peers, so a peer row there is a constant score.

    (This is about a degenerate *slice* - a held-out single component, a single-recording
    table - not about the shipped fleet: see
    ``test_peer_normalisation_applies_to_the_shipped_sim_door_layout`` for what sim/door
    really carries.)
    """
    n = 200
    X = np.random.default_rng(67).normal(0.0, 1.0, size=(n, 4)).astype(np.float32)
    t = fake_times(n)
    ser = np.full(n, "door_0000/door_L1", dtype=object)
    unit = np.full(n, "T01", dtype=object)
    for name in ("peer_normalisation", "peer_normalisation_shared", "peer_delta_temperature"):
        with pytest.raises(ValueError, match="no peer anywhere"):
            build(name).fit(X, t, series=ser, unit=unit)
        m = build(name, require_peers=False).fit(X, t, series=ser, unit=unit)
        assert np.isfinite(m.score(X, t, series=ser, unit=unit)).all()
        assert m.n_rows_with_peers_ == 0
    # no row context at all (direct use / unit tests) keeps the graceful degradation
    assert np.isfinite(build("peer_normalisation").fit(X, t).score(X, t)).all()


def test_series_sides_read_the_component_letter_only() -> None:
    got = Pm.series_sides(
        ["door_0000/door_L1", "door_0000/door_R4", "bearing_0000/axlebox_3R", "pneumatic_0000/apu_1"]
    )
    assert list(got) == ["L", "R", "R", ""]  # the R of "door" / L of "axlebox" are not sides


def test_peer_delta_temperature_keeps_trains_apart() -> None:
    """One hot box on T01 must not be averaged against T02's boxes at the same instant."""
    n_steps = 60
    series = [f"bearing_{r:04d}/axlebox_{i}L" for r in (0, 1) for i in (1, 2, 3, 4)]
    units = ["T01"] * 4 + ["T02"] * 4
    t, ser, unit = fleet_context(n_steps, series, units)
    rng = np.random.default_rng(71)
    step = np.repeat(np.arange(n_steps), 8)
    base = np.where(np.isin(np.arange(ser.size) % 8, [4, 5, 6, 7]), 55.0, 40.0)  # T02 runs hotter
    val = base + rng.normal(0.0, 0.1, ser.size)
    hot = (np.arange(ser.size) % 8 == 2) & (step > 40)
    val = val + np.where(hot, 6.0, 0.0)
    X = np.column_stack([val, rng.normal(0.0, 1.0, val.size)]).astype(np.float32)
    m = build("peer_delta_temperature", time_tol_s=1.0)
    fit = step <= 40
    m.fit(X[fit], t[fit], feature_names=["T_box_mean", "other"], series=ser[fit], unit=unit[fit])
    s = m.score(X, t, series=ser, unit=unit)
    assert s[hot].mean() > 5.0
    # T02 is a different train: neither its own 15 K offset nor T01's hot box reaches it
    assert abs(s[unit == "T02"]).mean() < 2.0
    assert abs(s[step <= 40]).mean() < 2.0
    # without the unit context every timestamp pools both trains, so the 15 K inter-train
    # offset lands in the peer delta and its healthy scatter swamps the real 6 K excursion
    pooled = build("peer_delta_temperature", time_tol_s=1.0)
    pooled.fit(X[fit], t[fit], feature_names=["T_box_mean", "other"])
    assert pooled.scale_ > 10.0 * m.scale_
    assert pooled.score(X, t)[hot].mean() < 2.0


def test_thermal_residual_refuses_a_same_sensor_regression() -> None:
    """Bearing window stats carry no speed/ambient/load/dwell; T_box_* is the same sensor."""
    names = SIM_BEARING_WS_NAMES
    n = 200
    X = np.random.default_rng(77).normal(40.0, 2.0, size=(n, len(names))).astype(np.float32)
    with pytest.raises(ValueError, match="speed / ambient / load / dwell"):
        build("thermal_residual_model").fit(X, fake_times(n), feature_names=names)
    # the escape hatch is explicit, never silent
    m = build("thermal_residual_model", require_physical=False).fit(X, fake_times(n), feature_names=names)
    assert m.covariate_index_


def test_thermal_residual_uses_the_adjacent_and_opposite_box_temperatures() -> None:
    """R110's stated inputs, delivered by the series/unit row context rather than columns."""
    n_steps = 150
    rng = np.random.default_rng(79)
    t, ser, unit = fleet_context(n_steps, BEARING_SERIES)
    common = np.repeat(rng.normal(40.0, 6.0, n_steps), 8)  # speed/ambient common mode
    val = common + rng.normal(0.0, 0.2, ser.size)
    step = np.repeat(np.arange(n_steps), 8)
    hot = (np.array([s.endswith("axlebox_3R") for s in ser])) & (step >= 120)
    val = val + np.where(hot, 7.0, 0.0)
    X = np.column_stack([val, rng.normal(0.0, 1.0, val.size), common]).astype(np.float32)
    names = ["T_box_mean", "vib_rms_mean", "T_box_wayside_mean"]
    m = build("thermal_residual_model")
    fit = step < 120
    m.fit(X[fit], t[fit], feature_names=names, series=ser[fit], unit=unit[fit])
    assert m.peer_columns_ == ["peer_same_side", "peer_opposite_side"]
    assert m.covariate_index_ == []  # no speed/ambient/load column, and never the own sensor
    s = m.score(X, t, series=ser, unit=unit)
    assert s[hot].mean() > 5.0 > abs(s[~hot]).mean()


def test_thermal_residual_named_covariates_exclude_the_targets_own_sensor() -> None:
    n = 400
    rng = np.random.default_rng(83)
    speed = rng.uniform(0.0, 30.0, n)
    t_box = 25.0 + 0.8 * speed + rng.normal(0.0, 0.3, n)
    X = np.column_stack([t_box, t_box * 1.02, speed]).astype(np.float32)
    names = ["T_box_mean", "T_box_max", "speed_mean"]
    m = build("thermal_residual_model").fit(X, fake_times(n), feature_names=names)
    assert m.covariate_names_ == ["speed_mean"]  # T_box_max is the same sensor, so it is out


def test_dwt_w8w9_startpeak_uses_the_named_door_current_channel() -> None:
    """Channel 0 is pos_ref on both shipped door layouts; the current channel is index 3."""
    for names in (SIM_DOOR_RAW_CHANNELS, CRANFIELD_RAW_CHANNELS):
        n, c = 120, len(names)
        rng = np.random.default_rng(89)
        X = rng.normal(0.0, 1.0, size=(n, 256, c)).astype(np.float32)
        X[60:, :40, 3] += 15.0  # a breakaway current peak, on the CURRENT channel only
        t = fake_times(n)
        m = build("dwt_w8w9_startpeak")
        m.fit(X[:50], t[:50], feature_names=names)
        assert m.channel_ == names.index("current") == 3
        s = m.score(X, t)
        assert s[60:].mean() > s[:50].mean() + 1.0
    with pytest.raises(ValueError, match="current"):
        build("dwt_w8w9_startpeak").fit(X, t, feature_names=["a", "b", "c", "d"][:c])


def test_raw_residual_rows_resolve_tp2_tp3_and_motor_current() -> None:
    """They resolved to [0] without channel names, dropping TP3 and Motor_current entirely."""
    n, c = 120, len(SIM_PNEUMATIC_RAW_CHANNELS)
    X = np.random.default_rng(97).normal(0.0, 1.0, size=(n, 32, c)).astype(np.float32)
    t = fake_times(n)
    for name in ("page_hinkley_residual", "adwin_residual"):
        m = build(name)
        m.fit(X, t, feature_names=SIM_PNEUMATIC_RAW_CHANNELS)
        assert m.channels_ == [0, 1, 6]
        assert m.channel_names_ == ["TP2", "TP3", "Motor_current"]
    # MetroPT-3's raw panel names the aggregates; the level column wins over the max column
    metro = ["TP2_mean", "TP2_max", "TP3_mean", "TP3_max", "Motor_current_mean", "Motor_current_max"]
    m = build("page_hinkley_residual")
    m.fit(X[:, :, :6], t, feature_names=metro)
    assert m.channel_names_ == ["TP2_mean", "TP3_mean", "Motor_current_mean"]
    # a panel without them is refused by name
    with pytest.raises(ValueError, match="TP2"):
        build("adwin_residual").fit(X[:, :, :4], t, feature_names=["pos_ref", "pos", "vel", "current"])


def test_k_of_n_votes_on_the_five_named_signals() -> None:
    names = METROPT3_WS_NAMES
    n = 300
    X = np.random.default_rng(103).normal(0.0, 1.0, size=(n, len(names))).astype(np.float32)
    m = build("k_of_n_corroboration")
    m.fit(X, fake_times(n), feature_names=names)
    assert m.member_names_ == [
        "TP2_mean_mean",
        "TP3_mean_mean",
        "Motor_current_mean_mean",
        "Oil_temperature_mean_mean",
        "COMP_duty_mean",  # documented stand-in: idle_run_ratio is a per-CYCLE quantity
    ]
    # on the cycle table the fifth member is the real thing
    cyc = SIM_PNEUMATIC_CYCLE_NAMES
    Xc = np.random.default_rng(107).normal(1.0, 1.0, size=(n, len(cyc))).astype(np.float32)
    m2 = build("k_of_n_corroboration", features=["idle_run_ratio", "duty_ratio", "T_oil_max"], k=2)
    m2.fit(Xc, fake_times(n), feature_names=cyc)
    assert m2.members_ == [15, 16, 7]


def test_k_of_n_refuses_a_member_the_table_does_not_have() -> None:
    n, names = 120, ["a_mean", "b_mean", "c_mean"]
    X = np.random.default_rng(109).normal(0.0, 1.0, size=(n, 3)).astype(np.float32)
    with pytest.raises(ValueError, match="TP2"):
        build("k_of_n_corroboration").fit(X, fake_times(n), feature_names=names)
    m = build("k_of_n_corroboration", features=["a", "TP2"], k=1, on_missing="drop")
    m.fit(X, fake_times(n), feature_names=names)
    assert m.member_names_ == ["a_mean"]


def test_kurtogram_default_features_are_the_ladder_row() -> None:
    m = build("kurtogram_envelope_bpfo")
    assert m.features == [
        "env_bpfo_energy",
        "env_bpfi_energy",
        "env_bsf_energy",
        "env_ftf_energy",
        "sk_band_fc",
        "sk_band_bw",
        "sk_max",
    ]
    assert build("kurtogram_envelope_bpfo", harmonics=True).features[:3] == [
        "env_bpfo_h1",
        "env_bpfo_h2",
        "env_bpfo_h3",
    ]


def test_kurtogram_picks_the_vibration_channel_by_name() -> None:
    n, length = 12, 1024
    X = np.random.default_rng(113).normal(0.0, 1.0, size=(n, length, 3)).astype(np.float32)
    m = build("kurtogram_envelope_bpfo", fs=20_000.0, max_level=2, max_fit_windows=8)
    m.fit(X, fake_times(n), feature_names=["T_box", "T_box_wayside", "vib_rms"])
    assert m.channel_ == 2


# ======================================================================================
# drift: monotone evidence
# ======================================================================================


@pytest.mark.parametrize(
    "name",
    [
        "page_hinkley_cycle_scalar",
        "adwin_kswin_residual",
        "page_hinkley_residual",
        "adwin_residual",
        "page_hinkley_thermal_residual",
    ],
)
def test_drift_rows_never_fall_back_after_a_permanent_step(name: str) -> None:
    """The verifier saw 100-125 negative post-step increments on every one of these rows."""
    n, cut = 600, 400
    rng = np.random.default_rng(127)
    kind = get_spec(name).input_kind
    kwargs: dict[str, Any] = {}
    if kind == "raw_window":
        X = rng.normal(0.0, 1.0, size=(n, 32, 3)).astype(np.float32)
        X[cut:, :, 0] += 6.0
        kwargs["channels"] = [0]
    elif name == "page_hinkley_thermal_residual":
        speed = rng.uniform(0.0, 30.0, n)
        t_box = 25.0 + 0.8 * speed + rng.normal(0.0, 0.3, n)
        t_box[cut:] += 8.0
        X = np.column_stack([t_box, speed]).astype(np.float32)
        kwargs["feature_names"] = ["T_box_mean", "speed_mean"]
    else:
        X = rng.normal(0.0, 1.0, size=(n, 4)).astype(np.float32)
        X[cut:, 0] += 6.0
        kwargs["feature_index"] = 0
    t = fake_times(n)
    s = build(name, **kwargs).fit(X[:300], t[:300]).score(X, t)
    assert (np.diff(s) >= -1e-12).all(), f"{name}: {int((np.diff(s) < -1e-12).sum())} score drops"
    assert s[cut:].mean() > s[:cut].mean()
    assert s[-1] == pytest.approx(s.max())


def test_every_drift_row_is_scored_one_series_at_a_time() -> None:
    for name in ("page_hinkley_cycle_scalar", "adwin_kswin_residual", "page_hinkley_residual",
                 "adwin_residual", "page_hinkley_thermal_residual"):
        assert type(build(name)).SEQUENTIAL is True, name


def test_the_three_statistics_are_monotone_unless_asked_otherwise() -> None:
    rng = np.random.default_rng(131)
    x = np.r_[rng.normal(0.0, 1.0, 300), rng.normal(5.0, 1.0, 300)]
    for fn, kw in (
        (Dm.page_hinkley_statistic, {"min_instances": 30}),
        (Dm.adwin_statistic, {"center": 0.0, "scale": 1.0}),
        (Dm.kswin_statistic, {"seed": 0}),
    ):
        mono, _ = fn(x, **kw)
        assert (np.diff(mono) >= -1e-12).all(), fn.__name__
        assert mono[-1] > mono[0]
        raw, _ = fn(x, monotone=False, **kw)
        assert (np.diff(raw) < -1e-12).any(), f"{fn.__name__}: monotone=False should saw-tooth"
        np.testing.assert_allclose(mono, np.maximum.accumulate(raw))


def test_drift_rows_resolve_a_named_cycle_feature_from_the_runner() -> None:
    names = SIM_PNEUMATIC_CYCLE_NAMES
    n = 300
    X = np.random.default_rng(137).normal(1.0, 1.0, size=(n, len(names))).astype(np.float32)
    m = build("page_hinkley_cycle_scalar", feature="idle_run_ratio")
    m.fit(X, fake_times(n), feature_names=names)
    assert m.feature_index_ == 15 and m.feature_name_ == "idle_run_ratio"


# ======================================================================================
# second-round verifier fixes
# ======================================================================================


def test_kurtogram_uses_the_delivered_sampling_rate_not_a_hard_coded_one() -> None:
    """Ottawa raw windows arrive decimated (fs_hz_out = 10,500 Hz), not at the sensor's 42 kHz.

    A hard-coded rate put every BPFO/BPFI/BSF/FTF line 4x off, so the model takes the rate
    from the runner's ``fs_hz`` row context and records what it used.
    """
    n, length = 12, 2048
    rng = np.random.default_rng(101)
    X = rng.normal(0.0, 1.0, size=(n, length, 1)).astype(np.float32)
    t = fake_times(n)

    # no context anywhere: the documented fallback is Ottawa's native rate
    bare = build("kurtogram_envelope_bpfo").fit(X, t)
    assert bare.fs_hz_ == Pm.KurtogramEnvelopeBpfo.DEFAULT_FS_HZ == 42_000.0

    # exactly how runner._context calls it for Ottawa raw at decimate=4
    m = build("kurtogram_envelope_bpfo")
    m.fit(X, t, feature_names=["vib_acc"], fs_hz=10_500.0)
    assert m.fs_hz_ == 10_500.0
    assert np.isfinite(m.score(X, t, fs_hz=10_500.0)).all()
    # the spectra really are different: the two rates disagree on the feature table
    assert not np.allclose(m._feature_table(X), bare._feature_table(X))
    # an explicit constructor kwarg is still an override that wins over the context
    pinned = build("kurtogram_envelope_bpfo", fs=20_000.0).fit(X, t, fs_hz=10_500.0)
    assert pinned.fs_hz_ == 20_000.0
    # and scoring a slice delivered at a different rate is refused, never silently rescaled
    with pytest.raises(ValueError, match="10500"):
        m.score(X, t, fs_hz=42_000.0)


def test_kurtogram_declares_fs_hz_so_the_runner_hands_it_over() -> None:
    """The row-context channel only reaches methods that declare the keyword."""
    from nebulax.bench.base import CONTEXT_KEYS, accepts_keyword

    m = build("kurtogram_envelope_bpfo")
    assert "fs_hz" in CONTEXT_KEYS
    assert accepts_keyword(m._fit, "fs_hz") and accepts_keyword(m._score, "fs_hz")


def test_thermal_residual_takes_the_named_adjacent_and_opposite_boxes() -> None:
    """R110's "adjacent and opposite axlebox temperatures" are two boxes, not two side means."""
    n_steps = 4
    t, ser, unit = fleet_context(n_steps, BEARING_SERIES)
    # a distinct temperature per box so each peer column identifies exactly one box
    per_box = {f"axlebox_{i}{sd}": 10.0 * i + (100.0 if sd == "R" else 0.0) for sd in "LR" for i in (1, 2, 3, 4)}
    y = np.array([per_box[str(c).rsplit("/", 1)[-1]] for c in ser], dtype=float)
    m = build("thermal_residual_model")
    peers, had = m._peer_temperatures(y, t, ser, unit)
    comp = Pm.component_id(ser)
    got = {str(c): (a, o) for c, a, o in zip(comp, peers[:, 0], peers[:, 1])}
    assert had.all()
    assert got["axlebox_1L"] == (20.0, 110.0)  # bogie partner 2L, opposite 1R
    assert got["axlebox_4L"] == (30.0, 140.0)  # bogie partner 3L, opposite 4R
    assert got["axlebox_3R"] == (140.0, 30.0)  # bogie partner 4R, opposite 3L
    # a whole-side average would give 1L the mean of {2L, 3L, 4L} = 30.0 and the mean of the
    # whole R side = 125.0 - neither is what the ladder asks for
    assert got["axlebox_1L"] != (30.0, 125.0)


def test_thermal_residual_peer_columns_never_read_the_boxs_own_sensor() -> None:
    """No position/side in the series id means no adjacent or opposite box - not "itself"."""
    n = 40
    y = np.linspace(30.0, 50.0, n)
    t = fake_times(n)
    ser = np.full(n, "pneumatic_0000/apu_1", dtype=object)
    unit = np.full(n, "T01", dtype=object)
    peers, had = build("thermal_residual_model")._peer_temperatures(y, t, ser, unit)
    assert not had.any() and np.isnan(peers).all()
    # ... so the row refuses rather than regressing T_box on T_box
    X = np.column_stack([y, y * 1.01]).astype(np.float32)
    with pytest.raises(ValueError, match="speed / ambient / load / dwell"):
        build("thermal_residual_model").fit(X, t, feature_names=["T_box_mean", "T_box_max"], series=ser, unit=unit)


def test_peer_normalisation_shared_uses_the_median_of_same_side_peers() -> None:
    """The ladder row says median of same-side peers; a leave-one-out mean is not that."""
    assert Pm.PeerNormalisationShared._DEFAULT_STATISTIC == "median"
    assert build("peer_normalisation_shared").statistic == "median"
    assert build("peer_normalisation").statistic == "mean"  # the door row is unchanged

    # four same-side boxes, one of them running away: the median reference is untouched
    V = np.array([[10.0], [10.0], [10.0], [30.0]])
    gid = np.zeros(4, dtype=np.int64)
    assert Pm.peer_delta(V, gid, statistic="median").ravel().tolist() == [0.0, 0.0, 0.0, 20.0]
    mean = Pm.peer_delta(V, gid, statistic="mean").ravel()
    assert np.allclose(mean, [-20 / 3, -20 / 3, -20 / 3, 20.0])  # the healthy boxes are smeared
    with pytest.raises(ValueError, match="statistic"):
        Pm.peer_delta(V, gid, statistic="trimmed_mean")


def test_peer_normalisation_shared_median_resists_a_second_degrading_box() -> None:
    """Two hot boxes a side corrupt a mean reference; the median of the other three does not."""
    n_steps = 80
    t, ser, unit = fleet_context(n_steps, BEARING_SERIES)
    rng = np.random.default_rng(107)
    step = np.repeat(np.arange(n_steps), len(BEARING_SERIES))
    comp = np.array([str(c).rsplit("/", 1)[-1] for c in ser], dtype=object)
    val = 40.0 + rng.normal(0.0, 0.05, ser.size)
    late = step >= 60
    val += np.where(late & (comp == "axlebox_1L"), 8.0, 0.0)  # the box we want flagged
    val += np.where(late & (comp == "axlebox_2L"), 8.0, 0.0)  # a second one, same side
    X = np.column_stack([val]).astype(np.float32)
    fit = step < 60
    scores = {}
    for stat in ("mean", "median"):
        m = build("peer_normalisation_shared", statistic=stat, standardise=False)
        m.fit(X[fit], t[fit], series=ser[fit], unit=unit[fit])
        scores[stat] = m.score(X, t, series=ser, unit=unit)
    hot = late & (comp == "axlebox_1L")
    assert scores["median"][hot].mean() > scores["mean"][hot].mean() + 1.0


def test_peer_normalisation_applies_to_the_shipped_sim_door_layout() -> None:
    """sim/door really carries TWO door series per train - it is not a peerless table.

    Measured on the uncached fleet (16 Sep 2026): 20 series over 10 units, e.g. T01 =
    door_0000/door_L1 + door_0010/door_L3. The old docstring/diagnostic claimed one door per
    train and that this row must raise here, which is false.
    """
    n_steps, per = 60, 2
    series = ["door_0000/door_L1", "door_0010/door_L3"]
    t, ser, unit = fleet_context(n_steps, series)  # both doors of T01 share each timestamp
    rng = np.random.default_rng(109)
    step = np.repeat(np.arange(n_steps), per)
    common = np.repeat(rng.normal(3.0, 0.4, n_steps), per)  # a fleet-wide common mode
    val = common + rng.normal(0.0, 0.02, ser.size)
    hot = (ser == series[0]) & (step >= 40)
    X = np.column_stack([val + np.where(hot, 1.5, 0.0)]).astype(np.float32)
    fit = step < 40
    m = build("peer_normalisation")  # require_peers=True: it must NOT raise here
    m.fit(X[fit], t[fit], series=ser[fit], unit=unit[fit])
    assert m.n_rows_with_peers_ == int(fit.sum())
    s = m.score(X, t, series=ser, unit=unit)
    assert s[hot].mean() > s[~hot].mean() + 3.0
    # the retracted claim is gone from the documentation as well
    assert "exactly one door series" not in (Pm.PeerNormalisation.__doc__ or "")
    assert "two doors" in (Pm.PeerNormalisation.__doc__ or "").lower()


def test_page_hinkley_statistic_has_no_silent_zero_fallback() -> None:
    """The statistic comes from our own running sums, and tracks river's own detections.

    The old code caught AttributeError on river's private attributes and substituted
    inc = dec = 0, i.e. a flat column, if river's internals ever moved.
    """
    import inspect

    src = inspect.getsource(Dm.page_hinkley_statistic)
    assert "except AttributeError" not in src  # no silent handler over river's internals
    assert "det._" not in src  # and no read of a private river attribute at all

    rng = np.random.default_rng(113)
    x = np.r_[rng.normal(0.0, 1.0, 400), rng.normal(6.0, 1.0, 400)]
    stat, flags = Dm.page_hinkley_statistic(x, accumulate=False, monotone=False)
    assert flags.any() and stat.max() > 0.0
    # river declares a drift exactly when our statistic crosses its threshold
    assert (stat[flags] > 50.0).all()
    assert not (stat[~flags] > 50.0).any()
    # a one-sided mode uses the matching one-sided quantity, and a flat stream stays flat
    up, _ = Dm.page_hinkley_statistic(x, mode="up", accumulate=False, monotone=False)
    down, _ = Dm.page_hinkley_statistic(x, mode="down", accumulate=False, monotone=False)
    assert up.max() > down.max()  # the step is upward
    assert np.allclose(Dm.page_hinkley_statistic(np.zeros(200))[0], 0.0)


@pytest.fixture(scope="module", autouse=True)
def _drop_the_test_only_inner():
    """The dummy inner is registered at import time (the wrappers build it by name); make
    sure it never leaks into another module's view of the registry."""
    yield
    unregister("_dummy_inner_for_tests")


def test_thermal_residual_refuses_explicit_same_sensor_covariates():
    import pytest
    from nebulax.models.physics import ThermalResidualModel
    names = ["T_box_mean", "T_box_max", "speed_mean", "T_amb_mean"]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4)).astype(np.float32)
    with pytest.raises(ValueError, match="own sensor"):
        ThermalResidualModel(target="T_box_mean", covariates=["T_box_max"], peer_covariates=False).fit(X, feature_names=names)
    m = ThermalResidualModel(target="T_box_mean", covariates=["speed_mean", "T_amb_mean"], peer_covariates=False).fit(X, feature_names=names)
    assert m.covariate_names_ == ["speed_mean", "T_amb_mean"]


def test_transition_mask_is_sequential():
    from nebulax.models.physics import TransitionMask
    assert TransitionMask.SEQUENTIAL is True


def test_page_hinkley_statistic_withholds_evidence_before_min_instances():
    from nebulax.models.drift import page_hinkley_statistic
    x = np.concatenate([np.zeros(5), np.full(60, 10.0)])
    stat, flags = page_hinkley_statistic(x, accumulate=False, monotone=False, min_instances=30, threshold=50.0, delta=0.005)
    assert (stat[:29] == 0).all()  # count reaches min_instances at index 29
    assert not flags[:29].any()
    crossed = np.flatnonzero(stat > 50.0)
    flagged = np.flatnonzero(flags)
    assert crossed.size and flagged.size and crossed[0] == flagged[0]
