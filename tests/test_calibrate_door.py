"""Tests for ``scripts/calibrate_door.py``.

The expensive part of the calibration is the simulator sweep, so almost everything here is
driven by **analytic** telemetry with known answers: a trapezoid profile whose current is
computed from the very force balance the estimator inverts, a square pulse of known width, a
sinusoid of known wavelength.  That is what makes these tests both fast (< 60 s) and
meaningful - a calibration you cannot check against a known answer is not a calibration.

One test does run the simulator, because the contract that ``stroke_table`` measures the
simulator and the adapter with the *same* estimator is the whole basis of the ratio table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.calibrate_door as C  # noqa: E402
from nebulax import schema as S  # noqa: E402
from nebulax.sim import door as D  # noqa: E402

T0 = pd.Timestamp("2026-09-01T00:00:00Z")


# --------------------------------------------------------------------------------------
# analytic fixtures
# --------------------------------------------------------------------------------------


def trapezoid(stroke: float, v: float, a: float, fs: float, hold_s: float = 0.5):
    """``(t, x)`` for one trapezoidal stroke followed by a stationary hold."""
    t_acc = v / a
    d_acc = 0.5 * a * t_acc**2
    t_cruise = max((stroke - 2 * d_acc) / v, 0.0)
    total = 2 * t_acc + t_cruise
    t = np.arange(0.0, total + hold_s, 1.0 / fs)
    x = np.where(
        t < t_acc,
        0.5 * a * t**2,
        np.where(
            t < t_acc + t_cruise,
            d_acc + v * (t - t_acc),
            np.where(
                t < total,
                stroke - 0.5 * a * np.maximum(total - t, 0.0) ** 2,
                stroke,
            ),
        ),
    )
    return t, x


def analytic_wide(
    *,
    m_eff: float = 8.0,
    f_c: float = 15.0,
    b: float = 30.0,
    c_i: float = 47.0,
    r_ohm: float = 2.0,
    k_e: float = 0.05,
    gear_gain: float = 1256.6,
    v: float = 0.05,
    a: float = 0.2,
    stroke: float = 0.10,
    fs: float = 200.0,
    n_strokes: int = 4,
    run_id: str = "an",
    with_voltage: bool = True,
) -> pd.DataFrame:
    """A wide frame whose current obeys ``c_i i = m_eff a + F_c sgn(v) + b v`` exactly."""
    ts, xs, cs, vs, rs = [], [], [], [], []
    t_off = 0.0
    for k in range(n_strokes):
        t, x = trapezoid(stroke, v, a, fs)
        if k % 2:  # alternate direction
            x = stroke - x
        vel = np.gradient(x, 1.0 / fs)
        acc = np.gradient(vel, 1.0 / fs)
        cur = (m_eff * acc + f_c * np.sign(vel) + b * vel) / c_i
        ts.append(t + t_off)
        xs.append(x)
        rs.append(x)
        cs.append(cur)
        vs.append(r_ohm * cur + k_e * vel * gear_gain if with_voltage else np.full(x.shape, np.nan))
        t_off += float(t[-1]) + 2.0  # a real gap between activities
    t = np.concatenate(ts)
    frame = pd.DataFrame(
        {
            "timestamp": T0 + pd.to_timedelta(t, unit="s"),
            "source": "sim",
            "run_id": run_id,
            "train_id": "T01",
            "car": np.int8(0),
            "component_id": "door_L1",
            "pos_ref": np.concatenate(rs),
            "pos": np.concatenate(xs),
            "current": np.concatenate(cs),
        }
    )
    if with_voltage:
        frame["voltage"] = np.concatenate(vs)
    return frame


# --------------------------------------------------------------------------------------
# stroke segmentation
# --------------------------------------------------------------------------------------


def test_segment_strokes_splits_a_triangle_and_drops_the_idle_hold():
    x = np.concatenate([np.linspace(0, 1, 60), np.ones(30), np.linspace(1, 0, 60), np.zeros(30)])
    segs = C.segment_strokes(x)
    assert len(segs) == 2
    (a0, b0), (a1, b1) = segs
    # the flat holds are excluded, not welded onto the neighbouring stroke
    assert b0 - a0 == pytest.approx(60, abs=2)
    assert b1 - a1 == pytest.approx(60, abs=2)
    assert a1 >= 90


def test_segment_strokes_survives_an_activity_boundary_jump():
    """A single huge discontinuity must not set the dead band for the whole file.

    Rig files concatenate activities, so the reference jumps between them.  Scaling the dead
    band off `max|dx|` made every genuine sample idle and found zero strokes.
    """
    ramp = np.linspace(0, 1, 60)
    x = np.concatenate([ramp, np.zeros(5), ramp, np.zeros(5)])  # two ramps, two hard resets
    segs = C.segment_strokes(x)
    assert len(segs) >= 2
    assert all(b - a >= C.MIN_STROKE_SAMPLES for a, b in segs)


def test_segment_strokes_rejects_traces_that_are_too_short():
    assert C.segment_strokes(np.linspace(0, 1, 3)) == []


# --------------------------------------------------------------------------------------
# per-stroke metrics
# --------------------------------------------------------------------------------------


def test_stroke_table_recovers_the_geometry_of_a_known_stroke():
    wide = analytic_wide(n_strokes=4)
    st = C.stroke_table(wide)
    assert len(st) == 4
    assert float(st["travel_m"].median()) == pytest.approx(0.10, rel=0.03)
    assert float(st["v_cruise"].median()) == pytest.approx(0.05, rel=0.05)
    assert float(st["fs_hz"].median()) == pytest.approx(200.0, rel=0.01)
    # cruise current must be the cruise force balance, not the whole-stroke mean
    assert float(st["i_cruise_mean"].median()) == pytest.approx((15.0 + 30.0 * 0.05) / 47.0, rel=0.05)


def test_stroke_table_needs_position_and_current():
    with pytest.raises(ValueError, match="missing"):
        C.stroke_table(pd.DataFrame({"timestamp": [T0], "pos": [0.0]}))


def test_reversal_spike_width_measures_a_known_pulse():
    dt = 0.01
    err = np.zeros(200)
    err[10:30] = 1.0  # 20 samples = 0.20 s at half height
    width, peak = C._reversal_spike_width(err, dt)
    assert peak == pytest.approx(1.0)
    assert width == pytest.approx(0.20, abs=dt)


def test_ripple_recovers_a_known_wavelength():
    pos = np.linspace(0.0, 0.10, 600)
    lam = 0.005
    cur = 1.0 + 0.2 * np.sin(2 * np.pi * pos / lam)
    lam_hat, amp, frac = C._ripple(pos, cur)
    assert lam_hat == pytest.approx(lam, rel=0.10)
    assert frac == pytest.approx(0.2 / np.sqrt(2), rel=0.25)


# --------------------------------------------------------------------------------------
# the healthy least-squares fit and its identifiability flags
# --------------------------------------------------------------------------------------


def test_fit_healthy_recovers_a_noiseless_plant():
    p = D.DoorParams.cranfield()
    c_i = p.force_per_amp
    wide = pd.concat(
        [
            analytic_wide(c_i=c_i, gear_gain=p.gear_gain, v=0.05, a=0.2, run_id="slow"),
            analytic_wide(c_i=c_i, gear_gain=p.gear_gain, v=0.10, a=0.4, run_id="fast"),
        ],
        ignore_index=True,
    )
    f = C.fit_healthy(wide, p)
    assert f.r2_mech > 0.99
    assert f.m_eff == pytest.approx(8.0, rel=0.15)
    assert f.f_c0 == pytest.approx(15.0, rel=0.15)
    assert f.b0 == pytest.approx(30.0, rel=0.30)
    assert f.stroke_m == pytest.approx(0.10, rel=0.05)
    # the electrical pair closes the system when a voltage channel exists
    assert f.k_t_identified
    assert f.r_ohm == pytest.approx(2.0, rel=0.10)
    assert f.k_t == pytest.approx(0.05, rel=0.10)


def test_fit_healthy_flags_kt_unidentifiable_without_voltage():
    p = D.DoorParams.cranfield()
    wide = analytic_wide(c_i=p.force_per_amp, gear_gain=p.gear_gain, with_voltage=False)
    f = C.fit_healthy(wide, p)
    assert not f.k_t_identified
    assert f.k_t == p.k_t  # held at the assumption, not invented
    assert np.isnan(f.r_ohm)
    assert any("assumed" in str(r["status"]) for r in f.as_rows())


def test_fit_healthy_flags_b0_unidentifiable_at_one_cruise_speed():
    """One motion profile cannot separate ``F_c0`` from ``b0`` - only ``F_c0 + b0 v_ref``."""
    p = D.DoorParams.cranfield()
    wide = analytic_wide(c_i=p.force_per_amp, gear_gain=p.gear_gain, v=0.05, a=0.2)
    f = C.fit_healthy(wide, p)
    assert f.v_span < C.B0_MIN_SPEED_SPAN
    assert not f.b0_identified
    # the combination that *is* well posed still comes out right
    assert f.f_cruise == pytest.approx(15.0 + 30.0 * 0.05, rel=0.10)
    assert any("NOT identifiable" in str(r["status"]) for r in f.as_rows())


def test_fit_healthy_rejects_empty_telemetry():
    with pytest.raises(ValueError):
        C.fit_healthy(pd.DataFrame({"timestamp": [T0] * 3, "pos": [0.0] * 3, "current": [0.0] * 3}), D.DoorParams.cranfield())


# --------------------------------------------------------------------------------------
# ratios, sweeps and inversion
# --------------------------------------------------------------------------------------


def test_class_summary_forms_ratios_against_healthy():
    st = pd.DataFrame(
        {
            "fault_type": ["healthy", "healthy", "friction", "friction"],
            "level": [0, 0, 1, 1],
            "i_cruise_mean": [1.0, 1.0, 2.0, 2.0],
            "i_cruise_rms": [1.0, 1.0, 2.0, 2.0],
            "v_cruise": [0.05, 0.05, 0.04, 0.04],
            "duration_s": [2.0, 2.0, 3.0, 3.0],
            "err_spike_width_s": [0.1, 0.1, 0.2, 0.2],
            "err_spike_peak_m": [0.001, 0.001, 0.003, 0.003],
            "ripple_frac": [0.02, 0.02, 0.10, 0.10],
        }
    )
    out = C.class_summary(st)
    row = out[out["fault_type"] == "friction"].iloc[0]
    assert row["i_cruise_mean_ratio"] == pytest.approx(2.0)
    assert row["v_cruise_ratio"] == pytest.approx(0.8)
    assert row["ripple_frac_ratio"] == pytest.approx(5.0)
    assert int(row["n"]) == 2


def test_class_summary_without_a_healthy_class_is_an_error():
    st = pd.DataFrame({"fault_type": ["friction"], "level": [1], **{m: [1.0] for m in C.RATIO_METRICS}})
    with pytest.raises(ValueError, match="healthy"):
        C.class_summary(st)


def _surrogate(gain_key: str, law):
    """A cheap stand-in for a simulator run: ``metric = law(gain, severity)``."""

    def measure(p: D.DoorParams, s: float) -> pd.Series:
        g = float(getattr(p, gain_key))
        return pd.Series({m: law(g, s) for m in C.RATIO_METRICS})

    return measure


def test_sweep_gain_and_invert_curve_round_trip_without_a_simulator():
    law = lambda g, s: 1.0 + g * s  # noqa: E731
    curve = C.sweep_gain(
        D.DoorParams.cranfield(),
        "k_friction_c",
        (0.5, 1.0, 2.0, 4.0),
        "friction",
        (0.5, 1.0),
        measure=_surrogate("k_friction_c", law),
    )
    assert set(curve["gain"]) == {0.5, 1.0, 2.0, 4.0}
    # ratio at severity 1 for gain g is (1 + g)/1
    target = 1.0 + 2.0
    assert C.invert_curve(curve, "i_cruise_mean", 1.0, target) == pytest.approx(2.0, rel=0.05)


def test_invert_curve_returns_nan_outside_the_swept_range():
    law = lambda g, s: 1.0 + g * s  # noqa: E731
    curve = C.sweep_gain(
        D.DoorParams.cranfield(), "k_friction_c", (0.5, 1.0), "friction", (1.0,), measure=_surrogate("k_friction_c", law)
    )
    assert np.isnan(C.invert_curve(curve, "i_cruise_mean", 1.0, 99.0))
    assert np.isnan(C.invert_curve(curve, "i_cruise_mean", 1.0, float("nan")))


def test_monotone_prefix_truncates_at_the_turning_point():
    g = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r = np.array([1.0, 2.0, 3.0, 2.5, 4.0])  # rises, turns over at index 3
    mask = C.monotone_prefix(g, r)
    assert mask.tolist() == [True, True, True, False, False]


def test_invert_curve_refuses_a_root_beyond_the_monotone_range():
    """A non-monotone map must not hand back a plausible-looking wrong root."""
    curve = pd.DataFrame(
        {
            "gain_name": "backlash_m",
            "gain": [0.001, 0.002, 0.004, 0.008, 0.016],
            "severity": 1.0,
            "err_spike_peak_m_ratio": [1.0, 1.5, 2.2, 3.4, 2.1],  # turns over at 16 mm
        }
    )
    # 3.0 is inside the monotone part -> interpolated
    assert C.invert_curve(curve, "err_spike_peak_m", 1.0, 3.0) == pytest.approx(0.008, rel=0.35)
    # 2.1 also occurs on the falling branch; only the monotone prefix is used
    assert C.invert_curve(curve, "err_spike_peak_m", 1.0, 3.6) != C.invert_curve(curve, "err_spike_peak_m", 1.0, 3.0)
    assert np.isnan(C.invert_curve(curve, "err_spike_peak_m", 1.0, 5.0))


def test_build_ratio_rows_marks_exactly_one_primary_observable_per_fault():
    rows = C.build_ratio_rows(D.DoorParams.cranfield(), pd.DataFrame(), {})
    assert rows, "the table must exist even with no data"
    primaries = {r["gain"] for r in rows if r["primary"]}
    assert primaries == {"k_friction_c", "backlash_m", "k_mis"}
    # with no measurement every Cranfield cell is PENDING, never a number
    assert all(not np.isfinite(float(r["cranfield_ratio"])) for r in rows)
    assert all(np.isfinite(float(r["gain_incumbent"])) for r in rows)


def test_level_severity_map_keeps_the_plans_anchors_for_all_three_classes():
    """The physics target is per class, and it keeps the plan's anchors for all of them.

    The plan pins the two-stage faults at ``s = 0.5`` and ``s = 1.0``.  Spalling has 8 stages
    in the real release, which the plan never saw; placing them at ``level/8`` is the placement
    that keeps those same anchors, at stages 4 and 8.
    """
    assert C.CRANFIELD_LEVEL_SEVERITY["lack_of_lubrication"] == {1: 0.5, 2: 1.0}
    assert C.CRANFIELD_LEVEL_SEVERITY["backlash"] == {1: 0.5, 2: 1.0}
    spall = C.CRANFIELD_LEVEL_SEVERITY["spalling"]
    assert len(spall) == C.N_SPALLING_STAGES == 8
    assert spall[4] == pytest.approx(0.5) and spall[8] == pytest.approx(1.0)
    assert sorted(spall) == list(range(1, 9))
    assert C.sim_severities("backlash") == (0.5, 1.0)
    assert len(C.sim_severities("spalling")) == 8
    # and it is deliberately NOT the adapter's ordinal ML label, which saturates at stage 5
    from nebulax.adapters import cranfield as A

    assert A.severity_from_level(8, "misalignment") == pytest.approx(1.0)
    assert A.severity_from_level(5, "misalignment") == pytest.approx(1.0)
    assert spall[5] != spall[8]


def test_build_ratio_rows_reports_every_spalling_stage():
    rows = C.build_ratio_rows(D.DoorParams.cranfield(), pd.DataFrame(), {})
    spall = [r for r in rows if r["cranfield_class"] == "spalling"]
    assert [r["level"] for r in spall] == list(range(1, 9))
    assert [float(r["severity"]) for r in spall] == pytest.approx([k / 8 for k in range(1, 9)])


def test_gains_from_curves_medians_only_the_levels_the_curve_could_invert():
    rows = [
        {"primary": True, "gain": "k_mis", "metric": "ripple_frac", "level": 1,
         "cranfield_ratio": 1.04, "gain_from_curve": float("nan"), "gain_incumbent": 3.0},
        {"primary": True, "gain": "k_mis", "metric": "ripple_frac", "level": 4,
         "cranfield_ratio": 1.34, "gain_from_curve": 0.20, "gain_incumbent": 3.0},
        {"primary": True, "gain": "k_mis", "metric": "ripple_frac", "level": 8,
         "cranfield_ratio": 1.45, "gain_from_curve": 0.10, "gain_incumbent": 3.0},
        {"primary": False, "gain": "k_mis", "metric": "x", "level": 8,
         "cranfield_ratio": 9.0, "gain_from_curve": 99.0, "gain_incumbent": 3.0},
    ]
    out = C.gains_from_curves(rows)["k_mis"]
    assert out["n_levels"] == 3 and out["n_invertible"] == 2  # the diagnostic row is not counted
    assert out["levels"] == [4, 8]
    assert out["value"] == pytest.approx(0.15)
    assert out["incumbent"] == pytest.approx(3.0)
    assert out["sign_ok"] and out["min_ratio"] == pytest.approx(1.04)


def test_sign_gate_fails_when_one_stage_moves_the_observable_the_wrong_way():
    """Each map only *raises* its observable, so a stage measuring below 1 means the observable
    does not track the fault at all - inverting the other stage would be reading noise."""
    rows = [
        {"primary": True, "gain": "backlash_m", "metric": "err_spike_peak_m", "level": 1,
         "cranfield_ratio": 0.977, "gain_from_curve": float("nan"), "gain_incumbent": 0.008},
        {"primary": True, "gain": "backlash_m", "metric": "err_spike_peak_m", "level": 2,
         "cranfield_ratio": 1.029, "gain_from_curve": 0.0008, "gain_incumbent": 0.008},
    ]
    out = C.gains_from_curves(rows)["backlash_m"]
    assert not out["sign_ok"]
    assert out["min_ratio"] == pytest.approx(0.977)
    # the inversion still happened; the gate is what stops it being adopted
    assert out["value"] == pytest.approx(0.0008)
    report = C.CalibrationReport(
        have_data=True, raw_dir=Path("x"), n_strokes=1, fit=None, measured=pd.DataFrame()
    )
    report.ratio_rows = rows
    report.gains = C.gains_from_curves(rows)
    row = next(r for r in C._gain_rows(report, D.DoorParams.cranfield()) if r["constant"] == "backlash_m")
    assert "unchanged" in str(row["new"])
    assert "wrong way" in str(row["evidence"])


def test_gains_from_curves_reports_nan_when_nothing_inverted():
    rows = [
        {"primary": True, "gain": "backlash_m", "metric": "err_spike_peak_m", "level": lvl,
         "cranfield_ratio": 1.2, "gain_from_curve": float("nan"), "gain_incumbent": 0.008}
        for lvl in (1, 2)
    ]
    out = C.gains_from_curves(rows)["backlash_m"]
    assert out["n_invertible"] == 0 and np.isnan(float(out["value"]))


def test_adoption_gate_blocks_exactly_the_mean_current_gains():
    """A mean-current ratio is compressed by the stepper's standing current; a ripple ratio is
    not, because the ripple is measured after detrending against position."""
    assert set(C.ADOPTION_BLOCKED) == {"k_friction_c", "k_friction_b"}
    assert "k_mis" not in C.ADOPTION_BLOCKED
    assert "backlash_m" not in C.ADOPTION_BLOCKED  # blocked by invertibility, not by the drive


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------


def test_write_doc_section_is_idempotent(tmp_path: Path):
    doc = tmp_path / "parameters.md"
    doc.write_text("# Title\n\nexisting content\n")
    C.write_doc_section(doc, "## Door\n\nfirst\n")
    once = doc.read_text()
    assert "existing content" in once and "first" in once
    C.write_doc_section(doc, "## Door\n\nsecond\n")
    twice = doc.read_text()
    assert twice.count(C.DOC_BEGIN) == 1
    assert "second" in twice and "first" not in twice
    assert "existing content" in twice


def test_write_doc_section_creates_a_missing_file(tmp_path: Path):
    doc = tmp_path / "sub" / "parameters.md"
    C.write_doc_section(doc, "body")
    assert C.DOC_BEGIN in doc.read_text()


def test_render_report_says_pending_when_there_is_no_data():
    params = D.DoorParams.cranfield()
    report = C.CalibrationReport(
        have_data=False, raw_dir=Path("x"), n_strokes=0, fit=None, measured=pd.DataFrame()
    )
    report.ratio_rows = C.build_ratio_rows(params, pd.DataFrame(), {})
    report.recommended = C.recommended_constants(report, params)
    text = C.render_report(report, params)
    assert "PENDING" in text
    assert "obs_base_seed" in text
    assert "No door constant below is calibrated against real actuator data" in text


# --------------------------------------------------------------------------------------
# the one simulator-backed test: both sides of every ratio use one estimator
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("fault_type,severity", [("friction", 1.0)])
def test_stroke_table_reads_the_simulator_the_same_way_it_reads_the_rig(fault_type, severity):
    params = D.DoorParams.cranfield()
    base = C.sim_stroke_metrics(params, None, 0.0)
    hot = C.sim_stroke_metrics(params, fault_type, severity)
    assert np.isfinite(base["i_cruise_mean"]) and base["i_cruise_mean"] > 0
    # friction must raise the cruise current and must do so through the current channel
    assert hot["i_cruise_mean"] / base["i_cruise_mean"] > 1.5


# --------------------------------------------------------------------------------------
# the real release layout: what the adapter hands over, and what goes back the other way
# --------------------------------------------------------------------------------------


def _release_dir(tmp_path: Path) -> Path:
    """A two-file CORD release in miniature: `Normal.mat` and `Spalling4.mat`, real naming."""
    scipy_io = pytest.importorskip("scipy.io")
    from nebulax.adapters import cranfield as A

    raw = tmp_path / "cranfield"
    raw.mkdir()
    fs = A.FS_HZ
    for fname, cls, stage, i_move in (("Normal.mat", "train", "", 0.85), ("Spalling4.mat", "point", "4th", 1.02)):
        variables = {}
        for profile, (travel_s, wait_s) in A.PROFILE_TIMING_S.items():
            n_tr, n_wt = int(round(travel_s * fs)), int(round(wait_s * fs))
            ramp = 120.0 * np.arange(n_tr) / n_tr
            ref = np.concatenate([np.zeros(n_wt), ramp, np.full(n_wt, 120.0), 120.0 - ramp]) + 20.0
            moving = np.abs(np.diff(ref, prepend=ref[0])) > 1e-9
            mat = np.column_stack(
                [ref, np.where(moving, 2.0, 0.0), np.where(moving, i_move, 0.40)]
            )
            for load in ("20kg", "40kg", "neg40kg"):
                variables[f"{cls}{profile}{stage}{load}1"] = mat
        scipy_io.savemat(raw / fname, variables)
    return raw


def test_load_strokes_takes_its_labels_from_the_feature_table(tmp_path: Path):
    """`fault_type`, `level` and `motion_profile` are columns the adapter emits, never
    substrings re-parsed out of the `run_id` - the run id is an identifier, not a schema."""
    strokes, meta = C.load_strokes(_release_dir(tmp_path))
    assert not strokes.empty
    assert {"fault_type", "level", "motion_profile", "load_kg"} <= set(strokes.columns)
    assert set(strokes["fault_type"]) == {"healthy", "misalignment"}
    assert set(strokes.loc[strokes["fault_type"] == "misalignment", "level"]) == {4}
    assert set(strokes.loc[strokes["fault_type"] == "healthy", "level"]) == {0}
    assert set(strokes["motion_profile"]) == {"sinusoidal", "trapezoidal"}
    assert meta["n_files"] == 2 and len(meta["missing_files"]) == 11


def test_load_dataset_caches_one_directory(tmp_path: Path):
    raw = _release_dir(tmp_path)
    a = C.load_dataset(raw)
    assert C.load_dataset(raw) is a
    assert C.load_dataset(tmp_path / "cranfield" / ".." / "cranfield") is a  # same resolved path


def test_class_summary_stratifies_by_motion_profile():
    """Pooling two motion profiles makes every median hostage to the trap:sin balance - which
    is exactly how `Backlash1.mat`, one repetition short of 60, faked a 1.077 duration ratio."""
    rows = []
    for profile, dur in (("trapezoidal", 5.0), ("sinusoidal", 6.0)):
        for ft, lvl, n, cur in (("healthy", 0, 10, 1.0), ("friction", 1, 10, 2.0)):
            # the faulty file is short of trapezoidal strokes, so pooling would tip the median
            k = n if not (profile == "trapezoidal" and ft == "friction") else 2
            for _ in range(k):
                rows.append(
                    {
                        "fault_type": ft, "level": lvl, "motion_profile": profile,
                        "i_cruise_mean": cur, "i_cruise_rms": cur, "v_cruise": 0.05,
                        "duration_s": dur, "err_spike_width_s": 0.1,
                        "err_spike_peak_m": 0.001, "ripple_frac": 0.02,
                    }
                )
    st = pd.DataFrame(rows)
    out = C.class_summary(st)
    fr = out[out["fault_type"] == "friction"].iloc[0]
    # within a profile the duration is identical, so the ratio is 1 - the pooled median is not
    assert fr["duration_s_ratio"] == pytest.approx(1.0)
    assert float(st[st["fault_type"] == "friction"]["duration_s"].median()) != pytest.approx(
        float(st[st["fault_type"] == "healthy"]["duration_s"].median())
    )
    assert fr["i_cruise_mean_ratio"] == pytest.approx(2.0)


def test_sim_block_to_matrix_round_trips_through_the_adapter(tmp_path: Path):
    """The self-check writer must produce something the *real-layout* adapter can read."""
    scipy_io = pytest.importorskip("scipy.io")
    from nebulax.adapters import cranfield as A

    wide = analytic_wide(n_strokes=4, fs=50.0, with_voltage=False)
    mat = C.sim_block_to_matrix(wide)
    assert mat.ndim == 2 and mat.shape[1] == 3
    assert mat[:, 0].max() == pytest.approx(100.0, rel=0.05)  # 0.10 m written as mm

    raw = tmp_path / "cranfield"
    raw.mkdir()
    scipy_io.savemat(raw / "Normal.mat", {"traintrap20kg1": mat})
    ds = A.load(raw)
    ds.validate()
    st = C.stroke_table(S.to_wide(ds.long, "door"), group_cols=("run_id",))
    assert len(st) == 4  # the four strokes went in and four came back
    assert float(st["travel_m"].median()) == pytest.approx(0.10, rel=0.05)
    assert float(st["fs_hz"].median()) == pytest.approx(A.FS_HZ, rel=0.01)
