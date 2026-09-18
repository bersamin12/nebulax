"""Simulator self-checks: run a short scenario per subsystem, assert the sanity anchors and
drop diagnostic plots into ``results/sim_checks/``.

::

    python -m nebulax.sim.selftest                     # every implemented subsystem
    python -m nebulax.sim.selftest --subsystem door    # just the door
    python -m nebulax.sim.selftest --out /tmp/checks --seed 7

Exit code 0 means every check passed; 1 names the first that did not.  This is the
``python -m sim.selftest`` step of the plan's Verification section: *each simulator runs
1 simulated day in < 10 s, the fault ramp is monotone, plots land in results/sim_checks/*.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from nebulax import schema as S
from nebulax.sim.common import SEC_PER_DAY, DegradationTrajectory, generate_service

DEFAULT_OUT = Path("results/sim_checks")
SUBSYSTEMS = ("door", "pneumatic", "bearing")


@dataclass(slots=True)
class Check:
    """One named assertion plus the value that produced it."""

    name: str
    ok: bool
    detail: str


def _say(checks: list[Check], name: str, ok: bool, detail: str) -> None:
    checks.append(Check(name, bool(ok), detail))


# --------------------------------------------------------------------------------------
# Door
# --------------------------------------------------------------------------------------


def _door_runs(seed: int) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]]:
    """One healthy day, one day held at ``s = 1``, one day ramping ``s: 0 -> 1``."""
    from nebulax.sim import door as dr

    out: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]] = {}
    specs: list[tuple[str, list[DegradationTrajectory]]] = [
        ("healthy", []),
        (
            "friction_s1",
            [
                DegradationTrajectory(
                    fault_type="friction",
                    subsystem="door",
                    component_id="door_L1",
                    t_onset=0.0,
                    t_failure=1.0,
                    gamma=1.0,
                    jitter_sigma=0.0,
                )
            ],
        ),
        (
            "friction_ramp",
            [
                DegradationTrajectory(
                    fault_type="friction",
                    subsystem="door",
                    component_id="door_L1",
                    t_onset=0.25 * SEC_PER_DAY,
                    t_failure=0.95 * SEC_PER_DAY,
                    gamma=1.5,
                )
            ],
        ),
    ]
    for name, faults in specs:
        rng = np.random.default_rng(seed)
        service = generate_service(1, rng, train_id="T01")
        t0 = time.perf_counter()
        long, feats, events = dr.simulate(
            dr.DoorParams(),
            faults,
            service,
            rng,
            store_every=5,
            run_id=f"selftest_door_{name}",
            car=3,
            component_id="door_L1",
        )
        out[name] = (long, feats, events, time.perf_counter() - t0)
    return out


def _closing_window(long: pd.DataFrame, feats: pd.DataFrame, cycle_id: int) -> pd.DataFrame:
    """The stored 50 Hz waveform of one cycle, wide, as the detail panel would show it."""
    row = feats[feats["cycle_id"] == cycle_id]
    if row.empty:
        return pd.DataFrame()
    t_start, t_end = row["t_start"].iloc[0], row["t_end"].iloc[0]
    door = long[long["subsystem"].astype("string") == "door"]
    wide = S.to_wide(door, "door")
    return wide[(wide["timestamp"] >= t_start) & (wide["timestamp"] <= t_end)].reset_index(drop=True)


def door(out_dir: Path = DEFAULT_OUT, seed: int = 0, make_plots: bool = True) -> list[Check]:
    """Run the door self-check and write ``results/sim_checks/door_*.png``."""
    from nebulax.sim import door as _dr

    checks: list[Check] = []
    dr_stroke = _dr.DoorParams().stroke_m
    runs = _door_runs(seed)

    for name, (long, feats, events, secs) in runs.items():
        S.validate_long(long)
        S.validate_features(feats, require_labels=True)
        S.validate_events(events)
        _say(checks, f"door[{name}] one day < 10 s", secs < 10.0, f"{secs:.2f} s")
        _say(
            checks,
            f"door[{name}] schema-conformant",
            True,
            f"{len(long):,} long rows / {len(feats)} cycles / {len(events)} events",
        )

    h = runs["healthy"][1]
    clean = h[h["reversal_count"] == 0]
    med_close = float(clean["closing_time"].median())
    _say(
        checks,
        "healthy closing time 2-3 s",
        2.0 <= med_close <= 3.0,
        f"median {med_close:.2f} s over {len(clean)} unobstructed cycles",
    )
    _say(
        checks,
        "healthy door never trips functional failure",
        not (runs["healthy"][2]["event"].astype("string") == "functional_failure").any(),
        f"{int((runs['healthy'][2]['event'].astype('string') == 'door_fault').sum())} door_fault event(s)",
    )
    _say(
        checks,
        "closing warning >= 2 s before movement",
        bool((h["warning_time"] >= 2.0).all()),
        f"min {float(h['warning_time'].min()):.2f} s [R165 PRM 4.2.2.3.2]",
    )
    obs = h[h["obstruction"] > 0]
    lat = obs["obs_detect_s"].to_numpy(dtype=float)
    lat = lat[np.isfinite(lat)]
    _say(
        checks,
        "obstruction detected within 0.3 s",
        lat.size == 0 or bool((lat <= 0.30).all()),
        f"{lat.size} detections, max {0.0 if lat.size == 0 else lat.max():.3f} s",
    )

    s1 = runs["friction_s1"][1]
    _say(
        checks,
        "friction s=1 reaches functional failure",
        bool((runs["friction_s1"][2]["event"].astype("string") == "functional_failure").any()),
        f"median closing {float(s1['closing_time'].median()):.2f} s, "
        f"{int(s1['ls_timeout'].sum())} ls_timeout cycles",
    )
    # The top of the severity axis must still be a *door*: force-limited and slow, not
    # seized.  A seized endpoint makes every severity above the cliff look identical and
    # leaves a severity regressor degenerate over most of its label range.
    r_close = float(s1["closing_time"].median()) / med_close
    r_cur = float(s1["i_mean_cruise"].median()) / float(clean["i_mean_cruise"].median())
    r_open = float(s1["pos_open_max"].median()) / float(dr_stroke)
    r_rev = float((s1["reversal_count"] > 0).mean()) - float((h["reversal_count"] > 0).mean())
    _say(
        checks,
        "friction s=1 is force-limited and slow, not seized",
        r_close >= 1.6 and r_cur >= 1.4 and r_open > 0.95 and r_rev <= 0.05,
        f"closing x{r_close:.2f}, cruise current x{r_cur:.2f}, still opens to "
        f"{100 * r_open:.0f} % of stroke, phantom-reversal rate {100 * r_rev:+.1f} pp",
    )

    ramp = runs["friction_ramp"][1].sort_values("cycle_id")
    sev = ramp["severity"].to_numpy(dtype=float)
    cur = ramp["i_rms_cruise"].to_numpy(dtype=float)
    # only the prognostic window: past functional failure the DCU has already taken the
    # door out of service, which is post-mortem data, not degradation.
    rul = ramp["rul_s"].to_numpy(dtype=float)
    ok = np.isfinite(cur) & (sev > 0) & (rul > 0)
    rho = (
        float(pd.Series(sev[ok]).corr(pd.Series(cur[ok]), method="spearman"))
        if ok.sum() > 10
        else float("nan")
    )
    _say(
        checks,
        "cruise current rises with severity up to functional failure",
        np.isfinite(rho) and rho > 0.8,
        f"Spearman rho = {rho:.3f} over {int(ok.sum())} faulted cycles",
    )
    healthy_cur = float(clean["i_rms_cruise"].median())
    _say(
        checks,
        "fault lifts cruise current well above healthy",
        float(np.nanmax(cur)) > 1.8 * healthy_cur,
        f"healthy {healthy_cur:.3f} A -> max {float(np.nanmax(cur)):.3f} A",
    )

    if make_plots:
        _door_plots(runs, out_dir)
    return checks


def _door_plots(runs: dict[str, tuple[Any, ...]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    h_long, h_feat = runs["healthy"][0], runs["healthy"][1]
    f_long, f_feat = runs["friction_ramp"][0], runs["friction_ramp"][1]

    # --- one healthy cycle vs one degraded cycle -------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex="col")
    panels = (
        (h_long, h_feat, "healthy", 0.0),
        (f_long, f_feat, "friction, mid-degradation", 0.28),
    )
    for col, (long, feat, title, target_s) in enumerate(panels):
        stored = feat[feat["cycle_id"] % 5 == 0]
        pick = (stored["severity"] - target_s).abs().to_numpy()
        cyc = int(stored["cycle_id"].iloc[int(np.argmin(pick))])
        sev_c = float(feat.loc[feat["cycle_id"] == cyc, "severity"].iloc[0])
        w = _closing_window(long, feat, cyc)
        if w.empty:
            continue
        t = (w["timestamp"] - w["timestamp"].iloc[0]).dt.total_seconds()
        axes[0, col].plot(t, w["pos_ref"], lw=1, ls="--", label="pos_ref")
        axes[0, col].plot(t, w["pos"], lw=1.2, label="pos")
        axes[0, col].set_ylabel("position (m)")
        axes[0, col].set_title(f"{title} - cycle {cyc}, s = {sev_c:.2f}")
        axes[0, col].legend(fontsize=7)
        axes[1, col].plot(t, w["current"], lw=1.0, color="tab:red")
        axes[1, col].set_ylabel("current (A)")
        axes[2, col].plot(t, w["pwm"], lw=1.0, color="tab:green", label="pwm")
        axes[2, col].plot(t, w["ls_closed"], lw=1.0, color="k", label="ls_closed")
        axes[2, col].plot(t, w["obstruction"], lw=1.0, color="tab:orange", label="obstruction")
        axes[2, col].set_ylabel("pwm / digitals")
        axes[2, col].set_xlabel("time in cycle (s)")
        axes[2, col].legend(fontsize=7)
    fig.suptitle("door: stored 50 Hz cycle waveforms (open activity, >=2 s warning, close activity)")
    fig.tight_layout()
    fig.savefig(out_dir / "door_cycle.png", dpi=110)
    plt.close(fig)

    # --- position-binned closing current profile --------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.linspace(0.0, 1.0, len(h_feat["current_profile_50"].iloc[0]))
    for feat, label, colour in ((h_feat, "healthy", "tab:blue"), (f_feat, "friction ramp", "tab:red")):
        prof = np.asarray([np.asarray(r, dtype=float) for r in feat["current_profile_50"]])
        sev = feat["severity"].to_numpy(dtype=float)
        if label == "healthy":
            ax.plot(xs, prof.mean(0), color=colour, label=f"{label} (n={len(prof)})")
            ax.fill_between(xs, np.percentile(prof, 5, 0), np.percentile(prof, 95, 0), color=colour, alpha=0.2)
        else:
            for lo, hi, alpha in ((0.05, 0.25, 0.4), (0.25, 0.6, 0.7), (0.6, 1.01, 1.0)):
                m = (sev >= lo) & (sev < hi)
                if m.any():
                    ax.plot(xs, prof[m].mean(0), color=colour, alpha=alpha, label=f"s in [{lo:.2f}, {hi:.2f})")
    ax.set_xlabel("closing travel (fraction of stroke)")
    ax.set_ylabel("|current| (A)")
    ax.set_title("door: current_profile_50 - the feature the literature actually uses [R64, R72]")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "door_current_profile.png", dpi=110)
    plt.close(fig)

    # --- per-cycle feature trends -------------------------------------------------------
    cols = ("closing_time", "i_rms_cruise", "pwm_mean", "energy_J", "pos_err_max", "T_motor")
    fig, axes = plt.subplots(len(cols), 1, figsize=(9, 11), sharex=True)
    for ax, col in zip(axes, cols):
        ax.plot(h_feat["cycle_id"], h_feat[col], lw=0.8, color="tab:blue", label="healthy")
        ax.plot(f_feat["cycle_id"], f_feat[col], lw=0.8, color="tab:red", label="friction ramp")
        ax.set_ylabel(col, fontsize=8)
    twin = axes[0].twinx()
    twin.plot(f_feat["cycle_id"], f_feat["severity"], lw=1.0, color="k", ls=":", label="severity")
    twin.set_ylabel("severity", fontsize=8)
    ff = runs["friction_ramp"][2]
    ff = ff[ff["event"].astype("string") == "functional_failure"]
    if len(ff):
        t_ff = ff["timestamp"].iloc[0]
        k = int((f_feat["t_end"] <= t_ff).sum())
        for ax in axes:
            ax.axvline(k, color="k", lw=1.0, ls="--")
        axes[0].annotate("functional failure", (k, axes[0].get_ylim()[1]), fontsize=7, ha="right", va="top")
    axes[0].legend(fontsize=7)
    axes[-1].set_xlabel("cycle index within the day")
    fig.suptitle("door: per-cycle features, healthy vs friction ramp (1 day, ~440 cycles)")
    fig.tight_layout()
    fig.savefig(out_dir / "door_features.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Pneumatic
# --------------------------------------------------------------------------------------


def _pneumatic_runs(seed: int) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]]:
    """One healthy day and one day with an ``air_leak`` ramping 0 -> 1 severity."""
    from nebulax.sim import pneumatic as pn

    out: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]] = {}
    specs: list[tuple[str, list[DegradationTrajectory]]] = [
        ("healthy", []),
        (
            "air_leak_ramp",
            [
                DegradationTrajectory(
                    fault_type="air_leak",
                    subsystem="pneumatic",
                    component_id="apu_1",
                    t_onset=0.15 * SEC_PER_DAY,
                    t_failure=1.0 * SEC_PER_DAY,
                    gamma=1.0,
                )
            ],
        ),
    ]
    for name, faults in specs:
        rng = np.random.default_rng(seed)
        service = generate_service(1, rng, train_id="T01")
        t0 = time.perf_counter()
        long, feats, events = pn.simulate(
            pn.PneumaticParams(),
            faults,
            service,
            rng,
            store_every=1,
            run_id=f"selftest_pneumatic_{name}",
        )
        out[name] = (long, feats, events, time.perf_counter() - t0)
    return out


def pneumatic(out_dir: Path = DEFAULT_OUT, seed: int = 0, make_plots: bool = True) -> list[Check]:
    """Run the pneumatic self-check and write ``results/sim_checks/pneumatic_*.png``."""
    checks: list[Check] = []
    runs = _pneumatic_runs(seed)

    for name, (long, feats, events, secs) in runs.items():
        S.validate_long(long)
        S.validate_features(feats, require_labels=True)
        S.validate_events(events)
        _say(checks, f"pneumatic[{name}] one day < 10 s", secs < 10.0, f"{secs:.2f} s")
        _say(
            checks,
            f"pneumatic[{name}] schema-conformant",
            True,
            f"{len(long):,} long rows / {len(feats)} cycles / {len(events)} events",
        )

    h_long, h_feat, h_events, _ = runs["healthy"]
    _say(
        checks,
        "healthy APU never trips functional failure",
        not (h_events["event"].astype("string") == "functional_failure").any(),
        f"{int((h_events['event'].astype('string') == 'functional_failure').sum())} functional_failure event(s)",
    )
    ok = h_feat["in_service_frac"].to_numpy() > 0.5
    duty = float(np.nanmean(h_feat["duty_ratio"].to_numpy()[ok]))
    _say(
        checks,
        "healthy compressor duty in a plausible band",
        0.03 <= duty <= 0.30,
        f"duty_ratio {duty:.3f} over {int(ok.sum())} in-service cycles [R101 measured ~0.09-0.12]",
    )

    f_long, f_feat, f_events, _ = runs["air_leak_ramp"]
    late = f_feat["severity"].to_numpy() > 0.5
    bad = late & (f_feat["in_service_frac"].to_numpy() > 0.5)
    idle_ok = float(np.nanmedian(h_feat["idle_run_ratio"].to_numpy()[ok]))
    idle_bad = float(np.nanmedian(f_feat["idle_run_ratio"].to_numpy()[bad])) if bad.any() else float("nan")
    _say(
        checks,
        "air leak shrinks idle_run_ratio well below healthy [R101]",
        bad.any() and idle_bad < 0.75 * idle_ok,
        f"healthy median {idle_ok:.2f} -> leak (s>0.5) median {idle_bad:.2f}",
    )
    duty_bad = float(np.nanmean(f_feat["duty_ratio"].to_numpy()[bad])) if bad.any() else float("nan")
    _say(
        checks,
        "air leak raises compressor duty above healthy",
        bad.any() and duty_bad > duty,
        f"healthy mean duty {duty:.3f} -> leak (s>0.5) mean duty {duty_bad:.3f}",
    )

    if make_plots:
        _pneumatic_plots(runs, out_dir)
    return checks


def _pneumatic_plots(runs: dict[str, tuple[Any, ...]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    _, h_feat, _, _ = runs["healthy"]
    _, f_feat, f_events, _ = runs["air_leak_ramp"]

    cols = ("duty_ratio", "idle_run_ratio", "T_oil_max", "P_res_min")
    fig, axes = plt.subplots(len(cols), 1, figsize=(9, 9), sharex=True)
    for ax, col in zip(axes, cols):
        ax.plot(h_feat["cycle_id"], h_feat[col], lw=0.8, color="tab:blue", label="healthy")
        ax.plot(f_feat["cycle_id"], f_feat[col], lw=0.8, color="tab:red", label="air_leak ramp")
        ax.set_ylabel(col, fontsize=8)
    twin = axes[0].twinx()
    twin.plot(f_feat["cycle_id"], f_feat["severity"], lw=1.0, color="k", ls=":", label="severity")
    twin.set_ylabel("severity", fontsize=8)
    ff = f_events[f_events["event"].astype("string") == "functional_failure"]
    if len(ff):
        t_ff = ff["timestamp"].iloc[0]
        k = int((f_feat["t_end"] <= t_ff).sum())
        for ax in axes:
            ax.axvline(k, color="k", lw=1.0, ls="--")
        axes[0].annotate("functional failure", (k, axes[0].get_ylim()[1]), fontsize=7, ha="right", va="top")
    axes[0].legend(fontsize=7)
    axes[-1].set_xlabel("compressor cycle index within the day")
    fig.suptitle("pneumatic: per-cycle features, healthy vs air_leak ramp (1 day)")
    fig.tight_layout()
    fig.savefig(out_dir / "pneumatic_features.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Bearing
# --------------------------------------------------------------------------------------


def _bearing_runs(seed: int) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]]:
    """One healthy day, one day with ``bearing_degradation`` ramping on axlebox_3R, one
    day with an acute ``hot_axle_box`` on axlebox_2L."""
    from nebulax.sim import bearing as br

    out: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]] = {}
    specs: list[tuple[str, list[DegradationTrajectory]]] = [
        ("healthy", []),
        (
            "outer_race_ramp",
            [
                DegradationTrajectory(
                    fault_type="bearing_degradation",
                    subsystem="bearing",
                    component_id="axlebox_3R",
                    t_onset=0.0,
                    t_failure=SEC_PER_DAY,
                    gamma=1.0,
                    jitter_sigma=0.0,
                )
            ],
        ),
        (
            "hot_axle_box",
            [
                DegradationTrajectory(
                    fault_type="hot_axle_box",
                    subsystem="bearing",
                    component_id="axlebox_2L",
                    t_onset=0.30 * SEC_PER_DAY,
                    t_failure=0.30 * SEC_PER_DAY + 10 * 3600.0,
                    gamma=3.0,
                    jitter_sigma=0.0,
                )
            ],
        ),
    ]
    for name, faults in specs:
        rng = np.random.default_rng(seed)
        service = generate_service(1, rng, train_id="T01")
        t0 = time.perf_counter()
        long, feats, events = br.simulate(
            br.BearingParams(),
            faults,
            service,
            rng,
            store_every=5,
            run_id=f"selftest_bearing_{name}",
            car=3,
        )
        out[name] = (long, feats, events, time.perf_counter() - t0)
    return out


def bearing(out_dir: Path = DEFAULT_OUT, seed: int = 0, make_plots: bool = True) -> list[Check]:
    """Run the bearing self-check and write ``results/sim_checks/bearing_*.png``."""
    checks: list[Check] = []
    runs = _bearing_runs(seed)

    for name, (long, feats, events, secs) in runs.items():
        S.validate_long(long)
        S.validate_features(feats, require_labels=True)
        S.validate_events(events)
        _say(checks, f"bearing[{name}] one day < 10 s", secs < 10.0, f"{secs:.2f} s")
        _say(
            checks,
            f"bearing[{name}] schema-conformant",
            True,
            f"{len(long):,} long rows / {len(feats)} cycles / {len(events)} events",
        )

    def _box(feat: pd.DataFrame, box: str) -> pd.DataFrame:
        return feat[feat["component_id"].astype("string") == box]

    h_events = runs["healthy"][2]
    _say(
        checks,
        "healthy fleet never trips a hot_box_alarm or functional failure",
        not (h_events["event"].astype("string").isin(["hot_box_alarm", "functional_failure"])).any(),
        f"{len(h_events)} event row(s) total",
    )

    vib_feat = runs["outer_race_ramp"][1]
    bad, good = _box(vib_feat, "axlebox_3R"), _box(vib_feat, "axlebox_3L")
    _say(
        checks,
        "outer-race ramp lifts kurtosis, BPFO and RMS well above the healthy peer box",
        bad["vib_kurt_mean"].max() > 12.0
        and good["vib_kurt_mean"].max() < 5.0
        and bad["vib_bpfo_mean"].max() > 4.0 * max(good["vib_bpfo_mean"].max(), 1e-6)
        and bad["vib_rms_mean"].max() > 1.5 * good["vib_rms_mean"].max(),
        f"kurt {good['vib_kurt_mean'].max():.1f} -> {bad['vib_kurt_mean'].max():.1f}, "
        f"bpfo {good['vib_bpfo_mean'].max():.3f} -> {bad['vib_bpfo_mean'].max():.3f}, "
        f"rms {good['vib_rms_mean'].max():.2f} -> {bad['vib_rms_mean'].max():.2f}",
    )

    hot_long, hot_feat, hot_events, _ = runs["hot_axle_box"]
    hot_bad, hot_good = _box(hot_feat, "axlebox_2L"), _box(hot_feat, "axlebox_2R")
    _say(
        checks,
        "hot_axle_box is the fault that drives temperature, not its healthy neighbour",
        hot_bad["T_box_max"].max() > 80.0 and hot_good["T_box_max"].max() < 80.0,
        f"axlebox_2L T_box_max {hot_bad['T_box_max'].max():.1f} C, "
        f"axlebox_2R T_box_max {hot_good['T_box_max'].max():.1f} C",
    )
    _say(
        checks,
        "hot_axle_box raises a hot_box_alarm event",
        (hot_events["event"].astype("string") == "hot_box_alarm").any(),
        f"{int((hot_events['event'].astype('string') == 'hot_box_alarm').sum())} hot_box_alarm event(s)",
    )

    if make_plots:
        _bearing_plots(runs, out_dir)
    return checks


def _bearing_plots(runs: dict[str, tuple[Any, ...]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    def _box(feat: pd.DataFrame, box: str) -> pd.DataFrame:
        return feat[feat["component_id"].astype("string") == box]

    _, vib_feat, _, _ = runs["outer_race_ramp"]
    bad, good = _box(vib_feat, "axlebox_3R").sort_values("cycle_id"), _box(vib_feat, "axlebox_3L").sort_values(
        "cycle_id"
    )
    _, hot_feat, _, _ = runs["hot_axle_box"]
    hot_bad = _box(hot_feat, "axlebox_2L").sort_values("cycle_id")
    hot_good = _box(hot_feat, "axlebox_2R").sort_values("cycle_id")

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=False)
    axes[0].plot(bad["cycle_id"], bad["vib_kurt_mean"], lw=0.8, color="tab:red", label="axlebox_3R (faulty)")
    axes[0].plot(good["cycle_id"], good["vib_kurt_mean"], lw=0.8, color="tab:blue", label="axlebox_3L (healthy)")
    axes[0].set_ylabel("vib_kurt_mean")
    axes[0].set_title("bearing: outer-race ramp, kurtosis")
    axes[0].legend(fontsize=7)

    axes[1].plot(bad["cycle_id"], bad["vib_bpfo_mean"], lw=0.8, color="tab:red", label="axlebox_3R (faulty)")
    axes[1].plot(good["cycle_id"], good["vib_bpfo_mean"], lw=0.8, color="tab:blue", label="axlebox_3L (healthy)")
    axes[1].set_ylabel("vib_bpfo_mean")
    axes[1].legend(fontsize=7)

    axes[2].plot(hot_bad["cycle_id"], hot_bad["T_box_max"], lw=0.8, color="tab:red", label="axlebox_2L (hot)")
    axes[2].plot(hot_good["cycle_id"], hot_good["T_box_max"], lw=0.8, color="tab:blue", label="axlebox_2R (healthy)")
    axes[2].set_ylabel("T_box_max (C)")
    axes[2].set_xlabel("5-minute window index within the day")
    axes[2].set_title("bearing: acute hot_axle_box")
    axes[2].legend(fontsize=7)

    fig.suptitle("bearing: per-window features (1 day)")
    fig.tight_layout()
    fig.savefig(out_dir / "bearing_features.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

_RUNNERS = {"door": door, "pneumatic": pneumatic, "bearing": bearing}


def run(subsystem: str, out_dir: Path = DEFAULT_OUT, seed: int = 0, make_plots: bool = True) -> list[Check]:
    """Run one subsystem's self-check."""
    if subsystem not in _RUNNERS:
        raise ValueError(
            f"selftest.run: no self-check for {subsystem!r}; implemented = {list(_RUNNERS)}"
        )
    return _RUNNERS[subsystem](out_dir=out_dir, seed=seed, make_plots=make_plots)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nebulax.sim.selftest", description=__doc__)
    ap.add_argument("--subsystem", default="all", choices=("all", *SUBSYSTEMS))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    names = list(SUBSYSTEMS) if args.subsystem == "all" else [args.subsystem]
    failed = 0
    for name in names:
        print(f"== {name} ==")
        for c in run(name, out_dir=args.out, seed=args.seed, make_plots=not args.no_plots):
            print(f"  [{'ok ' if c.ok else 'FAIL'}] {c.name}: {c.detail}")
            failed += not c.ok
    if not args.no_plots:
        print(f"plots -> {args.out}")
    print("OK" if not failed else f"{failed} check(s) FAILED")
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
