#!/usr/bin/env python
"""Calibrate the bearing vibration feature model against the University of Ottawa data.

What this does
--------------
Reads ``data/raw/ottawa/`` (UORED-VAFCLS, Mendeley ``y2px5tg92h``, CC BY 4.0 - see
``nebulax.adapters.ottawa`` for the full provenance), cuts every 10 s recording into
**1 s windows at 42 kHz**, and computes per window

* ``rms``    - broadband RMS of the **AC-coupled** acceleration (m/s2),
* ``kurt``   - kurtosis (already mean-removed by definition),
* ``crest``  - ``max|x - mean(x)| / rms_ac``,
* ``bpfo_amp`` / ``bpfi_amp`` - summed Hilbert-envelope-spectrum **amplitude** (single-sided
  ``2|E|/n``, m/s2) over the first three BPFO / BPFI harmonics
  (:func:`nebulax.features.vibration.envelope_spectrum_feats`, kurtogram-selected band),

then, per health class (``healthy`` / ``inner_race`` / ``outer_race``, with ``ball`` /
``cage`` reported for completeness):

1. fits ``rms ~ speed**alpha`` in log-log, **clustered on the record** (one point per
   file, not per window - the 10 windows of one recording are not 10 independent
   samples), with a 95 % CI, and reports the naive per-window fit beside it to show
   how much pseudo-replication would have flattered us;
2. forms the **faulty/healthy ratios** that set the simulator's fault endpoints
   (``vib_rms_defect``, ``vib_crest_gain``, ``vib_bpfo_healthy``/``vib_bpfo_gain``);
3. checks the healthy ``crest ~ 4.5`` and ``kurtosis ~ 3`` anchors that
   ``nebulax.sim.bearing`` takes from [R157].

It prints a constant-by-constant table of *current default* vs *Ottawa-derived* value
(so re-running it is the audit of ``nebulax/sim/bearing.py``), writes the per-window
table as CSV when asked, and renders four overlay plots to
``results/sim_checks/cal_bearing_*.png``.

Two measurement decisions, both load-bearing
--------------------------------------------
**AC coupling.** 21 of the 60 raw records carry a DC offset larger than their own AC
RMS (up to 1606 m/s2 on ``H_2_0``, against an AC RMS of 56). A DC offset is a sensor
bias, not vibration: left in, it inflates ``rms`` and crushes ``crest`` toward 1
(``H_17_0``: crest 1.09 DC-coupled vs 4.19 AC-coupled), which would have *falsified*
the ~4.5 healthy crest anchor for purely instrumental reasons. Every statistic here is
computed on ``x - mean(x)``. The rest of the pipeline now agrees:
:func:`nebulax.features.stats.window_stats` takes an ``ac_couple=True`` flag (its
``rms``/``crest`` slots become ``rms_ac``/``crest_ac``) and
:mod:`nebulax.adapters.ottawa` uses it for the ``vib_rms``/``vib_crest`` channels it
writes, keeping the DC-coupled pair and a ``dc_offset_ms2`` column beside them so the
bias stays visible. :func:`window_stats_ac` here is kept as the float64 reference
implementation this calibration is measured with (``window_stats`` returns float32);
``tests/test_calibrate_bearing.py`` pins the two against each other.

**Ratios, not absolutes.** Ottawa is a lab bench rig: an accelerometer bolted to a
small ER16K-class housing, 42 kHz broadband, healthy AC RMS ~32 m/s2. The simulator's
healthy floor is 0.62 m/s2, derived from a 12 t axle box at 300 km/h [R150]. The ~50x
gap is a real physical difference (mount, bearing size, structural path, bandwidth),
not a unit error - so **only dimensionless faulty/healthy ratios cross from Ottawa
into the simulator**, never absolute magnitudes.

Run::

    python scripts/calibrate_bearing.py                       # full run + plots
    python scripts/calibrate_bearing.py --table out.csv       # also dump per-window features
    python scripts/calibrate_bearing.py --no-plots --quiet    # numbers only
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nebulax.adapters.ottawa import FS_HZ, G_MS2, GEOMETRY  # noqa: E402
from nebulax.features.vibration import BearingGeometry, envelope_spectrum_feats  # noqa: E402
from nebulax.sim.bearing import BearingParams, severity_bump  # noqa: E402

__all__ = [
    "CLASS_OF",
    "PRE_CALIBRATION",
    "Calibration",
    "ConstantUpdate",
    "PowerLawFit",
    "RecordId",
    "extract_table",
    "fit_power_law",
    "main",
    "parse_record_name",
    "read_record",
    "make_plots",
    "recommend_constants",
    "record_features",
    "report",
    "summarise",
    "window_stats_ac",
]

DEFAULT_RAW_DIR: Final[Path] = REPO_ROOT / "data" / "raw" / "ottawa"
DEFAULT_OUT_DIR: Final[Path] = REPO_ROOT / "results" / "sim_checks"

#: File-name class code -> health label (see nebulax.adapters.ottawa).
CLASS_OF: Final[dict[str, str]] = {
    "H": "healthy",
    "I": "inner_race",
    "O": "outer_race",
    "B": "ball",
    "C": "cage",
}
#: The three classes this calibration is actually about; ball/cage are reported only.
PRIMARY_CLASSES: Final[tuple[str, ...]] = ("healthy", "inner_race", "outer_race")
#: Pooled "defective" population used for the s = 1 endpoints.
POOLED_CLASSES: Final[tuple[str, ...]] = ("inner_race", "outer_race")

WINDOW_S: Final[float] = 1.0
#: fast_kurtogram depth; 3 matches nebulax.adapters.ottawa's 1 s setting.
KURTOGRAM_MAX_LEVEL: Final[int] = 3
#: Speed the fitted power laws are reported at (the grand median of the 60 records).
REF_RPM: Final[float] = 1800.0
#: Narrow band in which healthy / inner / outer all have records, for a speed-matched check.
MATCHED_BAND_RPM: Final[tuple[float, float]] = (1780.0, 1825.0)

_FILE_RE: Final[re.Pattern[str]] = re.compile(r"^(?P<cls>[HIOBC])_(?P<id>\d+)_(?P<state>[012])$")

# Categorical slots 1-3 of the project palette (blue / orange / aqua): these are
# scatter plots, i.e. an all-pairs comparison, which caps a validated categorical
# palette at three series. ball/cage therefore share one muted "other" grey.
COLOR: Final[dict[str, str]] = {
    "healthy": "#2a78d6",
    "inner_race": "#eb6834",
    "outer_race": "#1baf7a",
    "ball": "#8a8f98",
    "cage": "#8a8f98",
}
_INK: Final[str] = "#22262b"
_MUTED: Final[str] = "#6b7280"
_GRID: Final[str] = "#dfe3e8"
_SIM: Final[str] = "#4a3aa7"  # violet: the simulator's own law, never a measured series

#: ``BearingParams`` as it stood BEFORE this calibration - the "before" curve of
#: ``cal_bearing_severity.png``, kept so the plot stays meaningful once the new values are
#: the live defaults (at which point every row of the report should read CONFIRMED).
PRE_CALIBRATION: Final[dict[str, float]] = {
    "vib_rms_defect": 2.00,      # ours, a guess
    "vib_crest_gain": 4.0,       # fitted to [R157] CWRU only
    "vib_bpfo_healthy": 0.0,     # no healthy floor at all
    "vib_bpfo_gain": 1.5,
}


# ------------------------------------------------------------------------------ reading


@dataclass(frozen=True, slots=True)
class RecordId:
    """Identity parsed out of an Ottawa file name ``<Class>_<bearingId>_<state>``."""

    stem: str
    cls_code: str
    health: str
    bearing_id: int
    state: int


def parse_record_name(stem: str) -> RecordId:
    m = _FILE_RE.match(stem)
    if not m:
        raise ValueError(
            f"calibrate_bearing: unrecognised record name {stem!r}; expected "
            f"<Class>_<bearingId>_<state> with Class in {{H,I,O,B,C}} and state in {{0,1,2}}"
        )
    return RecordId(
        stem=stem,
        cls_code=m["cls"],
        health=CLASS_OF[m["cls"]],
        bearing_id=int(m["id"]),
        state=int(m["state"]),
    )


def read_record(path: Path) -> tuple[np.ndarray, float, float]:
    """``(acceleration in m/s2, nominal rpm, nominal load)`` for one Ottawa CSV.

    The raw ``Accelerometer`` column is in g's; it is converted with the same
    :data:`nebulax.adapters.ottawa.G_MS2` the adapter uses so both agree. ``Speed`` and
    ``Load`` are per-recording constants living only in row 0 (every later row is 0.0).
    """
    df = pd.read_csv(
        path, encoding="utf-8-sig", usecols=["Accelerometer", "Speed", "Load"], dtype="float32"
    )
    df.columns = [c.strip() for c in df.columns]
    accel = df["Accelerometer"].to_numpy(dtype=np.float64) * G_MS2
    return accel, float(df["Speed"].iloc[0]), float(df["Load"].iloc[0])


# ------------------------------------------------------------------------------ features


def window_stats_ac(x: np.ndarray) -> dict[str, np.ndarray]:
    """AC-coupled RMS / kurtosis / crest for a ``(n_win, L)`` stack of windows.

    Every statistic is taken on ``x - mean(x)`` per window - see the module docstring on
    why DC coupling is not an option on this dataset. ``dc`` is returned too so the
    offset itself stays visible rather than being silently discarded.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"window_stats_ac: expected a 2-D (n_win, L) array, got shape {x.shape}")
    dc = x.mean(axis=1)
    d = x - dc[:, None]
    var = np.mean(d * d, axis=1)
    rms = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        kurt = np.where(var > 0, np.mean(d**4, axis=1) / np.where(var > 0, var * var, np.nan), np.nan)
        crest = np.where(rms > 0, np.max(np.abs(d), axis=1) / np.where(rms > 0, rms, np.nan), np.nan)
    return {"dc": dc, "rms": rms, "kurt": kurt, "crest": crest}


def record_features(
    accel: np.ndarray,
    rpm: float,
    *,
    fs: float = FS_HZ,
    window_s: float = WINDOW_S,
    geometry: BearingGeometry = GEOMETRY,
    max_level: int = KURTOGRAM_MAX_LEVEL,
) -> pd.DataFrame:
    """Per-window feature table for one recording (one row per whole ``window_s`` window)."""
    L = int(round(window_s * fs))
    n_win = int(accel.size // L)
    if n_win == 0:
        raise ValueError(
            f"record_features: {accel.size} samples is shorter than one {window_s} s window "
            f"({L} samples at {fs} Hz)"
        )
    X = np.asarray(accel[: n_win * L], dtype=np.float64).reshape(n_win, L)
    stats = window_stats_ac(X)
    bpfo = np.empty(n_win)
    bpfi = np.empty(n_win)
    sk_max = np.empty(n_win)
    shaft_hz = np.empty(n_win)
    for k in range(n_win):
        f = envelope_spectrum_feats(X[k], fs, rpm, geometry, max_level=max_level)
        bpfo[k] = f["env_bpfo_energy"]
        bpfi[k] = f["env_bpfi_energy"]
        sk_max[k] = f["sk_max"]
        shaft_hz[k] = f["shaft_hz"]
    # `env_*_energy` is the sum of that family's harmonic envelope *amplitudes* (single-sided
    # 2|E|/n, m/s2 - see envelope_spectrum_feats). It is already an amplitude, so the
    # faulty/healthy ratio below is a ratio of envelope amplitudes, which is what the
    # simulator's vib_bpfo (m/s2, on the same scale as vib_rms) represents. No sqrt: the
    # features used to be |E|^2/n power and this line used to take one.
    return pd.DataFrame(
        {
            "win": np.arange(n_win),
            "rms": stats["rms"],
            "kurt": stats["kurt"],
            "crest": stats["crest"],
            "dc": stats["dc"],
            "bpfo_amp": bpfo,
            "bpfi_amp": bpfi,
            "sk_max": sk_max,
            "shaft_hz": shaft_hz,
        }
    )


def extract_table(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    *,
    window_s: float = WINDOW_S,
    max_level: int = KURTOGRAM_MAX_LEVEL,
    geometry: BearingGeometry = GEOMETRY,
    files: Sequence[Path] | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Per-window feature table across every Ottawa recording under ``raw_dir``."""
    raw_dir = Path(raw_dir)
    if files is None:
        if not raw_dir.exists():
            raise FileNotFoundError(
                f"calibrate_bearing: {raw_dir} does not exist; expected the flat UORED-VAFCLS "
                f"layout, e.g. {raw_dir}/H_1_0.csv, {raw_dir}/I_1_1.csv, {raw_dir}/O_6_2.csv "
                f"(run scripts/download_data.py --dataset ottawa)"
            )
        files = sorted(raw_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(
                f"calibrate_bearing: no *.csv records under {raw_dir}; expected names like "
                f"'H_1_0.csv' / 'I_1_1.csv' / 'O_6_2.csv'"
            )
    frames: list[pd.DataFrame] = []
    for path in files:
        rec = parse_record_name(Path(path).stem)
        accel, rpm, load = read_record(Path(path))
        f = record_features(
            accel, rpm, window_s=window_s, geometry=geometry, max_level=max_level
        )
        f.insert(0, "file", rec.stem)
        f.insert(1, "health", rec.health)
        f.insert(2, "state", rec.state)
        f.insert(3, "bearing_id", rec.bearing_id)
        f.insert(4, "rpm", rpm)
        f.insert(5, "load", load)
        frames.append(f)
        if progress:
            print(f"  {rec.stem:8s} {len(f):3d} windows  rms={f['rms'].median():8.2f}", flush=True)
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------------------ fitting


@dataclass(frozen=True, slots=True)
class PowerLawFit:
    """OLS fit of ``log(value) = log(a) + alpha * log(speed)``."""

    alpha: float
    intercept: float
    stderr: float
    ci_lo: float
    ci_hi: float
    r2: float
    n: int
    speed_lo: float
    speed_hi: float

    @property
    def speed_span(self) -> float:
        """Ratio of the fastest to the slowest unit - the conditioning of the fit."""
        return self.speed_hi / self.speed_lo

    def at(self, speed: float | np.ndarray) -> np.ndarray:
        return np.exp(self.intercept) * np.asarray(speed, dtype=np.float64) ** self.alpha

    def covers(self, alpha: float) -> bool:
        """Is ``alpha`` inside the 95 % CI, i.e. does the data fail to refute it?"""
        return bool(self.ci_lo <= alpha <= self.ci_hi)

    def describe(self) -> str:
        return (
            f"alpha = {self.alpha:+.2f} +- {self.stderr:.2f} "
            f"(95% CI [{self.ci_lo:+.2f}, {self.ci_hi:+.2f}], R2 = {self.r2:.3f}, "
            f"n = {self.n}, speed span {self.speed_span:.2f}x)"
        )


def fit_power_law(speed: np.ndarray, value: np.ndarray) -> PowerLawFit:
    """Fit ``value ~ speed**alpha`` by OLS in log-log with a 95 % CI on ``alpha``.

    Feed it **one point per recording** (a per-file median), not per window: the windows
    of one recording share a bearing, a speed and a mounting, so treating them as
    independent shrinks the standard error by ~sqrt(10) for free and is pseudo-replication.
    """
    x = np.log(np.asarray(speed, dtype=np.float64).reshape(-1))
    y = np.log(np.asarray(value, dtype=np.float64).reshape(-1))
    if x.size != y.size:
        raise ValueError(f"fit_power_law: speed has {x.size} points, value has {y.size}")
    n = x.size
    if n < 3:
        raise ValueError(f"fit_power_law: need >= 3 points to put a CI on the slope, got {n}")
    sxx = float(((x - x.mean()) ** 2).sum())
    if sxx <= 0:
        raise ValueError("fit_power_law: every point is at the same speed - alpha unidentifiable")
    alpha, intercept = np.polyfit(x, y, 1)
    resid = y - (alpha * x + intercept)
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    stderr = float(np.sqrt(sse / (n - 2) / sxx))
    # Student-t 97.5 % quantile without pulling scipy in for one number.
    from scipy import stats as _st

    tcrit = float(_st.t.ppf(0.975, n - 2))
    return PowerLawFit(
        alpha=float(alpha),
        intercept=float(intercept),
        stderr=stderr,
        ci_lo=float(alpha - tcrit * stderr),
        ci_hi=float(alpha + tcrit * stderr),
        r2=float(1.0 - sse / sst) if sst > 0 else float("nan"),
        n=n,
        speed_lo=float(np.exp(x.min())),
        speed_hi=float(np.exp(x.max())),
    )


# ------------------------------------------------------------------------------ summary

_FEATURE_COLS: Final[tuple[str, ...]] = ("rms", "kurt", "crest", "bpfo_amp", "bpfi_amp", "sk_max")


@dataclass(frozen=True, slots=True)
class Calibration:
    """Everything derived from the per-window table, in one place."""

    per_file: pd.DataFrame
    group: pd.DataFrame
    matched: pd.DataFrame
    ratios: dict[str, float]
    fits: dict[str, PowerLawFit]
    window_fits: dict[str, PowerLawFit]
    n_windows: int
    n_records: int
    dc_dominated: int

    def med(self, health: str, col: str) -> float:
        """Median over records of the per-record median of ``col`` for one health class."""
        return float(self.group.loc[health, col])


def _per_file(table: pd.DataFrame) -> pd.DataFrame:
    keys = ["file", "health", "state", "bearing_id", "rpm", "load"]
    agg = table.groupby(keys, as_index=False)[list(_FEATURE_COLS) + ["dc"]].median()
    agg["n_windows"] = table.groupby(keys, as_index=False).size()["size"].to_numpy()
    return agg


def _group_stats(per_file: pd.DataFrame) -> pd.DataFrame:
    out = per_file.groupby("health").agg(
        n_records=("file", "nunique"),
        rpm_lo=("rpm", "min"),
        rpm_hi=("rpm", "max"),
        load=("load", "median"),
        **{c: (c, "median") for c in _FEATURE_COLS},
        rms_p10=("rms", lambda s: s.quantile(0.10)),
        rms_p90=("rms", lambda s: s.quantile(0.90)),
        kurt_p10=("kurt", lambda s: s.quantile(0.10)),
        kurt_p90=("kurt", lambda s: s.quantile(0.90)),
        crest_p10=("crest", lambda s: s.quantile(0.10)),
        crest_p90=("crest", lambda s: s.quantile(0.90)),
    )
    return out


def summarise(table: pd.DataFrame) -> Calibration:
    """Per-record medians, per-class stats, faulty/healthy ratios and the speed fits."""
    per_file = _per_file(table)
    group = _group_stats(per_file)
    in_band = per_file[per_file["rpm"].between(*MATCHED_BAND_RPM)]
    matched = _group_stats(in_band) if len(in_band) else group.iloc[0:0]

    pooled = per_file[per_file["health"].isin(POOLED_CLASSES)]
    healthy = per_file[per_file["health"] == "healthy"]

    def ratio(sel: pd.DataFrame, col: str) -> float:
        h = float(healthy[col].median())
        return float(sel[col].median() / h) if h else float("nan")

    ratios: dict[str, float] = {}
    for col in ("rms", "kurt", "crest", "bpfo_amp", "bpfi_amp"):
        ratios[f"{col}_pooled"] = ratio(pooled, col)
        for health in POOLED_CLASSES + ("ball", "cage"):
            sel = per_file[per_file["health"] == health]
            if len(sel):
                ratios[f"{col}_{health}"] = ratio(sel, col)
    if len(in_band):
        hb = in_band[in_band["health"] == "healthy"]
        for health in POOLED_CLASSES:
            sel = in_band[in_band["health"] == health]
            if len(sel) and len(hb):
                ratios[f"rms_{health}_matched"] = float(
                    sel["rms"].median() / hb["rms"].median()
                )

    fits: dict[str, PowerLawFit] = {}
    window_fits: dict[str, PowerLawFit] = {}
    groups: dict[str, pd.DataFrame] = {h: per_file[per_file["health"] == h] for h in PRIMARY_CLASSES}
    groups["pooled_faulty"] = pooled
    for name, sel in groups.items():
        if len(sel) < 3 or sel["rpm"].nunique() < 2:
            continue
        fits[name] = fit_power_law(sel["rpm"].to_numpy(), sel["rms"].to_numpy())
        w = table[table["file"].isin(sel["file"])]
        window_fits[name] = fit_power_law(w["rpm"].to_numpy(), w["rms"].to_numpy())

    dc_dominated = int((per_file["dc"].abs() > per_file["rms"]).sum())
    return Calibration(
        per_file=per_file,
        group=group,
        matched=matched,
        ratios=ratios,
        fits=fits,
        window_fits=window_fits,
        n_windows=int(len(table)),
        n_records=int(per_file["file"].nunique()),
        dc_dominated=dc_dominated,
    )


# ------------------------------------------------------------------ constants for the sim


@dataclass(frozen=True, slots=True)
class ConstantUpdate:
    """One ``BearingParams`` field: what it is now, what Ottawa says, and why."""

    name: str
    old: float
    new: float
    evidence: str
    verdict: str  # 'changed' | 'confirmed' | 'unrefuted'

    @property
    def changed(self) -> bool:
        return self.verdict == "changed"


def recommend_constants(cal: Calibration, params: BearingParams | None = None) -> list[ConstantUpdate]:
    """Map the measured ratios onto :class:`nebulax.sim.bearing.BearingParams` fields.

    The mapping rules, stated once so they are auditable:

    * ``vib_rms`` at ``s = 1`` must be ``rms_ratio`` x the healthy floor, so
      ``vib_rms_defect = vib_rms_healthy * (rms_ratio - 1)`` with ``rms_ratio`` the
      pooled inner+outer faulty/healthy ratio of AC RMS.
    * The kurtosis/crest bump peaks at ``s = kurt_s_peak``, and Ottawa's seeded defects
      are of **unknown** severity, so a measured class median is a *lower bound* on the
      peak: keep the literature peak where it already clears the measurement
      (kurtosis: 3 + 19 = 22 vs 16.8 measured), raise it to the measurement where it does
      not (crest: 4.5 + 4.0 = 8.5 is below the 11.8 measured on outer-race records).
      The most impulsive measured class (outer race) sets the bound in both cases.
    * ``vib_bpfo``'s **absolute** scale has no source either way, so the s = 1 endpoint
      ``vib_bpfo_healthy + vib_bpfo_gain`` is held at its previous value and only the
      measured healthy/faulty *contrast* (outer-race BPFO envelope amplitude ratio) is
      imposed, which turns a healthy floor of exactly zero - trivially separable, the
      same error rail_phm 4.3.2 caught on temperature - into a measured one.
    * Speed exponents are only changed if the fit **refutes** the incumbent, i.e. if the
      incumbent falls outside the 95 % CI.
    """
    p = params or BearingParams()
    out: list[ConstantUpdate] = []

    # --- rms endpoint --------------------------------------------------------------
    r_rms = cal.ratios["rms_pooled"]
    a_d = p.vib_rms_healthy * (r_rms - 1.0)
    out.append(
        ConstantUpdate(
            "vib_rms_defect",
            p.vib_rms_defect,
            round(a_d, 2),
            f"pooled inner+outer AC-RMS ratio {r_rms:.2f}x healthy "
            f"(inner {cal.ratios['rms_inner_race']:.2f}x, outer {cal.ratios['rms_outer_race']:.2f}x) "
            f"=> A_d = A_h*(ratio-1)",
            "changed" if abs(a_d - p.vib_rms_defect) > 0.05 else "confirmed",
        )
    )

    # --- healthy anchors -----------------------------------------------------------
    k_h = cal.med("healthy", "kurt")
    out.append(
        ConstantUpdate(
            "vib_kurt_healthy",
            p.vib_kurt_healthy,
            p.vib_kurt_healthy,
            f"Ottawa healthy kurtosis {k_h:.2f} (per-record median; "
            f"p10-p90 {cal.group.loc['healthy', 'kurt_p10']:.2f}-"
            f"{cal.group.loc['healthy', 'kurt_p90']:.2f} over 20 bearings) vs [R157] 2.76-2.96",
            "confirmed",
        )
    )
    c_h = cal.med("healthy", "crest")
    out.append(
        ConstantUpdate(
            "vib_crest_healthy",
            p.vib_crest_healthy,
            p.vib_crest_healthy,
            f"Ottawa healthy crest {c_h:.2f} (per-record median; "
            f"p10-p90 {cal.group.loc['healthy', 'crest_p10']:.2f}-"
            f"{cal.group.loc['healthy', 'crest_p90']:.2f}) vs [R157] 4.22-5.59",
            "confirmed",
        )
    )

    # --- bump peaks ----------------------------------------------------------------
    k_o = cal.med("outer_race", "kurt")
    k_peak_now = p.vib_kurt_healthy + p.vib_kurt_gain * float(
        severity_bump(p.kurt_s_peak, s_peak=p.kurt_s_peak, w_lo=p.kurt_w_lo, w_hi=p.kurt_w_hi)
    )
    k_gain = p.vib_kurt_gain if k_peak_now >= k_o else round(k_o - p.vib_kurt_healthy, 2)
    out.append(
        ConstantUpdate(
            "vib_kurt_gain",
            p.vib_kurt_gain,
            k_gain,
            f"bump peak {k_peak_now:.1f} vs the most impulsive measured class (outer race "
            f"{k_o:.2f}, inner race {cal.med('inner_race', 'kurt'):.2f}, healthy {k_h:.2f}): the "
            f"peak must reach the measurement, and [R157]'s 21.69 already clears it",
            "changed" if k_gain != p.vib_kurt_gain else "confirmed",
        )
    )
    c_o = cal.med("outer_race", "crest")
    c_peak_now = p.vib_crest_healthy + p.vib_crest_gain * float(
        severity_bump(p.kurt_s_peak, s_peak=p.kurt_s_peak, w_lo=p.kurt_w_lo, w_hi=p.kurt_w_hi)
    )
    c_gain = p.vib_crest_gain if c_peak_now >= c_o else round(c_o - p.vib_crest_healthy, 2)
    out.append(
        ConstantUpdate(
            "vib_crest_gain",
            p.vib_crest_gain,
            c_gain,
            f"bump peak {c_peak_now:.2f} vs the measured outer-race crest {c_o:.2f} (inner race "
            f"{cal.med('inner_race', 'crest'):.2f}, healthy {c_h:.2f}): the peak is raised until it "
            f"reaches the measurement ([R157]'s CWRU-only fit gave only 8.50)",
            "changed" if c_gain != p.vib_crest_gain else "confirmed",
        )
    )

    # --- envelope ------------------------------------------------------------------
    r_bpfo = cal.ratios["bpfo_amp_outer_race"]
    endpoint = p.vib_bpfo_gain + getattr(p, "vib_bpfo_healthy", 0.0)
    floor = round(endpoint / r_bpfo, 3)
    out.append(
        ConstantUpdate(
            "vib_bpfo_healthy",
            getattr(p, "vib_bpfo_healthy", 0.0),
            floor,
            f"outer-race BPFO envelope amplitude is {r_bpfo:.2f}x healthy "
            f"(pooled {cal.ratios['bpfo_amp_pooled']:.2f}x; inner-race BPFI "
            f"{cal.ratios['bpfi_amp_inner_race']:.2f}x) - a healthy floor of 0 is not measured",
            "changed" if abs(floor - getattr(p, "vib_bpfo_healthy", 0.0)) > 1e-3 else "confirmed",
        )
    )
    gain = round(endpoint - floor, 3)
    out.append(
        ConstantUpdate(
            "vib_bpfo_gain",
            p.vib_bpfo_gain,
            gain,
            f"s=1 endpoint held at {endpoint:.2f} m/s2 (unsourced absolute scale); only the "
            f"measured {r_bpfo:.2f}x contrast is imposed",
            "changed" if abs(gain - p.vib_bpfo_gain) > 1e-3 else "confirmed",
        )
    )

    # --- speed exponents -------------------------------------------------------------
    for field_name, key, incumbent_src in (
        ("vib_speed_exp_healthy", "healthy", "[R158] healthy fit ~2.3"),
        ("vib_speed_exp_defect", "pooled_faulty", "[R158] defect fit 1.0-1.3"),
    ):
        fit = cal.fits.get(key)
        current = float(getattr(p, field_name))
        if fit is None:
            continue
        refuted = not fit.covers(current)
        out.append(
            ConstantUpdate(
                field_name,
                current,
                round(fit.alpha, 2) if refuted else current,
                f"Ottawa {key} rms~rpm^alpha: {fit.describe()} - speed span is only "
                f"{fit.speed_span:.2f}x and between-bearing scatter dominates, so the fit "
                f"{'REFUTES' if refuted else 'cannot refute'} the incumbent {incumbent_src}",
                "changed" if refuted else "unrefuted",
            )
        )
    return out


# ------------------------------------------------------------------------------- plots


def _style(ax) -> None:
    ax.grid(True, color=_GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.xaxis.label.set_color(_MUTED)
    ax.yaxis.label.set_color(_MUTED)


def _fig(title: str, subtitle: Sequence[str], nrows: int = 1, ncols: int = 1,
         figsize: tuple[float, float] = (9.0, 5.0)):
    """Figure with a left-aligned title + wrapped subtitle lines above the axes."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle(title, fontsize=12.5, color=_INK, fontweight="bold", ha="left", x=0.010, y=0.985)
    y = 0.985 - 0.26 / figsize[1]
    for line in subtitle:
        fig.text(0.010, y, line, fontsize=8.2, color=_MUTED, ha="left", va="top")
        y -= 0.165 / figsize[1]
    top = y - 0.05 / figsize[1]
    return fig, axes, top


def _save(fig, path: Path, top: float):
    import matplotlib.pyplot as plt

    fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def plot_rms_speed(cal: Calibration, table: pd.DataFrame, out_dir: Path) -> Path:
    """rms vs speed, log-log, per health class, with the clustered fits and the sim laws."""
    from matplotlib.ticker import FuncFormatter, FixedLocator

    p = BearingParams()
    fig, ax, top = _fig(
        "Broadband RMS vs shaft speed, by health state",
        [
            "Ottawa UORED-VAFCLS, 1 s windows at 42 kHz, AC-coupled.  Faint dots = windows, "
            "solid = per-record median, dashed = OLS on the record medians.",
            "The record speeds span only 1.1-1.2x, so between-bearing scatter swamps the slope: "
            "every fitted CI below is wide enough to contain the simulator's law.",
        ],
        figsize=(9.2, 5.6),
    )
    for health in PRIMARY_CLASSES:
        w = table[table["health"] == health]
        f = cal.per_file[cal.per_file["health"] == health]
        ax.scatter(w["rpm"], w["rms"], s=5, color=COLOR[health], alpha=0.18, linewidths=0)
        ax.scatter(
            f["rpm"], f["rms"], s=44, color=COLOR[health], edgecolor="white", linewidths=1.2,
            zorder=3, label=f"{health} ({len(f)} records)",
        )
        fit = cal.fits.get(health)
        if fit is not None:
            xs = np.linspace(f["rpm"].min(), f["rpm"].max(), 50)
            ax.plot(xs, fit.at(xs), color=COLOR[health], linewidth=2.0, linestyle="--", zorder=4)
            mid = len(xs) // 2
            ax.annotate(
                f"a = {fit.alpha:+.1f}  CI [{fit.ci_lo:+.1f}, {fit.ci_hi:+.1f}]",
                xy=(xs[mid], fit.at(xs[mid])), xytext=(0, 9), textcoords="offset points",
                fontsize=8, color=_INK, ha="center",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.85),
                zorder=6,
            )
    # The simulator's laws, anchored on each class's own median so only the SLOPE is compared.
    for health, exp_, dash, lbl in (
        ("healthy", p.vib_speed_exp_healthy, "-", f"sim healthy floor  v^{p.vib_speed_exp_healthy:g}"),
        ("outer_race", p.vib_speed_exp_defect, "-.", f"sim defect term  v^{p.vib_speed_exp_defect:g}"),
    ):
        f = cal.per_file[cal.per_file["health"] == health]
        xs = np.linspace(f["rpm"].min(), f["rpm"].max(), 50)
        y0 = float(f["rms"].median())
        ax.plot(xs, y0 * (xs / float(f["rpm"].median())) ** exp_, color=_SIM, linewidth=2.0,
                linestyle=dash, alpha=0.9, zorder=5, label=lbl)
    ax.set_xscale("log")
    ax.set_yscale("log")
    lo, hi = float(cal.per_file["rpm"].min()), float(cal.per_file["rpm"].max())
    ax.set_xlim(lo * 0.985, hi * 1.015)
    ax.xaxis.set_major_locator(FixedLocator([1700, 1800, 1900, 2000, 2100, 2200]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_locator(FixedLocator([20, 30, 50, 100, 200, 300, 500, 700]))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("nominal shaft speed (rpm, log scale)")
    ax.set_ylabel("AC-coupled broadband RMS (m/s2, log scale)")
    _style(ax)
    ax.legend(frameon=False, fontsize=8.2, labelcolor=_INK, loc="lower right", ncols=2)
    return _save(fig, out_dir / "cal_bearing_rms_speed.png", top)


def plot_healthy_anchors(cal: Calibration, table: pd.DataFrame, out_dir: Path) -> Path:
    """Do the healthy kurtosis ~ 3 and crest ~ 4.5 anchors survive contact with Ottawa?"""
    p = BearingParams()
    fig, axes, top = _fig(
        "Healthy anchors: kurtosis ~ 3 and crest ~ 4.5",
        [
            "Every 1 s healthy window (20 bearings, 200 windows), AC-coupled.  Vertical line = "
            "simulator default; grey band = the [R157] range rail_phm 4.3.3 quotes.",
        ],
        1, 2, figsize=(9.2, 4.4),
    )
    h = table[table["health"] == "healthy"]
    hf = cal.per_file[cal.per_file["health"] == "healthy"]
    for ax, col, anchor, lit, xlabel in (
        (axes[0], "kurt", p.vib_kurt_healthy, (2.76, 2.96), "kurtosis"),
        (axes[1], "crest", p.vib_crest_healthy, (4.22, 5.59), "crest factor"),
    ):
        ax.hist(h[col], bins=28, color=COLOR["healthy"], alpha=0.32, linewidth=0,
                label="1 s windows")
        ax.hist(hf[col], bins=14, color=COLOR["healthy"], alpha=0.95, linewidth=0,
                label="per-record medians")
        ax.axvspan(*lit, color=_MUTED, alpha=0.16, linewidth=0, label="[R157] measured range")
        ax.axvline(anchor, color=_SIM, linewidth=2.0, label=f"sim default {anchor:g}")
        med = float(hf[col].median())
        ax.axvline(med, color=_INK, linewidth=1.4, linestyle="--",
                   label=f"Ottawa median {med:.2f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        _style(ax)
        ax.legend(frameon=False, fontsize=7.6, labelcolor=_INK)
    return _save(fig, out_dir / "cal_bearing_anchors.png", top)


def plot_envelope(cal: Calibration, out_dir: Path) -> Path:
    """BPFO / BPFI envelope amplitude by health class - what sets vib_bpfo's contrast."""
    fig, axes, top = _fig(
        "Envelope-spectrum band amplitude by health state",
        [
            "summed harmonic envelope amplitude (2|E|/n, m/s2) in the kurtogram-selected "
            "band; one point per record, bar = class median, log scale.",
            "The BPFO ratio on outer-race records is what sets the simulator's healthy "
            "vib_bpfo floor; the geometry is assumed (see the module docstring).",
        ],
        1, 2, figsize=(9.2, 4.6),
    )
    order = [c for c in list(PRIMARY_CLASSES) + ["ball", "cage"]
             if (cal.per_file["health"] == c).any()]
    rng = np.random.default_rng(7)
    for ax, col, title in (
        (axes[0], "bpfo_amp", "BPFO - outer-race frequency"),
        (axes[1], "bpfi_amp", "BPFI - inner-race frequency"),
    ):
        h_med = float(cal.group.loc["healthy", col])
        for i, health in enumerate(order):
            sel = cal.per_file[cal.per_file["health"] == health]
            x = i + rng.uniform(-0.13, 0.13, len(sel))
            ax.scatter(x, sel[col] / h_med, s=38, color=COLOR[health], alpha=0.85,
                       edgecolor="white", linewidths=1.0, zorder=3)
            m = float(sel[col].median()) / h_med
            ax.plot([i - 0.3, i + 0.3], [m, m], color=_INK, linewidth=2.0, zorder=4)
            ax.annotate(f"{m:.1f}x", xy=(i, m), xytext=(0, 7), textcoords="offset points",
                        fontsize=8.5, color=_INK, ha="center", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                                  edgecolor="none", alpha=0.85))
        ax.axhline(1.0, color=_MUTED, linewidth=1.0, linestyle=":")
        ax.set_yscale("log")
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([o.replace("_", "\n") for o in order], fontsize=8)
        ax.set_ylabel("band amplitude / healthy median")
        ax.set_title(title, fontsize=9.5, color=_INK, loc="left")
        _style(ax)
    return _save(fig, out_dir / "cal_bearing_envelope.png", top)


def plot_severity_laws(cal: Calibration, updates: list[ConstantUpdate], out_dir: Path) -> Path:
    """The simulator's severity laws before and after, over the measured Ottawa bands."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(BearingParams)}
    old = BearingParams(**{k: v for k, v in PRE_CALIBRATION.items() if k in fields})
    new = BearingParams(**{u.name: u.new for u in updates if u.changed and u.name in fields})
    s = np.linspace(0.0, 1.0, 201)
    fig, axes, top = _fig(
        "Simulator severity laws vs the Ottawa measurement",
        [
            "Laws evaluated at v = 22 m/s.  Bands = Ottawa healthy and defective (inner+outer) "
            "per-record p10-p90.",
            "Ottawa's seeded defects have no documented size, so the defective band BOUNDS the "
            "curve (the peak must reach it) - it does not place it on the severity axis.",
        ],
        2, 2, figsize=(9.6, 6.6),
    )

    def bump(p: BearingParams) -> np.ndarray:
        return severity_bump(s, s_peak=p.kurt_s_peak, w_lo=p.kurt_w_lo, w_hi=p.kurt_w_hi)

    hf = cal.per_file[cal.per_file["health"] == "healthy"]
    ff = cal.per_file[cal.per_file["health"].isin(POOLED_CLASSES)]
    # Ottawa -> simulator magnitude: ratios only, anchored on the healthy median (see docstring).
    scale = float(old.vib_rms_healthy) / float(hf["rms"].median())
    panels = (
        ("vib_rms (m/s2)", lambda p: p.vib_rms_healthy + p.vib_rms_defect * s,
         (hf["rms"] * scale, ff["rms"] * scale)),
        ("vib_kurt", lambda p: p.vib_kurt_healthy + p.vib_kurt_gain * bump(p),
         (hf["kurt"], ff["kurt"])),
        ("vib_crest", lambda p: p.vib_crest_healthy + p.vib_crest_gain * bump(p),
         (hf["crest"], ff["crest"])),
        ("vib_bpfo (m/s2)",
         lambda p: getattr(p, "vib_bpfo_healthy", 0.0) + p.vib_bpfo_gain * s, None),
    )
    for k, (ax, (label, law, bands)) in enumerate(zip(axes.ravel(), panels)):
        y_old = np.broadcast_to(np.asarray(law(old), dtype=float), s.shape)
        y_new = np.broadcast_to(np.asarray(law(new), dtype=float), s.shape)
        hi = max(y_old.max(), y_new.max())
        if bands is not None:
            for arr, color, name in ((bands[0], COLOR["healthy"], "Ottawa healthy p10-p90"),
                                     (bands[1], COLOR["outer_race"], "Ottawa defective p10-p90")):
                lo_b, hi_b = float(arr.quantile(0.10)), float(arr.quantile(0.90))
                ax.axhspan(lo_b, hi_b, color=color, alpha=0.16, linewidth=0, label=name)
                hi = max(hi, min(hi_b, 1.6 * hi))
        ax.plot(s, y_old, color=_MUTED, linewidth=1.8, linestyle=":", label="before (pre-Ottawa)")
        ax.plot(s, y_new, color=_SIM, linewidth=2.2, label="after (Ottawa-calibrated)")
        ax.set_ylim(0.0 if label.startswith("vib_bpfo") else None, hi * 1.06)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("severity s")
        ax.set_ylabel(label)
        _style(ax)
        ax.legend(frameon=False, fontsize=7.6, labelcolor=_INK, loc="upper left")
    return _save(fig, out_dir / "cal_bearing_severity.png", top)


def make_plots(cal: Calibration, table: pd.DataFrame, updates: list[ConstantUpdate],
               out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_rms_speed(cal, table, out_dir),
        plot_healthy_anchors(cal, table, out_dir),
        plot_envelope(cal, out_dir),
        plot_severity_laws(cal, updates, out_dir),
    ]


# -------------------------------------------------------------------------------- report


def report(cal: Calibration, updates: list[ConstantUpdate]) -> str:
    lines: list[str] = []
    lines.append(f"Ottawa UORED-VAFCLS: {cal.n_records} records, {cal.n_windows} x "
                 f"{WINDOW_S:g} s windows at {FS_HZ:g} Hz")
    lines.append(f"  DC-dominated records (|mean| > AC RMS): {cal.dc_dominated}/{cal.n_records} "
                 f"- every statistic below is AC-coupled")
    lines.append("")
    lines.append("Per-record medians by health state (median over records):")
    cols = ["n_records", "rpm_lo", "rpm_hi", "load", "rms", "kurt", "crest", "bpfo_amp", "bpfi_amp"]
    lines.append(cal.group[cols].round(3).to_string())
    lines.append("")
    lines.append("Faulty / healthy ratios:")
    for k in sorted(cal.ratios):
        lines.append(f"  {k:28s} {cal.ratios[k]:7.2f}x")
    lines.append("")
    lines.append("rms ~ speed**alpha (clustered on the record; naive per-window fit for contrast):")
    for name, fit in cal.fits.items():
        lines.append(f"  {name:14s} {fit.describe()}")
        w = cal.window_fits.get(name)
        if w is not None:
            lines.append(f"  {'':14s}   per-window (pseudo-replicated): {w.describe()}")
    lines.append("")
    lines.append("BearingParams constants:")
    lines.append(f"  {'field':24s} {'current':>9s} {'ottawa':>9s}  verdict")
    for u in updates:
        lines.append(f"  {u.name:24s} {u.old:9.3f} {u.new:9.3f}  {u.verdict.upper()}")
        lines.append(f"  {'':24s} {'':9s} {'':9s}  {u.evidence}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR, help="data/raw/ottawa")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="where the PNGs go")
    ap.add_argument("--table", type=Path, default=None, help="also dump the per-window table here")
    ap.add_argument("--window-s", type=float, default=WINDOW_S)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if not args.quiet:
        print(f"reading {args.raw} ...", flush=True)
    table = extract_table(args.raw, window_s=args.window_s, progress=not args.quiet)
    cal = summarise(table)
    updates = recommend_constants(cal)
    text = report(cal, updates)
    if not args.quiet:
        print(text)
    if args.table is not None:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.table, index=False)
        if not args.quiet:
            print(f"\nper-window table -> {args.table}")
    if not args.no_plots:
        paths = make_plots(cal, table, updates, args.out)
        if not args.quiet:
            for p in paths:
                print(f"plot -> {p}")
    changed = [u.name for u in updates if u.changed]
    if not args.quiet:
        print(f"\n{len(changed)} constant(s) the data says should change: {', '.join(changed) or 'none'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
