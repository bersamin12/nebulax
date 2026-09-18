"""``nebulax.adapters.ottawa`` - University of Ottawa variable-speed bearing dataset
(UORED-VAFCLS), the real-data proxy for the ``bearing`` subsystem [rail_phm 3.4, 4.3.4].

Source: Mendeley ``data.mendeley.com/datasets/y2px5tg92h`` ("UORED-VAFCLS"), DOI
10.17632/y2px5tg92h.5 (version 5), licence CC BY 4.0.

Real-data notes - verified this session by direct inspection of the 60 files actually
downloaded under ``data/raw/ottawa/``, not by re-reading the dataset's own documentation
(unreachable offline)
--------------------------------------------------------------------------------------------
* **This is UORED-VAFCLS (``y2px5tg92h``), the constant-speed ablation set - NOT the
  encoder-equipped variable-speed set.** ``docs/research/rail_phm.md`` SS4.3.4 describes a
  *separate* University of Ottawa dataset - "uOttawa bearing vibration under time-varying
  rotational speed", DOI ``10.17632/v43hmbwxpm.2`` [R118] - as shipping "a 1024-CPR encoder
  channel at 200 kHz" with speed that "sweeps 13.7-28.9 Hz shaft speed WITHIN a single 10 s
  record", and explicitly promotes *that* dataset to primary for the bearing subsystem.
  ``docs/research/datasets.md`` names UORED-VAFCLS [R119] as the "constant-speed ablation" and
  ``configs/model_ladder.yaml``'s ``bearing.datasets`` list carries both as separate ids
  (``ottawa_variable_speed`` and ``ottawa_uored_vafcls``). rail_phm.md is internally consistent
  about this - it is not self-contradictory. Only UORED-VAFCLS (``y2px5tg92h``) has been
  downloaded to this repo (``data/raw/ottawa/`` = the 60 ``H/I/O/B/C_*.csv`` files this adapter
  reads); ``ottawa_variable_speed`` (``v43hmbwxpm``) has **no adapter and no download** here yet
  - flagged for whoever owns data acquisition next, not silently worked around.
  This adapter's own choices (tacholess-only features, no encoder use, reading ``Speed``/
  ``Load`` from row 0 only) are correct **for the file layout it actually reads**: UORED-VAFCLS
  carries exactly **5 columns at 42 kHz** - ``Accelerometer, Acoustic, Speed, Load, Temperature
  Difference`` - no encoder/tacho column, and ``Speed``/``Load`` are **not per-sample**: only
  row 0 carries the nominal value, every other one of the remaining 419,999 rows is exactly 0.0.
  Speed is a **per-recording constant** here - it varies *across* the 60 files (that is the
  "variable-speed" part of the family name shared with v43hmbwxpm, misleading as it is for
  *this* member of the family), not *within* one.
* **A temperature channel does exist** in UORED-VAFCLS (``Temperature Difference``, degC),
  which is consistent with rail_phm.md SS4.3.3's "no temperature channel" note being about the
  *v43hmbwxpm* variable-speed set, not this one. Its reference point (difference from what?) is
  undocumented in anything reachable offline this session, so it is carried into the long table
  as ``T_box`` with that caveat attached in ``meta`` - a coarse proxy, not a calibrated absolute
  axle-box temperature. Bearing-thermal ground truth for our simulator still comes from
  [R149]/[R150] per rail_phm.md SS4.3.1-4.3.4; this is not that.
* **Raw ``Accelerometer`` is in g's, not m/s2** (confirmed by direct inspection and recorded
  verbatim in ``data/raw/ottawa/MANIFEST.json``'s inspection note, "Accelerometer (g)"). This
  adapter converts to **m/s2** (``x9.80665``) immediately on read, before any feature or long-
  table signal is derived, so that ``vib_rms``/``vib_bpfo`` land in the same unit
  :data:`nebulax.schema.SIGNAL_SPECS` declares (``unit='m/s2'``) and that
  ``nebulax.sim.bearing`` uses for the identical signal names (healthy floor ~0.6-2.6 m/s2).
* **The accelerometer is DC-biased on a third of the records, so the amplitude features are
  AC-coupled.** 21 of the 60 records carry a window mean larger than their own AC RMS (worst
  ``H_2_0``: mean 1606 m/s2 against an AC RMS of 56). A constant offset is a sensor bias, not
  vibration: DC-coupled it inflates the RMS of those records by 1.5-44x (worst ``H_17_0``,
  43.9x) and crushes the crest factor towards 1 (``H_17_0``: 1.09 DC-coupled vs 4.19
  AC-coupled). The ``vib_rms``/``vib_crest`` long-table
  channels are therefore computed with :func:`nebulax.features.stats.window_stats`'s
  ``ac_couple=True`` (``vib_kurt`` is central by definition and was never affected), matching
  ``scripts/calibrate_bearing.py``, which derives the bearing simulator's constants on
  ``x - mean(x)``. The offset is **kept visible, never hidden**: the feature table carries the
  DC-coupled ``rms``/``crest`` columns unchanged *and* the new ``rms_ac``/``crest_ac``/
  ``dc_offset_ms2`` (= the window's ``mean``) columns beside them, and ``meta``
  ``["dc_offset_ms2"]`` records the whole-record offset per file. See ``docs/parameters.md``
  bearing section 4.
* ``Acoustic`` (raw microphone) is present in the files but not feature-ized in this pass -
  noted in ``meta`` as a future extension, not silently dropped.
* ``.mat`` copies of every file are not read: byte-for-byte the same values as the matching
  ``.csv`` (verified by the download agent), so reading both would double the work for nothing.
* **Isolated sensor-dropout glitches in ``Temperature Difference``.** A handful of raw files
  (e.g. ``C_16_1.csv``, ``C_16_2.csv``, ``C_17_1.csv``, ``C_17_2.csv``) contain single-sample
  spikes reaching magnitudes up to ``-8e11`` degC - almost certainly a sensor/logging dropout
  sentinel, not a real temperature. ``T_box`` is therefore computed with samples outside a
  physically plausible range (``_TEMP_PLAUSIBLE_C``) rejected before averaging (falling back to
  the window's raw median in the rare case every sample in a window is rejected); the total
  number of rejected samples is recorded in ``meta["temp_glitches_rejected"]``.

What we extract
----------------
Ground truth ("labels = condition") comes from the file name
``<Class>_<bearingId>_<state>.csv``: ``Class in {H, I, O, B, C}`` (healthy / inner race / outer
race / ball / cage) and ``state in {0, 1, 2}`` (healthy / developing / faulty). All 20 physical
bearings get one healthy recording (``H_<id>_0``); ids 1-5/6-10/11-15/16-20 additionally get
inner-race/outer-race/ball/cage recordings at both fault states. There is no run-to-failure
trace, so this is a hard-coded *per-recording health-state* fault log (``t_failure``/``rul_s``
unknown -> NaT/NaN; ``shape='step'`` since the seeded defect is present for the whole recording,
not evolving within it) - exactly the "labels = condition" the adapter contract asks for.

Per record we convert ``Accelerometer`` from g's to m/s2 (``x9.80665``), then cut 0.25 s
(10,500-sample) and 1 s (42,000-sample) tiled windows and compute, per window: time-domain stats via
:func:`nebulax.features.stats.window_stats` (mean/std/min/max/slope/rms/kurtosis/crest,
plus the AC-coupled ``rms_ac``/``crest_ac`` and the ``dc_offset_ms2`` bias behind them) plus
skew/peak-to-peak/shape-factor/impulse-factor, and the kurtogram-band Hilbert-envelope spectrum
via :func:`nebulax.features.vibration.envelope_spectrum_feats` (BPFO/BPFI/BSF/FTF harmonics),
using the row-0 nominal RPM and an **assumed ER16K-class bearing geometry**
(``n_elements=9, ball_diameter_m=7.94e-3, pitch_diameter_m=39.0e-3`` - the same geometry already
used in ``tests/test_features_vibration.py``; UORED-VAFCLS's own geometry table was not
reachable offline this session, so treat the harmonic *frequency assignment* as approximate, not
the raw envelope statistics themselves). The 1 s window's engineered channels
(``vib_rms``/``vib_crest`` = the AC-coupled ``window_stats`` pair, ``vib_kurt`` = its
(mean-removed by definition) kurtosis, ``vib_bpfo`` = the summed BPFO-harmonic
envelope-spectrum *amplitude* (single-sided ``2|E|/n``, m/s2 - an amplitude, not an energy),
``T_box`` = the outlier-rejected window mean of
``Temperature Difference``, see :func:`_robust_temp_mean`) are also
written to the long telemetry table at a synthetic 1 Hz cadence, matching
:data:`nebulax.schema.SIGNAL_SPECS`'s documented ``fs_hz`` for the ``bearing`` subsystem.

Features vs metadata. The file name *is* the ground truth, so every column that repeats it
wears the reserved ``meta_`` prefix (:data:`nebulax.schema.METADATA_PREFIX`) and is dropped
from ``X`` by :func:`nebulax.schema.feature_columns`: ``meta_class`` (``H/I/O/B/C``, a
relabelling of ``fault_type``), ``meta_state`` (``0/1/2``) and ``meta_state_label``
(``healthy``/``developing``/``faulty``) - both bijections of the ``severity`` label - and
``meta_bearing_id``, the physical bearing 1..20, which exists only to group the splits. The
measured columns stay features, and so do ``window_s`` and the two operating conditions the
rig sets and a real controller would know, ``nominal_speed_rpm`` and ``nominal_load_units``:
they are constant across classes by design of the test matrix, so they confound rather than
encode the target.

Grouping and splits. ``train_id`` carries the physical bearing id (``bearing_01``..
``bearing_20``) so a bearing-wise split - required per rail_phm.md 3.4/4.3.4 to avoid the
leakage that segment-wise/condition-wise splits are documented to cause on Ottawa - is a plain
``groupby("train_id")`` (or ``groupby("meta_bearing_id")`` on the feature table). ``run_id`` is the file stem (one 10 s recording). ``component_id``
cycles through :data:`nebulax.schema.AXLEBOX_COMPONENT_IDS` (``(bearing_id - 1) % 8``) since the
registry only names 8 axle-box slots per train and this rig has 20 physical units; the real
identity is never lost, it lives in ``train_id``/``run_id``.

Timestamps are synthetic: Ottawa ships no wall-clock time, so each recording is anchored 20 s
apart from a fixed epoch purely to give every row a legal, sortable ``datetime64[ms, UTC]``
(recorded in ``meta["timestamp_epoch"]``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from nebulax.features.stats import window_stats
from nebulax.features.vibration import BearingGeometry, envelope_spectrum_feats
from nebulax.schema import (
    AXLEBOX_COMPONENT_IDS,
    Dataset,
    coerce_fault_log,
    coerce_features,
    coerce_long,
    empty_events,
    empty_fault_log,
    empty_long,
)

__all__ = ["load", "GEOMETRY", "FS_HZ"]

SOURCE: Final[str] = "ottawa"
SUBSYSTEM: Final[str] = "bearing"

#: Verified this session: every raw file is exactly 420,000 rows spanning a 10 s recording.
FS_HZ: Final[float] = 42_000.0
RECORD_S: Final[float] = 10.0
#: Raw ``Accelerometer`` is in g's (confirmed by MANIFEST.json's inspection note); converted to
#: m/s2 on read so it matches nebulax.schema.SIGNAL_SPECS's declared unit for vib_rms/vib_bpfo.
G_MS2: Final[float] = 9.80665
#: Physically plausible range for "Temperature Difference" (degC); samples outside this range
#: are isolated sensor-dropout glitches (verified up to -8e11 in a handful of raw files) and are
#: rejected before averaging into T_box - see _robust_temp_mean.
_TEMP_PLAUSIBLE_C: Final[tuple[float, float]] = (-50.0, 150.0)
#: (name -> window length in seconds) - both are extracted for every record.
WINDOW_LENGTHS_S: Final[dict[str, float]] = {"w1_0s": 1.0, "w0_25s": 0.25}
#: Cadence of the engineered channels written into the long telemetry table.
LONG_WINDOW_S: Final[float] = 1.0
#: fast_kurtogram search depth per window length - bounded so the full 60-file adapter run
#: stays well under a minute (timed empirically on the real data: max_level=3 at 1 s and
#: max_level=2 at 0.25 s costs ~35 s total across all 60 files on this machine).
_KURTOGRAM_MAX_LEVEL: Final[dict[float, int]] = {1.0: 3, 0.25: 2}

_CLASS_FAULT: Final[dict[str, str]] = {
    "H": "healthy",
    "I": "inner_race",
    "O": "outer_race",
    "B": "ball",
    "C": "cage",
}
_STATE_LABEL: Final[dict[str, str]] = {"0": "healthy", "1": "developing", "2": "faulty"}
_STATE_SEVERITY: Final[dict[str, float]] = {"0": 0.0, "1": 0.5, "2": 1.0}
_FILE_RE: Final[re.Pattern[str]] = re.compile(r"^(?P<cls>[HIOBC])_(?P<id>\d+)_(?P<state>[012])$")

#: ER16K-class deep-groove ball bearing geometry - see module docstring caveat: not re-verified
#: against UORED-VAFCLS's own descriptor this session, reused from the existing project test
#: fixture (``tests/test_features_vibration.py``) for a single consistent assumption.
GEOMETRY: Final[BearingGeometry] = BearingGeometry(
    n_elements=9, ball_diameter_m=7.94e-3, pitch_diameter_m=39.0e-3
)

_EPOCH: Final[pd.Timestamp] = pd.Timestamp("2020-01-01T00:00:00Z")
#: Synthetic, arbitrary spacing between recordings - just keeps rows sortable/non-overlapping.
_RECORD_SPACING_S: Final[float] = 20.0

#: envelope_spectrum_feats also returns DC-coupled time-domain rms/kurtosis/crest on the raw
#: window; window_stats already gives us those (AC-coupled as well), so they are not re-emitted.
_ENV_SKIP_KEYS: Final[frozenset[str]] = frozenset({"rms", "kurtosis", "crest"})


def _record_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"nebulax.adapters.ottawa: {raw_dir} does not exist; expected the University of "
            f"Ottawa UORED-VAFCLS flat layout, e.g. {raw_dir}/H_1_0.csv, {raw_dir}/I_1_1.csv, "
            f"{raw_dir}/O_6_2.csv, ... (run scripts/download_data.py --dataset ottawa)"
        )
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"nebulax.adapters.ottawa: no *.csv files found under {raw_dir}; expected file "
            f"names like 'H_1_0.csv' / 'I_1_1.csv' / 'O_6_2.csv' (<Class>_<bearingId>_<state>.csv, "
            f"Class in H|I|O|B|C, state in 0|1|2)"
        )
    return files


def _parse_name(path: Path) -> tuple[str, int, str]:
    m = _FILE_RE.match(path.stem)
    if not m:
        raise ValueError(
            f"nebulax.adapters.ottawa: unrecognised file name {path.name!r}; expected "
            f"<Class>_<bearingId>_<state>.csv with Class in {{H,I,O,B,C}}, state in {{0,1,2}}"
        )
    return m["cls"], int(m["id"]), m["state"]


def _component_id(bearing_id: int) -> str:
    return AXLEBOX_COMPONENT_IDS[(bearing_id - 1) % len(AXLEBOX_COMPONENT_IDS)]


def _read_record(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=["Accelerometer", "Speed", "Load", "Temperature Difference"],
        dtype="float32",
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def _robust_temp_mean(temp_win: np.ndarray) -> tuple[np.ndarray, int]:
    """Per-window mean of ``temp_win`` (shape ``(n_win, L)``), rejecting samples outside
    :data:`_TEMP_PLAUSIBLE_C`. A handful of raw files carry isolated single-sample sensor-dropout
    glitches (magnitudes up to -8e11 degC) that would otherwise poison the whole window's mean.
    Falls back to the raw median for a window where every sample was rejected. Returns
    ``(means, n_rejected)`` with ``n_rejected`` the total sample count rejected across all windows.
    """
    lo, hi = _TEMP_PLAUSIBLE_C
    mask = (temp_win >= lo) & (temp_win <= hi)
    n_rejected = int((~mask).sum())
    masked = np.where(mask, temp_win, np.nan)
    with np.errstate(invalid="ignore"):
        means = np.nanmean(masked, axis=1)
    all_rejected = ~np.isfinite(means)
    if all_rejected.any():
        means[all_rejected] = np.median(temp_win[all_rejected], axis=1)
    return means.astype(np.float64), n_rejected


def _extra_time_stats(x: np.ndarray) -> tuple[float, float, float, float]:
    """skew, peak-to-peak, shape_factor, impulse_factor - not covered by window_stats."""
    x = np.asarray(x, dtype=np.float64)
    mean = float(x.mean())
    c = x - mean
    std = float(np.std(c))
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    mean_abs = float(np.mean(np.abs(x)))
    skew = float(np.mean(c**3) / std**3) if std > 0 else float("nan")
    p2p = float(x.max() - x.min())
    shape_factor = float(rms / mean_abs) if mean_abs > 0 else float("nan")
    impulse_factor = float(peak / mean_abs) if mean_abs > 0 else float("nan")
    return skew, p2p, shape_factor, impulse_factor


def load(raw_dir: Path) -> Dataset:
    """Load the University of Ottawa UORED-VAFCLS bearing set as a ``bearing`` Dataset.

    Pure and offline: reads ``raw_dir`` only, never writes it. See the module docstring for
    the full column/label/feature mapping and the corrections to rail_phm.md's description of
    this specific download.
    """
    raw_dir = Path(raw_dir)
    files = _record_files(raw_dir)

    long_records: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    fault_rows: list[dict[str, Any]] = []
    sample_count_anomalies: list[str] = []
    bearings_seen: set[int] = set()
    temp_glitches_rejected = 0
    dc_offsets: dict[str, float] = {}

    for i, path in enumerate(files):
        cls, bearing_id, state = _parse_name(path)
        bearings_seen.add(bearing_id)
        fault_type = _CLASS_FAULT[cls]
        state_label = _STATE_LABEL[state]
        severity = float(_STATE_SEVERITY[state])
        is_faulty = state != "0"
        run_id = f"ottawa_{path.stem}"
        train_id = f"bearing_{bearing_id:02d}"
        component_id = _component_id(bearing_id)
        anchor = _EPOCH + pd.Timedelta(seconds=i * _RECORD_SPACING_S)

        df = _read_record(path)
        # Raw file is in g's; convert to m/s2 once, up front, so every downstream feature and
        # long-table signal is in the unit nebulax.schema.SIGNAL_SPECS declares (see G_MS2).
        accel = df["Accelerometer"].to_numpy(dtype=np.float64) * G_MS2
        temp = df["Temperature Difference"].to_numpy(dtype=np.float64)
        nominal_rpm = float(df["Speed"].iloc[0])
        nominal_load = float(df["Load"].iloc[0])
        n_samples = accel.size
        # Whole-record DC bias (m/s2). Kept as provenance: on 21 of the 60 records this is
        # larger than the record's own AC RMS, which is why the amplitude features below are
        # AC-coupled - see the module docstring.
        dc_offsets[run_id] = float(np.mean(accel))
        expected = int(round(RECORD_S * FS_HZ))
        if n_samples != expected:
            sample_count_anomalies.append(f"{path.name}: {n_samples} rows, expected {expected}")

        for w_name, w_s in WINDOW_LENGTHS_S.items():
            L = int(round(w_s * FS_HZ))
            n_win = n_samples // L
            if n_win == 0:
                continue
            usable = n_win * L
            X = accel[:usable].reshape(n_win, L, 1)
            stats = window_stats(X)  # (n_win, 8): mean,std,min,max,slope,rms,kurtosis,crest
            # Same eight slots, but rms/crest taken on x - mean(x): slot 5 is rms_ac and slot 7
            # is crest_ac (nebulax.features.stats.AC_STAT_NAMES). Both variants are written to
            # the feature table so the DC bias stays auditable; the long table carries the AC one.
            stats_ac = window_stats(X, ac_couple=True)
            temp_mean, n_rejected = _robust_temp_mean(temp[:usable].reshape(n_win, L))
            temp_glitches_rejected += n_rejected

            for k in range(n_win):
                x = X[k, :, 0]
                t_start = anchor + pd.Timedelta(seconds=k * w_s)
                t_end = t_start + pd.Timedelta(seconds=w_s)
                env: dict[str, float] = (
                    envelope_spectrum_feats(
                        x, FS_HZ, nominal_rpm, GEOMETRY, max_level=_KURTOGRAM_MAX_LEVEL[w_s]
                    )
                    if nominal_rpm > 0
                    else {}
                )
                skew, p2p, shape_factor, impulse_factor = _extra_time_stats(x)

                row: dict[str, Any] = {
                    "run_id": run_id,
                    "source": SOURCE,
                    "train_id": train_id,
                    "car": np.int8(0),
                    "subsystem": SUBSYSTEM,
                    "component_id": component_id,
                    "cycle_id": np.int64(k if w_name == "w1_0s" else k + 10_000),
                    "t_start": t_start,
                    "t_end": t_end,
                    "window_s": np.float32(w_s),
                    # meta_*: the file name's own ground truth + the grouping id; never
                    # model inputs (see nebulax.schema.feature_columns).
                    "meta_class": cls,
                    "meta_bearing_id": np.int32(bearing_id),
                    "meta_state": state,
                    "meta_state_label": state_label,
                    "nominal_speed_rpm": np.float32(nominal_rpm),
                    "nominal_load_units": np.float32(nominal_load),
                    "temp_diff_mean_c": np.float32(temp_mean[k]),
                    "mean": np.float32(stats[k, 0]),
                    "std": np.float32(stats[k, 1]),
                    "min": np.float32(stats[k, 2]),
                    "max": np.float32(stats[k, 3]),
                    "slope": np.float32(stats[k, 4]),
                    "rms": np.float32(stats[k, 5]),
                    "kurtosis": np.float32(stats[k, 6]),
                    "crest": np.float32(stats[k, 7]),
                    "rms_ac": np.float32(stats_ac[k, 5]),
                    "crest_ac": np.float32(stats_ac[k, 7]),
                    "dc_offset_ms2": np.float32(stats[k, 0]),
                    "skew": np.float32(skew),
                    "p2p": np.float32(p2p),
                    "shape_factor": np.float32(shape_factor),
                    "impulse_factor": np.float32(impulse_factor),
                    "fault_type": fault_type,
                    "severity": np.float32(severity),
                    "rul_s": np.float32("nan"),
                    "is_faulty": bool(is_faulty),
                    "alarm_window_3d": False,
                }
                for name, val in env.items():
                    if name in _ENV_SKIP_KEYS:
                        continue
                    row[name] = np.float32(val)
                feature_rows.append(row)

                if w_s == LONG_WINDOW_S:
                    vib_bpfo = float(env.get("env_bpfo_energy", np.nan))
                    for signal, value in (
                        ("vib_rms", float(stats_ac[k, 5])),
                        ("vib_kurt", float(stats[k, 6])),
                        ("vib_crest", float(stats_ac[k, 7])),
                        ("vib_bpfo", vib_bpfo),
                        ("T_box", float(temp_mean[k])),
                    ):
                        long_records.append(
                            {
                                "timestamp": t_start,
                                "source": SOURCE,
                                "run_id": run_id,
                                "train_id": train_id,
                                "car": np.int8(0),
                                "subsystem": SUBSYSTEM,
                                "component_id": component_id,
                                "signal": signal,
                                "value": np.float32(value),
                            }
                        )

        if is_faulty:
            fault_rows.append(
                {
                    "run_id": run_id,
                    "train_id": train_id,
                    "car": np.int8(0),
                    "subsystem": SUBSYSTEM,
                    "component_id": component_id,
                    "fault_type": fault_type,
                    "t_onset": anchor,
                    "t_failure": pd.NaT,
                    "t_functional_failure": pd.NaT,
                    "gamma": np.float32(1.0),
                    "shape": "step",
                    "params_json": json.dumps(
                        {
                            "ottawa_class": cls,
                            "ottawa_bearing_id": bearing_id,
                            "ottawa_state": state,
                            "ottawa_state_label": state_label,
                            "nominal_speed_rpm": nominal_rpm,
                            "nominal_load_units": nominal_load,
                        }
                    ),
                }
            )

    long = coerce_long(pd.DataFrame(long_records)) if long_records else empty_long()
    features = coerce_features(pd.DataFrame(feature_rows))
    fault_log = coerce_fault_log(pd.DataFrame(fault_rows)) if fault_rows else empty_fault_log()
    events = empty_events()

    meta: dict[str, Any] = {
        "doi": "10.17632/y2px5tg92h.5",
        "dataset_name": "UORED-VAFCLS",
        "licence": "CC BY 4.0",
        "source_url": "https://data.mendeley.com/datasets/y2px5tg92h/5",
        "raw_dir": str(raw_dir),
        "n_files": len(files),
        "n_bearings": len(bearings_seen),
        "fs_hz": FS_HZ,
        "record_s": RECORD_S,
        "window_lengths_s": list(WINDOW_LENGTHS_S.values()),
        "bearing_geometry": {
            "n_elements": GEOMETRY.n_elements,
            "ball_diameter_m": GEOMETRY.ball_diameter_m,
            "pitch_diameter_m": GEOMETRY.pitch_diameter_m,
            "contact_angle_deg": GEOMETRY.contact_angle_deg,
            "assumption": "ER16K-class deep-groove ball bearing; not re-verified against "
            "UORED-VAFCLS's own descriptor this session (unreachable offline) - reused from "
            "tests/test_features_vibration.py for one consistent assumption across the repo.",
        },
        "columns_used": ["Accelerometer", "Speed", "Load", "Temperature Difference"],
        "columns_ignored": ["Acoustic (raw mic, not feature-ized in this pass)"],
        "formats_ignored": [".mat (byte-identical to the .csv per the download manifest)"],
        "accelerometer_units": "raw file is in g's (confirmed by MANIFEST.json's inspection "
        "note); converted to m/s2 (x9.80665, see G_MS2) before any feature or long-table "
        "signal is derived, to match nebulax.schema.SIGNAL_SPECS unit='m/s2' for vib_rms/"
        "vib_bpfo and nebulax.sim.bearing's own m/s2-calibrated vib_rms for the same signal. "
        "Absolute magnitudes still differ sharply from the simulator's healthy floor "
        "(~0.6-2.6 m/s2) even after unit conversion - this rig's accelerometer is mounted "
        "directly on a small lab bearing housing at 42 kHz broadband, not on a 12 t axle box "
        "through a heavier structural path, so a large real-vs-simulated magnitude gap is "
        "expected on top of the unit fix, not evidence of a remaining unit bug.",
        "temp_glitches_rejected": int(temp_glitches_rejected),
        "ac_coupling": "vib_rms/vib_crest (long) and the rms_ac/crest_ac feature columns are "
        "computed on x - mean(x) per window (nebulax.features.stats.window_stats(ac_couple="
        "True)); vib_kurt/kurtosis are central by definition and unaffected. 21 of the 60 "
        "records carry a DC bias larger than their own AC RMS (worst H_2_0: 1606 m/s2 vs an "
        "AC RMS of 56), which DC-coupled inflates their RMS by 1.5-44x (worst H_17_0, 43.9x) "
        "and drives the crest factor to ~1 (H_17_0: 1.09 DC vs 4.19 AC) - an accelerometer "
        "bias, not vibration. The bias is "
        "not hidden: the DC-coupled rms/crest columns stay in the feature table beside the AC "
        "pair, every window's offset is the dc_offset_ms2 (== mean) column, and the "
        "whole-record offsets are in meta['dc_offset_ms2']. Matches "
        "scripts/calibrate_bearing.py, which was AC-coupled from the start; see "
        "docs/parameters.md bearing section 4.",
        "dc_offset_ms2": {run: float(v) for run, v in dc_offsets.items()},
        "renames": {"T_box": "outlier-rejected window-mean of 'Temperature Difference' (degC, "
                    "see _robust_temp_mean/_TEMP_PLAUSIBLE_C); NOT a calibrated absolute "
                    "axle-box temperature - see module docstring caveat",
                    "vib_bpfo": "summed BPFO-harmonic envelope-spectrum amplitude "
                    "(single-sided 2|E|/n, m/s2) from "
                    "nebulax.features.vibration.envelope_spectrum_feats under the assumed "
                    "ER16K geometry above, computed on the m/s2-converted accelerometer signal"},
        "corrections_to_rail_phm": [
            "This adapter loads UORED-VAFCLS (y2px5tg92h), the constant-speed ablation member "
            "of the 'variable-speed' Ottawa bearing family - docs/research/datasets.md's own "
            "term for it. docs/research/rail_phm.md SS4.3.4's description of a 1024-CPR "
            "encoder channel at 200 kHz with speed sweeping 13.7-28.9 Hz WITHIN a single 10 s "
            "record is about a DIFFERENT, separate Ottawa dataset ('uOttawa bearing vibration "
            "under time-varying rotational speed', DOI 10.17632/v43hmbwxpm.2, [R118]) that "
            "rail_phm.md promotes to primary for the bearing subsystem and that "
            "configs/model_ladder.yaml's bearing.datasets list carries as the separate id "
            "'ottawa_variable_speed' alongside this dataset's own id 'ottawa_uored_vafcls'. "
            "rail_phm.md is not self-contradictory on this point. That primary dataset "
            "(v43hmbwxpm) has NOT been downloaded to this repo - no data/raw/ottawa_variable_"
            "speed directory and no adapter for it exist yet; flagged here for whoever owns "
            "data acquisition next, not silently worked around.",
            "Separately, and correctly attributed to THIS dataset (UORED-VAFCLS): the 60 "
            "files actually present here have no encoder/tachometer channel, are sampled at "
            "42 kHz, and carry a single nominal RPM per recording (row 0 only; the rest of "
            "the Speed/Load columns are exactly 0.0) - so every window in this adapter is "
            "necessarily tacholess, and rail_phm.md SS4.3.3's separate note that this family "
            "'carries no temperature channel' does not describe UORED-VAFCLS either, which "
            "does carry a 'Temperature Difference' column (see T_box caveat above).",
        ],
        "timestamp_epoch": _EPOCH.isoformat(),
        "record_spacing_s": _RECORD_SPACING_S,
        "timestamps_are_synthetic": True,
        "labels": "hard-coded from the file-name health state (Class/state), not from a "
        "run-to-failure trace; t_failure/rul_s are unknown for every row",
        "groups_for_cv": "train_id (bearing_01..bearing_20) - bearing-wise split required, "
        "per rail_phm.md 3.4/4.3.4",
        "sample_count_anomalies": sample_count_anomalies,
    }

    return Dataset(long=long, features=features, fault_log=fault_log, events=events, meta=meta)
