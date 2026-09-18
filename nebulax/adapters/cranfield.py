"""Cranfield linear-actuator adapter (door proxy) - REAL file layout.

Public contract: :func:`load(raw_dir) -> nebulax.schema.Dataset`, per
``nebulax/adapters/__init__.py``. Everything else in this module is private.

Provenance
----------
Cranfield Online Research Data (CORD), C. Ruiz-Carcel and A. Starr, *"Data set for
'Data-based Detection and Diagnosis of Faults in Linear Actuators'"*, Through-Life
Engineering Services Institute, Cranfield University, DOI ``10.17862/cranfield.rd.5097649``.
The files arrived on 2026-09-15 together with the official **"Data description.pdf"**, which
is the primary source for everything asserted below; read it with ``pdftotext -layout``.
Real train doors are electro-mechanical (motor + lead/ball screw), so this bench-rig linear
actuator is our door-drivetrain proxy [rail_phm.md 1.1, plan "Door <-> Cranfield"].

The real layout (verified against the 13 delivered files)
---------------------------------------------------------
::

    data/raw/cranfield/
        Normal.mat              LackLubrication1.mat   Backlash1.mat   Spalling1..8.mat
        Data description.pdf    LackLubrication2.mat   Backlash2.mat

Thirteen ``.mat`` files, one per condition (PDF section 4, Fig. 7). Each holds **60 matrices**
- 2 motion profiles x 3 loads x 10 repetitions - except ``Backlash1.mat``, which has **59**
(``backtrap1st40kg`` has only 9 repetitions; ``backtrap1st40kg2`` is absent from the release).
Every matrix is ``(2000, 3) float64`` = **80 s at 25 Hz** (PDF section 2: "All the data was
acquired at 25 Hz"; section 4: "0.04 s intervals").

Columns, per PDF section 4 and Fig. 6, in this order:

==== ============================================= =======================================
col  channel                                        this adapter emits
==== ============================================= =======================================
0    position **set point** (mm)                    ``pos_ref`` (m)
1    position **error** = set point - measurement    (not emitted directly; see below)
2    **motor current** (A)                          ``current`` (A)
==== ============================================= =======================================

There is no measured-position column: the rig logs the *error*, so
``pos = pos_ref - pos_err`` reconstructs the potentiometer reading exactly (Vishay REC 115L
linear potentiometer, PDF section 2). ``vel`` is ``np.gradient(pos) * fs`` - derived, not
measured. Millimetres are converted to metres; the schema's door channels are SI.

**There is no voltage channel**, which bounds the healthy fit: see "What this costs the
calibration" below.

Variable naming (PDF Fig. 8)
----------------------------
``<class><profile><level><load><rep>``:

* ``class``  - ``train`` (normal/healthy), ``back`` (backlash), ``lub`` (lack of lubrication),
  ``point`` (spalling - the PDF's Fig. 6 legend calls it "point defect").
* ``profile`` - ``sin`` (sinusoidal: 120 mm stroke in 6 s, 2 s wait at both ends) or
  ``trap`` (trapezoidal: 120 mm stroke in 5 s, 3 s wait at both ends). Either way one
  out-and-back sequence takes 16 s and the **full sequence is repeated 5 times per test**, so
  each 2000-sample matrix holds 5 out-and-back sequences = 10 strokes.
* ``level``  - ``1st``..``8th``, the degradation stage; **absent for ``train``** (normal has
  no stage).
* ``load``   - ``20kg``, ``40kg`` or ``neg40kg``, the external load in kgf applied by the
  second actuator through the load cell; ``neg40kg`` is the same 40 kgf *opposing* the
  motion, and is emitted as ``load_kg = -40``.
* ``rep``    - the test repetition, 1..10.

Examples: ``lubsin1st20kg1``, ``trainsin20kg3``, ``pointtrap8thneg40kg10``.

Rig facts (PDF section 2), carried into ``meta`` and used by the door calibration: ball screw
RM1605-C7 with **5 mm lead**, Nema 34 stepper motor with 4.6 N.m holding torque, 120 mm
stroke, max +/-40 kgf external load, position by linear potentiometer, current by a Honeywell
CSLA2CD Hall-effect sensor. Every test was preceded by ~30 min of running so the motor and nut
reached a steady temperature, which is why the healthy files are usable as one population.

Faults (PDF section 3)
----------------------
* **lack of lubrication** - stage 1: lubricant removed with degreaser (the PDF notes "no
  dramatic changes... due to the inherent low friction of the ball-screw architecture");
  stage 2: the bolts holding the nut's plastic seals tightened to create more friction.
* **spalling** - **8 stages**: a 1 mm surface defect on the screw raceway grown to 2, 3 and
  4 mm (stages 2-4), replicated into a neighbouring channel (5), both enlarged through the
  sidewall between them (6), a third 4 mm defect on the other neighbour (7), and part of a
  sidewall removed (8).
* **backlash** - 2 stages: the original 3.15 mm balls replaced by 3.0 mm (stage 1) and 2.5 mm
  (stage 2) balls.

THE MOTOR IS A STEPPER - read this before trusting any current number
---------------------------------------------------------------------
The door simulator (:mod:`nebulax.sim.door`) models a **PMDC** drive, where the quasi-static
current is proportional to the load torque (``i = F / (eta k_t G / r)``). The Cranfield rig
uses a **Nema 34 stepper** (PDF section 2). A stepper is driven by a chopper current
regulator at a commanded phase current, so as long as it does not lose synchronism its current
is governed mainly by the drive's set point and only weakly by the mechanical load; the
measured Hall-effect current is a *drive-level* quantity, not a torque-proportional one.

The consequence is concrete and measurable in these very files: the healthy rig draws
**0.41 A standing still against 0.87 A while travelling**, so roughly half the cruise current
carries no load information at all, and the faulty/healthy cruise-current ratios this adapter
feeds the calibration come out at only **1.17 (lubrication stage 1) and 1.25 (stage 2)** - a
ratio compressed towards 1 by that standing offset. Every faulty/healthy current ratio derived
from this dataset is therefore a **proxy** for the simulator's torque-proportional observable,
not the same quantity, and a gain read off a PMDC inversion curve with a mean-current ratio is
a **lower bound** on the real force change. ``scripts/calibrate_door.py`` quantifies the
compression and the door section of ``docs/parameters.md`` repeats it beside every number it
touches.

What this costs the calibration (identifiability)
-------------------------------------------------
With ``(pos_ref, pos, current)`` and **no voltage**, the mechanical regression
``i = (m_eff/c_i) a + (F_c0/c_i) sgn(v) + (b0/c_i) v + i0`` identifies only the three ratios
``m_eff/c_i``, ``F_c0/c_i``, ``b0/c_i``; ``R`` and ``k_e`` are **not identifiable at all**
(they need ``V = R i + k_e omega``), so ``k_t`` cannot be separated from the mechanical
constants. On top of that, the stepper makes even those three ratios weak, because the
regression's premise - current proportional to force - is not this drive's physics.
``stroke_m``, ``v_ref`` and ``a_ref`` come straight off the reference profile and are always
identified.

Mapping to the frozen schema
----------------------------
``nebulax.schema.FAULT_TYPES["door"]``, per the approved plan's door<->Cranfield note:
``back -> backlash``, ``lub -> friction`` (lack of lubrication *is* a friction fault -
rail_phm 4.1 groups them), ``point``/spalling ``-> misalignment`` (the spalling ripple is
carried through the simulator's position-periodic misalignment term), ``train -> healthy``.

The ordinal stage maps onto the feature table's ``severity`` **ML label** as ``0.0`` for
healthy, else ``min(1.0, 0.2 + 0.2*(level - 1))``. Spalling has **8** stages, so stages 5-8
all saturate at ``1.0`` on this label - that is deliberate (the label scale is the plan's
ordinal 0.2-1.0 ML target, shared with every other adapter), and it is *not* the map the
physics calibration uses: ``scripts/calibrate_door.py`` maps the spalling stages linearly
onto ``s = level/8`` so that stage 4 -> ``s = 0.5`` and stage 8 -> ``s = 1.0``. The two maps
are deliberately different and must not be unified.

Granularity: **one run per (file, variable)**, i.e. one 80 s test recording = one ``run_id``
(``cranfield_<Stem>_<variable>``), 779 runs in total. One fault-log row per run
(``shape="step"`` - a seeded bench fixture holds its severity for the whole test, there is no
run-to-failure trajectory) and one feature-table row per stroke (extend or retract
half-cycle).

Features vs metadata
--------------------
The release's own test metadata encodes the target, so it is written under the reserved
``meta_`` prefix (:data:`nebulax.schema.METADATA_PREFIX`) and is therefore excluded from ``X``
by :func:`nebulax.schema.feature_columns`:

* ``meta_class``  - the Cranfield class name (``normal``/``lack_of_lubrication``/``backlash``/
  ``spalling``); a relabelling of ``fault_type``.
* ``meta_level``  - the ordinal degradation stage; a bijection of the ``severity`` label.
* ``meta_test_id`` - ``<Stem>:<variable>``, the recording identity, for grouping splits
  (leave-one-test-out) without parsing ``run_id``.
* ``meta_motion_profile`` - ``sinusoidal``/``trapezoidal``, the commanded profile. It is a
  legitimate operating condition, but it is a *string* describing the test matrix rather than
  a measured quantity, and the calibration stratifies by it, so it is carried as metadata.
* ``meta_rep`` - the repetition index 1..10, a counter of the test matrix, never an input.

``load_kg`` stays a **feature**: the opposing/aiding load (kgf, negative = opposing) is an
operating condition a real door controller knows at inference time, it is applied across every
class of the test matrix, and it is exactly the confounder a load-aware model must handle. So
are the per-stroke signal statistics from :func:`_stroke_features`.

No wall clock exists in the source, so timestamps are synthetic: each run gets its own
1-hour-spaced slot from a fixed epoch (recorded in ``meta``), purely to keep runs from
overlapping in time; do not read any calendar meaning into them.

Missing files are a WARNING, not an error: :func:`load` logs every one of the 13 expected
files it cannot find and parses whatever is present, so a partial download still produces a
usable (and honestly labelled) Dataset.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from nebulax import schema as S

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

DOI: Final[str] = "10.17862/cranfield.rd.5097649"
SOURCE: Final[str] = "cranfield"
SUBSYSTEM: Final[str] = "door"
TRAIN_ID: Final[str] = "cranfield_rig"
COMPONENT_ID: Final[str] = "door_L1"  # generic bench rig, not tied to any specific fleet door
CAR: Final[int] = 0

#: Verified from the PDF (section 2 "All the data was acquired at 25 Hz"; section 4 "0.04 s
#: intervals") and from the file geometry (2000 samples = 80 s = 5 x 16 s sequences).
FS_HZ: Final[float] = 25.0
#: Nominal samples per matrix in the release. Files are parsed whatever their length; this is
#: only used to flag a surprise in ``meta``.
NOMINAL_SAMPLES: Final[int] = 2000
#: Rig geometry, PDF section 2. Carried into ``meta`` so the calibration need not re-read the PDF.
STROKE_MM: Final[float] = 120.0
SCREW_LEAD_MM: Final[float] = 5.0
BALL_DIAM_MM: Final[dict[str, float]] = {"original": 3.15, "1st": 3.0, "2nd": 2.5}
MOTOR: Final[str] = "Nema 34 stepper, 4.6 N.m holding torque (NOT a PMDC drive)"
SCREW: Final[str] = "RM1605-C7 anti-backlash ball nut, 5 mm lead"

EPOCH: Final[pd.Timestamp] = pd.Timestamp("2000-01-01T00:00:00Z")  # synthetic, see module docstring

#: The licence is not stated in the delivered files (neither in "Data description.pdf" nor in
#: any sidecar); only the CORD landing page carries it, and that page is behind a Cloudflare
#: challenge. UNVERIFIED until somebody opens the DOI in a real browser.
LICENCE: Final[str] = (
    "UNVERIFIED - the delivered files (13 .mat + 'Data description.pdf') state no licence "
    "anywhere; the CORD landing page that would carry it is behind a Cloudflare challenge. "
    f"Check https://doi.org/{DOI} in a browser before redistributing."
)

CITATION: Final[str] = (
    "C. Ruiz-Carcel and A. Starr, \"Data set for 'Data-based Detection and Diagnosis of Faults "
    "in Linear Actuators'\", Through-Life Engineering Services Institute, Cranfield University, "
    f"CORD, DOI {DOI}."
)

# --------------------------------------------------------------------------------------
# The real layout
# --------------------------------------------------------------------------------------

#: Variable-name class token -> canonical ``FAULT_TYPES['door']`` entry (plan's door map).
CLASS_FAULT_TYPE: Final[dict[str, str]] = {
    "train": "healthy",
    "back": "backlash",
    "lub": "friction",
    "point": "misalignment",  # "point defect" = spalling, PDF Fig. 6 legend
}

#: Variable-name class token -> the dataset's own class name (used in ``meta`` and messages).
CLASS_NAME: Final[dict[str, str]] = {
    "train": "normal",
    "back": "backlash",
    "lub": "lack_of_lubrication",
    "point": "spalling",
}

#: The 13 files of the release -> (class token, degradation stage). Stage 0 = no stage.
EXPECTED_FILES: Final[dict[str, tuple[str, int]]] = {
    "Normal.mat": ("train", 0),
    "Backlash1.mat": ("back", 1),
    "Backlash2.mat": ("back", 2),
    "LackLubrication1.mat": ("lub", 1),
    "LackLubrication2.mat": ("lub", 2),
    **{f"Spalling{i}.mat": ("point", i) for i in range(1, 9)},
}

#: Number of spalling stages in the release (PDF section 3 / Fig. 4).
N_SPALLING_STAGES: Final[int] = 8

#: ``<class><profile><level><load><rep>``; ``level`` absent for the normal class.
VAR_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<cls>train|back|lub|point)"
    r"(?P<profile>sin|trap)"
    r"(?P<level>1st|2nd|3rd|[4-8]th)?"
    r"(?P<load>neg40kg|40kg|20kg)"
    r"(?P<rep>\d{1,2})$"
)

_ORDINAL_LEVEL: Final[dict[str, int]] = {
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6, "7th": 7, "8th": 8,
}

_PROFILE_NAME: Final[dict[str, str]] = {"sin": "sinusoidal", "trap": "trapezoidal"}
#: Motion timing per profile, PDF section 3: (stroke time s, wait-at-each-end s).
PROFILE_TIMING_S: Final[dict[str, tuple[float, float]]] = {"sin": (6.0, 2.0), "trap": (5.0, 3.0)}
#: Sequences (out-and-back) per test recording, PDF section 3.
SEQUENCES_PER_TEST: Final[int] = 5

_LOAD_KGF: Final[dict[str, float]] = {"20kg": 20.0, "40kg": 40.0, "neg40kg": -40.0}

#: Minimum samples for a stroke to be kept as a feature row.
_MIN_STROKE_SAMPLES: Final[int] = 4
#: Reference-derivative dead band as a fraction of the p95 step: below it the rig is idle.
#: The set point is a *commanded* trajectory with exact zeros during the 2-3 s end waits, so
#: the band only has to clear zero - and it must stay small, because the sinusoidal profile
#: spends its first and last half-second below any generous threshold. At 0.25 it clipped
#: 3.8 mm off each end of every sinusoidal stroke (116.2 mm measured against the rig's real
#: 120 mm); at 0.05 both profiles come back at the full stroke.
_STROKE_DEAD_FRAC: Final[float] = 0.05


def severity_from_level(level: int, fault_type: str) -> float:
    """The ML label: ``0.0`` healthy, else ``min(1.0, 0.2 + 0.2*(level-1))``.

    Spalling has :data:`N_SPALLING_STAGES` = 8 stages, so stages 5-8 all saturate at 1.0 on
    this scale. That is intentional - see the module docstring; the *physics* map used by
    ``scripts/calibrate_door.py`` is a different, linear one.
    """
    if fault_type == "healthy":
        return 0.0
    return float(min(1.0, 0.2 + 0.2 * (level - 1)))


@dataclass(frozen=True, slots=True)
class _Test:
    """One 80 s recording = one matrix inside one ``.mat`` file = one ``run_id``."""

    file: str
    variable: str
    fault_type: str
    cranfield_class: str
    level: int
    severity: float
    motion_profile: str
    load_kg: float
    rep: int


def parse_variable(name: str, file_stem: str) -> _Test | None:
    """Parse one ``.mat`` variable name into its test metadata, or ``None`` if it is not one."""
    m = VAR_RE.match(name)
    if m is None:
        return None
    cls = m.group("cls")
    fault_type = CLASS_FAULT_TYPE[cls]
    level = _ORDINAL_LEVEL[m.group("level")] if m.group("level") else 0
    return _Test(
        file=file_stem,
        variable=name,
        fault_type=fault_type,
        cranfield_class=CLASS_NAME[cls],
        level=level,
        severity=severity_from_level(level, fault_type),
        motion_profile=_PROFILE_NAME[m.group("profile")],
        load_kg=_LOAD_KGF[m.group("load")],
        rep=int(m.group("rep")),
    )


# --------------------------------------------------------------------------------------
# Channel extraction
# --------------------------------------------------------------------------------------

#: Column order inside every matrix (PDF section 4 / Fig. 6).
COLUMNS: Final[tuple[str, str, str]] = ("pos_ref_mm", "pos_err_mm", "current_a")
_MM_TO_M: Final[float] = 1.0e-3


def extract_channels(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """``(n, 3)`` matrix -> SI channels ``pos_ref, pos_err, pos, current, vel``.

    Position arrives in millimetres and is converted to metres. The rig logs the *error*
    (set point minus measurement), so the measured position is reconstructed as
    ``pos = pos_ref - pos_err``; ``vel`` is differentiated from it and is therefore derived,
    not measured.
    """
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(
            f"cranfield adapter: expected an (n, 3) matrix [set point mm, error mm, current A] "
            f"per the dataset's 'Data description.pdf' section 4; got shape {a.shape}"
        )
    pos_ref = a[:, 0] * _MM_TO_M
    pos_err = a[:, 1] * _MM_TO_M
    pos = pos_ref - pos_err
    current = a[:, 2]
    vel = np.gradient(pos) * FS_HZ if pos.size > 1 else np.zeros_like(pos)
    return {"pos_ref": pos_ref, "pos_err": pos_err, "pos": pos, "current": current, "vel": vel}


def segment_strokes(pos_ref: np.ndarray) -> list[tuple[int, int]]:
    """Split a set-point trace into monotone extend/retract strokes, dropping the end waits.

    The commanded profile is piecewise constant-or-moving (PDF section 3: 5 s / 6 s of travel
    with 3 s / 2 s held at each end), so a dead band at :data:`_STROKE_DEAD_FRAC` of the p95
    step separates travel from wait cleanly and vectorised. Idle samples are *dropped*, not
    absorbed: a 3 s pause welded onto a 5 s stroke would corrupt every duration and current
    mean taken from it.
    """
    x = np.asarray(pos_ref, dtype=np.float64)
    if x.size < _MIN_STROKE_SAMPLES:
        return []
    d = np.diff(x)
    absd = np.abs(d)
    peak = float(np.percentile(absd[absd > 0], 95)) if np.any(absd > 0) else 0.0
    if peak <= 0.0:
        return []
    dead = _STROKE_DEAD_FRAC * peak
    sgn = np.where(d > dead, 1, np.where(d < -dead, -1, 0)).astype(np.int8)
    if not np.any(sgn):
        return []
    breaks = np.flatnonzero(np.diff(sgn) != 0) + 1
    edges = [0, *breaks.tolist(), sgn.size]
    out: list[tuple[int, int]] = []
    for a, b in zip(edges[:-1], edges[1:]):
        if sgn[a] == 0:
            continue
        lo, hi = int(a), int(min(b + 1, x.size))  # d[i] spans x[i]..x[i+1]
        if hi - lo >= _MIN_STROKE_SAMPLES:
            out.append((lo, hi))
    return out


def _stroke_features(
    seg: tuple[int, int], ch: dict[str, np.ndarray], fs: float
) -> dict[str, float]:
    s, e = seg
    n = e - s
    cur = ch["current"][s:e]
    pos_ref = ch["pos_ref"][s:e]
    err = ch["pos_err"][s:e]
    k = max(1, min(5, cur.size))
    return {
        "n_samples": float(n),
        "duration_s": float(n / fs),
        "i_mean": float(np.mean(np.abs(cur))),
        "i_rms": float(np.sqrt(np.mean(cur**2))),
        "i_peak": float(np.max(np.abs(cur))),
        "i_start_peak": float(np.max(np.abs(cur[:k]))),
        "pos_ref_start": float(pos_ref[0]),
        "pos_ref_end": float(pos_ref[-1]),
        "pos_ref_range": float(pos_ref.max() - pos_ref.min()),
        "direction": float(np.sign(pos_ref[-1] - pos_ref[0])),
        "v_ref_mps": float(abs(pos_ref[-1] - pos_ref[0]) * fs / max(n - 1, 1)),
        "pos_err_max": float(np.max(np.abs(err))),
        "pos_err_rms": float(np.sqrt(np.mean(err**2))),
    }


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------


def discover(raw_dir: Path) -> tuple[list[Path], list[str]]:
    """The 13 expected ``.mat`` files that are present, and the names of those that are not.

    Discovery is by the release's own filenames (:data:`EXPECTED_FILES`), case-insensitively,
    because the condition and its stage are encoded there and nowhere else; any *other*
    ``.mat`` in the directory is reported as unexpected rather than silently parsed, since its
    class would be unknown.
    """
    present: dict[str, Path] = {}
    for p in sorted(raw_dir.rglob("*.mat")):
        if p.is_file() and not p.name.startswith("."):
            key = next((e for e in EXPECTED_FILES if e.lower() == p.name.lower()), None)
            if key is not None and key not in present:
                present[key] = p
    found = [present[k] for k in EXPECTED_FILES if k in present]
    missing = [k for k in EXPECTED_FILES if k not in present]
    return found, missing


def _expected_layout_message(raw_dir: Path, *, reason: str) -> str:
    return (
        f"nebulax.adapters.cranfield.load: raw_dir {raw_dir} {reason}.\n"
        "Expected layout (the real CORD release, verified 2026-09-15 against its own "
        "'Data description.pdf'):\n"
        + "".join(f"  {raw_dir}/{name}\n" for name in EXPECTED_FILES)
        + "  (plus 'Data description.pdf', which this adapter does not need at run time)\n"
        "Each .mat holds 60 matrices named <class><profile><level><load><rep> "
        "(e.g. trainsin20kg3, lubtrap2ndneg40kg7), every one (2000, 3) float64 = 80 s at 25 Hz, "
        "columns [position set point mm, position error mm, motor current A].\n"
        "Missing files are only a WARNING - load() parses whatever is present. This error means "
        "there is nothing at all to parse.\n"
        "If raw_dir contains a MANIFEST.json recording a partial/failed download, load() returns "
        "an empty, schema-valid Dataset instead of raising - see "
        "nebulax.adapters.cranfield.__doc__."
    )


def _empty_dataset_from_manifest(raw_dir: Path, manifest: dict[str, Any]) -> S.Dataset:
    """No ``.mat`` files, but a MANIFEST.json explains why. Returning an honest, schema-valid,
    empty Dataset here - rather than raising - keeps the validate CLI able to report the real
    situation instead of crashing on an already-diagnosed missing-data condition."""
    return S.Dataset(
        long=S.empty_long(),
        features=S.empty_features(),
        fault_log=S.empty_fault_log(),
        events=S.empty_events(),
        meta={
            "doi": DOI,
            "citation": CITATION,
            "licence": manifest.get("licence", LICENCE),
            "source": SOURCE,
            "subsystem": SUBSYSTEM,
            "n_files": 0,
            "n_runs": 0,
            "missing_files": list(EXPECTED_FILES),
            "download_status": manifest.get("status", "unknown"),
            "download_notes": manifest.get("notes", []),
            "download_errors": manifest.get("errors", []),
            "unverified": [
                f"None of the 13 expected Cranfield .mat files were found under {raw_dir}. "
                f"{raw_dir / 'MANIFEST.json'} records a non-ok download status - see its "
                "'notes'/'errors' above. This is a deliberate empty Dataset, not a crash: place "
                "the release's 13 .mat files under this directory and re-run load()."
            ],
        },
    )


# --------------------------------------------------------------------------------------
# public contract
# --------------------------------------------------------------------------------------

_SIGNALS: Final[tuple[str, ...]] = ("pos_ref", "pos", "current", "vel")


def _long_block(tests: list[_Test], chans: list[dict[str, np.ndarray]], t0: list[pd.Timestamp]) -> pd.DataFrame:
    """One long-table block for a whole file, built by concatenation rather than per-run
    DataFrames: 60 matrices x 4 signals would otherwise be 240 frames to concatenate per file."""
    ts = np.concatenate(
        [
            (t.value + (np.arange(chans[i]["pos"].size, dtype=np.int64) * int(1e9 / FS_HZ)))
            for i, t in enumerate(t0)
        ]
    )
    run = np.repeat(
        np.array([f"cranfield_{t.file}_{t.variable}" for t in tests], dtype=object),
        [c["pos"].size for c in chans],
    )
    n = ts.size
    values = np.concatenate(
        [np.concatenate([c[sig] for c in chans]).astype(np.float32) for sig in _SIGNALS]
    )
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(np.tile(ts, len(_SIGNALS)), unit="ns", utc=True),
            "source": SOURCE,
            "run_id": pd.Categorical(np.tile(run, len(_SIGNALS))),
            "train_id": TRAIN_ID,
            "car": np.int8(CAR),
            "subsystem": SUBSYSTEM,
            "component_id": COMPONENT_ID,
            "signal": pd.Categorical(np.repeat(np.array(_SIGNALS, dtype=object), n)),
            "value": values,
        }
    )


def load(raw_dir: Path) -> S.Dataset:
    """Load the real Cranfield linear-actuator fault set as a door-proxy Dataset.

    One run per (file, ``.mat`` variable) - one 80 s recording of 5 out-and-back sequences -
    with one feature row per stroke carrying ``load_kg`` (negative = opposing load) and the
    per-stroke signal statistics as features, the test matrix as ``meta_*`` metadata
    (``meta_class``, ``meta_level``, ``meta_test_id``, ``meta_motion_profile``, ``meta_rep``)
    and the ``fault_type``/``severity`` labels.

    Any of the 13 expected files that is absent is logged as a **WARNING** and skipped;
    ``load`` still succeeds on the rest. It raises ``FileNotFoundError`` only if ``raw_dir``
    does not exist, or exists with neither any expected ``.mat`` file nor a ``MANIFEST.json``
    explaining their absence.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise FileNotFoundError(_expected_layout_message(raw_dir, reason="does not exist"))

    files, missing = discover(raw_dir)
    for name in missing:
        LOGGER.warning(
            "nebulax.adapters.cranfield: expected release file %s is missing from %s "
            "(condition %s stage %s) - continuing without it",
            name,
            raw_dir,
            CLASS_NAME[EXPECTED_FILES[name][0]],
            EXPECTED_FILES[name][1] or "-",
        )
    if not files:
        manifest_path = raw_dir / "MANIFEST.json"
        if manifest_path.exists():
            LOGGER.warning(
                "nebulax.adapters.cranfield: none of the 13 release files found under %s; "
                "returning an empty Dataset explained by %s",
                raw_dir,
                manifest_path,
            )
            return _empty_dataset_from_manifest(raw_dir, json.loads(manifest_path.read_text()))
        raise FileNotFoundError(
            _expected_layout_message(raw_dir, reason="contains none of the 13 expected .mat files")
        )

    from scipy.io import loadmat

    long_blocks: list[pd.DataFrame] = []
    feature_rows: list[dict[str, Any]] = []
    fault_rows: list[dict[str, Any]] = []
    all_tests: list[_Test] = []
    unexpected_vars: list[str] = []
    odd_lengths: dict[str, int] = {}
    run_idx = 0

    for path in files:
        stem = path.stem
        raw: dict[str, Any] = loadmat(path)
        tests: list[_Test] = []
        chans: list[dict[str, np.ndarray]] = []
        t0: list[pd.Timestamp] = []
        for var in sorted(k for k in raw if not k.startswith("__")):
            meta_t = parse_variable(var, stem)
            if meta_t is None:
                unexpected_vars.append(f"{path.name}:{var}")
                continue
            arr = np.asarray(raw[var])
            if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < _MIN_STROKE_SAMPLES:
                unexpected_vars.append(f"{path.name}:{var} (shape {arr.shape})")
                continue
            if arr.shape[0] != NOMINAL_SAMPLES:
                odd_lengths[f"{path.name}:{var}"] = int(arr.shape[0])
            ch = extract_channels(arr)
            tests.append(meta_t)
            chans.append(ch)
            t0.append(EPOCH + pd.Timedelta(hours=run_idx))
            run_idx += 1

        if not tests:
            LOGGER.warning(
                "nebulax.adapters.cranfield: %s held no matrix with a parsable "
                "<class><profile><level><load><rep> name - skipped",
                path.name,
            )
            continue

        long_blocks.append(_long_block(tests, chans, t0))
        all_tests.extend(tests)

        for t, ch, start in zip(tests, chans, t0):
            run_id = f"cranfield_{t.file}_{t.variable}"
            n = ch["pos"].size
            ts = start + pd.to_timedelta(np.arange(n) / FS_HZ, unit="s")
            for cyc_id, seg in enumerate(segment_strokes(ch["pos_ref"])):
                s, e = seg
                feature_rows.append(
                    {
                        "run_id": run_id,
                        "source": SOURCE,
                        "train_id": TRAIN_ID,
                        "car": np.int8(CAR),
                        "subsystem": SUBSYSTEM,
                        "component_id": COMPONENT_ID,
                        "cycle_id": np.int64(cyc_id),
                        "t_start": ts[s],
                        "t_end": ts[e - 1],
                        "fault_type": t.fault_type,
                        "severity": np.float32(t.severity),
                        "is_faulty": bool(t.fault_type != "healthy"),
                        "rul_s": np.float32(np.nan),
                        "alarm_window_3d": False,
                        # meta_*: describes the test, never a model input (schema.feature_columns)
                        "meta_class": t.cranfield_class,
                        "meta_level": np.int64(t.level),
                        "meta_test_id": f"{t.file}:{t.variable}",
                        "meta_motion_profile": t.motion_profile,
                        "meta_rep": np.int64(t.rep),
                        # operating condition the controller knows -> a legitimate feature
                        "load_kg": np.float32(t.load_kg),
                        **_stroke_features(seg, ch, FS_HZ),
                    }
                )
            fault_rows.append(
                {
                    "run_id": run_id,
                    "train_id": TRAIN_ID,
                    "car": np.int8(CAR),
                    "subsystem": SUBSYSTEM,
                    "component_id": COMPONENT_ID,
                    "fault_type": t.fault_type,
                    "t_onset": ts[0],
                    "t_failure": pd.NaT,
                    "t_functional_failure": pd.NaT,
                    "gamma": np.float32(np.nan),
                    "shape": "step",
                    "params_json": json.dumps(
                        {
                            "source_file": path.name,
                            "variable": t.variable,
                            "cranfield_class": t.cranfield_class,
                            "level": t.level,
                            "severity_label": t.severity,
                            "motion_profile": t.motion_profile,
                            "load_kg": t.load_kg,
                            "rep": t.rep,
                            "note": (
                                "seeded bench fixture; severity constant for the whole 80 s test, "
                                "no run-to-failure"
                            ),
                        }
                    ),
                }
            )

    long = S.coerce_long(pd.concat(long_blocks, ignore_index=True))
    features = S.coerce_features(pd.DataFrame(feature_rows))
    fault_log = S.coerce_fault_log(pd.DataFrame(fault_rows))

    counts: dict[str, int] = {}
    per_file: list[dict[str, Any]] = []
    for path in files:
        sub = [t for t in all_tests if t.file == path.stem]
        if not sub:
            continue
        per_file.append(
            {
                "file": path.name,
                "cranfield_class": sub[0].cranfield_class,
                "level": sub[0].level,
                "fault_type": sub[0].fault_type,
                "n_tests": len(sub),
            }
        )
    for t in all_tests:
        counts[t.fault_type] = counts.get(t.fault_type, 0) + 1

    meta: dict[str, Any] = {
        "doi": DOI,
        "citation": CITATION,
        "licence": LICENCE,
        "source": SOURCE,
        "subsystem": SUBSYSTEM,
        "n_files": len(files),
        "n_runs": len(all_tests),
        "missing_files": missing,
        "unexpected_variables": unexpected_vars,
        "non_nominal_lengths": odd_lengths,
        "fs_hz": FS_HZ,
        "component_id": COMPONENT_ID,
        "train_id": TRAIN_ID,
        "fault_class_counts": counts,
        "files": per_file,
        "rig": {
            "screw": SCREW,
            "motor": MOTOR,
            "stroke_mm": STROKE_MM,
            "screw_lead_mm": SCREW_LEAD_MM,
            "ball_diameter_mm": BALL_DIAM_MM,
            "loads_kgf": sorted(_LOAD_KGF.values()),
            "profiles_stroke_wait_s": PROFILE_TIMING_S,
            "sequences_per_test": SEQUENCES_PER_TEST,
            "warm_up": "~30 min of running before every test (PDF section 3)",
        },
        "channels": {
            "pos_ref": "position set point, mm in file -> m here (PDF section 4, Fig. 6a)",
            "pos": "pos_ref - pos_err; the rig logs the error, not the measurement",
            "current": "motor current, A, Honeywell CSLA2CD Hall-effect sensor (Fig. 6c)",
            "vel": "DERIVED: np.gradient(pos) * 25 Hz, not a measured channel",
            "voltage": "ABSENT - no voltage channel exists in this release",
        },
        "stepper_caveat": (
            "The rig's motor is a Nema 34 STEPPER (PDF section 2), not the PMDC drive "
            "nebulax.sim.door models. A chopper-regulated stepper holds a commanded phase "
            "current, so the logged current is a drive-level quantity only weakly coupled to "
            "load torque. Every faulty/healthy current ratio taken from this dataset is a "
            "PROXY for the simulator's torque-proportional current, not the same observable."
        ),
        "unverified": [
            LICENCE,
            "Timestamps are SYNTHETIC (the source has no wall clock): each of the "
            f"{len(all_tests)} runs gets its own 1-hour-spaced slot from {EPOCH.isoformat()}. "
            "Do not read calendar meaning into them.",
            f"component_id is fixed to {COMPONENT_ID!r} for every test: this is a generic bench "
            "rig, not tied to a specific fleet door.",
            "The door<->rig mapping itself (back->backlash, lub->friction, spalling->"
            "misalignment) is the approved plan's calibration choice, not a claim that a "
            "ball-screw spall and a door-leaf misalignment are the same physics.",
        ],
    }
    if missing:
        meta["unverified"].append(
            f"{len(missing)} of the 13 release files are missing from {raw_dir}: {missing}. "
            "Every class count and ratio below is computed on what was present."
        )
    return S.Dataset(long=long, features=features, fault_log=fault_log, events=S.empty_events(), meta=meta)
