"""ACV (train air-conditioning) loader and per-car ranker features.

The ACV task ranks the 8 cars of one train by how likely each is the refrigerant-leaking car
(`docs/ps3_contract.md` section 3, ACV Info Kit section 3). Everything here is **header driven**:
the Info Kit warns that "the exact parameter set differs between case files" - one of the six
training cases carries ~60 parameters per car (and four empty cars), two spell the outdoor sensor
differently, and modes are strings. So the loader never assumes a column list; it matches
``^Car (\\d{1,2}) - <param>`` (:data:`CAR_COL_RE`, a widened `common.CAR_COL_RE`: a file spelling
its headers ``Car 1 - ...`` is read, and the id is emitted exactly as that header spells it) and
maps each file's own parameter names onto a
small canonical schema through :data:`SYNONYMS`, listing whatever it could not map in
``ACVCase.unmapped`` (shown in the explanation payload).

Canonical fields (all optional - a case is usable with ``indoor`` + ``running_mode`` alone):

==============  ==========================================================================
``indoor``      saloon temperature: *Indoor Average Temperature* / *Passenger Cabin
                Temperature Detected Value*
``setpoint``    cooling setpoint: *ACV Control Temperature (Cooling)* / *Target Temperature
                Value*
``setpoint_heat`` *ACV Control Temperature (Heating)* (kept only to detect heating season)
``outdoor``     *Outdoor Average Temperature* / *Outside Temperature Sensor Reading* /
                *Fresh Air Temperature Detected Value*
``running_mode``  *ACV Running Mode* (strings: Automatic / Full / Half Cooling, Stop(ped),
                Ventilation, Emergency Ventilation, Self-Check, Invalid)
``setting_mode``  *ACV Setting Mode* / *ACV Control Mode*
``valid``       *ACV Information Valid* ("Valid" / "Invalid")
``load_halved`` *Load Halved* / *Load Shedding*
==============  ==========================================================================

Fold-local rule: nothing in this module fits anything across cases. Every feature is computed
**inside one case file** from that file's own eight cars (peer normalisation), so a feature can
never carry information from another case, let alone from the held-out one. The only fitted
objects in the task are the ranker weights, and `nebulax.ps3.acv` fits those on the training
cases of the outer fold only.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from nebulax.ps3.common import CACHE_DIR

__all__ = [
    "FEATURE_VERSION",
    "CAR_COL_RE",
    "SYNONYMS",
    "COOLING_MODES",
    "FULL_COOLING_MODES",
    "OFF_MODES",
    "ACVCase",
    "load_case",
    "case_masks",
    "car_features",
    "RANKERS",
    "GUO_INDEX_FEATURES",
    "cached_features",
]

#: Bump when a feature definition changes (part of the cache key and of every results json).
FEATURE_VERSION = "acv-f5"

#: ``Car <n> - <parameter>``, **one or two digits**.
#:
#: `common.CAR_COL_RE` requires two (``Car 01 - ...``, which is how all seven organiser files
#: spell it). A held-out file that spells its headers ``Car 1 - ...`` would then be refused as
#: "not an ACV case file" rather than read, so the loader uses this widened pattern and keeps the
#: id **exactly as the header spells it** - ``Car 1`` stays ``"1"``, ``Car 01`` stays ``"01"`` -
#: because `acv_predictions.csv:ranked_cars` must echo the file's own car identifiers.
CAR_COL_RE: re.Pattern[str] = re.compile(r"^Car\s+(\d{1,2})\s*-\s*(.+)$")

#: A feature with at most this many distinct values across the cars of a case cannot order an
#: eight-car fleet on its own; `nebulax.ps3.acv._is_degenerate` marks such rows tie-break-only.
MIN_DISTINCT_LEVELS = 3

#: Canonical field -> parameter names seen in the organisers' files, most specific first.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "indoor": (
        "Indoor Average Temperature",
        "Passenger Cabin Temperature Detected Value",
        "Observation Area Temperature Detected Value",
    ),
    "setpoint": (
        "ACV Control Temperature (Cooling)",
        "Target Temperature Value",
    ),
    "setpoint_heat": ("ACV Control Temperature (Heating)",),
    "outdoor": (
        "Outdoor Average Temperature",
        "Outside Temperature Sensor Reading",
        "Fresh Air Temperature Detected Value",
    ),
    "running_mode": ("ACV Running Mode",),
    "setting_mode": ("ACV Setting Mode", "ACV Control Mode"),
    "valid": ("ACV Information Valid",),
    "load_halved": ("Load Halved", "Load Shedding"),
}

#: Any mode in which the compressor is expected to be making cold air (lower-cased, stripped).
COOLING_MODES: frozenset[str] = frozenset(
    {"automatic cooling", "full cooling", "half cooling", "cooling", "auto cooling"}
)

#: The hardest-working cooling states (ORNL 2024: duty in full cooling rises with undercharge).
FULL_COOLING_MODES: frozenset[str] = frozenset({"full cooling"})

#: Modes in which no cooling is demanded - excluded from every temperature feature.
OFF_MODES: frozenset[str] = frozenset(
    {"stop", "stopped", "ventilation", "emergency ventilation", "self-check", "invalid", "off"}
)

_VALID_TOKENS: frozenset[str] = frozenset({"valid", "normal", "1", "true", "yes"})
_LOAD_HALVED_TOKENS: frozenset[str] = frozenset({"halved", "load halved", "load shedding", "1", "true", "yes"})

#: Tokens the organisers use for "no reading" inside an otherwise typed column.
_NA_TOKENS: frozenset[str] = frozenset({"", "none", "nan", "null", "n/a", "na", "-", "--", "invalid"})

_ID_COLS: tuple[str, ...] = ("Car model", "Train number", "Time")


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


@dataclass
class ACVCase:
    """One case file: the canonical panel plus everything the explanation needs.

    ``panel`` maps a canonical field to a ``DataFrame`` indexed by timestamp whose columns are the
    file's own two-digit car ids (``"01"`` ... ``"08"``), so a peer comparison is one pandas call.
    Numeric fields are floats; ``running_mode`` / ``setting_mode`` are lower-cased strings.
    """

    file_id: str
    cars: list[str]
    time: pd.DatetimeIndex
    panel: dict[str, pd.DataFrame]
    unmapped: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def numeric(self, name: str) -> pd.DataFrame:
        """Numeric canonical field, all-NaN (right shape) when the file does not carry it."""
        df = self.panel.get(name)
        if df is None:
            return pd.DataFrame(np.nan, index=self.time, columns=self.cars)
        return df

    def text(self, name: str) -> pd.DataFrame:
        """String canonical field, empty strings when the file does not carry it."""
        df = self.panel.get(name)
        if df is None:
            return pd.DataFrame("", index=self.time, columns=self.cars, dtype=object)
        return df

    @property
    def dt_seconds(self) -> float:
        """Median sample interval in seconds (30 s in five cases, 10 s in ``acv_case_04``)."""
        if len(self.time) < 2:
            return 30.0
        d = pd.Series(self.time).diff().dt.total_seconds().median()
        return float(d) if np.isfinite(d) and d > 0 else 30.0


def _clean_text(s: pd.Series) -> pd.Series:
    out = s.astype("object").where(s.notna(), "")
    return out.map(lambda v: str(v).strip().lower())


def _clean_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype(float)
    txt = s.astype("string").str.strip()
    txt = txt.where(~txt.str.lower().isin(list(_NA_TOKENS)), other=pd.NA)
    return pd.to_numeric(txt, errors="coerce").astype(float)


def load_case(path: Path | str, *, sheet: Any = 0) -> ACVCase:
    """Read one ACV ``.xlsx`` into the canonical panel.

    Car headers may spell the id with one or two digits (``Car 1 - ...`` as well as
    ``Car 01 - ...``); the id is kept **verbatim** and ``ACVCase.cars`` is ordered numerically, so
    a prediction row echoes the file's own spelling.

    Raises ``ValueError`` with a readable message for an empty sheet, a missing ``Time`` column or
    a file whose headers contain no ``Car <N> - ...`` columns at all (the three ways a wrong file
    arrives on the upload page).
    """
    p = Path(path)
    try:
        raw = pd.read_excel(p, sheet_name=sheet)
    except Exception as exc:  # pragma: no cover - engine-specific message
        raise ValueError(f"{p.name}: could not be read as an Excel workbook ({exc})") from exc
    if isinstance(raw, dict):  # sheet_name=None
        raw = next(iter(raw.values()))
    if raw.empty:
        raise ValueError(f"{p.name}: the sheet is empty (no data rows)")
    time_col = next((c for c in raw.columns if str(c).strip().lower() == "time"), None)
    if time_col is None:
        raise ValueError(f"{p.name}: no 'Time' column (found {list(raw.columns)[:6]}...)")
    with warnings.catch_warnings():
        # A file whose Time column is junk is a normal thing to be handed on the upload page; the
        # clear ValueError below is the answer, not pandas' "could not infer format" warning.
        warnings.simplefilter("ignore", UserWarning)
        time = pd.to_datetime(raw[time_col], errors="coerce")
    if time.isna().all():
        raise ValueError(f"{p.name}: no parseable timestamp in column {time_col!r}")
    keep = time.notna().to_numpy()
    raw = raw.loc[keep].reset_index(drop=True)
    time = pd.DatetimeIndex(time[keep].to_numpy(), name="Time")

    by_param: dict[str, dict[str, pd.Series]] = {}
    for col in raw.columns:
        m = CAR_COL_RE.match(str(col))
        if m is None:
            continue
        car, param = m.group(1), m.group(2).strip()
        by_param.setdefault(param, {})[car] = raw[col]
    if not by_param:
        raise ValueError(
            f"{p.name}: no 'Car <N> - <parameter>' columns; this does not look like an ACV case file"
        )
    # numeric order, but the id string stays exactly as the header spells it
    cars = sorted({c for cols in by_param.values() for c in cols}, key=lambda c: (int(c), c))

    wanted = {syn.lower(): canon for canon, syns in SYNONYMS.items() for syn in syns}
    text_fields = {"running_mode", "setting_mode", "valid", "load_halved"}
    panel: dict[str, pd.DataFrame] = {}
    mapped_params: set[str] = set()
    for canon, syns in SYNONYMS.items():
        frame = pd.DataFrame(index=time, columns=cars, dtype=object)
        filled = False
        for car in cars:
            for syn in syns:  # synonyms in priority order: first one with data wins
                series = by_param.get(syn, {}).get(car)
                if series is None:
                    continue
                vals = _clean_text(series) if canon in text_fields else _clean_numeric(series)
                vals = vals.loc[keep] if len(vals) != len(time) else vals
                if canon in text_fields:
                    has = bool((vals.to_numpy() != "").any())
                else:
                    has = bool(np.isfinite(vals.to_numpy(dtype=float)).any())
                if has:
                    frame[car] = vals.to_numpy()
                    mapped_params.add(syn)
                    filled = True
                    break
        if canon in text_fields:
            frame = frame.fillna("").astype(object)
        else:
            frame = frame.apply(pd.to_numeric, errors="coerce").astype(float)
        if filled or canon in ("indoor", "running_mode"):
            panel[canon] = frame

    unmapped = sorted(set(by_param) - mapped_params, key=str.lower)
    meta: dict[str, Any] = {"n_rows": int(len(time)), "n_params": len(by_param)}
    for col in _ID_COLS[:2]:
        if col in raw.columns:
            vals = raw[col].dropna().unique()
            if len(vals):
                meta[col.lower().replace(" ", "_")] = str(vals[0])
    return ACVCase(file_id=p.name, cars=cars, time=time, panel=panel, unmapped=unmapped, meta=meta)


# --------------------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------------------


def case_masks(
    case: ACVCase, *, hot_quantile: float = 0.5, cooling_only: bool = True, steady_minutes: float = 10.0
) -> dict[str, Any]:
    """Row/car masks every feature shares.

    ``valid`` - the car's *ACV Information Valid* flag (all-True when the file has no such
    column); ``cooling`` - running mode in :data:`COOLING_MODES`; ``usable`` - valid, cooling and
    a finite indoor reading; ``hot`` - timestamps whose fleet-median outdoor temperature is at or
    above the record's ``hot_quantile`` (the "hotter half of the record"; a leak is only visible
    when the plant is loaded), falling back to the fleet-median *indoor* temperature when no
    outdoor sensor is mapped, and to every row when neither exists.
    """
    ind = case.numeric("indoor")
    rm = case.text("running_mode")
    valid_txt = case.panel.get("valid")
    if valid_txt is None:
        valid = pd.DataFrame(True, index=case.time, columns=case.cars)
    else:
        valid = valid_txt.isin(list(_VALID_TOKENS)) | (valid_txt == "")
    cooling = rm.isin(list(COOLING_MODES))
    full_cooling = rm.isin(list(FULL_COOLING_MODES))
    on = ~rm.isin(list(OFF_MODES)) & (rm != "")
    usable = (cooling if cooling_only else on) & valid & ind.notna()

    # Kim et al. [C13] steady-state gate: rolling sigma of the indoor temperature over a short
    # window. Pull-down transients and steady operation are different regimes and the literature
    # scores them as different features, so the mask is exposed rather than applied here.
    win = max(int(round(steady_minutes * 60.0 / case.dt_seconds)), 3)
    roll_sd = ind.rolling(win, min_periods=max(win // 2, 2), center=True).std()
    sd_ref = roll_sd.stack().median() if roll_sd.notna().any().any() else np.nan
    steady = roll_sd.le(sd_ref) if np.isfinite(sd_ref) else ind.notna()

    out = case.numeric("outdoor")
    driver = out.median(axis=1, skipna=True)
    driver_name = "outdoor"
    if not np.isfinite(driver.to_numpy(dtype=float)).any():
        driver = ind.median(axis=1, skipna=True)
        driver_name = "indoor"
    finite = driver[np.isfinite(driver.to_numpy(dtype=float))]
    if len(finite) == 0:
        hot = pd.Series(True, index=case.time)
        threshold = float("nan")
    else:
        threshold = float(finite.quantile(hot_quantile))
        hot = driver.ge(threshold).fillna(False)
    return {
        "valid": valid,
        "cooling": cooling,
        "steady": steady.fillna(False),
        "full_cooling": full_cooling,
        "on": on,
        "usable": usable,
        "hot": hot,
        "hot_driver": driver,
        "hot_driver_name": driver_name,
        "hot_threshold": threshold,
    }


# --------------------------------------------------------------------------------------
# Per-car features
# --------------------------------------------------------------------------------------


def _masked(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    return df.where(mask)


def _peer_median(df: pd.DataFrame, *, leave_one_out: bool = False) -> pd.DataFrame:
    """Fleet median at each timestamp, broadcast back over the cars.

    ``leave_one_out`` excludes the car itself from its own reference (the strict peer
    normalisation of [R78]); the default includes it, which is what the plan's baseline names.
    """
    if not leave_one_out:
        med = df.median(axis=1, skipna=True)
        return pd.DataFrame(np.repeat(med.to_numpy()[:, None], df.shape[1], axis=1), index=df.index, columns=df.columns)
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for car in df.columns:
        out[car] = df.drop(columns=[car]).median(axis=1, skipna=True)
    return out


def _col_mean(df: pd.DataFrame, mask: pd.DataFrame | pd.Series) -> pd.Series:
    return _masked(df, _as_frame(mask, df)).mean(axis=0, skipna=True)


def _as_frame(mask: pd.DataFrame | pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    if isinstance(mask, pd.Series):
        return pd.DataFrame(np.repeat(mask.to_numpy()[:, None], like.shape[1], axis=1), index=like.index, columns=like.columns)
    return mask


def _relative(s: pd.Series) -> pd.Series:
    """Centre a per-car statistic on the fleet median (peer normalisation of a scalar)."""
    med = s.median(skipna=True)
    return s - (med if np.isfinite(med) else 0.0)


def _slope_per_hour(y: np.ndarray, t_hours: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(t_hours)
    if ok.sum() < 3:
        return float("nan")
    tt, yy = t_hours[ok], y[ok]
    if np.ptp(tt) <= 0:
        return float("nan")
    return float(np.polyfit(tt, yy, 1)[0])


def car_features(
    case: ACVCase, *, hot_quantile: float = 0.5, cooling_only: bool = True, min_rows: int = 30
) -> pd.DataFrame:
    """One row per car, every ranker's raw statistic as a column.

    Higher is always "more like a leak" (the rankers are sorted descending), so features whose
    physical direction is downward under undercharge - the pull-down rate, for instance - are
    stored already negated and named accordingly. Cars with fewer than ``min_rows`` usable rows
    (``acv_case_04`` carries four entirely empty cars) get NaN and are ranked last.
    """
    cars = case.cars
    m = case_masks(case, hot_quantile=hot_quantile, cooling_only=cooling_only)
    usable, hot = m["usable"], m["hot"]
    hot_f = _as_frame(hot, usable)
    hot_usable = usable & hot_f

    ind = case.numeric("indoor")
    sp = case.numeric("setpoint")
    outd = case.numeric("outdoor")
    rm = case.text("running_mode")

    n_usable = usable.sum(axis=0).reindex(cars).astype(float)
    n_hot = hot_usable.sum(axis=0).reindex(cars).astype(float)
    enough = (n_usable >= min_rows) & (n_hot >= max(5, min_rows // 3))

    ind_u = _masked(ind, usable)
    peer = _peer_median(ind_u)
    peer_loo = _peer_median(ind_u, leave_one_out=True)
    delta = ind_u - peer
    delta_loo = ind_u - peer_loo

    feats = pd.DataFrame(index=pd.Index(cars, name="car"))
    feats["n_rows"] = float(len(case.time))
    feats["n_usable"] = n_usable
    feats["n_hot_usable"] = n_hot

    # --- peer deltas (the plan's baseline family) --------------------------------------
    feats["peer_delta_hot"] = delta.where(hot_f).mean(axis=0, skipna=True).reindex(cars)
    feats["peer_delta_all"] = delta.mean(axis=0, skipna=True).reindex(cars)
    feats["peer_delta_hot_loo"] = delta_loo.where(hot_f).mean(axis=0, skipna=True).reindex(cars)

    # robust peer z: per-timestamp MAD across cars, then the car's median standardised delta
    mad = (delta - delta.median(axis=1, skipna=True).to_numpy()[:, None]).abs().median(axis=1, skipna=True)
    scale = (1.4826 * mad).replace(0.0, np.nan)
    z = delta.div(scale, axis=0)
    feats["robust_peer_z"] = z.where(hot_f).median(axis=0, skipna=True).reindex(cars)

    # --- control residual: how far above its own setpoint the car sits ------------------
    resid = _masked(ind - sp, usable)
    feats["ctrl_residual_raw"] = resid.where(hot_f).mean(axis=0, skipna=True).reindex(cars)
    feats["ctrl_residual"] = _relative(feats["ctrl_residual_raw"])
    feats["ctrl_residual_all"] = _relative(resid.mean(axis=0, skipna=True).reindex(cars))

    # time above setpoint and unmet degree-minutes (ORNL: only visible at deep charge loss)
    above = (resid > 0.5).astype(float).where(resid.notna())
    feats["above_setpoint_frac"] = _relative(
        above.where(hot_f).mean(axis=0, skipna=True).reindex(cars).astype(float)
    )
    unmet = resid.clip(lower=0.0)
    feats["unmet_degree_min"] = _relative(
        (unmet.where(hot_f).mean(axis=0, skipna=True) * (case.dt_seconds / 60.0)).reindex(cars)
    )

    # --- duty and cycling (ORNL: runtime fraction and cycle counts move first) ----------
    on_rows = m["on"] & m["valid"]
    n_on = on_rows.sum(axis=0).reindex(cars).astype(float).replace(0.0, np.nan)
    feats["cooling_duty"] = _relative((m["cooling"] & on_rows).sum(axis=0).reindex(cars) / n_on)
    feats["full_cooling_duty"] = _relative((m["full_cooling"] & on_rows).sum(axis=0).reindex(cars) / n_on)
    hours = max(float(len(case.time)) * case.dt_seconds / 3600.0, 1e-9)
    switches = (rm != rm.shift(1)) & (rm != "") & (rm.shift(1) != "")
    feats["mode_switch_rate"] = _relative(switches.sum(axis=0).reindex(cars).astype(float) / hours)
    lh = case.panel.get("load_halved")
    if lh is None:
        feats["load_halved_share"] = 0.0
    else:
        feats["load_halved_share"] = _relative(lh.isin(list(_LOAD_HALVED_TOKENS)).mean(axis=0).reindex(cars))

    # --- pull-down: how fast the car recovers once cooling is demanded ------------------
    feats["pulldown_penalty"] = _pulldown_penalty(case, usable, resid, cars)

    # --- persistence of the peer delta --------------------------------------------------
    hourly = delta.where(hot_f).resample("1h").mean()
    hourly = hourly.loc[hourly.notna().any(axis=1)]
    if len(hourly):
        leader = hourly.idxmax(axis=1).dropna()
        share = leader.value_counts(normalize=True) if len(leader) else pd.Series(dtype=float)
        feats["persistence_frac"] = share.reindex(cars).fillna(0.0)
        daily = delta.where(hot_f).resample("1D").mean()
        t_h = (daily.index - daily.index[0]).total_seconds().to_numpy() / 3600.0
        feats["peer_delta_trend"] = pd.Series(
            {c: _slope_per_hour(daily[c].to_numpy(dtype=float), t_h) for c in cars}
        ).reindex(cars)
    else:
        feats["persistence_frac"] = 0.0
        feats["peer_delta_trend"] = np.nan

    # --- steady-state-only peer delta (Kim gate) ----------------------------------------
    steady_f = m["steady"].reindex(index=delta.index, columns=delta.columns).fillna(False)
    feats["peer_delta_steady"] = delta.where(hot_f & steady_f).mean(axis=0, skipna=True).reindex(cars)

    # --- Guo/Chen/Xiao peer-regression residual and signed fault index [C1] --------------
    feats["guo_residual"] = _guo_residual(case, usable, hot_f, cars)
    feats["guo_index"] = _guo_index(feats, cars)

    feats["enough_data"] = enough.astype(bool)
    for col in feats.columns:
        if col in ("n_rows", "n_usable", "n_hot_usable", "enough_data"):
            continue
        feats.loc[~enough, col] = np.nan
    feats.attrs["file_id"] = case.file_id
    feats.attrs["feature_version"] = FEATURE_VERSION
    # The mask this frame was built with. `acv.ACVTask.predict` refuses a ranker whose
    # `feature_config` disagrees, so the deployed path can never use a different mask from the
    # one the cross-validation reported.
    feats.attrs["hot_quantile"] = round(float(hot_quantile), 3)
    feats.attrs["cooling_only"] = bool(cooling_only)
    feats.attrs["feature_config"] = [round(float(hot_quantile), 3), bool(cooling_only)]
    feats.attrs["hot_threshold"] = m["hot_threshold"]
    feats.attrs["hot_driver"] = m["hot_driver_name"]
    feats.attrs["unmapped"] = list(case.unmapped)
    return feats


def _pulldown_penalty(case: ACVCase, usable: pd.DataFrame, resid: pd.DataFrame, cars: Sequence[str]) -> pd.Series:
    """Negated pull-down rate (K/h) while the car is above setpoint: high = recovers slowly.

    A capacity-starved unit pulls the saloon down more slowly, so the *negated* slope is the
    leak-shaped direction. Peer-normalised on the fleet median so it is comparable across cases.
    """
    ind = case.numeric("indoor")
    step = max(int(round(600.0 / case.dt_seconds)), 1)  # 10 minutes
    dt_h = (step * case.dt_seconds) / 3600.0
    rate = (ind.shift(-step) - ind) / dt_h
    hot_demand = usable & (resid > 0.5)
    vals = rate.where(hot_demand & rate.notna()).mean(axis=0, skipna=True).reindex(cars)
    return _relative(-vals)


def _guo_residual(
    case: ACVCase, usable: pd.DataFrame, hot_f: pd.DataFrame, cars: Sequence[str]
) -> pd.Series:
    """Peer-regression residual of the indoor temperature, Guo/Chen/Xiao 2024 [C1] style.

    For every ordered pair (target car, peer car) a ridge of the **peer's** indoor temperature on
    the operating conditions ``(T_outdoor, T_setpoint, sin/cos hour-of-day)`` is fitted on the
    peer's own usable rows and used to predict the target car; the **median** over the seven peer
    predictions is the target's "what a healthy car would have read" reference (the median is what
    makes the method survive a faulty peer - they show up to a third of the fleet may be faulty).
    The feature is the mean residual ``measured - predicted`` over the hot rows: positive means
    the car runs warmer than its healthy siblings would under the same conditions, the direction
    undercharge pushes it.
    """
    from sklearn.linear_model import Ridge

    ind = case.numeric("indoor")
    sp = case.numeric("setpoint")
    outd = case.numeric("outdoor")
    hour = case.time.hour.to_numpy(dtype=float) + case.time.minute.to_numpy(dtype=float) / 60.0
    base = np.column_stack([np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0)])

    def design(car: str) -> np.ndarray:
        o = outd[car].to_numpy(dtype=float) if car in outd.columns else np.full(len(case.time), np.nan)
        if not np.isfinite(o).any():
            o = outd.median(axis=1, skipna=True).to_numpy(dtype=float)
        t = sp[car].to_numpy(dtype=float) if car in sp.columns else np.full(len(case.time), np.nan)
        return np.column_stack([o, t, base])

    X = {c: design(c) for c in cars}
    y = {c: ind[c].to_numpy(dtype=float) for c in cars}
    ok = {c: (usable[c].to_numpy(dtype=bool) & np.isfinite(y[c]) & np.isfinite(X[c]).all(axis=1)) for c in cars}

    out: dict[str, float] = {}
    for target in cars:
        rows = ok[target] & hot_f[target].to_numpy(dtype=bool)
        if rows.sum() < 20:
            out[target] = float("nan")
            continue
        preds = []
        for peer in cars:
            if peer == target or ok[peer].sum() < 20:
                continue
            try:
                model = Ridge(alpha=1.0).fit(X[peer][ok[peer]], y[peer][ok[peer]])
                preds.append(model.predict(X[target][rows]))
            except Exception:  # pragma: no cover - a degenerate peer is simply skipped
                continue
        if len(preds) < 2:
            out[target] = float("nan")
            continue
        ref = np.median(np.vstack(preds), axis=0)
        out[target] = float(np.nanmean(y[target][rows] - ref))
    return pd.Series(out).reindex(cars)


#: Feature -> thermodynamic direction under undercharge (+1 = the value rises when charge is low).
#: Every feature in this module is already stored "higher = more leak-like", so all signs are +1;
#: the table is kept explicit because [C1]'s index is only meaningful with a direction per feature.
GUO_INDEX_FEATURES: dict[str, float] = {
    "guo_residual": 1.0,
    "unmet_degree_min": 1.0,
    "full_cooling_duty": 1.0,
}


def _guo_index(feats: pd.DataFrame, cars: Sequence[str]) -> pd.Series:
    """``I = sum_k sign_k * sign(R_k) * (R_k / IQR_k)^2`` over :data:`GUO_INDEX_FEATURES`.

    [C1] sums ``sign_k * (R_k / IQR(R_k))**2``; squaring a residual throws away its sign, which is
    fine for their *detection* threshold but useless for *ranking* cars, so the extra
    ``sign(R_k)`` keeps the term signed. ``IQR_k`` is taken across the cars of this case, so the
    index is scale free and nothing is fitted across cases.
    """
    total = pd.Series(0.0, index=pd.Index(cars, name="car"))
    any_ok = pd.Series(False, index=total.index)
    for name, sign in GUO_INDEX_FEATURES.items():
        if name not in feats.columns:
            continue
        v = pd.to_numeric(feats[name], errors="coerce").astype(float).reindex(cars)
        if v.dropna().nunique() < max(4, len(cars) // 2):
            # Near-degenerate across the fleet (a duty that is 0.000 for six cars and 0.002 for
            # one): its IQR is a rounding width, and dividing by it would let quantisation noise
            # dominate the whole index.
            continue
        iqr = float(v.quantile(0.75) - v.quantile(0.25))
        if not np.isfinite(iqr) or iqr <= 0:
            # A feature the middle six cars agree on exactly (a duty that is 0 for everyone but the
            # outlier) has no interquartile scale; [C1]'s normalisation is undefined, and rescaling
            # by the full range would let a rounding-width difference dominate the index. Skip it.
            continue
        r = v / iqr
        term = sign * np.sign(r) * r.pow(2)
        any_ok |= term.notna()
        total = total.add(term.fillna(0.0), fill_value=0.0)
    return total.where(any_ok)


#: Ranker name -> the feature column it sorts on. Every one is "higher = more likely faulty".
RANKERS: dict[str, str] = {
    "peer_delta_hot": "peer_delta_hot",
    "peer_delta_all": "peer_delta_all",
    "peer_delta_hot_loo": "peer_delta_hot_loo",
    "robust_peer_z": "robust_peer_z",
    "ctrl_residual": "ctrl_residual",
    "ctrl_residual_all": "ctrl_residual_all",
    "above_setpoint_frac": "above_setpoint_frac",
    "unmet_degree_min": "unmet_degree_min",
    "cooling_duty": "cooling_duty",
    "full_cooling_duty": "full_cooling_duty",
    "mode_switch_rate": "mode_switch_rate",
    "load_halved_share": "load_halved_share",
    "pulldown_penalty": "pulldown_penalty",
    "persistence_frac": "persistence_frac",
    "peer_delta_trend": "peer_delta_trend",
    "peer_delta_steady": "peer_delta_steady",
    "guo_residual": "guo_residual",
    "guo_index": "guo_index",
}


# --------------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------------


def cache_dir() -> Path:
    d = CACHE_DIR / "acv"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cached_features(
    path: Path | str, *, hot_quantile: float = 0.5, cooling_only: bool = True, use_cache: bool = True
) -> pd.DataFrame:
    """:func:`car_features` for one file, cached to parquet by name + mtime + feature version.

    ``acv_case_04.xlsx`` takes ~55 s to parse (22k rows x 483 columns), so the leave-one-case-out
    loop would otherwise re-read it six times.

    The frame's ``attrs`` carry the cost of producing it, so a results file can report what the
    run cost **cold** as well as warm (``results/ps3/acv_cv.md``):

    ``build_seconds``
        the measured cost of parsing the xlsx and computing the features, recorded when this
        frame was first built and kept in the sidecar json for every later cache hit;
    ``read_seconds``
        the cost of this call (a parquet read on a hit, the full build on a miss);
    ``cache_hit``
        whether this call was served from the parquet cache.
    """
    p = Path(path)
    if not use_cache:
        t0 = time.perf_counter()
        feats = car_features(load_case(p), hot_quantile=hot_quantile, cooling_only=cooling_only)
        elapsed = time.perf_counter() - t0
        feats.attrs["build_seconds"] = round(elapsed, 3)
        feats.attrs["read_seconds"] = round(elapsed, 3)
        feats.attrs["cache_hit"] = False
        return feats
    stamp = f"{p.name}|{p.stat().st_mtime_ns}|{FEATURE_VERSION}|{hot_quantile}|{int(cooling_only)}"
    key = hashlib.sha1(stamp.encode("utf-8")).hexdigest()[:12]
    cached = cache_dir() / f"{p.stem}__{key}.parquet"
    meta = cached.with_suffix(".attrs.json")
    if cached.exists():
        t0 = time.perf_counter()
        feats = pd.read_parquet(cached)
        if meta.exists():
            feats.attrs.update(json.loads(meta.read_text(encoding="utf-8")))
        feats.attrs["read_seconds"] = round(time.perf_counter() - t0, 3)
        feats.attrs["cache_hit"] = True
        return feats
    t0 = time.perf_counter()
    feats = car_features(load_case(p), hot_quantile=hot_quantile, cooling_only=cooling_only)
    elapsed = time.perf_counter() - t0
    feats.attrs["build_seconds"] = round(elapsed, 3)
    feats.attrs["read_seconds"] = round(elapsed, 3)
    feats.attrs["cache_hit"] = False
    try:
        feats.to_parquet(cached)
        meta.write_text(
            json.dumps({k: v for k, v in feats.attrs.items() if k != "cache_hit"}, default=str),
            encoding="utf-8",
        )
    except Exception:  # pragma: no cover - a cache miss is never fatal
        pass
    return feats
