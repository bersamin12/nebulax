"""Validator and packer for the PS3 `predictions.zip` deliverable.

The judges run `judge_leaderboard.py` over the CSVs inside `predictions.zip`; a schema slip costs
a whole subsystem ("a submission that errors out, or has zero overlap with the held-out set ...
will not receive a score for that subsystem", specification section 5.1). This module is the gate
every path to a submission goes through - the CLI, the API and the app's download button:

    report = validate_csv("rail", "rail_predictions.csv", expected_ids_for("rail"))
    report.raise_for_errors()
    pack([...], "submission/<team>/predictions.zip")
    validate_zip("submission/<team>/predictions.zip").raise_for_errors()

Rules enforced (from `04_Example_Submission/` and the four Info Kits):

* header exactly the example's, in order (door may carry one extra trailing ``confidence``, which
  the Door Info Kit explicitly permits and the judge ignores - it is reported as a warning);
* ``file_id`` is an exact, case-sensitive basename **with** its extension, no directory part;
* the ``file_id`` set equals the distributed test files when they are known, with no duplicates;
* no empty cells, no NaN/Inf anywhere;
* label vocabularies: door ``Normal`` / ``Abnormal resistance``, rail ``Normal`` / ``Side I`` /
  ``Side II``;
* ACV ``ranked_cars`` lists every car in that file exactly once, two-digit ids joined by ``|``;
* door timestamps parse, ``start < end``, and no two predicted segments overlap;
* SHM predictions are finite and positive.
"""

from __future__ import annotations

import csv
import math
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from nebulax.ps3.common import (
    ACCEPTED_SUFFIXES,
    DOOR_LABELS,
    OUTPUT_FILENAMES,
    RAIL_LABELS,
    TASK_NAMES,
    acv_car_ids,
    natural_key,
    parse_door_timestamp,
)

__all__ = [
    "CSV_HEADERS",
    "SubmissionError",
    "ValidationReport",
    "validate_csv",
    "validate_zip",
    "pack",
    "expected_ids_for",
    "task_for_filename",
]

#: Header row of each `*_predictions.csv`, copied from `04_Example_Submission/`.
CSV_HEADERS: dict[str, tuple[str, ...]] = {
    "door": ("start_time", "end_time", "prediction"),
    "acv": ("file_id", "ranked_cars"),
    "rail": ("file_id", "prediction"),
    "shm": ("file_id", "prediction"),
}

#: Extra columns tolerated after the required header (the Door Info Kit allows a `confidence`).
_OPTIONAL_TRAILING: dict[str, tuple[str, ...]] = {"door": ("confidence",)}

_CAR_ID_RE = re.compile(r"^\d{2}$")
_NAN_TOKENS = {"nan", "-nan", "inf", "-inf", "+inf", "infinity", "-infinity", "na", "n/a", "null", "none"}
#: Fixed zip timestamp so the same CSVs always pack to byte-identical bytes.
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


class SubmissionError(ValueError):
    """Raised by :meth:`ValidationReport.raise_for_errors`."""


@dataclass
class ValidationReport:
    """Outcome of one validation. ``ok`` is ``not errors``; warnings never fail a submission."""

    task: str
    path: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0
    ids: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> "ValidationReport":
        if self.errors:
            joined = "\n  - ".join(self.errors)
            raise SubmissionError(f"{self.path} ({self.task}) is not a valid submission:\n  - {joined}")
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "path": self.path,
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "n_rows": self.n_rows,
            "ids": list(self.ids),
        }

    def __str__(self) -> str:  # pragma: no cover - operator sugar
        state = "OK" if self.ok else f"{len(self.errors)} error(s)"
        return f"[{self.task}] {self.path}: {state}, {self.n_rows} rows"


def task_for_filename(name: str) -> str:
    """``"rail_predictions.csv"`` -> ``"rail"``; raises for anything else."""
    base = Path(str(name)).name
    for task, fname in OUTPUT_FILENAMES.items():
        if base == fname:
            return task
    raise ValueError(f"{base!r} is not a PS3 prediction file; expected one of {sorted(OUTPUT_FILENAMES.values())}")


# --------------------------------------------------------------------------------------
# expected ids
# --------------------------------------------------------------------------------------


def expected_ids_for(task: str, test_dir: Path | str | None = None) -> list[str] | None:
    """File ids the distributed test set should produce, in natural order.

    ACV: the xlsx basenames. Rail / SHM: every csv basename. Door: ``None`` - the door submission
    is one row per *segment* found in the single `Test.csv` stream, so there is no id column.
    ``test_dir`` defaults to the dataset's own ``Test`` folder under ``NEBULAX_PS3_DATA``.
    """
    from nebulax.ps3.common import test_dir as default_test_dir

    key = str(task).strip().lower()
    if key not in TASK_NAMES:
        raise KeyError(f"unknown PS3 task {task!r}; expected one of {list(TASK_NAMES)}")
    if key == "door":
        return None
    d = Path(test_dir) if test_dir is not None else default_test_dir(key)
    if not d.exists():
        raise FileNotFoundError(f"no test directory at {d} (set NEBULAX_PS3_DATA or pass test_dir=)")
    suffixes = ACCEPTED_SUFFIXES[key]
    names = [p.name for p in d.iterdir() if p.is_file() and p.suffix.lower() in suffixes and not p.name.startswith("~$")]
    return sorted(names, key=natural_key)


# --------------------------------------------------------------------------------------
# CSV validation
# --------------------------------------------------------------------------------------


def _read_rows(path: Path) -> tuple[list[str], list[tuple[int, list[str]]]]:
    """Header (stripped) and the data rows, each carrying its 1-based source line number.

    Wholly blank lines are dropped but never renumber the rows after them, so an error message
    points at the line a human will find in an editor.
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        raw = list(csv.reader(fh))
    if not raw:
        return [], []
    rows = [(i, r) for i, r in enumerate(raw[1:], start=2) if any(c.strip() for c in r)]
    return [c.strip() for c in raw[0]], rows


def _looks_nan(text: str) -> bool:
    return text.strip().lower() in _NAN_TOKENS


def validate_csv(
    task: str,
    path: Path | str,
    expected_ids: Sequence[str] | None = None,
    *,
    expected_cars: Mapping[str, Sequence[str]] | None = None,
) -> ValidationReport:
    """Validate one `*_predictions.csv` against the organiser schema.

    ``expected_ids`` (from :func:`expected_ids_for`) turns on set equality against the distributed
    test files; pass ``None`` to skip that check. ``expected_cars`` maps an ACV ``file_id`` to the
    car ids in that file's own headers (see :func:`nebulax.ps3.common.acv_car_ids`); without it the
    ACV check falls back to format + duplicate checks.
    """
    key = str(task).strip().lower()
    if key not in TASK_NAMES:
        raise KeyError(f"unknown PS3 task {task!r}; expected one of {list(TASK_NAMES)}")
    p = Path(path)
    rep = ValidationReport(task=key, path=str(p))
    if not p.exists():
        rep.errors.append(f"file does not exist: {p}")
        return rep
    if p.name != OUTPUT_FILENAMES[key]:
        rep.warnings.append(f"file is named {p.name!r}; the zip member must be {OUTPUT_FILENAMES[key]!r}")

    header, rows = _read_rows(p)
    required = list(CSV_HEADERS[key])
    if not header:
        rep.errors.append("file is empty (no header row)")
        return rep
    optional = list(_OPTIONAL_TRAILING.get(key, ()))
    if header[: len(required)] != required:
        rep.errors.append(f"header is {header!r}, expected {required!r}")
        return rep
    extra = header[len(required) :]
    if extra:
        if extra == optional:
            rep.warnings.append(f"extra column(s) {extra!r} are ignored by the judge")
        else:
            rep.errors.append(f"unexpected extra column(s) {extra!r}; header must be {required!r}")
            return rep

    rep.n_rows = len(rows)
    if not rows:
        rep.errors.append("no prediction rows")
        return rep
    width = len(header)
    for i, row in rows:
        if len(row) != width:
            rep.errors.append(f"line {i}: {len(row)} fields, expected {width}")
    if rep.errors:
        return rep
    for i, row in rows:
        for col, value in zip(required, row, strict=False):
            if not value.strip():
                rep.errors.append(f"line {i}: empty {col!r}")
            elif _looks_nan(value):
                rep.errors.append(f"line {i}: {col!r} is {value!r} (NaN/Inf are not acceptable predictions)")

    if key == "door":
        _validate_door(rows, rep, expected_ids)
    else:
        _validate_keyed(key, rows, rep, expected_ids, expected_cars)
    return rep


def _validate_door(
    rows: list[tuple[int, list[str]]], rep: ValidationReport, expected_ids: Sequence[str] | None
) -> None:
    if expected_ids is not None:
        rep.errors.append("door predictions have no file_id column; expected_ids must be None")
    spans: list[tuple[int, int, int]] = []  # (start_ms, end_ms, line)
    seen: set[tuple[str, str]] = set()
    for i, row in rows:
        start_s, end_s, label = row[0].strip(), row[1].strip(), row[2].strip()
        if label not in DOOR_LABELS:
            rep.errors.append(f"line {i}: prediction {label!r} not in {list(DOOR_LABELS)}")
        try:
            start = parse_door_timestamp(start_s)
            end = parse_door_timestamp(end_s)
        except ValueError as exc:
            rep.errors.append(f"line {i}: {exc}")
            continue
        if not start < end:
            rep.errors.append(f"line {i}: start_time {start_s!r} is not before end_time {end_s!r}")
            continue
        pair = (start_s, end_s)
        if pair in seen:
            rep.errors.append(f"line {i}: duplicate segment {start_s} .. {end_s}")
        seen.add(pair)
        spans.append((int(start.value // 1_000_000), int(end.value // 1_000_000), i))
    spans.sort()
    for (s0, e0, l0), (s1, _e1, l1) in zip(spans, spans[1:], strict=False):
        if s1 < e0:
            rep.errors.append(f"lines {l0} and {l1}: predicted segments overlap")
    rep.ids = [f"{r[0].strip()}..{r[1].strip()}" for _, r in rows]


def _validate_keyed(
    task: str,
    rows: list[tuple[int, list[str]]],
    rep: ValidationReport,
    expected_ids: Sequence[str] | None,
    expected_cars: Mapping[str, Sequence[str]] | None,
) -> None:
    suffixes = ACCEPTED_SUFFIXES[task]
    ids: list[str] = []
    for i, row in rows:
        fid, value = row[0].strip(), row[1].strip()
        ids.append(fid)
        if "/" in fid or "\\" in fid:
            rep.errors.append(f"line {i}: file_id {fid!r} must be a bare basename, not a path")
        if not any(fid.endswith(s) for s in suffixes):
            rep.errors.append(f"line {i}: file_id {fid!r} must keep its extension ({' or '.join(suffixes)})")
        if task == "rail":
            if value not in RAIL_LABELS:
                rep.errors.append(f"line {i}: prediction {value!r} not in {list(RAIL_LABELS)}")
        elif task == "shm":
            try:
                number = float(value)
            except ValueError:
                rep.errors.append(f"line {i}: prediction {value!r} is not a number")
                continue
            if not math.isfinite(number):
                rep.errors.append(f"line {i}: prediction {value!r} is not finite")
            elif number <= 0.0:
                rep.errors.append(f"line {i}: cumulative damage {number!r} must be positive")
        elif task == "acv":
            _validate_ranked_cars(fid, value, i, rep, expected_cars)

    dupes = sorted({x for x in ids if ids.count(x) > 1}, key=natural_key)
    if dupes:
        rep.errors.append(f"duplicate file_id(s): {dupes}")
    if expected_ids is not None:
        want, got = set(expected_ids), set(ids)
        missing, extra = sorted(want - got, key=natural_key), sorted(got - want, key=natural_key)
        if missing:
            rep.errors.append(f"missing {len(missing)} expected file_id(s): {missing[:10]}")
        if extra:
            rep.errors.append(f"unexpected file_id(s) not in the test set: {extra[:10]}")
    rep.ids = ids


def _validate_ranked_cars(
    fid: str,
    value: str,
    line: int,
    rep: ValidationReport,
    expected_cars: Mapping[str, Sequence[str]] | None,
) -> None:
    cars = [c.strip() for c in value.split("|")]
    bad = [c for c in cars if not _CAR_ID_RE.match(c)]
    if bad:
        rep.errors.append(f"line {line}: ranked_cars entries {bad} are not two-digit car ids joined by '|'")
        return
    dupes = sorted({c for c in cars if cars.count(c) > 1})
    if dupes:
        rep.errors.append(f"line {line}: ranked_cars repeats {dupes}")
    if expected_cars is not None and fid in expected_cars:
        want = {str(c).strip() for c in expected_cars[fid]}
        got = set(cars)
        if want != got:
            rep.errors.append(
                f"line {line}: ranked_cars must list every car in {fid} exactly once; "
                f"missing {sorted(want - got)}, unexpected {sorted(got - want)}"
            )
    elif len(cars) != 8:
        rep.warnings.append(f"line {line}: {len(cars)} cars ranked for {fid} (the released ACV files have 8)")


# --------------------------------------------------------------------------------------
# zip
# --------------------------------------------------------------------------------------


def pack(csv_paths: Iterable[Path | str], out_zip: Path | str) -> Path:
    """Write ``predictions.zip``: the given CSVs at the archive root, nothing else.

    Member names are the organiser file names (whatever the source file was called), duplicates
    are refused, and the archive is deterministic - the same CSV bytes always pack to the same zip
    bytes, so a re-run of the submission script is diffable.
    """
    out = Path(out_zip)
    members: dict[str, Path] = {}
    for raw in csv_paths:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"cannot pack missing file {p}")
        task = task_for_filename(p.name)
        name = OUTPUT_FILENAMES[task]
        if name in members:
            raise ValueError(f"two files map to the same zip member {name!r}: {members[name]} and {p}")
        members[name] = p
    if not members:
        raise ValueError("pack(): nothing to pack")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, members[name].read_bytes())
    return out


def validate_zip(
    out_zip: Path | str,
    expected_ids: Mapping[str, Sequence[str] | None] | None = None,
    *,
    expected_cars: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> ValidationReport:
    """Re-open a packed ``predictions.zip`` and re-run :func:`validate_csv` on every member.

    Members must be `*_predictions.csv` files at the archive root - no subfolders, no `__MACOSX`,
    no stray extras. ``expected_ids`` / ``expected_cars`` are keyed by task and forwarded.
    """
    p = Path(out_zip)
    rep = ValidationReport(task="zip", path=str(p))
    if not p.exists():
        rep.errors.append(f"file does not exist: {p}")
        return rep
    if not zipfile.is_zipfile(p):
        rep.errors.append("not a zip archive")
        return rep
    with zipfile.ZipFile(p) as zf:
        names = zf.namelist()
        tasks: dict[str, str] = {}
        for name in names:
            if name.endswith("/"):
                rep.errors.append(f"zip contains a directory entry {name!r}; the CSVs must sit at the archive root")
                continue
            if "/" in name or "\\" in name:
                rep.errors.append(f"zip member {name!r} is inside a folder; the CSVs must sit at the archive root")
                continue
            try:
                tasks[task_for_filename(name)] = name
            except ValueError as exc:
                rep.errors.append(str(exc))
        if not tasks:
            rep.errors.append("zip contains no *_predictions.csv")
        rep.ids = sorted(tasks)
        if rep.errors:
            return rep
        with tempfile.TemporaryDirectory() as tmp:
            for task, name in sorted(tasks.items()):
                target = Path(tmp) / name
                target.write_bytes(zf.read(name))
                sub = validate_csv(
                    task,
                    target,
                    (expected_ids or {}).get(task),
                    expected_cars=(expected_cars or {}).get(task),
                )
                rep.n_rows += sub.n_rows
                rep.errors += [f"{name}: {e}" for e in sub.errors]
                rep.warnings += [f"{name}: {w}" for w in sub.warnings]
    return rep
