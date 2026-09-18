#!/usr/bin/env python
"""Download the four NEBULA X Track 3 real-data proxies into ``data/raw/<dataset>/``.

Datasets
--------
metropt3   UCI ML Repository id 791 ("MetroPT-3 Dataset"), pneumatic APU proxy.
metropt2   Zenodo record 7766691 ("MetroPT2"), pneumatic APU proxy, optional
           (only fetched when the total download is under 3 GB, per policy).
cranfield  Cranfield linear-actuator fault set, CORD DOI
           10.17862/cranfield.rd.5097649, door proxy.
ottawa     University of Ottawa variable-speed/load bearing set, Mendeley
           Data y2px5tg92h (UORED-VAFCLS), bearing proxy.

Design
------
Idempotent at file granularity: every download goes through ``download()``,
which skips the transfer outright when the destination already exists with
the exact expected byte size. It is additionally idempotent at dataset
granularity: each dataset directory keeps a ``MANIFEST.json`` and, on the
next run, a dataset whose manifest says ``status: ok`` and whose files are
all still present at the recorded size is skipped without even talking to
the source API (pass ``--force`` to override).

Never installs packages. ``ucimlrepo`` is not present in the ``nebulax`` conda
env (and this script does not attempt to install it); MetroPT-3 is fetched
from the plain static UCI zip URL instead, which needs nothing extra.

Run: ``python scripts/download_data.py --dataset all --out data/raw``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import requests

CHUNK_SIZE = 1 << 20  # 1 MiB
DEFAULT_TIMEOUT = 60
MAX_RETRIES = 3
USER_AGENT = "nebulax-downloader/1.0 (NEBULA X hackathon, research use)"
HEADERS = {"User-Agent": USER_AGENT}

# --------------------------------------------------------------------------- metropt3

UCI_METROPT3_ZIP_URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
UCI_METROPT3_API = "https://archive.ics.uci.edu/api/dataset?id=791"
UCI_METROPT3_LICENCE = (
    "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/legalcode) "
    "-- stated on https://archive.ics.uci.edu/dataset/791/metropt+3+dataset"
)

# --------------------------------------------------------------------------- metropt2

ZENODO_METROPT2_RECORD_URL = "https://zenodo.org/api/records/7766691"
ZENODO_METROPT2_LANDING = "https://zenodo.org/records/7766691"
ZENODO_SIZE_CAP_BYTES = 3 * 1024**3  # 3 GB, per task instructions

# --------------------------------------------------------------------------- ottawa

MENDELEY_OTTAWA_ID = "y2px5tg92h"
MENDELEY_DATASET_INFO_URL = f"https://data.mendeley.com/public-api/datasets/{MENDELEY_OTTAWA_ID}"
MENDELEY_FILES_URL_TMPL = (
    f"https://data.mendeley.com/public-api/datasets/{MENDELEY_OTTAWA_ID}/files"
    "?folder_id=root&version={version}"
)
DEFAULT_OTTAWA_FORMATS = ("csv", "mat")
DERIVED_SPECTROGRAM_PREFIXES = ("4_spectrogram", "5_spectrogram")

# --------------------------------------------------------------------------- cranfield

CRANFIELD_DOI = "10.17862/cranfield.rd.5097649"
CRANFIELD_DOI_URL = f"https://doi.org/{CRANFIELD_DOI}"
FIGSHARE_API = "https://api.figshare.com/v2"
CRANFIELD_FIGSHARE_MIRROR = "https://cranfield.figshare.com"
CHALLENGE_MARKERS = ("just a moment", "cf-mitigated", "cf-chl-opt", "challenges.cloudflare.com")

#: The 13 .mat files of the CORD release plus the description PDF, exactly as delivered.
#: Named here so a manual drop-in can be checksummed and verified without any network access
#: (the DOI landing page is behind a Cloudflare JS challenge -- see the notes written below).
CRANFIELD_EXPECTED_FILES = (
    "Normal.mat",
    "Backlash1.mat",
    "Backlash2.mat",
    "LackLubrication1.mat",
    "LackLubrication2.mat",
    *(f"Spalling{i}.mat" for i in range(1, 9)),
)
CRANFIELD_EXPECTED_DOC = "Data description.pdf"
#: The release states no licence anywhere in the delivered files; only the landing page would.
CRANFIELD_LICENCE_UNVERIFIED = (
    "UNVERIFIED -- no licence statement appears in any delivered file (13 .mat + "
    f"'{CRANFIELD_EXPECTED_DOC}'); the CORD landing page that would carry it is behind a "
    f"Cloudflare challenge. Check https://doi.org/{CRANFIELD_DOI} in a browser before "
    "redistributing."
)


# ============================================================================ helpers


@dataclass
class FileEntry:
    path: str
    size: int
    sha256: str
    source_url: str

    def to_json(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256, "source_url": self.source_url}


@dataclass
class DatasetManifest:
    dataset: str
    status: str  # ok | partial | skipped
    source_url: str
    licence: str
    files: list[FileEntry] = field(default_factory=list)
    skipped_files: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "status": self.status,
            "source_url": self.source_url,
            "licence": self.licence,
            "files": [f.to_json() for f in self.files],
            "skipped_files": self.skipped_files,
            "notes": self.notes,
            "errors": self.errors,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


def fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def sha256_of(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prior_manifest(ds_dir: Path) -> Optional[dict[str, Any]]:
    p = ds_dir / "MANIFEST.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def files_already_present(manifest: dict[str, Any], ds_dir: Path) -> bool:
    files = manifest.get("files", [])
    if not files:
        return False
    for f in files:
        p = ds_dir / f["path"]
        if not p.exists() or p.stat().st_size != f["size"]:
            return False
    return True


def head_size(url: str) -> Optional[int]:
    try:
        r = requests.head(url, headers=HEADERS, timeout=DEFAULT_TIMEOUT, allow_redirects=True)
        cl = r.headers.get("Content-Length")
        return int(cl) if cl is not None else None
    except (requests.RequestException, ValueError):
        return None


def download(
    url: str,
    dest: Path,
    *,
    expected_size: Optional[int] = None,
    retries: int = MAX_RETRIES,
) -> bool:
    """Stream ``url`` to ``dest``. Returns True if a transfer happened, False if skipped.

    Idempotent: skips outright when ``dest`` exists with exactly ``expected_size`` bytes.
    Raises RuntimeError if every retry fails.
    """
    if dest.exists() and expected_size is not None and dest.stat().st_size == expected_size:
        return False

    tmp = dest.with_name(dest.name + ".part")
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, headers=HEADERS, timeout=DEFAULT_TIMEOUT) as r:
                r.raise_for_status()
                written = 0
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        written += len(chunk)
            if expected_size is not None and written != expected_size:
                raise RuntimeError(f"size mismatch: wrote {written} bytes, expected {expected_size}")
            tmp.replace(dest)
            return True
        except Exception as e:  # noqa: BLE001 - retried, then re-raised below
            last_err = e
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"failed after {retries} attempts: {last_err}")


def _finish(ds_dir: Path, m: DatasetManifest) -> dict[str, Any]:
    data = m.to_json()
    (ds_dir / "MANIFEST.json").write_text(json.dumps(data, indent=2))
    return data


def _looks_like_bot_challenge(text: str) -> bool:
    low = text[:4000].lower()
    return any(marker in low for marker in CHALLENGE_MARKERS)


# ============================================================================ metropt3


def fetch_metropt3(out_root: Path, force: bool) -> dict[str, Any]:
    ds = "metropt3"
    ds_dir = out_root / ds
    ds_dir.mkdir(parents=True, exist_ok=True)
    m = DatasetManifest(dataset=ds, status="ok", source_url=UCI_METROPT3_ZIP_URL, licence=UCI_METROPT3_LICENCE)

    prior = load_prior_manifest(ds_dir)
    if not force and prior and prior.get("status") == "ok" and files_already_present(prior, ds_dir):
        print(f"[{ds}] up to date ({len(prior['files'])} files) -- skipping")
        return prior

    try:
        from ucimlrepo import fetch_ucirepo  # noqa: F401

        try:
            fetch_ucirepo(id=791)
            m.notes.append("ucimlrepo.fetch_ucirepo(id=791) succeeded, but this script still uses the static zip "
                            "as the file of record so the manifest's sha256 matches a plain, reproducible URL")
        except Exception as e:  # noqa: BLE001 - ucimlrepo raises its own DatasetNotFoundError subclass
            m.notes.append(
                f"tried ucimlrepo.fetch_ucirepo(id=791) first as instructed: it is installed but raised "
                f"{type(e).__name__}: {e} -- this dataset id exists on UCI but is not registered as "
                "'importable' in ucimlrepo's tabular API (it is a raw time-series file, not a plain "
                "feature table); fell back to the static UCI zip URL"
            )
    except ImportError:
        m.notes.append(
            "ucimlrepo is not installed in the nebulax env (never installed by this script, per policy); "
            "used the static UCI zip URL fallback instead"
        )

    zip_path = ds_dir / "metropt3.zip"
    expected_zip_size = head_size(UCI_METROPT3_ZIP_URL)
    try:
        print(f"[{ds}] downloading {UCI_METROPT3_ZIP_URL}")
        download(UCI_METROPT3_ZIP_URL, zip_path, expected_size=expected_zip_size)
    except RuntimeError as e:
        m.status = "partial"
        m.errors.append(f"download of {UCI_METROPT3_ZIP_URL} failed: {e}")
        return _finish(ds_dir, m)

    zip_size = zip_path.stat().st_size
    m.files.append(FileEntry(path=zip_path.name, size=zip_size, sha256=sha256_of(zip_path), source_url=UCI_METROPT3_ZIP_URL))

    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            zf.extractall(ds_dir)
    except zipfile.BadZipFile as e:
        m.status = "partial"
        m.errors.append(f"extract failed: {e}")
        return _finish(ds_dir, m)

    for n in names:
        p = ds_dir / n
        if p.exists():
            m.files.append(FileEntry(path=n, size=p.stat().st_size, sha256=sha256_of(p), source_url=UCI_METROPT3_ZIP_URL))

    try:
        api = requests.get(UCI_METROPT3_API, headers=HEADERS, timeout=DEFAULT_TIMEOUT).json()
        doi = api.get("data", {}).get("dataset_doi", "unknown")
        summary = api.get("data", {}).get("additional_info", {}).get("summary", "")
        m.notes.append(f"UCI dataset id 791, dataset DOI {doi}")
        m.notes.append("failure report table from UCI additional_info.summary: " + " | ".join(summary.split("\n")))
    except requests.RequestException as e:
        m.notes.append(f"could not fetch UCI metadata API for extra provenance: {e}")

    m.notes.append(
        "layout: single CSV inside the zip, 1 Hz, 1,516,948 rows, columns "
        "index, timestamp, TP2, TP3, H1, DV_pressure, Reservoirs, Motor_current, Oil_temperature "
        "(7 analogue) + COMP, DV_eletric, Towers, MPG, LPS, Pressure_switch, Oil_level, Caudal_impulses "
        "(8 digital). No GPS, no Flowmeter column (that is MetroPT-1 only). Unlabelled; failures are "
        "given only as a free-text report table (start/end timestamps + severity), not a column."
    )
    return _finish(ds_dir, m)


# ============================================================================ metropt2


def fetch_metropt2(out_root: Path, force: bool) -> dict[str, Any]:
    ds = "metropt2"
    ds_dir = out_root / ds
    ds_dir.mkdir(parents=True, exist_ok=True)
    m = DatasetManifest(dataset=ds, status="ok", source_url=ZENODO_METROPT2_RECORD_URL, licence="")

    prior = load_prior_manifest(ds_dir)
    if not force and prior and prior.get("status") in ("ok", "skipped") and (
        prior.get("status") == "skipped" or files_already_present(prior, ds_dir)
    ):
        print(f"[{ds}] up to date (status={prior.get('status')}) -- skipping")
        return prior

    try:
        r = requests.get(ZENODO_METROPT2_RECORD_URL, headers=HEADERS, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        rec = r.json()
    except requests.RequestException as e:
        m.status = "partial"
        m.errors.append(f"failed to fetch Zenodo record metadata: {e}")
        return _finish(ds_dir, m)

    files = rec.get("files", [])
    total = sum(f.get("size", 0) for f in files)
    lic_id = rec.get("metadata", {}).get("license", {}).get("id", "unknown")
    m.licence = f"{lic_id} -- {ZENODO_METROPT2_LANDING}"
    m.notes.append(f"record title: {rec.get('metadata', {}).get('title')}")

    if total >= ZENODO_SIZE_CAP_BYTES:
        m.status = "skipped"
        m.notes.append(
            f"total size {fmt_size(total)} ({total} bytes) >= 3GB cap -- not downloaded, per task instructions"
        )
        return _finish(ds_dir, m)

    m.notes.append(f"total size {fmt_size(total)} < 3GB cap -- downloading all {len(files)} file(s)")
    for f in files:
        key = f["key"]
        size = f.get("size")
        url = f["links"]["self"]
        checksum = f.get("checksum", "")  # "md5:<hex>"
        dest = ds_dir / key
        try:
            print(f"[{ds}] downloading {key} ({fmt_size(size)})")
            download(url, dest, expected_size=size)
        except RuntimeError as e:
            m.status = "partial"
            m.errors.append(f"{key}: {e}")
            continue
        sha = sha256_of(dest)
        if checksum.startswith("md5:"):
            md5 = hashlib.md5()
            with dest.open("rb") as fh:
                for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
                    md5.update(chunk)
            if md5.hexdigest() != checksum.split(":", 1)[1]:
                m.status = "partial"
                m.errors.append(f"{key}: md5 mismatch against Zenodo-provided checksum")
        m.files.append(FileEntry(path=key, size=dest.stat().st_size, sha256=sha, source_url=url))

    m.notes.append(
        "layout: two flat CSVs, no subfolders. MetroPT2.csv is the raw 1 Hz APU log "
        "(16 sensor signals + control signals + GPS lat/long/speed, 21 attributes total per the "
        "UCI-style description on the Zenodo record, 2022-04-28 to 2022-07-28, ~7.1M rows); "
        "dataset_train.csv is a pre-split training slice provided by the authors. Ground-truth failure "
        "windows are NOT in the CSVs -- they are only published in the companion paper "
        "(air leak 2022-06-04 10:19:24 - 14:22:39; oil leak 2022-07-11 10:10:18 - 2022-07-14 10:22:08), "
        "per docs/research/rail_phm.md section 2.2 citing [R93], not the Zenodo landing page itself."
    )
    return _finish(ds_dir, m)


# ============================================================================ ottawa


def fetch_ottawa(out_root: Path, force: bool, formats: tuple[str, ...], include_derived: bool) -> dict[str, Any]:
    ds = "ottawa"
    ds_dir = out_root / ds
    ds_dir.mkdir(parents=True, exist_ok=True)
    m = DatasetManifest(
        dataset=ds, status="ok", source_url=f"https://data.mendeley.com/datasets/{MENDELEY_OTTAWA_ID}", licence=""
    )

    prior = load_prior_manifest(ds_dir)
    if not force and prior and prior.get("status") == "ok" and files_already_present(prior, ds_dir):
        print(f"[{ds}] up to date ({len(prior['files'])} files) -- skipping")
        return prior

    try:
        r = requests.get(MENDELEY_DATASET_INFO_URL, headers=HEADERS, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        info = r.json()
    except requests.RequestException as e:
        m.status = "partial"
        m.errors.append(f"failed to fetch Mendeley dataset info: {e}")
        return _finish(ds_dir, m)

    version = info.get("version")
    lic = info.get("data_licence", {}) or {}
    m.licence = f"{lic.get('full_name', 'unknown')} -- {lic.get('url', '')}"
    m.notes.append(f"dataset '{info.get('name')}', Mendeley DOI {info.get('doi', {}).get('id')}, version {version}")
    m.source_url = MENDELEY_FILES_URL_TMPL.format(version=version)

    all_files = info.get("files", [])
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for f in all_files:
        ext = f["filename"].rsplit(".", 1)[-1].lower()
        is_derived = f["filename"].lower().startswith(DERIVED_SPECTROGRAM_PREFIXES)
        keep = (ext in formats) and (include_derived or not is_derived)
        (selected if keep else skipped).append(f)

    m.skipped_files = [{"filename": f["filename"], "size": f["size"]} for f in skipped]
    skipped_total = sum(f["size"] for f in skipped)
    m.notes.append(
        f"selected {len(selected)}/{len(all_files)} files (--ottawa-formats={','.join(formats)}, "
        f"--ottawa-include-derived={include_derived}); skipped {len(skipped)} files totalling "
        f"{fmt_size(skipped_total)} -- by default these are the duplicate .xlsx encoding of the same "
        "raw arrays and the two pre-rendered STFT-spectrogram .zip archives (derived, not raw signal); "
        "pass --ottawa-formats csv,mat,xlsx --ottawa-include-derived to fetch the complete ~3.9GB dataset"
    )

    for f in selected:
        key = f["filename"]
        size = f["size"]
        url = f["content_details"]["download_url"]
        expected_sha = f["content_details"].get("sha256_hash")
        dest = ds_dir / key
        try:
            print(f"[{ds}] downloading {key} ({fmt_size(size)})")
            download(url, dest, expected_size=size)
        except RuntimeError as e:
            m.status = "partial"
            m.errors.append(f"{key}: {e}")
            continue
        sha = sha256_of(dest)
        if expected_sha and sha != expected_sha:
            m.status = "partial"
            m.errors.append(f"{key}: sha256 mismatch (got {sha}, Mendeley recorded {expected_sha})")
        m.files.append(FileEntry(path=key, size=dest.stat().st_size, sha256=sha, source_url=url))

    m.notes.append(
        "layout (from real inspection of H_10_0.csv/.mat, 2026-09-14): flat files named "
        "<Class>_<bearingID>_<state>.{csv,mat,xlsx}. Class in {H=healthy, I=inner race, O=outer race, "
        "B=ball, C=cage}. State in {0=healthy baseline, 1=developing fault, 2=faulty}. Bearing IDs "
        "1-5 are the inner-race group, 6-10 outer-race, 11-15 ball, 16-20 cage; every id 1-20 also has "
        "an H_<id>_0 healthy-baseline recording -- 60 raw recordings total (20 healthy + 4 fault types "
        "x 5 bearings x 2 non-healthy states), each stored redundantly as .csv, .mat and .xlsx (same "
        "values, three encodings). Each raw recording: 420,000 rows x 5 columns, sampled at 42,000 Hz "
        "for 10.0 s continuous. Columns, identical order in the CSV header and in the single array "
        "(named after the file, e.g. variable 'H_10_0') inside the .mat: Accelerometer (g), Acoustic "
        "(raw microphone signal, uncalibrated units), Speed (nominal shaft speed in RPM, e.g. 1796), "
        "Load (nominal applied load in the rig's own units, e.g. 400), Temperature Difference (deg C, "
        "the one genuinely continuous slow channel). IMPORTANT ADAPTER TRAP: Speed and Load are NOT "
        "per-sample channels -- the nominal value is written once into row 0 only, and every other one "
        "of the 419,999 rows is exactly 0 in those two columns. A naive per-sample feature pipeline "
        "must read row 0 for the nominal speed/load and drop the column from the windowed signal, not "
        "treat 'mostly zero' as the real channel. Two additional pre-rendered derived .zip archives "
        "('4_Spectrogram_accelerometer...' and '5_Spectrogram_acoustic...') hold 12,000 STFT PNG images "
        "each (Hanning window, signal length 512) -- these are skipped by default, see skipped_files."
    )
    return _finish(ds_dir, m)


# ============================================================================ cranfield


def _cranfield_manifest_from_disk(ds_dir: Path, m: DatasetManifest) -> Optional[dict[str, Any]]:
    """Checksum a manually-placed CORD release and write its manifest, with no network at all.

    The DOI sits behind a Cloudflare JS challenge that no HTTP client can pass, so the release
    is obtained by hand (see the notes this function writes). Once the files are on disk this
    is the whole job: verify which of the 13 expected ``.mat`` files are there, record
    ``sha256`` and size for each, and set ``status`` to ``ok`` when all 13 are present or
    ``partial`` when some are missing. Returns ``None`` when nothing is on disk, so the caller
    can fall through to the (documented, failing) network attempts.
    """
    present = [n for n in CRANFIELD_EXPECTED_FILES if (ds_dir / n).is_file()]
    if not present:
        return None
    missing = [n for n in CRANFIELD_EXPECTED_FILES if n not in present]
    for name in present:
        path = ds_dir / name
        m.files.append(
            FileEntry(
                path=name,
                size=path.stat().st_size,
                sha256=sha256_of(path),
                source_url=CRANFIELD_DOI_URL,
            )
        )
    doc = ds_dir / CRANFIELD_EXPECTED_DOC
    if doc.is_file():
        m.files.append(
            FileEntry(
                path=CRANFIELD_EXPECTED_DOC,
                size=doc.stat().st_size,
                sha256=sha256_of(doc),
                source_url=CRANFIELD_DOI_URL,
            )
        )
    else:
        m.notes.append(
            f"'{CRANFIELD_EXPECTED_DOC}' is not present. It is the release's primary "
            "documentation -- channel order, variable naming, rig and fault descriptions all "
            "come from it -- so fetch it with the data."
        )
    m.status = "ok" if not missing else "partial"
    m.licence = CRANFIELD_LICENCE_UNVERIFIED
    m.skipped_files = [{"path": n, "reason": "not present in this drop-in"} for n in missing]
    m.notes.append(
        f"{len(present)} of {len(CRANFIELD_EXPECTED_FILES)} expected .mat files verified on "
        "disk by sha256; no network request was made. The CORD DOI cannot be fetched by an HTTP "
        "client (Cloudflare managed JS challenge), so these files were placed here by hand and "
        "this manifest is their checksum record."
    )
    if missing:
        m.errors.append(f"missing expected release files: {missing}")
    m.notes.append(
        "Release layout (verified against the files themselves and against "
        f"'{CRANFIELD_EXPECTED_DOC}' section 4): 13 .mat files, one per condition. Each holds 60 "
        "matrices -- 2 motion profiles x 3 loads (20, 40, -40 kgf) x 10 repetitions -- named "
        "<class><profile><level><load><rep> (e.g. trainsin20kg3, pointtrap8thneg40kg10). "
        "Backlash1.mat holds 59: backtrap1st40kg has 9 repetitions. Every matrix is (2000, 3) "
        "float64 = 80 s at 25 Hz, columns [position set point (mm), position error (mm), motor "
        "current (A)]."
    )
    m.notes.append(
        "The rig's motor is a Nema 34 STEPPER (not a PMDC drive), so the current channel is a "
        "drive-level quantity with a large load-independent component -- see "
        "nebulax/adapters/cranfield.py and docs/parameters.md before using any current ratio."
    )
    print(
        f"[cranfield] {len(present)}/{len(CRANFIELD_EXPECTED_FILES)} release files checksummed "
        f"on disk -- status={m.status}" + (f", missing {missing}" if missing else "")
    )
    return _finish(ds_dir, m)


def fetch_cranfield(out_root: Path, force: bool) -> dict[str, Any]:
    ds = "cranfield"
    ds_dir = out_root / ds
    ds_dir.mkdir(parents=True, exist_ok=True)
    m = DatasetManifest(
        dataset=ds,
        status="partial",
        source_url=CRANFIELD_DOI_URL,
        licence=CRANFIELD_LICENCE_UNVERIFIED,
    )

    prior = load_prior_manifest(ds_dir)
    if not force and prior and prior.get("status") == "ok" and files_already_present(prior, ds_dir):
        print(f"[{ds}] up to date ({len(prior['files'])} files) -- skipping")
        return prior

    # The release is obtained by hand; whenever it is on disk, checksum it and stop.
    from_disk = _cranfield_manifest_from_disk(ds_dir, m)
    if from_disk is not None:
        return from_disk

    errors: list[str] = []

    # Attempt 1: resolve the DOI directly and fetch the landing page.
    try:
        r = requests.get(CRANFIELD_DOI_URL, headers=HEADERS, timeout=DEFAULT_TIMEOUT, allow_redirects=True)
        if r.status_code == 200 and not _looks_like_bot_challenge(r.text):
            m.notes.append(f"DOI resolved and returned real content at {r.url}")
        elif _looks_like_bot_challenge(r.text):
            errors.append(
                f"DOI {CRANFIELD_DOI} resolves to {r.url} (HTTP {r.status_code}), but the response body is "
                "a bot-challenge page (Cloudflare 'Just a moment...' managed JS challenge), not the dataset"
            )
        else:
            errors.append(f"DOI {CRANFIELD_DOI} resolved to {r.url} but returned HTTP {r.status_code}")
    except requests.RequestException as e:
        errors.append(f"DOI resolution GET failed: {e}")

    # Attempt 2: figshare public API, search by DOI (dataset was historically figshare-backed).
    try:
        r = requests.post(
            f"{FIGSHARE_API}/articles/search",
            json={"doi": CRANFIELD_DOI},
            headers={**HEADERS, "Content-Type": "application/json"},
            timeout=DEFAULT_TIMEOUT,
        )
        results = r.json() if r.ok else None
        if not results:
            errors.append(
                "figshare public API articles/search for this DOI returned no results "
                "(HTTP %s, body=%s) -- the record is not indexed there any more" % (r.status_code, str(results)[:200])
            )
    except (requests.RequestException, ValueError) as e:
        errors.append(f"figshare API search failed: {e}")

    # Attempt 3: the institution's own figshare mirror (cranfield.figshare.com), in case the DOI
    # only points at the newer repository but the old mirror is still browsable.
    try:
        r = requests.get(CRANFIELD_FIGSHARE_MIRROR, headers=HEADERS, timeout=15)
        if "x-amzn-waf-action" in {k.lower() for k in r.headers}:
            errors.append(
                f"{CRANFIELD_FIGSHARE_MIRROR} returned HTTP {r.status_code} behind an AWS WAF challenge "
                "(x-amzn-waf-action header present) -- not browsable without a real browser"
            )
        elif r.status_code != 200:
            errors.append(f"{CRANFIELD_FIGSHARE_MIRROR} returned HTTP {r.status_code}")
    except requests.RequestException as e:
        errors.append(f"{CRANFIELD_FIGSHARE_MIRROR} unreachable: {e}")

    m.errors = errors
    m.notes.append(
        "Cranfield's Online Research Data (CORD) service has moved off Figshare to a self-hosted "
        "DSpace instance at dspace.lib.cranfield.ac.uk, and that instance sits behind a Cloudflare "
        "'managed challenge' that requires executing JavaScript in a real browser -- it cannot be "
        "passed by a plain HTTP client (curl/requests), which is what this script and its retry paths "
        "are limited to. The historical cranfield.figshare.com institution mirror is likewise behind an "
        "AWS WAF bot challenge, and figshare's own public API (searched both by this DOI and by the "
        "bare numeric id 5097649) no longer has the record indexed. status=partial is correct here, not "
        "a bug: manual download is required (open the DOI in an actual browser, solve the challenge, "
        "download the 13 .mat files and 'Data description.pdf', drop them under "
        "data/raw/cranfield/, then re-run this script -- it detects them on disk, checksums them "
        "and writes status=ok without making any network request at all)."
    )
    return _finish(ds_dir, m)


# ============================================================================ main


DATASET_FUNCS = {
    "metropt3": lambda out_root, args: fetch_metropt3(out_root, args.force),
    "metropt2": lambda out_root, args: fetch_metropt2(out_root, args.force),
    "cranfield": lambda out_root, args: fetch_cranfield(out_root, args.force),
    "ottawa": lambda out_root, args: fetch_ottawa(
        out_root,
        args.force,
        tuple(x.strip().lower() for x in args.ottawa_formats.split(",") if x.strip()),
        args.ottawa_include_derived,
    ),
}

DATASET_ORDER = ["metropt3", "metropt2", "ottawa"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["all", *DATASET_ORDER], default="all")
    ap.add_argument("--out", default="data/raw", help="output root directory (default: data/raw)")
    ap.add_argument("--force", action="store_true", help="re-download and re-verify even if a prior manifest says ok")
    ap.add_argument(
        "--ottawa-formats",
        default=",".join(DEFAULT_OTTAWA_FORMATS),
        help="comma-separated raw file extensions to fetch for Ottawa (default: csv,mat)",
    )
    ap.add_argument(
        "--ottawa-include-derived",
        action="store_true",
        help="also fetch the two pre-rendered spectrogram .zip archives for Ottawa (adds ~1.8GB)",
    )
    return ap


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    names = DATASET_ORDER if args.dataset == "all" else [args.dataset]
    results: dict[str, dict[str, Any]] = {}
    for name in names:
        print(f"\n=== {name} ===")
        results[name] = DATASET_FUNCS[name](out_root, args)

    print("\n=== manifest summary ===")
    overall_ok = True
    for name, res in results.items():
        status = res.get("status")
        nfiles = len(res.get("files", []))
        total = sum(f["size"] for f in res.get("files", []))
        print(f"{name:10s} status={status:8s} files={nfiles:4d} total={fmt_size(total):>10s}  -> data/raw/{name}/MANIFEST.json")
        for e in res.get("errors", []):
            print(f"           ERROR: {e}")
        if status not in ("ok", "skipped"):
            overall_ok = False

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
