"""Real-dataset adapters. One module per source, all with the same tiny contract.

The contract
------------
Every adapter module exposes exactly one public function::

    def load(raw_dir: Path) -> Dataset

where :class:`nebulax.schema.Dataset` bundles ``(long, features, fault_log, events, meta)``:

``long``
    Schema-conformant long telemetry (:data:`nebulax.schema.LONG_COLUMNS`), ``source`` set to
    this adapter's source token, ``run_id`` a stable identifier the adapter invents (e.g.
    ``"metropt3"``, ``"cranfield_p1_l2_r03"``, ``"ottawa_H_A_1"``). Signal names MUST come from
    :data:`nebulax.schema.SIGNALS` for the subsystem - for the pneumatic subsystem that means
    the **MetroPT-3 names verbatim**, typo and all (``DV_eletric``).
``features``
    The per-cycle / per-window feature table (:data:`nebulax.schema.FEATURE_KEY_COLUMNS` plus
    feature columns). An adapter may return an empty, correctly-typed frame
    (:func:`nebulax.schema.empty_features`) and leave feature extraction to ``nebulax.features``.
``fault_log``
    Ground truth, :data:`nebulax.schema.FAULT_LOG_COLUMNS`. MetroPT-3 rows are hard-coded from
    the failure reports in the source paper; Cranfield rows come from its ordinal fault level
    mapped to severity 0.2-1.0; Ottawa rows come from the health state of each recording.
    Empty is legal only when the source genuinely has no labels.
``events``
    Discrete events (:data:`nebulax.schema.EVENT_TYPES`), possibly empty.
``meta``
    Free-form provenance: source paths, file count, sampling rate, DOI/accession, licence,
    any column renaming applied, and anything the calibration agents need. Written as
    ``meta.json`` by :func:`nebulax.schema.write_dataset`.

Rules
-----
* ``load`` must be **pure and offline**: it reads ``raw_dir`` and nothing else. No downloads.
* ``load`` must not modify anything in ``data/raw``.
* Units are converted to the registry's units at load time; the conversion is recorded in ``meta``.
* Timestamps are UTC. Sources with a local-time clock state the assumed offset in ``meta``.
* If ``raw_dir`` is missing or empty, raise ``FileNotFoundError`` with the expected layout.

Subsystem mapping (proxy datasets)
----------------------------------
``metropt3`` / ``metropt2`` -> ``pneumatic`` (MetroPT-3: UCI 791; MetroPT-2: Zenodo 7766691)
``cranfield``               -> ``door``      (CORD DOI 10.17862/cranfield.rd.5097649)
``ottawa``                  -> ``bearing``   (Mendeley y2px5tg92h and the variable-speed set)

Validate any adapter with::

    /home/administrator/miniconda3/envs/nebulax/bin/python -m nebulax.adapters.validate \\
        --source metropt3 --raw data/raw/metropt3
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Protocol, runtime_checkable

from nebulax.schema import Dataset

__all__ = [
    "Dataset",
    "AdapterProtocol",
    "ADAPTER_MODULES",
    "ADAPTER_SUBSYSTEM",
    "DEFAULT_RAW_DIRS",
    "load_source",
    "resolve_adapter",
]


@runtime_checkable
class AdapterProtocol(Protocol):
    """What every adapter module looks like from the outside."""

    def load(self, raw_dir: Path) -> Dataset:  # pragma: no cover - typing only
        ...


#: source token -> candidate module names, tried in order (lazy import).
ADAPTER_MODULES: Final[dict[str, tuple[str, ...]]] = {
    "metropt3": ("nebulax.adapters.metropt3", "nebulax.adapters.metropt"),
    "metropt2": ("nebulax.adapters.metropt2", "nebulax.adapters.metropt"),
    "cranfield": ("nebulax.adapters.cranfield",),
    "ottawa": ("nebulax.adapters.ottawa",),
    "sim": ("nebulax.adapters.synthetic",),
}

#: Which subsystem each source is a proxy for.
ADAPTER_SUBSYSTEM: Final[dict[str, str]] = {
    "metropt3": "pneumatic",
    "metropt2": "pneumatic",
    "cranfield": "door",
    "ottawa": "bearing",
    "sim": "all",
}

#: Conventional raw-data locations, relative to the repo root (gitignored).
DEFAULT_RAW_DIRS: Final[dict[str, str]] = {
    "metropt3": "data/raw/metropt3",
    "metropt2": "data/raw/metropt2",
    "cranfield": "data/raw/cranfield",
    "ottawa": "data/raw/ottawa",
    "sim": "data/sim",
}


def resolve_adapter(source: str):
    """Lazily import the adapter module for ``source`` and return it.

    Raises ``ValueError`` for an unknown source and ``ModuleNotFoundError`` (with the module
    names tried) when the adapter has not been written yet.
    """
    import importlib

    if source not in ADAPTER_MODULES:
        raise ValueError(
            f"resolve_adapter: unknown source {source!r}; expected one of {sorted(ADAPTER_MODULES)}"
        )
    tried = ADAPTER_MODULES[source]
    last: Exception | None = None
    for mod_name in tried:
        try:
            module = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            if exc.name is not None and not mod_name.startswith(exc.name):
                raise  # a real missing dependency inside the adapter, not the adapter itself
            last = exc
            continue
        if not hasattr(module, "load"):
            raise AttributeError(
                f"resolve_adapter: {mod_name} exists but exposes no load(raw_dir) function; "
                f"see the contract in nebulax/adapters/__init__.py"
            )
        return module
    raise ModuleNotFoundError(
        f"resolve_adapter: no adapter module for source {source!r}; tried {list(tried)}. "
        f"Write one exposing load(raw_dir: Path) -> Dataset."
    ) from last


def load_source(source: str, raw_dir: str | Path | None = None) -> Dataset:
    """Import the adapter for ``source`` and run its ``load(raw_dir)``.

    ``raw_dir`` defaults to :data:`DEFAULT_RAW_DIRS` relative to the current directory.
    """
    module = resolve_adapter(source)
    path = Path(raw_dir) if raw_dir is not None else Path(DEFAULT_RAW_DIRS[source])
    return module.load(path)
