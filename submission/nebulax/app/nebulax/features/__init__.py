"""Feature extraction: windows -> summary statistics / envelope-spectrum vibration features
-> per-cycle feature tables joined against the fault log.

``nebulax.features.windows``    :class:`Windows`, :func:`make_windows`.
``nebulax.features.stats``      :func:`window_stats` (time-domain, per-channel, vectorised).
``nebulax.features.vibration``  :class:`BearingGeometry`, :func:`fast_kurtogram`,
                                 :func:`envelope_spectrum_feats` (bearing envelope-spectrum).
``nebulax.features.cycles``     :func:`label_from_fault_log`, :func:`peer_normalise`.
"""

from __future__ import annotations

from nebulax.features.cycles import DEFAULT_ALARM_WINDOW_S, label_from_fault_log, peer_normalise
from nebulax.features.stats import STAT_NAMES, feature_names, window_stats
from nebulax.features.vibration import BearingGeometry, envelope_spectrum_feats, fast_kurtogram
from nebulax.features.windows import Windows, make_windows

__all__ = [
    "Windows",
    "make_windows",
    "STAT_NAMES",
    "window_stats",
    "feature_names",
    "BearingGeometry",
    "fast_kurtogram",
    "envelope_spectrum_feats",
    "label_from_fault_log",
    "peer_normalise",
    "DEFAULT_ALARM_WINDOW_S",
]
