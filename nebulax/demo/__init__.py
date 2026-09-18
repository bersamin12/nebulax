"""Demo-side glue: run the frozen benchmark winners over the whole fleet for the app.

``docs/app_contract.md`` section 3 is the specification; :mod:`nebulax.demo.scoring` is the
implementation and ``scripts/score_for_demo.py`` the batch entry point that writes
``data/scores/``.
"""

from __future__ import annotations

__all__ = ["scoring"]
