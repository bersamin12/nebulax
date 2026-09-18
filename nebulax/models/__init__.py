"""Importing this package registers every model module with
``nebulax.bench.registry`` (each module's ``@register`` decorators run at import time).

``nebulax.bench.runner._import_models`` does ``import nebulax.models`` for exactly this
reason. Add new model modules here as an import line; never remove another module's import.
"""

from __future__ import annotations

from nebulax.models import boosting as boosting  # noqa: F401
from nebulax.models import classical as classical  # noqa: F401
from nebulax.models import foundation as foundation  # noqa: F401
from nebulax.models import statistical as statistical  # noqa: F401
from nebulax.models import deep as deep  # noqa: F401
from nebulax.models import tsc as tsc  # noqa: F401
from nebulax.models import physics as physics  # noqa: F401
from nebulax.models import drift as drift  # noqa: F401
