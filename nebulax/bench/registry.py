"""Model registry: ``@register`` on the class, ``build(name, **params)`` in the runner.

::

    from nebulax.bench.base import AnomalyDetector
    from nebulax.bench.registry import register

    @register("iforest", input_kind="window_stats", family="ensemble_outlier")
    class IForest(AnomalyDetector):
        def _fit(self, X, t=None, **kw): ...
        def _score(self, X, t=None): ...

    model = build("iforest", n_estimators=200)   # -> IForest(n_estimators=200)

The decorator stamps ``name``/``input_kind``/``family``/``task`` onto the class, so the
runner can ask a model what input it needs without instantiating it. ``name``,
``input_kind`` and ``family`` must match the row in ``configs/model_ladder.yaml``; ``task``
defaults to ``"ad"`` because ladder rows omit it for anomaly detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, TypeVar

from nebulax.bench.base import INPUT_KINDS, TASKS, BaseModel

__all__ = ["ModelSpec", "register", "build", "get_spec", "list_models", "is_registered", "unregister"]

T = TypeVar("T", bound=type[BaseModel])


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One registered model: its class plus the metadata the runner reports."""

    name: str
    cls: type[BaseModel]
    input_kind: str
    family: str
    task: str
    meta: dict[str, Any] = field(default_factory=dict)


_REGISTRY: dict[str, ModelSpec] = {}


def register(
    name: str,
    input_kind: str,
    family: str,
    task: str = "ad",
    **meta: Any,
) -> Callable[[T], T]:
    """Class decorator registering a model under ``name``.

    Raises ``ValueError`` on a duplicate name, an unknown ``input_kind`` (must be one of
    :data:`nebulax.bench.base.INPUT_KINDS`), an unknown ``task``
    (:data:`nebulax.bench.base.TASKS`), or a class that is not a :class:`BaseModel`.
    Extra keywords (``tier``, ``priority``, ``reference_id``, ``needs_gpu``, ...) are kept in
    :attr:`ModelSpec.meta` and written into the results row.
    """
    if input_kind not in INPUT_KINDS:
        raise ValueError(
            f"register({name!r}): unknown input_kind {input_kind!r}; expected one of {list(INPUT_KINDS)}"
        )
    if task not in TASKS:
        raise ValueError(f"register({name!r}): unknown task {task!r}; expected one of {list(TASKS)}")

    def decorator(cls: T) -> T:
        if not isinstance(cls, type) or not issubclass(cls, BaseModel):
            raise ValueError(
                f"register({name!r}): {getattr(cls, '__name__', cls)!r} must subclass "
                f"nebulax.bench.base.BaseModel (AnomalyDetector / Classifier / Regressor)"
            )
        if name in _REGISTRY and _REGISTRY[name].cls is not cls:
            existing = _REGISTRY[name].cls
            raise ValueError(
                f"register({name!r}): already registered to "
                f"{existing.__module__}.{existing.__qualname__}; pick a unique model name"
            )
        cls.name = name
        cls.input_kind = input_kind
        cls.family = family
        cls.task = task
        _REGISTRY[name] = ModelSpec(name=name, cls=cls, input_kind=input_kind, family=family, task=task, meta=dict(meta))
        return cls

    return decorator


def build(name: str, **params: Any) -> BaseModel:
    """Instantiate the registered model ``name`` with ``params``.

    Raises ``ValueError`` naming the closest registered alternatives when ``name`` is unknown.
    """
    spec = get_spec(name)
    return spec.cls(**params)


def get_spec(name: str) -> ModelSpec:
    """The :class:`ModelSpec` for ``name``; ``ValueError`` if it is not registered."""
    try:
        return _REGISTRY[name]
    except KeyError:
        near = [n for n in sorted(_REGISTRY) if n.startswith(name[:3])] or sorted(_REGISTRY)[:10]
        raise ValueError(
            f"build/get_spec: model {name!r} is not registered "
            f"({len(_REGISTRY)} registered). Did you mean one of {near}? "
            f"Model modules must be imported before use (nebulax.models.*)."
        ) from None


def is_registered(name: str) -> bool:
    """True if ``name`` is in the registry."""
    return name in _REGISTRY


def list_models(
    *,
    input_kind: str | None = None,
    family: str | None = None,
    task: str | None = None,
    names: Iterable[str] | None = None,
) -> list[ModelSpec]:
    """Registered models, filtered and sorted by name. All filters are exact matches."""
    if input_kind is not None and input_kind not in INPUT_KINDS:
        raise ValueError(f"list_models: unknown input_kind {input_kind!r}; expected one of {list(INPUT_KINDS)}")
    if task is not None and task not in TASKS:
        raise ValueError(f"list_models: unknown task {task!r}; expected one of {list(TASKS)}")
    wanted = set(names) if names is not None else None
    out = [
        spec
        for spec in _REGISTRY.values()
        if (input_kind is None or spec.input_kind == input_kind)
        and (family is None or spec.family == family)
        and (task is None or spec.task == task)
        and (wanted is None or spec.name in wanted)
    ]
    return sorted(out, key=lambda s: s.name)


def unregister(name: str) -> None:
    """Remove ``name`` from the registry. For tests only - the runner never calls this."""
    _REGISTRY.pop(name, None)
