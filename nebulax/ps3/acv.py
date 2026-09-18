"""ACV: rank the eight cars of a train by refrigerant-leak likelihood.

One case file (`.xlsx`, 30 s telemetry over 3-4 days, eight cars) in, one CSV row out::

    file_id,ranked_cars
    acv_test_case.xlsx,01|03|04|08|07|06|02|05

scored by the organisers' rank decay ``(n - (r - 1)) / n`` (ACV Info Kit section 4,
`nebulax.ps3.scoring.rank_decay`). The features live in :mod:`nebulax.ps3.acv_features`; this
module holds the ranker (how the per-car features become an order), the leave-one-case-out
validation, and the `Task` the app and the submission packer call.

**The ranker is a rule chosen inside the fold, not a classifier.** Six labelled cases is far too
little to train a per-car model, and a per-car model would be free to learn a car-*identity*
prior, which is exactly the wrong thing (the faulty car differs between cases and the test train
is a different train). So:

* every feature is peer-normalised **inside one case file** - a car is only ever compared with the
  seven other cars of its own train at the same timestamps, so no statistic can cross cases;
* the ranker's shape (which features, their weights, the aggregation) is chosen on the training
  cases of the outer fold and frozen before the held-out case is read;
* ties and cars with too little data fall back to the file's own header order, never to a learnt
  car-id preference;
* the pool a fold may choose from is pre-registered and ordered on physical grounds
  (:func:`selection_pool`), because with five training cases most of a 26-rule pool ties on rank
  decay - `results/ps3/acv_ladder.md` keeps that row as the evidence - and a fold that still has
  to choose does it on :func:`separation`, a margin read off its own five training cases;
* **the committed headline is the pre-registered baseline rule, not a ladder winner.**
  :func:`train` never promotes: with six cases, a row that tops a table read on those six cases
  cannot then be scored on them. The ladder's best row (`peer_delta_hot_loo`, 1.0000) is reported
  as `ladder_best`, labelled "selected on all six cases". Both produce the identical row for the
  organisers' Test file, so only the claim differs.

With six cases the leave-one-case-out number is **exploratory** (plan, W4 validation schemes) and
is reported as such: 0.9792 +/- 0.051, where one case moving one rank changes the mean by 0.021
and the whole distance to 1.0000 is one rank step on a four-car case decided by 4.2 mK.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from nebulax.ps3 import acv_features as af
from nebulax.ps3.common import (
    BaseTask,
    PredictionResult,
    RESULTS_DIR,
    Trace,
    Viewport,
    git_rev,
    load_model,
    register_task,
    save_model,
    train_dir,
    labels_path,
    test_dir,
)
from nebulax.ps3.scoring import rank_decay

__all__ = [
    "ACVRanker",
    "BASELINE_RANKER",
    "ACVTask",
    "rank_cars",
    "score_cars",
    "loco_scores",
    "fit_ranker",
    "separation",
    "leave_one_case_out",
    "chance_floors",
    "FeatureBank",
    "candidate_rankers",
    "selection_pool",
    "run_ladder",
    "train",
]

#: Maximum points in the explanation trace (the contract asks for a few hundred, not 10,000).
MAX_TRACE_POINTS = 2000


# --------------------------------------------------------------------------------------
# The ranker
# --------------------------------------------------------------------------------------


@dataclass
class ACVRanker:
    """A frozen ranking rule: which per-car features, how they are combined, how ties break.

    ``aggregation``:

    ``single``
        sort on ``features[0]`` alone (the plan's baseline).
    ``zsum``
        weighted sum of each feature standardised **within the case** by the across-car MAD, so a
        hot case and a mild case contribute comparably and nothing is fitted across cases.
    ``borda``
        weighted sum of within-case rank positions - robust to a feature with one wild car.

    ``weights`` are the only numbers fitted outside a single file, and
    :func:`fit_ranker` fits them on the training cases of the outer fold only.
    """

    features: tuple[str, ...] = ("peer_delta_hot",)
    weights: tuple[float, ...] = (1.0,)
    aggregation: str = "single"
    tie_break: tuple[str, ...] = ("ctrl_residual",)
    hot_quantile: float = 0.5
    cooling_only: bool = True
    feature_version: str = af.FEATURE_VERSION
    name: str = "baseline_peer_delta_hot"
    trained_on: tuple[str, ...] = ()
    cv: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.features = tuple(str(f) for f in self.features)
        if not self.features:
            raise ValueError("ACVRanker needs at least one feature")
        unknown = [f for f in self.features + tuple(self.tie_break) if f not in af.RANKERS]
        if unknown:
            raise ValueError(f"unknown ACV ranker feature(s) {unknown}; known: {sorted(af.RANKERS)}")
        if self.aggregation not in ("single", "zsum", "borda"):
            raise ValueError(f"unknown aggregation {self.aggregation!r}")
        w = tuple(float(x) for x in self.weights)
        if len(w) != len(self.features):
            w = (1.0,) * len(self.features)
        self.weights = w
        self.tie_break = tuple(str(f) for f in self.tie_break)
        self.hot_quantile = float(self.hot_quantile)
        self.cooling_only = bool(self.cooling_only)

    @property
    def feature_config(self) -> tuple[float, bool]:
        """The ``(hot_quantile, cooling_only)`` the feature frame must have been built with."""
        return (round(self.hot_quantile, 3), bool(self.cooling_only))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


#: The plan's baseline: hot-hour peer delta, ties broken by the control residual.
BASELINE_RANKER = ACVRanker()


def _within_case_z(col: pd.Series) -> pd.Series:
    """Standardise one feature across the cars of a single case (median / MAD, no fitting)."""
    v = pd.to_numeric(col, errors="coerce").astype(float)
    med = v.median(skipna=True)
    mad = (v - med).abs().median(skipna=True)
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        spread = v.max(skipna=True) - v.min(skipna=True)
        scale = float(spread) if np.isfinite(spread) and spread > 0 else 1.0
    return (v - med) / scale


def score_cars(feats: pd.DataFrame, ranker: ACVRanker = BASELINE_RANKER) -> pd.DataFrame:
    """Per-car score frame: ``score``, the tie-break columns and the contributing features.

    Cars flagged ``enough_data = False`` (``acv_case_04`` has four entirely empty cars) score
    ``NaN`` and :func:`rank_cars` puts them last.
    """
    out = pd.DataFrame(index=feats.index)
    if ranker.aggregation == "single":
        out["score"] = pd.to_numeric(feats[ranker.features[0]], errors="coerce").astype(float)
    elif ranker.aggregation == "zsum":
        total = pd.Series(0.0, index=feats.index)
        any_ok = pd.Series(False, index=feats.index)
        for name, w in zip(ranker.features, ranker.weights):
            z = _within_case_z(feats[name])
            any_ok |= z.notna()
            total = total.add((w * z).fillna(0.0), fill_value=0.0)
        out["score"] = total.where(any_ok)
    else:  # borda
        n = float(len(feats.index))
        total = pd.Series(0.0, index=feats.index)
        any_ok = pd.Series(False, index=feats.index)
        for name, w in zip(ranker.features, ranker.weights):
            v = pd.to_numeric(feats[name], errors="coerce").astype(float)
            any_ok |= v.notna()
            pos = v.rank(ascending=False, method="average", na_option="keep")
            total = total.add((w * (n - pos) / n).fillna(0.0), fill_value=0.0)
        out["score"] = total.where(any_ok)
    for name in ranker.tie_break:
        out[name] = pd.to_numeric(feats[name], errors="coerce").astype(float)
    enough = feats["enough_data"].astype(bool) if "enough_data" in feats.columns else pd.Series(True, index=feats.index)
    out["score"] = out["score"].where(enough)
    out["enough_data"] = enough
    return out


def rank_cars(feats: pd.DataFrame, ranker: ACVRanker = BASELINE_RANKER) -> tuple[list[str], pd.DataFrame]:
    """``(ranked car ids, score frame)`` - most- to least-likely faulty.

    Order: score descending, then each tie-break feature descending, then the car id ascending
    (the file's own header order, a deterministic fallback - **never** a learnt car-id prior).
    Cars with no usable data sort last.
    """
    scores = score_cars(feats, ranker)
    order = scores.copy()
    order["_car"] = [str(c) for c in order.index]
    by = ["score"] + [c for c in ranker.tie_break if c in order.columns]
    ranked = order.sort_values(
        by=by + ["_car"],
        ascending=[False] * len(by) + [True],
        na_position="last",
        kind="mergesort",
    )
    return [str(c) for c in ranked.index], scores


# --------------------------------------------------------------------------------------
# Validation (leave-one-case-out) and weight fitting
# --------------------------------------------------------------------------------------


def _labels(root: Path | str | None = None) -> dict[str, str]:
    lab = pd.read_csv(labels_path("acv", root=root), dtype=str)
    return {str(r.filename).strip(): str(r.faulty_car).strip() for r in lab.itertuples()}


def _case_paths(directory: Path) -> list[Path]:
    return [p for p in sorted(directory.glob("*.xlsx")) if not p.name.startswith("~$")]


class FeatureBank:
    """Per-case feature frames, one set per ``(hot_quantile, cooling_only)`` configuration.

    The mask and season-half ablations need the features recomputed, and ``acv_case_04.xlsx``
    takes ~55 s to parse, so every configuration is computed once, memoised here and cached to
    parquet by :func:`nebulax.ps3.acv_features.cached_features`.
    """

    def __init__(self, paths: Iterable[Path], *, use_cache: bool = True) -> None:
        self.paths = [Path(p) for p in paths]
        self.use_cache = bool(use_cache)
        self._bank: dict[tuple[float, bool], dict[str, pd.DataFrame]] = {}
        #: One record per (file, configuration) actually materialised - see :meth:`cost`.
        self.loads: list[dict[str, Any]] = []

    def for_config(self, hot_quantile: float, cooling_only: bool) -> dict[str, pd.DataFrame]:
        key = (round(float(hot_quantile), 3), bool(cooling_only))
        if key not in self._bank:
            frames: dict[str, pd.DataFrame] = {}
            for p in self.paths:
                t0 = time.perf_counter()
                feats = af.cached_features(
                    p, hot_quantile=key[0], cooling_only=key[1], use_cache=self.use_cache
                )
                frames[p.name] = feats
                self.loads.append(
                    {
                        "file": p.name,
                        "config": list(key),
                        "cache_hit": bool(feats.attrs.get("cache_hit", False)),
                        "seconds": round(time.perf_counter() - t0, 3),
                        "build_seconds": float(feats.attrs.get("build_seconds", float("nan"))),
                    }
                )
            self._bank[key] = frames
        return self._bank[key]

    def cost(self) -> dict[str, Any]:
        """What the feature frames of this run cost, warm and cold.

        ``warm_seconds`` is what this run actually spent on them (a parquet read per hit);
        ``cold_seconds`` is the sum of each frame's **recorded** build time - the measured cost of
        parsing the xlsx and computing the features when that cache entry was first written. The
        two differ by two orders of magnitude (``acv_case_04.xlsx`` alone is ~55 s a
        configuration), so a wall-clock number from a warm run is not the cost of the method.
        """
        warm = float(sum(r["seconds"] for r in self.loads))
        builds = [r["build_seconds"] for r in self.loads]
        cold = float(np.nansum(builds)) if builds else 0.0
        return {
            "n_frames": len(self.loads),
            "n_cache_hits": sum(1 for r in self.loads if r["cache_hit"]),
            "n_cache_misses": sum(1 for r in self.loads if not r["cache_hit"]),
            "warm_seconds": round(warm, 2),
            "cold_seconds": round(cold, 2),
            "cold_seconds_known": bool(builds) and not bool(np.isnan(builds).any()),
        }

    def for_ranker(self, ranker: ACVRanker) -> dict[str, pd.DataFrame]:
        return self.for_config(*ranker.feature_config)

    @property
    def files(self) -> list[str]:
        return [p.name for p in self.paths]


def loco_scores(bank: FeatureBank, labels: Mapping[str, str], ranker: ACVRanker, *, files: Sequence[str] | None = None) -> dict[str, Any]:
    """Apply one **fixed** ranker to a set of cases and score it with the organisers' rank decay.

    Used both for a ladder row (all six cases - the frozen outer CV, since a fixed rule has
    nothing to fit) and for a fold's inner search (the five training cases only).
    """
    feats_by_file = bank.for_ranker(ranker)
    names = list(files) if files is not None else sorted(feats_by_file)
    per_file: dict[str, float] = {}
    ranks: dict[str, int] = {}
    orders: dict[str, list[str]] = {}
    for fid in names:
        ranked, _ = rank_cars(feats_by_file[fid], ranker)
        true = labels[fid]
        per_file[fid] = rank_decay(ranked, true, n=len(ranked))
        ranks[fid] = ranked.index(true) + 1 if true in ranked else 0
        orders[fid] = ranked
    vals = list(per_file.values())
    return {
        "score": float(np.mean(vals)) if vals else 0.0,
        "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "top1": float(np.mean([r == 1 for r in ranks.values()])) if ranks else 0.0,
        "mean_rank": float(np.mean(list(ranks.values()))) if ranks else 0.0,
        "per_file": per_file,
        "ranks": ranks,
        "orders": orders,
    }


def separation(
    bank: FeatureBank,
    labels: Mapping[str, str],
    ranker: ACVRanker,
    *,
    files: Sequence[str] | None = None,
) -> float:
    """Mean within-case margin (in MAD units) between the true car and the best other car.

    Rank decay over five cases is a coarse, heavily tied statistic - three of the five pool rules
    score exactly 1.000 on most folds - so on its own it cannot tell a fold's candidates apart.
    This is the finer reading of the *same* training cases: how decisively the rule separates the
    labelled car from its peers, standardised inside each case by the across-car MAD
    (:func:`_within_case_z`) so rules on different scales are comparable. Negative when the true
    car is not first. It is computed on the fold's training files only and never on the held-out
    case, exactly like the score it refines.
    """
    feats_by_file = bank.for_ranker(ranker)
    names = list(files) if files is not None else sorted(feats_by_file)
    margins: list[float] = []
    for fid in names:
        z = _within_case_z(score_cars(feats_by_file[fid], ranker)["score"])
        z = z[z.notna()]
        true = labels[fid]
        if true not in z.index:
            margins.append(-float(len(z)))  # the true car is unrankable: worst possible evidence
            continue
        others = z.drop(index=true)
        margins.append(float(z[true] - others.max()) if len(others) else float(z[true]))
    return float(np.mean(margins)) if margins else 0.0


def fit_ranker(
    bank: FeatureBank,
    labels: Mapping[str, str],
    candidates: Sequence[ACVRanker],
    *,
    files: Sequence[str] | None = None,
) -> ACVRanker:
    """Pick the best candidate rule **on the given (training-fold) cases only**.

    Order of preference: the training rank decay, then the training top-1 rate, then the training
    *separation* (:func:`separation`, the margin the rule puts between the labelled car and its
    peers), then the fewest features, then the candidate's position in the pool.

    The separation term matters: without it the first three criteria tie for most of the pool on
    most folds and the winner is decided by the pool order alone - a constant, which would make
    the nested loop decorative (the W4 verifier's finding). All five criteria are deterministic
    and every one of them reads the training cases only; none looks at the held-out case.
    """
    names = list(files) if files is not None else bank.files
    scored = []
    for i, cand in enumerate(candidates):
        s = loco_scores(bank, labels, cand, files=names)
        sep = round(separation(bank, labels, cand, files=names), 6)
        scored.append((-s["score"], -s["top1"], -sep, len(cand.features), i, cand, s, sep))
    scored.sort(key=lambda t: t[:5])
    best = scored[0]
    n_tied = sum(1 for t in scored if t[0] == best[0] and t[1] == best[1])
    n_tied_after_sep = sum(1 for t in scored if t[:3] == best[:3])
    return ACVRanker(
        **{
            **best[5].as_dict(),
            "trained_on": tuple(sorted(names)),
            "cv": {
                "selection": "chosen on the training cases of this fold",
                "train_score": best[6]["score"],
                "train_top1": best[6]["top1"],
                "train_separation": best[7],
                "n_candidates": len(candidates),
                "n_tied_at_top": n_tied,
                "n_tied_after_separation": n_tied_after_sep,
            },
        }
    )


def leave_one_case_out(
    bank: FeatureBank,
    labels: Mapping[str, str],
    candidates: Sequence[ACVRanker] | None = None,
) -> dict[str, Any]:
    """The frozen ACV scheme: for each case, freeze the rule on the other five, then rank it.

    ``candidates=None`` means the rule is fixed (the baseline): the fold still re-runs, it simply
    has nothing to choose. With candidates, the choice is made on the five training cases and the
    held-out case is read only afterwards - the fold-local rule for a rule-based ranker.
    """
    files = sorted(bank.files)
    folds: list[dict[str, Any]] = []
    for held in files:
        train_files = [f for f in files if f != held]
        chosen = (
            BASELINE_RANKER
            if not candidates
            else fit_ranker(bank, labels, candidates, files=train_files)
        )
        feats = bank.for_ranker(chosen)[held]
        ranked, scores = rank_cars(feats, chosen)
        true = labels[held]
        r = ranked.index(true) + 1 if true in ranked else 0
        folds.append(
            {
                "held_out": held,
                "chosen_ranker": chosen.name,
                "chosen_features": list(chosen.features),
                "chosen_weights": list(chosen.weights),
                "chosen_aggregation": chosen.aggregation,
                "train_score": float(chosen.cv["train_score"]) if chosen.cv.get("train_score") is not None else None,
                "train_separation": chosen.cv.get("train_separation"),
                "n_tied_at_top": chosen.cv.get("n_tied_at_top"),
                "n_tied_after_separation": chosen.cv.get("n_tied_after_separation"),
                "n_candidates": chosen.cv.get("n_candidates"),
                "true_car": true,
                "rank": int(r),
                "ranked_cars": ranked,
                "score": float(rank_decay(ranked, true, n=len(ranked))),
                "margin": _margin(scores, ranked),
            }
        )
    vals = [f["score"] for f in folds]
    return {
        "scheme": "leave-one-case-out (6 cases, exploratory)",
        "folds": folds,
        "score": float(np.mean(vals)) if vals else 0.0,
        "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "top1": float(np.mean([f["rank"] == 1 for f in folds])) if folds else 0.0,
        "n_cases": len(folds),
    }


def chance_floors(feats_by_file: Mapping[str, pd.DataFrame]) -> dict[str, float]:
    """The two floors a rank-decay number has to beat, both averaged over the cases.

    ``uniform``
        a uniformly random permutation of the *n* cars of a case: ``(n + 1) / (2 n)``, i.e.
        0.5625 for eight cars. This is what the table quoted until now.
    ``blind``
        the honest floor, because it is what a **signal-free** ranker actually scores here:
        `rank_cars` sorts cars with no usable data last for free, and ``acv_case_04.xlsx`` has
        four entirely empty cars, so a blind ranking of that case draws the true car uniformly
        from the four rankable ones and expects ``(2n - k + 1) / (2n)`` = 0.8125. Over the six
        cases that is **0.6042**, not 0.5625 - every row in the ladder has that much less headroom
        than the uniform floor suggests.
    """
    uniform: list[float] = []
    blind: list[float] = []
    for frame in feats_by_file.values():
        n = len(frame.index)
        if n == 0:
            continue
        k = int(frame["enough_data"].astype(bool).sum()) if "enough_data" in frame.columns else n
        k = k or n  # a case with no usable car at all still produces a full (header-order) ranking
        uniform.append((n + 1) / (2 * n))
        blind.append(float(np.mean([(n - (r - 1)) / n for r in range(1, k + 1)])))
    return {
        "uniform": float(np.mean(uniform)) if uniform else 0.0,
        "blind": float(np.mean(blind)) if blind else 0.0,
    }


def _margin(scores: pd.DataFrame, ranked: Sequence[str]) -> float:
    s = pd.to_numeric(scores["score"], errors="coerce")
    if len(ranked) < 2:
        return float("nan")
    a, b = s.get(ranked[0], np.nan), s.get(ranked[1], np.nan)
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(a - b)


# --------------------------------------------------------------------------------------
# The Task
# --------------------------------------------------------------------------------------


class ACVTask(BaseTask):
    """`load` -> one `ACVCase`; `featurise` -> per-car feature frame; `predict` -> one CSV row."""

    name = "acv"

    def load(self, path: Path | str) -> af.ACVCase:
        return af.load_case(path)

    def featurise(self, raw: af.ACVCase, model: Any = None) -> pd.DataFrame:
        """Per-car feature frame, built with **the model's own mask**.

        ``hot_quantile`` / ``cooling_only`` are part of the rule, not of the loader: 14 of the 106
        ladder rows use a non-default mask. Building the frame at the default while the ranker
        says otherwise would ship predictions computed differently from the cross-validated rule,
        silently - so the frame follows ``ranker.feature_config`` here and :meth:`predict` refuses
        a frame that disagrees. ``model=None`` keeps the committed artefact's mask (the baseline's
        0.5 / cooling-only when there is no artefact).
        """
        if not isinstance(raw, af.ACVCase):
            raise TypeError(f"ACVTask.featurise expects an ACVCase, got {type(raw).__name__}")
        hot_quantile, cooling_only = _as_ranker(model).feature_config
        return af.car_features(raw, hot_quantile=hot_quantile, cooling_only=cooling_only)

    def predict(self, feats: pd.DataFrame, model: Any = None) -> PredictionResult:
        ranker = _as_ranker(model)
        _check_feature_config(feats, ranker)
        ranked, scores = rank_cars(feats, ranker)
        file_id = str(feats.attrs.get("file_id", "") or "")
        top = ranked[0] if ranked else ""
        numbers: dict[str, float] = {}
        for car in feats.index:
            v = pd.to_numeric(feats.loc[car, "peer_delta_hot"], errors="coerce")
            numbers[f"car_{car}_peer_delta_hot_k"] = float(v) if np.isfinite(v) else 0.0
        if top:
            numbers["top_car"] = float(int(top))
            numbers["top_car_peer_delta_hot_k"] = numbers.get(f"car_{top}_peer_delta_hot_k", 0.0)
            cool = pd.to_numeric(feats.loc[top, "n_usable"], errors="coerce") / max(
                float(pd.to_numeric(feats.loc[top, "n_rows"], errors="coerce")), 1.0
            )
            numbers["top_car_cooling_fraction"] = float(cool) if np.isfinite(cool) else 0.0
            numbers["rank_margin_to_2nd"] = float(_margin(scores, ranked))
            if not np.isfinite(numbers["rank_margin_to_2nd"]):
                numbers["rank_margin_to_2nd"] = 0.0
        # The Info Kit warns the parameter set differs between files; say how many of this file's
        # own parameters the canonical schema could not place (the names are in ``extras``).
        numbers["n_unmapped_parameters"] = float(len(feats.attrs.get("unmapped", [])))
        return PredictionResult(
            task="acv",
            file_id=file_id,
            rows=[{"file_id": file_id, "ranked_cars": "|".join(ranked)}],
            numbers=numbers,
            trace=None,
            viewport=_viewport(top),
            extras={
                "ranked_cars": ranked,
                "scores": scores,
                "features": feats,
                "ranker": ranker.as_dict(),
                "unmapped_parameters": list(feats.attrs.get("unmapped", [])),
                "hot_threshold": feats.attrs.get("hot_threshold"),
                "hot_driver": feats.attrs.get("hot_driver"),
            },
        )

    def run(self, path: Path | str, model: Any = None) -> PredictionResult:
        """`load` -> `featurise` -> `predict`, keeping the trace (which needs the raw panel)."""
        case = self.load(path)
        feats = self.featurise(case, model)
        feats.attrs.setdefault("file_id", case.file_id)
        result = self.predict(feats, model)
        if not result.file_id:
            result.file_id = case.file_id
            for row in result.rows:
                row["file_id"] = case.file_id
        result.trace = _peer_trace(
            case, result.extras.get("ranked_cars", [None])[0], ranker=_as_ranker(model)
        )
        return result


def _viewport(car: str) -> Viewport:
    if not car:
        return Viewport(health="warn", component="")
    n = int(car)
    return Viewport(car=n if 1 <= n <= 8 else None, health="crit", component=f"car{n}_ac1")


def _check_feature_config(feats: pd.DataFrame, ranker: ACVRanker) -> None:
    """Refuse a feature frame built with a different mask from the one the ranker was chosen on.

    The frame records its own ``(hot_quantile, cooling_only)`` (`acv_features.car_features`), so
    the deployed path cannot silently disagree with the cross-validated one. Frames from before
    feature version ``acv-f5`` carry no configuration and are accepted unchecked.
    """
    got = feats.attrs.get("feature_config")
    if got is None:
        return
    got = (round(float(got[0]), 3), bool(got[1]))
    if got != ranker.feature_config:
        raise ValueError(
            f"ACV feature frame was built with hot_quantile/cooling_only {got}, but ranker "
            f"{ranker.name!r} needs {ranker.feature_config}; re-featurise with this ranker "
            "(ACVTask.featurise(case, ranker))"
        )


def _peer_trace(
    case: af.ACVCase, car: str | None, *, ranker: ACVRanker = BASELINE_RANKER
) -> Trace | None:
    """Indoor minus the fleet median for the top-ranked car, over the whole record.

    Downsampled to :data:`MAX_TRACE_POINTS` by block mean so the payload stays small; the mark is
    the worst sustained excursion.
    """
    if not car or car not in case.cars:
        return None
    hot_quantile, cooling_only = ranker.feature_config
    m = af.case_masks(case, hot_quantile=hot_quantile, cooling_only=cooling_only)
    ind = case.numeric("indoor").where(m["usable"])
    peer = ind.median(axis=1, skipna=True)
    delta = (ind[car] - peer).astype(float)
    delta = delta[delta.notna()]
    if delta.empty:
        return None
    if len(delta) > MAX_TRACE_POINTS:
        step = int(np.ceil(len(delta) / MAX_TRACE_POINTS))
        block_mean = delta.groupby(np.arange(len(delta)) // step).mean()
        delta = pd.Series(block_mean.to_numpy(), index=delta.index[::step][: len(block_mean)])
    x = [pd.Timestamp(t).isoformat() for t in delta.index]
    y = [float(v) for v in delta.to_numpy(dtype=float)]
    peak = int(np.nanargmax(y))
    marks = [{"x": x[peak], "label": f"+{y[peak]:.2f} K vs fleet", "kind": "peak"}]
    return Trace(x=x, y=y, marks=marks, label=f"Car {car} indoor - fleet median (K), cooling rows")


def _as_ranker(model: Any) -> ACVRanker:
    if model is None:
        try:
            model = load_model("acv")
        except FileNotFoundError:
            return BASELINE_RANKER
    if isinstance(model, ACVRanker):
        return model
    if isinstance(model, Mapping):
        return ACVRanker(**dict(model))
    raise TypeError(f"ACV model must be an ACVRanker or a mapping, got {type(model).__name__}")


register_task(ACVTask())


# --------------------------------------------------------------------------------------
# Training entry point (scripts/ps3_train.py --task acv)
# --------------------------------------------------------------------------------------


def _write_predictions(
    ranker: ACVRanker, out_csv: Path, *, root: Path | str | None = None
) -> list[dict[str, Any]]:
    """Run every distributed Test file through the same code path the app uses."""
    task = ACVTask()
    rows: list[dict[str, Any]] = []
    for p in sorted(test_dir("acv", root=root).glob("*.xlsx")):
        if p.name.startswith("~$"):
            continue
        res = task.run(p, ranker)
        rows.extend(task.to_rows(res))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["file_id", "ranked_cars"]).to_csv(out_csv, index=False, lineterminator="\n")
    return rows


def _wall_seconds(wall_s: float, cost: Mapping[str, Any]) -> dict[str, Any]:
    """This run's wall clock, split into what it cost **warm** and what it costs **cold**.

    A warm run reads six small parquet frames per mask configuration; a cold one parses the six
    workbooks, and ``acv_case_04.xlsx`` alone (22k rows x 483 columns) is ~55 s of that per
    configuration. Quoting only the warm number - 3.3 s in the first committed artefact - reads as
    the cost of the method and is off by two orders of magnitude, so both are recorded here:

    ``warm``
        this run, as measured (``cache_hits`` of ``n_frames`` feature frames came from parquet);
    ``cold``
        the same run with every feature frame rebuilt from the xlsx, using each frame's own
        **recorded** build time (`acv_features.cached_features` stores it in the cache sidecar
        when the frame is first computed).
    """
    warm = float(wall_s)
    cold = warm - float(cost.get("warm_seconds", 0.0)) + float(cost.get("cold_seconds", 0.0))
    return {
        "warm": round(warm, 2),
        "cold": round(cold, 2) if cost.get("cold_seconds_known") else None,
        "measured": "cold" if cost.get("n_cache_hits", 0) == 0 else (
            "warm" if cost.get("n_cache_misses", 0) == 0 else "mixed"
        ),
        "feature_frames": cost.get("n_frames"),
        "cache_hits": cost.get("n_cache_hits"),
        "feature_seconds_warm": cost.get("warm_seconds"),
        "feature_seconds_cold": cost.get("cold_seconds"),
    }


def _cv_payload(
    cv: Mapping[str, Any],
    ranker: ACVRanker,
    wall_s: float,
    *,
    floors: Mapping[str, float],
    cost: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    wall = _wall_seconds(wall_s, cost)
    payload = {
        "task": "acv",
        "scheme": cv["scheme"],
        "folds": cv["n_cases"],
        "seeds": [0],
        "git_rev": git_rev(),
        "feature_version": af.FEATURE_VERSION,
        "wall_seconds": wall["warm"],
        "wall_seconds_warm": wall["warm"],
        "wall_seconds_cold": wall["cold"],
        "wall_seconds_detail": wall,
        "metric": "rank_decay (ACV Info Kit section 4)",
        "score": cv["score"],
        "sd": cv["sd"],
        "top1": cv["top1"],
        "headline": (
            f"{cv['score']:.4f} +/- {cv['sd']:.3f} (exploratory, {cv['n_cases']} cases)"
        ),
        # The floor a signal-free ranker actually reaches on these six cases (empty cars sort
        # last for free); the uniform floor is kept beside it for comparability.
        "chance_floor": round(float(floors["blind"]), 4),
        "chance_floor_blind": round(float(floors["blind"]), 4),
        "chance_floor_uniform": round(float(floors["uniform"]), 4),
        "exploratory": True,
        "ranker": ranker.as_dict(),
        "per_fold": cv["folds"],
    }
    payload.update(dict(extra or {}))
    return payload


def _cv_markdown(payload: Mapping[str, Any]) -> str:
    r = payload["ranker"]
    wall = payload.get("wall_seconds_detail", {})
    cold = wall.get("cold")
    best = payload.get("ladder_best") or {}
    lines = [
        "# ACV - leave-one-case-out (exploratory)",
        "",
        f"Rule `{r['name']}`: features {list(r['features'])}, aggregation `{r['aggregation']}`, "
        f"tie-break {list(r['tie_break'])}, hot quantile {r['hot_quantile']}.",
        f"Feature version `{payload['feature_version']}`, git `{payload['git_rev']}`, "
        f"{payload['wall_seconds']} s wall warm"
        + (f" / {cold} s cold (every feature frame rebuilt from the xlsx)." if cold else "."),
        "",
        f"**Mean rank-decay {payload['score']:.4f} +/- {payload['sd']:.3f}** over {payload['folds']} cases "
        f"(top-1 {payload['top1']:.3f}), **exploratory**. A signal-free ranker that leaves the empty "
        f"cars last scores {payload['chance_floor_blind']:.4f} on these six cases - that, not the "
        f"{payload['chance_floor_uniform']:.4f} a uniformly random ranking of 8 cars scores, is the "
        "floor this number has to beat (`acv_case_04.xlsx` has four entirely empty cars, so a blind "
        "ranking of it already scores 0.8125).",
        "",
    ]
    if best:
        lines += [
            f"This headline is the **pre-registered baseline** rule (`{r['name']}`), evaluated on all "
            f"six cases. The ladder's best row, `{best.get('row')}`, reaches "
            f"{float(best.get('score', 0.0)):.4f} - but it was **{best.get('label')}**, so that number "
            "is a selection result, not a held-out one, and `train()` deliberately does not promote "
            "it: a rule picked on the same six cases it is then scored on cannot be the headline "
            "under the plan's fold-local rule. Both rules produce the identical row for the "
            "organisers' Test file, so nothing about the deliverable turns on the choice - only the "
            "claim does. `acv_ladder.md` keeps the full table.",
            "",
        ]
    if payload.get("ladder_selected_score") is not None:
        chosen = payload.get("ladder_selected_per_fold") or {}
        counts = sorted(
            {name: sum(1 for v in chosen.values() if v == name) for name in set(chosen.values())}.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        picks = ", ".join(f"`{name}` on {n} of {len(chosen)} folds" for name, n in counts)
        same = abs(float(payload["ladder_selected_score"]) - float(payload["score"])) < 5e-4
        lines += [
            "Re-choosing the rule inside every fold (the pre-registered pool of "
            "`acv.selection_pool`, five training cases per fold, rank decay then top-1 then "
            f"`acv.separation`) scores {float(payload['ladder_selected_score']):.4f}"
            + (" - the same as this fixed baseline" if same else "")
            + (f". The folds do not all pick the same rule: {picks}." if len(counts) > 1 else
               f". Every fold picks {picks}."),
            "",
        ]
    lines += [
        "Six cases is too few for a confidence interval: one case moving one rank changes the mean by "
        "0.021, and the 0.9792 -> 1.0000 gap in the ladder is exactly that - one rank step on "
        "`acv_case_04.xlsx`, whose top two cars are 4.2 mK apart. The ranker is peer-normalised "
        "inside each case, so no statistic can cross cases.",
        "",
        "| held-out case | true car | rank | score | ranked_cars | rule chosen on the other 5 |",
        "|---|---|---|---|---|---|",
    ]
    for f in payload["per_fold"]:
        order = "|".join(f["ranked_cars"])
        lines.append(
            f"| `{f['held_out']}` | {f['true_car']} | {f['rank']} | {f['score']:.3f} | "
            f"`{order}` | `{f['chosen_ranker']}` |"
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# The ladder (plan W4, `docs/research/ps3_addendum.md` / finder C1, C4, C6, C13)
# --------------------------------------------------------------------------------------

#: Every single-feature ranker the plan and the research finder name, in ladder order.
SINGLE_FEATURE_ROWS: tuple[tuple[str, str], ...] = (
    ("peer_delta_hot", "hot-hour peer delta (plan baseline)"),
    ("peer_delta_all", "all-hours peer delta"),
    ("peer_delta_hot_loo", "hot-hour peer delta, leave-one-car-out reference"),
    ("peer_delta_steady", "hot-hour peer delta, steady-state rows only [C13]"),
    ("robust_peer_z", "robust peer z (per-timestamp MAD)"),
    ("ctrl_residual", "mode/validity-conditioned control residual (indoor - setpoint)"),
    ("ctrl_residual_all", "control residual, all hours"),
    ("above_setpoint_frac", "time above setpoint [C6]"),
    ("unmet_degree_min", "unmet degree-minutes [C6]"),
    ("cooling_duty", "cooling-mode duty"),
    ("full_cooling_duty", "full-cooling duty [C6]"),
    ("mode_switch_rate", "mode-transition rate (drive cycles) [C4]"),
    ("load_halved_share", "load-halved share"),
    ("pulldown_penalty", "negated pull-down rate [C4]"),
    ("persistence_frac", "persistence: share of hot hours ranked worst"),
    ("peer_delta_trend", "trend of the daily peer delta [C16]"),
    ("guo_residual", "peer-regression residual, median over peers [C1]"),
    ("guo_index", "signed fault index over the peer residuals [C1]"),
)

#: Multi-feature aggregations (rank aggregation / weighted within-case z).
COMBINATION_ROWS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("borda_peer_ctrl", ("peer_delta_hot", "ctrl_residual"), "borda"),
    ("borda_peer_ctrl_unmet", ("peer_delta_hot", "ctrl_residual", "unmet_degree_min"), "borda"),
    ("borda_guo_peer", ("guo_residual", "peer_delta_hot"), "borda"),
    ("borda_all_five", ("peer_delta_hot", "ctrl_residual", "unmet_degree_min", "guo_residual", "pulldown_penalty"), "borda"),
    ("zsum_peer_ctrl", ("peer_delta_hot", "ctrl_residual"), "zsum"),
    ("zsum_peer_ctrl_unmet", ("peer_delta_hot", "ctrl_residual", "unmet_degree_min"), "zsum"),
    ("zsum_guo_peer", ("guo_residual", "peer_delta_hot"), "zsum"),
    ("zsum_all_five", ("peer_delta_hot", "ctrl_residual", "unmet_degree_min", "guo_residual", "pulldown_penalty"), "zsum"),
)

#: Mask x season-half ablation grid: ``(label, hot_quantile, cooling_only)``.
MASK_ROWS: tuple[tuple[str, float, bool], ...] = (
    ("cooling/hot", 0.5, True),
    ("cooling/all", 0.0, True),
    ("cooling/hottest-quarter", 0.75, True),
    ("all-on-rows/hot", 0.5, False),
)


def selection_pool() -> list[ACVRanker]:
    """The **pre-registered, ordered** pool a fold may choose from.

    Order is a prior, and :func:`fit_ranker` uses it to break ties, so it has to be justified
    rather than alphabetical:

    1. ``peer_delta_hot_loo`` - the strict peer reference: a car's own reading is excluded from
       the median it is compared against. This is the form [C1] (median over *peer* predictions)
       and [R78] (`features.peer_normalise`) prescribe, and including a car in its own reference
       measurably dilutes a large excursion, so it is the a-priori first choice.
    2. ``robust_peer_z`` - the same delta, scaled by the across-car MAD at each timestamp, so a
       case whose fleet is noisy does not inflate the number.
    3. ``peer_delta_hot`` - the pooled-median approximation, i.e. the plan's baseline.
    4. ``peer_delta_steady`` - the same on steady-state rows only ([C13]'s transient gate).
    5. a Borda aggregation of (1), (2) and the control residual - the rank-aggregation row.

    **Why the pool is this small.** Not because the open pool scores worse - it does not. Letting
    a fold choose from all 26 rankers scores 0.9792 over the six cases, the same as this pool and
    the same as the fixed baseline (`acv_ladder.md` keeps that row; an earlier version of this
    docstring quoted 0.854, which the committed ladder contradicts - the W4 verifier's finding).
    The reason is that on five training cases a dozen open-pool candidates tie at 1.000 on rank
    decay, so most of the pool is separated only by :func:`separation`, a margin statistic that a
    single wild car can dominate. Restricting the pool to one physically motivated family -
    peer-normalised indoor-temperature deltas, which is what the whole peer-fleet FDD literature
    says is the primary observable when there are no pressures - keeps the choice inside a family
    whose members differ only in their reference, so a noisy pick costs little. The restriction
    was made *after* seeing the full ladder, so any number produced by selecting on these six
    cases keeps an optimistic bias; that is why the committed headline is the **pre-registered
    baseline** and not a selected row (see :func:`train` and `results/ps3/acv_cv.md`).
    """
    return [
        ACVRanker(features=("peer_delta_hot_loo",), tie_break=("ctrl_residual",), name="peer_delta_hot_loo"),
        ACVRanker(features=("robust_peer_z",), tie_break=("peer_delta_hot_loo",), name="robust_peer_z"),
        ACVRanker(features=("peer_delta_hot",), tie_break=("ctrl_residual",), name="peer_delta_hot"),
        ACVRanker(features=("peer_delta_steady",), tie_break=("ctrl_residual",), name="peer_delta_steady"),
        ACVRanker(
            features=("peer_delta_hot_loo", "robust_peer_z", "ctrl_residual"),
            aggregation="borda",
            tie_break=("peer_delta_hot_loo",),
            name="borda_peer_family",
        ),
    ]


def candidate_rankers(
    *, masks: Sequence[tuple[str, float, bool]] = MASK_ROWS, full_grid: bool = False
) -> list[ACVRanker]:
    """Every ladder row: all rankers x the mask/season-half ablations.

    Without ``full_grid`` this is the whole ranker list under the default mask - the "open" pool
    the ladder keeps as a cautionary row; the pool a fold actually selects from is
    :func:`selection_pool`.
    """
    grid = list(masks) if full_grid else [("cooling/hot", 0.5, True)]
    out: list[ACVRanker] = []
    for mask_label, hq, cooling_only in grid:
        suffix = "" if mask_label == "cooling/hot" else f" [{mask_label}]"
        for feat, _desc in SINGLE_FEATURE_ROWS:
            out.append(
                ACVRanker(
                    features=(feat,),
                    aggregation="single",
                    tie_break=("ctrl_residual",) if feat != "ctrl_residual" else ("peer_delta_hot",),
                    hot_quantile=hq,
                    cooling_only=cooling_only,
                    name=f"{feat}{suffix}",
                )
            )
        for name, feats, agg in COMBINATION_ROWS:
            out.append(
                ACVRanker(
                    features=feats,
                    weights=(1.0,) * len(feats),
                    aggregation=agg,
                    tie_break=("peer_delta_hot",),
                    hot_quantile=hq,
                    cooling_only=cooling_only,
                    name=f"{name}{suffix}",
                )
            )
    return out


def run_ladder(bank: FeatureBank, labels: Mapping[str, str]) -> dict[str, Any]:
    """Every ranker x ablation on the frozen outer CV, plus the nested per-fold selection row.

    A fixed rule has nothing to fit, so "applied to all six cases" *is* its leave-one-case-out
    score; the `selected (nested)` row is the honest number for the *procedure* that chooses a
    rule, because there the choice is remade on five cases per fold.
    """
    rows: list[dict[str, Any]] = []
    for cand in candidate_rankers(full_grid=True):
        t0 = time.perf_counter()
        s = loco_scores(bank, labels, cand)
        rows.append(
            {
                "row": cand.name,
                "degenerate": _is_degenerate(bank, cand),
                "degenerate_reason": _degenerate_reason(bank, cand),
                "features": list(cand.features),
                "n_features": len(cand.features),
                "aggregation": cand.aggregation,
                "hot_quantile": cand.hot_quantile,
                "cooling_only": cand.cooling_only,
                "score": s["score"],
                "sd": s["sd"],
                "top1": s["top1"],
                "mean_rank": s["mean_rank"],
                "per_file": s["per_file"],
                "ranks": s["ranks"],
                "fit_seconds": round(time.perf_counter() - t0, 3),
            }
        )
    for pool_name, pool in (
        ("selection pool (peer family, 5)", selection_pool()),
        ("open pool (all 26 rankers)", candidate_rankers()),
    ):
        t0 = time.perf_counter()
        nested = leave_one_case_out(bank, labels, pool)
        rows.append(_nested_row(f"selected (nested, {pool_name})", nested, time.perf_counter() - t0))
    rows.sort(key=lambda r: (-r["score"], -r["top1"], r["n_features"], r["row"]))
    return {"rows": rows, "nested": nested}


def _nested_row(label: str, nested: Mapping[str, Any], seconds: float) -> dict[str, Any]:
    return (
        {
            "row": label,
            "features": sorted({f for fold in nested["folds"] for f in fold["chosen_features"]}),
            "n_features": len({f for fold in nested["folds"] for f in fold["chosen_features"]}),
            "aggregation": "per-fold",
            "hot_quantile": 0.5,
            "cooling_only": True,
            "nested": True,
            "score": nested["score"],
            "sd": nested["sd"],
            "top1": nested["top1"],
            "mean_rank": float(np.mean([f["rank"] for f in nested["folds"]])),
            "per_file": {f["held_out"]: f["score"] for f in nested["folds"]},
            "ranks": {f["held_out"]: f["rank"] for f in nested["folds"]},
            "fit_seconds": round(seconds, 3),
            "chosen_per_fold": {f["held_out"]: f["chosen_ranker"] for f in nested["folds"]},
            "n_candidates": next((f.get("n_candidates") for f in nested["folds"]), None),
            "tied_at_top_per_fold": {f["held_out"]: f.get("n_tied_at_top") for f in nested["folds"]},
        "separation_per_fold": {f["held_out"]: f.get("train_separation") for f in nested["folds"]},
        }
    )


def _degenerate_reason(bank: FeatureBank, ranker: ACVRanker) -> str | None:
    """Why a ladder row's score is its tie-break's, not its feature's - or ``None``.

    Two ways a row can claim a signal it does not have, both judged on the row's **own** features
    over the cars of each case (a multi-feature row only counts as degenerate when *every* one of
    its features is):

    ``constant``
        the feature takes one value across the cars, so it orders nothing. ``load_halved_share``
        is exactly this: the organisers' `Load Halved` column reads `Normal` for every car of
        every case, so the feature is 0 everywhere. ``cooling_duty`` likewise.
    ``few-valued``
        the feature takes at most :data:`nebulax.ps3.acv_features.MIN_DISTINCT_LEVELS` distinct
        values across the cars, so it cannot order an eight-car fleet and its top tier is a tie
        the tie-break resolves. ``robust_peer_z`` is this row: the per-timestamp MAD scaling
        quantises the median z to multiples of 0.6745, three levels per case, and stripping its
        `peer_delta_hot_loo` tie-break drops it from 1.0000 to the baseline's 0.9792 - so it is
        not independent corroboration of the peer-delta family, it *is* the peer delta.

    A row is flagged when the condition holds in at least half of the cases.
    """
    feats = bank.for_ranker(ranker)
    if not feats:
        return None
    half = max(1, len(feats) // 2)
    levels = {
        name: [
            int(pd.to_numeric(frame[name], errors="coerce").dropna().nunique())
            for frame in feats.values()
        ]
        for name in ranker.features
    }
    if all(sum(1 for n in v if n <= 1) >= half for v in levels.values()):
        return "constant"
    if all(sum(1 for n in v if n <= af.MIN_DISTINCT_LEVELS) >= half for v in levels.values()):
        return f"<= {af.MIN_DISTINCT_LEVELS} distinct values per case"
    return None


def _is_degenerate(bank: FeatureBank, ranker: ACVRanker) -> bool:
    """True when the row's score is produced by its tie-break rather than by its own feature.

    See :func:`_degenerate_reason`; such a row must be marked, or the table claims a signal that
    does not exist.
    """
    return _degenerate_reason(bank, ranker) is not None


def _ladder_markdown(payload: Mapping[str, Any]) -> str:
    rows = payload["rows"]
    best = rows[0]["score"]
    open_row = next((r for r in rows if r.get("nested") and "open pool" in r["row"]), None)
    worst_tie = ""
    if open_row:
        tied = [v for v in (open_row.get("tied_at_top_per_fold") or {}).values() if v]
        if tied:
            worst_tie = (
                f" (on one fold {max(tied)} of the {open_row.get('n_candidates')} open-pool "
                "candidates are tied)"
            )
    default_rows = [r for r in rows if r["cooling_only"] and r["hot_quantile"] == 0.5]
    top_fixed = next((r for r in rows if not r.get("nested")), None)
    top_txt = f"`{top_fixed['row']}`, {top_fixed['score']:.4f}" if top_fixed else "-"
    lines = [
        "# ACV ladder - leave-one-case-out rank decay (exploratory, 6 cases)",
        "",
        f"Feature version `{payload['feature_version']}`, git `{payload['git_rev']}`, "
        f"{payload['wall_seconds']} s wall, {len(rows)} rows.",
        "",
        "Every row is a **rule**, not a fitted model: with 6 cases x 8 cars and one positive per case "
        "there is nothing to train a classifier on, and a classifier would be free to learn a car-id "
        "prior, which is exactly wrong here. A fixed rule has no parameters to fit inside a fold, so "
        "applying it to all six cases *is* its leave-one-case-out score. The two `selected (nested, "
        "...)` rows are the ones that choose something: they re-choose the rule on the five training "
        "cases of every fold, and are the honest numbers for a selection *procedure* rather than for a "
        "rule.",
        "",
        "**No row in this table is the committed headline.** `train()` does not promote a ladder "
        "winner: every fixed row here is scored on the same six cases the table is read on, so the "
        f"top row ({top_txt}) is **selected on all six cases**, not a "
        "held-out number. The committed artefact and the headline in `acv_cv.md` are the "
        "pre-registered baseline rule at 0.9792 +/- 0.051. Both produce the identical row for the "
        "organisers' Test file.",
        "",
        "**Why the fold selects from a five-rule pool, not from all 26.** The `tied at top` column "
        "below counts how many candidates were tied at the best training rank decay when that fold "
        f"made its choice. With the open pool most of it ties on five cases{worst_tie}, so the pick "
        "rests on `acv.separation` (the margin the rule puts between the labelled car and its peers "
        "on the training cases), a statistic one wild car can dominate; `acv.selection_pool` "
        "therefore restricts the pool to one physically motivated family - peer-normalised "
        "indoor-temperature deltas, the primary observable when the telemetry has no pressures - and "
        "pre-registers its order (strict leave-one-car-out reference first, as [C1] and [R78] "
        "prescribe). Both nested rows land on 0.9792, i.e. on the baseline. That restriction was "
        "made after seeing this table, which is the other reason the headline is the baseline and "
        "not a row from here.",
        "",
        f"Chance floor: a signal-free ranker that leaves the empty cars last scores "
        f"**{payload.get('chance_floor', 0.0):.4f}** on these six cases (`acv_case_04.xlsx` has four "
        f"entirely empty cars, so a blind ranking of it already scores 0.8125); a uniformly random "
        f"ranking of 8 cars scores {payload.get('chance_floor_uniform', 0.5625):.4f}. "
        f"sd is over the {payload['n_cases']} cases, not over seeds (the rules are deterministic).",
        "",
        "## Main table (cooling-mode rows, hotter half of the record)",
        "",
        "| rank | row | score | sd | top-1 | mean rank of true car | n feat | fit s | note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(default_rows, start=1):
        bold = "**" if r["score"] >= best - 1e-12 else ""
        if r.get("degenerate"):
            note = f"tie-break only ({r.get('degenerate_reason', 'feature is constant')})"
        elif r.get("nested"):
            note = "rule re-chosen per fold"
        elif top_fixed is not None and r["row"] == top_fixed["row"]:
            note = "selected on all six cases (not held out; not the artefact)"
        else:
            note = ""
        lines.append(
            f"| {i} | {bold}`{r['row']}`{bold} | {bold}{r['score']:.4f}{bold} | {r['sd']:.4f} | "
            f"{r['top1']:.3f} | {r['mean_rank']:.2f} | {r['n_features']} | {r['fit_seconds']:.3f} | {note} |"
        )
    lines += [
        "",
        "## Mask x season-half ablation",
        "",
        "`cooling/hot` = cooling-mode rows in the hotter half of the record (default); `cooling/all` = "
        "every cooling row; `cooling/hottest-quarter` = the top 25 % of ambient; `all-on-rows/hot` = "
        "every row the unit is switched on for, cooling or not.",
        "",
        "| row | score | sd | top-1 |",
        "|---|---|---|---|",
    ]
    lines[-1] = "|---|---|---|---|---|"
    lines[-2] = "| row | score | sd | top-1 | note |"
    for r in rows:
        if r in default_rows:
            continue
        note = f"tie-break only ({r['degenerate_reason']})" if r.get("degenerate") else ""
        lines.append(
            f"| `{r['row']}` | {r['score']:.4f} | {r['sd']:.4f} | {r['top1']:.3f} | {note} |"
        )
    lines += [
        "",
        "## Per-fold selection",
        "",
        "A fold's rule is chosen on its five training cases by rank decay, then top-1, then "
        "`acv.separation` (the training margin between the labelled car and its peers, in MAD "
        "units), then the fewest features, then the pool order. `tied at top` counts the candidates "
        "the first two criteria could not separate - the reason the third exists.",
        "",
        "| pool | held-out case | rule chosen on the other five | tied at top | train separation |",
        "|---|---|---|---|---|",
    ]
    for nested_row in [r for r in rows if r.get("nested")]:
        tied = nested_row.get("tied_at_top_per_fold", {})
        sep = nested_row.get("separation_per_fold", {})
        n_cand = nested_row.get("n_candidates")
        for fid, name in nested_row.get("chosen_per_fold", {}).items():
            sep_txt = f"{sep[fid]:.2f}" if isinstance(sep.get(fid), (int, float)) else "-"
            lines.append(
                f"| {nested_row['row']} | `{fid}` | `{name}` | {tied.get(fid)} of {n_cand} | {sep_txt} |"
            )
    lines += [
        "",
        "## Skipped for budget, and why",
        "",
        "* **Supervised classifiers / deep models** - 6 cases x 8 cars = 48 rows with 6 positives. The "
        "finder's explicit \"do not\" ([C1] practitioner notes); nothing to gain and a car-id prior to "
        "lose.",
        "* **Virtual refrigerant-charge sensors and RP-1043-style decoupling features** ([C7], [C8]) - "
        "they need subcooling, superheat, pressures or power. Five of the seven files carry eight "
        "temperature/mode parameters per car and none of those signals; only `acv_case_04.xlsx` has "
        "pressures, and the **test file does not**, so a feature built on them could never run on the "
        "held-out case.",
        "* **3R2C grey-box + extended Kalman filter** ([C4]) - 4-6 h and non-convex identification "
        "for a pull-down feature the seven peer cars already supply for free; the cheap version "
        "(`pulldown_penalty`) is in the table.",
        "* **Per-position (car-id) offset correction** - the finder suggests subtracting a per-position "
        "feature offset when the faulty car is not uniform across cases. It is not uniform here "
        "(cars 01, 02, 03, 01, 04, 06), but the offset would have to be estimated from training cases "
        "whose own faulty car sits in the same skewed positions, so it would systematically discount "
        "exactly the cars that are usually faulty. Deliberately **not implemented**: the ranker must "
        "never carry a car-id prior, in either direction.",
        "",
    ]
    return "\n".join(lines)


def _best_ladder_row(ladder_payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """The ladder's top fixed rule, labelled for what it is: selected on all six cases."""
    rows = [r for r in ladder_payload.get("rows", []) if not r.get("nested")]
    if not rows:
        return None
    best = max(rows, key=lambda r: (r["score"], r["top1"], -r["n_features"]))
    return {
        "row": best["row"],
        "score": best["score"],
        "sd": best["sd"],
        "top1": best["top1"],
        "label": "selected on all six cases",
        "note": (
            "not a held-out number: this row tops a table read on the same six cases it is scored "
            "on, so it is reported here and never promoted to the artefact."
        ),
    }


# --------------------------------------------------------------------------------------
# Training entry point (scripts/ps3_train.py --task acv)
# --------------------------------------------------------------------------------------


def train(args: Any = None) -> dict[str, Any]:
    """Fit/freeze the ACV ranker, run leave-one-case-out, write the artefact and the results.

    ``args`` is ``scripts/ps3_train.py``'s namespace (``--ladder``, ``--tag``, ``--data-root``,
    ``--out-dir``, ``--model-dir``, ``--no-predict``); ``None`` means "all defaults, baseline".

    **The committed artefact and the headline are always the pre-registered baseline rule.**
    ``--ladder`` runs the whole table and the per-fold selection, and reports both, but it no
    longer promotes a winner: with six cases, promoting the row that tops the ladder on the same
    six cases the ladder was read on - and then publishing that row's score - is selection on the
    CV used for the headline, which the plan's fold-local rule forbids. The ladder's best row and
    the nested selection score are written to `acv_cv.json` as clearly labelled ladder results
    (`ladder_best`, `ladder_selected_score`) instead. On the organisers' Test file the baseline
    and the ladder's best row produce the identical row, so the deliverable does not depend on
    this at all; only the claim does.
    """
    t0 = time.perf_counter()
    root = getattr(args, "data_root", None)
    ladder = bool(getattr(args, "ladder", False))
    tag = str(getattr(args, "tag", "baseline") or "")
    out_dir = Path(getattr(args, "out_dir", None) or RESULTS_DIR)
    model_dir = getattr(args, "model_dir", None)
    do_predict = bool(getattr(args, "predict", True))
    use_cache = not bool(getattr(args, "no_cache", False))

    labels = _labels(root)
    bank = FeatureBank(_case_paths(train_dir("acv", root=root)), use_cache=use_cache)
    missing = sorted(set(labels) - set(bank.files))
    if missing:
        raise FileNotFoundError(f"Train_Labels.csv lists case(s) with no xlsx: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ladder_payload: dict[str, Any] | None = None
    if ladder:
        lad = run_ladder(bank, labels)
        ladder_payload = {
            "task": "acv",
            "scheme": "leave-one-case-out (6 cases, exploratory)",
            "n_cases": len(bank.files),
            "seeds": [0],
            "git_rev": git_rev(),
            "feature_version": af.FEATURE_VERSION,
            "metric": "rank_decay (ACV Info Kit section 4)",
            "chance_floor": round(float(chance_floors(bank.for_config(0.5, True))["blind"]), 4),
            "chance_floor_uniform": round(float(chance_floors(bank.for_config(0.5, True))["uniform"]), 4),
            "wall_seconds": round(time.perf_counter() - t0, 2),
            "wall_seconds_detail": _wall_seconds(time.perf_counter() - t0, bank.cost()),
            "rows": lad["rows"],
        }
        (out_dir / "acv_ladder.json").write_text(
            json.dumps(ladder_payload, indent=2, default=str) + "\n", encoding="utf-8"
        )
        (out_dir / "acv_ladder.md").write_text(_ladder_markdown(ladder_payload), encoding="utf-8")

    candidates = selection_pool() if ladder else None
    cv = leave_one_case_out(bank, labels, candidates) if candidates else None
    baseline_cv = leave_one_case_out(bank, labels, None)

    # No promotion: the artefact and the headline are the pre-registered baseline, whatever the
    # ladder says. See this function's docstring - the alternative is selecting on the CV the
    # headline is quoted from, on six cases.
    final = ACVRanker(**{**BASELINE_RANKER.as_dict(), "trained_on": tuple(sorted(bank.files))})
    headline = baseline_cv
    best_row = _best_ladder_row(ladder_payload) if ladder_payload else None
    wall = time.perf_counter() - t0
    payload = _cv_payload(
        headline,
        final,
        wall,
        floors=chance_floors(bank.for_ranker(final)),
        cost=bank.cost(),
        extra={
            "baseline_score": baseline_cv["score"],
            "ladder_selected_score": cv["score"] if cv else None,
            "ladder_selected_per_fold": (
                {f["held_out"]: f["chosen_ranker"] for f in cv["folds"]} if cv else None
            ),
            "ladder_best": best_row,
            "artefact": "baseline (pre-registered; the ladder never promotes on this CV)",
            "selection_note": (
                "train() does not promote a ladder winner: the ladder is read on the same six "
                "cases the headline is computed on, so a promoted row's score would be a "
                "selection result. The ladder's best row is reported as `ladder_best`."
            ),
        },
    )

    cv_json = out_dir / "acv_cv.json"
    cv_json.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    (out_dir / "acv_cv.md").write_text(_cv_markdown(payload), encoding="utf-8")
    save_model(
        "acv",
        final,
        {
            "cv": {k: payload[k] for k in ("scheme", "score", "sd", "top1", "folds")},
            "feature_version": af.FEATURE_VERSION,
            "ranker": final.as_dict(),
        },
        model_dir=model_dir,
    )
    csv_path = out_dir / (f"acv_predictions_{tag}.csv" if tag else "acv_predictions.csv")
    n_rows = 0
    if do_predict:
        n_rows = len(_write_predictions(final, csv_path, root=root))
    return {
        "task": "acv",
        "scheme": payload["scheme"],
        "score": payload["score"],
        "sd": payload["sd"],
        "top1": payload["top1"],
        "baseline_score": baseline_cv["score"],
        "ladder_selected_score": payload["ladder_selected_score"],
        "ladder_best": payload["ladder_best"],
        "chance_floor": payload["chance_floor"],
        "chance_floor_uniform": payload["chance_floor_uniform"],
        "wall_seconds_cold": payload["wall_seconds_cold"],
        "artefact": payload["artefact"],
        "n_cases": payload["folds"],
        "ranker": final.name,
        "features": list(final.features),
        "feature_version": af.FEATURE_VERSION,
        "wall_seconds": payload["wall_seconds"],
        "cv_json": str(cv_json),
        "ladder_json": str(out_dir / "acv_ladder.json") if ladder_payload else None,
        "predictions_csv": str(csv_path) if do_predict else None,
        "n_prediction_rows": n_rows,
    }
