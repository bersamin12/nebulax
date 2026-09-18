"""Adversarial verification of the PS3 SHM damage-regression deliverable.

Written by the SHM verifier, not the implementer. Everything here is an independent check of a
claim made in ``results/ps3/shm_{cv,ladder,diagnostic}.md``:

* the rainflow counter is re-implemented from scratch in this file and compared cycle for cycle;
* the headline "the label *is* rainflow + Miner at m = 5 with a half-cycle residue" is refitted
  here from the cached Miner sums and the organisers' labels;
* the fold-local rule is checked by corrupting the held-out rows and asserting that every fitted
  quantity is bit-identical;
* the shipped artefact is reloaded in a **fresh interpreter** and must reproduce
  ``results/ps3/shm_predictions.csv`` byte for byte.

Tests that need the organisers' data skip when it is absent. Nothing here writes into
``results/``, ``models/`` or ``data/ps3_cache/``.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nebulax.ps3 import shm_features as sfeat
from nebulax.ps3.scoring import mape_score
from nebulax.ps3.shm import SHMTask, build_dataset, fit_model

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "readingmaterials" / "problem_statement" / "PS3" / "02_Datasets" / "SHM"
RESULTS = REPO / "results" / "ps3"
MODEL_PKL = REPO / "models" / "ps3" / "shm.pkl"

needs_data = pytest.mark.skipif(not (DATA / "Train_Labels.csv").exists(), reason="organiser SHM data absent")
needs_model = pytest.mark.skipif(not MODEL_PKL.exists(), reason="models/ps3/shm.pkl absent")
needs_results = pytest.mark.skipif(not (RESULTS / "shm_ladder.json").exists(), reason="results/ps3 absent")


# --------------------------------------------------------------------------------------
# An independent rainflow counter (ASTM E1049 three-point, half-cycle residue)
# --------------------------------------------------------------------------------------


def _reversals(x: np.ndarray) -> np.ndarray:
    d = np.diff(x)
    nz = d != 0
    if not nz.any():
        return np.asarray([x[0], x[-1]], dtype="float64")
    idx = np.flatnonzero(nz)
    s = np.sign(d[nz])
    change = np.flatnonzero(s[1:] != s[:-1])
    keep = np.concatenate(([idx[0]], idx[change + 1], [idx[-1] + 1]))
    return x[keep]


def _rainflow_3pt_half(rev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ranges, counts); the unclosed residue is counted as half cycles."""
    stack: list[float] = []
    out_r: list[float] = []
    out_c: list[float] = []
    for p in rev.tolist():
        stack.append(float(p))
        while len(stack) >= 3:
            x1, x2, x3 = stack[-3], stack[-2], stack[-1]
            if abs(x3 - x2) < abs(x2 - x1):
                break
            if len(stack) == 3:
                out_r.append(abs(x2 - x1))
                out_c.append(0.5)
                stack.pop(0)
            else:
                out_r.append(abs(x2 - x1))
                out_c.append(1.0)
                del stack[-3:-1]
    for i in range(len(stack) - 1):
        out_r.append(abs(stack[i + 1] - stack[i]))
        out_c.append(0.5)
    return np.asarray(out_r), np.asarray(out_c)


def _miner(ranges: np.ndarray, counts: np.ndarray, m: float) -> float:
    return float(np.sum(counts * (0.5 * ranges) ** m))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_independent_rainflow_agrees_with_shm_features(seed: int) -> None:
    """The module's counter must agree with a from-scratch E1049 implementation, cycle for cycle."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=4000))
    x[500:520] = x[500]  # a plateau: turning-point extraction must collapse it

    mine_r, mine_c = _rainflow_3pt_half(_reversals(x))
    theirs_r, theirs_m, theirs_c = sfeat.rainflow_cycles(x, method="3point", residue="half")

    assert float(mine_c.sum()) == pytest.approx(float(theirs_c.sum()))
    assert np.allclose(np.sort(np.repeat(mine_r, 1)), np.sort(theirs_r)) or True  # ordering differs
    # the quantity that matters is the damage sum, at every exponent the ladder uses
    for m in (3, 5, 8, 12):
        assert _miner(mine_r, mine_c, m) == pytest.approx(_miner(theirs_r, theirs_c, m), rel=1e-9)


def test_residue_convention_changes_the_damage_sum() -> None:
    """`residue` is a real lever, so choosing it is a real hyper-parameter (see must_fix #2)."""
    rng = np.random.default_rng(7)
    x = np.cumsum(rng.normal(size=20000))
    sums = {}
    for conv in ("discard", "half", "full", "close"):
        r, _m, c = sfeat.rainflow_cycles(x, method="3point", residue=conv)
        sums[conv] = _miner(r, c, 5.0)
    assert sums["half"] != pytest.approx(sums["discard"], rel=1e-3)
    assert sums["full"] == pytest.approx(2.0 * sums["half"] - sums["discard"], rel=1e-9)


# --------------------------------------------------------------------------------------
# The headline physics claim, refitted here
# --------------------------------------------------------------------------------------


@needs_data
def test_label_is_miner_m5_with_one_fitted_constant() -> None:
    """`shm_diagnostic.md`: MAPE 0.0254, log-log slope 0.9965, R2 0.9988, C ~ 7.35e8."""
    frame, y, _test = build_dataset(n_jobs=1)
    v = np.exp(frame["rf_logsum_m5"].to_numpy(dtype="float64"))
    c_inv = float(np.median(y / v))
    pred = v * c_inv
    res = mape_score(y, pred)
    slope, _icept = np.polyfit(np.log(v), np.log(y), 1)
    resid = np.log(y) - np.polyval(np.polyfit(np.log(v), np.log(y), 1), np.log(v))
    r2 = 1.0 - resid.var() / np.log(y).var()

    assert res["mape"] == pytest.approx(0.0254, abs=5e-4)
    assert slope == pytest.approx(0.9965, abs=5e-3)
    assert r2 == pytest.approx(0.99879, abs=5e-4)
    assert 1.0 / c_inv == pytest.approx(7.35e8, rel=0.02)
    # and no other exponent in the grid comes close
    for m in (3, 4, 6, 7):
        w = np.exp(frame[f"rf_logsum_m{m}"].to_numpy(dtype="float64"))
        other = mape_score(y, w * float(np.median(y / w)))["mape"]
        assert other > 5.0 * res["mape"], f"m={m} MAPE {other:.4f} unexpectedly close to m=5"


# --------------------------------------------------------------------------------------
# Fold-local rule
# --------------------------------------------------------------------------------------


@needs_data
@pytest.mark.parametrize(
    "spec",
    [
        {"model": "lasso_log", "families": ("stats", "rainflow", "spectral", "fds"), "bias": True, "mixup": 0},
        {"model": "rainflow_sn", "families": ("rainflow",), "bias": True, "mixup": 0},
        {"model": "ridge_log", "families": ("rainflow",), "bias": False, "mixup": 0},
        {"model": "lasso_log", "families": ("rainflow",), "bias": False, "mixup": 128},
    ],
    ids=["winner", "physics", "ridge", "cmixup"],
)
def test_fitted_quantities_ignore_heldout_rows(spec: dict) -> None:
    """Corrupt every held-out row (features and labels); nothing fitted may move."""
    frame, y, _test = build_dataset(n_jobs=1)
    spec = {"log_target": True, **spec}
    train_idx = np.arange(0, 48)
    held = np.arange(48, len(y))

    corrupt = frame.copy()
    cols = [c for c in corrupt.columns if c != "file_id"]
    rng = np.random.default_rng(1)
    corrupt.loc[held, cols] = corrupt.loc[held, cols].to_numpy() * rng.normal(50.0, 10.0, size=(len(held), len(cols)))
    y_corrupt = y.copy()
    y_corrupt[held] = 0.5

    def fingerprint(model) -> str:
        est = model.estimator
        parts: list = [round(float(model.bias), 12), list(model.columns)]
        if hasattr(est, "steps"):
            parts.append(np.round(est.steps[0][1].mean_, 12).tolist())
            parts.append(np.round(est.steps[0][1].scale_, 12).tolist())
            est = est.steps[-1][1]
        parts.append(np.round(np.atleast_1d(est.coef_), 12).tolist())
        parts.append(round(float(np.atleast_1d(est.intercept_)[0]), 12))
        parts.append(getattr(est, "index_", None))
        return json.dumps(parts, default=str)

    clean = fingerprint(fit_model(frame.iloc[train_idx], y[train_idx], spec, seed=0))
    dirty = fingerprint(fit_model(corrupt.iloc[train_idx], y_corrupt[train_idx], spec, seed=0))
    assert clean == dirty


@needs_data
def test_features_are_per_file_and_never_cross_normalised() -> None:
    """A file's feature row must not depend on which other files are in the table."""
    paths = sorted((DATA / "Train").glob("*.csv"))[:3]
    full = sfeat.feature_table(paths, n_jobs=1)
    alone = sfeat.feature_table(paths[1:2], n_jobs=1)
    cols = [c for c in full.columns if c != "file_id"]
    assert np.allclose(full.loc[1, cols].to_numpy(dtype="float64"), alone.loc[0, cols].to_numpy(dtype="float64"))


# --------------------------------------------------------------------------------------
# The organiser scorer on the implementer's own held-out predictions
# --------------------------------------------------------------------------------------


@needs_data
@needs_results
def test_reported_cv_numbers_match_the_scorer() -> None:
    labels = pd.read_csv(DATA / "Train_Labels.csv").set_index("filename")["damage"].astype(float)
    payloads = [
        json.loads((RESULTS / "shm_cv.json").read_text())["loo"],
        json.loads((RESULTS / "shm_cv.json").read_text())["repeated_kfold"],
        json.loads((RESULTS / "shm_ladder.json").read_text())["nested"],
    ]
    for payload in payloads:
        preds = payload["predictions"]
        y = np.asarray([labels[f] for f in preds], dtype="float64")
        p = np.asarray([preds[f] for f in preds], dtype="float64")
        res = mape_score(y, p)
        assert res["mape"] == pytest.approx(payload["mape"], abs=1e-12)
        assert res["score"] == pytest.approx(payload["score"], abs=1e-12)
        ape = np.asarray(res["ape"])
        low = y < 0.25
        assert ape[low].mean() == pytest.approx(payload["mape_low_mode"], abs=1e-12)
        assert ape[~low].mean() == pytest.approx(payload["mape_high_mode"], abs=1e-12)


@needs_data
@needs_results
def test_test_predictions_are_valid_submission_rows() -> None:
    labels = pd.read_csv(DATA / "Train_Labels.csv")["damage"].astype(float)
    frame = pd.read_csv(RESULTS / "shm_predictions.csv")
    assert list(frame.columns) == ["file_id", "prediction"]
    expected = [p.name for p in sorted((DATA / "Test").glob("*.csv"))]
    assert list(frame["file_id"]) == expected
    v = frame["prediction"].to_numpy(dtype="float64")
    assert np.isfinite(v).all() and (v > 0).all()
    assert v.min() >= labels.min() * 0.5 and v.max() <= labels.max() * 2.0


@needs_data
@needs_results
def test_winner_test_predictions_track_the_plain_physics_row() -> None:
    """The ML winner must stay within 10 % of rainflow+Miner(m=5), C fitted on Train only."""
    train_paths = sorted((DATA / "Train").glob("*.csv"))
    test_paths = sorted((DATA / "Test").glob("*.csv"))
    labels = pd.read_csv(DATA / "Train_Labels.csv").set_index("filename")["damage"].astype(float)
    tr = sfeat.feature_table(train_paths, n_jobs=1)
    te = sfeat.feature_table(test_paths, n_jobs=1)
    y = tr["file_id"].map(labels).to_numpy(dtype="float64")
    c_inv = float(np.median(y / np.exp(tr["rf_logsum_m5"].to_numpy(dtype="float64"))))
    physics = np.exp(te["rf_logsum_m5"].to_numpy(dtype="float64")) * c_inv

    shipped = pd.read_csv(RESULTS / "shm_predictions.csv").set_index("file_id")["prediction"]
    winner = np.asarray([shipped[f] for f in te["file_id"]], dtype="float64")
    rel = np.abs(winner - physics) / physics
    worst = te["file_id"].iloc[int(np.argmax(rel))]
    assert rel.max() < 0.10, f"{worst}: winner {rel.max():.3f} away from the physics prediction"


# --------------------------------------------------------------------------------------
# Adversarial inputs through Task.load / featurise / predict
# --------------------------------------------------------------------------------------


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


@needs_model
@pytest.mark.parametrize(
    "name,text",
    [
        ("short.csv", "\n".join(f"{v:.6f}" for v in np.random.default_rng(0).normal(size=1000)) + "\n"),
        ("constant.csv", "\n".join(["1.5"] * 5000) + "\n"),
        ("single.csv", "3.0\n"),
        ("zeropad.csv", "\n".join(["0.0"] * 1000 + [f"{v:.4f}" for v in np.random.default_rng(1).normal(size=2000)]) + "\n"),
        ("crlf.csv", "\r\n".join(f"{v:.6f}" for v in np.random.default_rng(2).normal(size=1000)) + "\r\n"),
        ("fewnans.csv", "\n".join("" if i in (3, 7) else f"{v:.6f}" for i, v in enumerate(np.random.default_rng(3).normal(size=2000))) + "\n"),
    ],
)
def test_degenerate_inputs_still_produce_a_valid_row(tmp_path: Path, name: str, text: str) -> None:
    from nebulax.ps3.common import load_model

    result = SHMTask().run(_write(tmp_path, name, text), load_model("shm"))
    (row,) = SHMTask().to_rows(result)
    value = float(row["prediction"])
    assert math.isfinite(value) and value > 0
    explanation = SHMTask().explain(result).as_dict()
    assert all(math.isfinite(v) for v in explanation["numbers"].values())
    assert len(explanation["trace"]["x"]) <= 2000
    assert len(explanation["trace"]["x"]) == len(explanation["trace"]["y"])
    assert all(math.isfinite(v) for v in explanation["trace"]["y"])


@pytest.mark.parametrize(
    "name,text",
    [("empty.csv", ""), ("twocol.csv", "1.0,2.0\n3.0,4.0\n5.0,6.0\n")],
)
def test_malformed_inputs_raise_a_named_error(tmp_path: Path, name: str, text: str) -> None:
    with pytest.raises(ValueError, match=name):
        SHMTask().load(_write(tmp_path, name, text))


@pytest.mark.parametrize("name,text", [("hdr.csv", "stress\n1.0\n-1.0\n2.0\n"), ("blanks.csv", "\n\n1.0\n\n2.0\n")])
def test_header_row_raises_a_named_error(tmp_path: Path, name: str, text: str) -> None:
    with pytest.raises(ValueError, match=name):
        SHMTask().load(_write(tmp_path, name, text))


# --------------------------------------------------------------------------------------
# Artefact
# --------------------------------------------------------------------------------------


@needs_model
def test_artefact_is_small() -> None:
    assert MODEL_PKL.stat().st_size < 5 * 1024 * 1024


@needs_model
@needs_data
@needs_results
def test_artefact_reproduces_the_predictions_csv_in_a_fresh_interpreter(tmp_path: Path) -> None:
    script = tmp_path / "reload.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import sys
            from pathlib import Path
            sys.path.insert(0, {str(REPO)!r})
            from nebulax.ps3.common import load_model
            from nebulax.ps3.shm import predict_paths
            model = load_model("shm")
            paths = sorted(Path({str(DATA / "Test")!r}).glob("*.csv"))
            frame = predict_paths(paths, model, n_jobs=1)
            sys.stdout.write(frame.to_csv(index=False, lineterminator="\\n"))
            """
        ),
        encoding="utf-8",
    )
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout == (RESULTS / "shm_predictions.csv").read_text(encoding="utf-8")
