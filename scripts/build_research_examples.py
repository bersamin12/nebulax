"""Build small, reproducible research excerpts for the offline Predict view.

MetroPT-3 uses the already scored model windows, preserving their alert flags. Ottawa uses
100 ms AC-coupled RMS windows from one healthy and one inner-race record. These are examples,
not extra PS3 predictions or complete copies of either dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web/src/predict/researchExamples.json"


def metro_examples() -> list[dict]:
    frame = pd.read_parquet(ROOT / "data/scores/metropt3.parquet").sort_values("timestamp")
    first_alert = int(np.flatnonzero(frame["alert"].to_numpy())[0])
    picks = [
        ("normal", "Routine operation", frame.iloc[:40]),
        ("anomaly", "Elevated anomaly score", frame.iloc[first_alert - 12 : first_alert + 32]),
    ]
    out = []
    for key, label, rows in picks:
        points = [
            {
                "label": row.timestamp.strftime("%d %b %H:%M UTC"),
                "value": round(float(row.score), 3),
                "alert": bool(row.alert),
            }
            for row in rows.itertuples()
        ]
        out.append({
            "id": key,
            "label": label,
            "record": "MetroPT-3 · 2020 air compressor",
            "measure": "LGBM residual anomaly score",
            "unit": "score",
            "threshold": round(float(rows.threshold.iloc[0]), 3),
            "points": points,
        })
    return out


def bearing_examples() -> list[dict]:
    out = []
    for key, label, filename, condition in [
        ("healthy", "Healthy bearing", "H_3_0.csv", "healthy"),
        ("inner_race", "Inner-race fault", "I_1_1.csv", "faulty"),
    ]:
        values = pd.read_csv(ROOT / "data/raw/ottawa" / filename, usecols=[0]).iloc[:, 0].to_numpy(dtype=float)
        window = 4200  # 100 ms at 42 kHz
        values = values[: (len(values) // window) * window].reshape(-1, window)
        centered = values - values.mean(axis=1, keepdims=True)
        rms = np.sqrt(np.mean(centered * centered, axis=1))
        out.append({
            "id": key,
            "label": label,
            "record": f"Ottawa UORED-VAFCLS · {filename}",
            "measure": "Measured acceleration · 100 ms RMS",
            "unit": "recorded acceleration units",
            "condition": condition,
            "points": [{"label": f"{i / 10:.1f} s", "value": round(float(v), 3)} for i, v in enumerate(rms)],
        })
    return out


def main() -> None:
    payload = {"pneumatic": metro_examples(), "bearing": bearing_examples()}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{OUT}: {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
