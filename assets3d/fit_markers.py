"""Refit the 2D fallback markers to a fresh elevation render.

    python assets3d/fit_markers.py --n-cars 8 --png docs/design/r151_elevation.png \
        --out web/src/components/viewport/elevationMarkers.json

The elevation render (build_r151.py) is orthographic, so pixel <- world is linear:

    x_px = a + b * x_world        y_px = c - b * y_world      (y_world = Blender z, rail top = 0)

`a`, `b` are re-derived by least squares from the inter-car body edges measured in the PNG's alpha
channel (a row at roof-band height, where the gangways do not reach), `c` from the rail-top row
at a column outside the set. The analytic transform of the camera setup is printed next to the
fit as a sanity check; the fit is what gets written. Part positions come from
``build_r151.layout(n_cars)`` (pure python, no Blender needed).

Marker sizes are given in world metres and converted with the fitted scale, so they stay in
proportion whatever the pixel density of the render.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_r151 as R  # noqa: E402

# marker geometry in world metres (tuned on the six-car render at 27 px/m)
DOOR_W = 1.48
DOOR_TOP, DOOR_BOTTOM = 3.05, 1.10              # near-side door opening
FAR_PLATE_DROP, FAR_PLATE_H = 0.48, 0.33        # far-side door plate in the roof band
BOX_W, BOX_H, BOX_OFF = 0.48, 0.41, 0.30        # axle box, L / R offset
APU_W, APU_H = 1.55, 0.67
AC_W, AC_TOP, AC_BOTTOM = 2.2, 4.55, 3.70       # roof AC plate, above the roof line
BOGIE_HALF_W, BOGIE_TOP, BOGIE_BOTTOM = 1.55, 1.10, 0.62
RAIL_PLATE_H = 0.22
RAIL_I_TOP, RAIL_II_TOP = -0.08, -0.40          # long plates below the near rail
LABEL_DROP = 1.77


def measure(png, lay, guess):
    """Measure inter-car edges and the rail top in the PNG; return the least-squares a, b, c."""
    im = np.asarray(Image.open(png).convert("RGBA"))
    h, w = im.shape[:2]
    alpha = im[..., 3] > 127
    # body edges: a row through the upper body, above the gangway top (FLOOR_Z + 2.15 = 3.25 m)
    row = int(round(guess["c"] - guess["b"] * 3.45))
    line = alpha[row]
    edges = np.flatnonzero(np.diff(line.astype(np.int8)))  # index i: change between i and i+1
    rises = [i + 1 for i in edges if not line[i] and line[i + 1]]
    falls = [i + 0.5 for i in edges if line[i] and not line[i + 1]]
    rises = [r - 0.5 for r in rises]
    # every car body is one opaque run; keep the runs longer than half a car (drop noise)
    runs = []
    for r in rises:
        f = next((f for f in falls if f > r), None)
        if f is not None and f - r > guess["b"] * R.LEN_M * 0.5:
            runs.append((r, f))
    cars = lay["cars"]
    if len(runs) != len(cars):
        raise SystemExit(f"expected {len(cars)} body runs on row {row}, found {len(runs)}: {runs}")
    xs, ps = [], []
    for i in range(len(cars) - 1):          # only inter-car edges (the cab noses are shaped)
        xs.append(cars[i]["x1"]); ps.append(runs[i][1])
        xs.append(cars[i + 1]["x0"]); ps.append(runs[i + 1][0])
    A = np.vstack([np.ones(len(xs)), xs]).T
    (a, b), *_ = np.linalg.lstsq(A, np.asarray(ps), rcond=None)
    resid = np.asarray(ps) - (a + b * np.asarray(xs))
    # rail top: a column outside the set (2 m before car 1), the first opaque row from the top
    col = int(round(a + b * (cars[0]["x0"] - 2.0)))
    rows = np.flatnonzero(alpha[:, col])
    if not len(rows):
        raise SystemExit(f"no rail found in column {col}")
    c = float(rows[0]) - 0.5
    return {"a": float(a), "b": float(b), "c": c, "width": w, "height": h,
            "edge_residual_px": float(np.abs(resid).max()), "row": row, "col": col}


def markers(lay, fit, image):
    a, b, c = fit["a"], fit["b"], fit["c"]
    px = lambda x: round(a + b * x, 1)          # noqa: E731
    py = lambda y: round(c - b * y, 1)          # noqa: E731
    m = lambda metres: round(b * metres, 1)     # noqa: E731
    out = {"image": image, "width": fit["width"], "height": fit["height"],
           "fit": {"a": round(a, 3), "b": round(b, 4), "c": round(c, 2)}, "cars": {}, "rails": {}}
    for car in lay["cars"]:
        n = car["n"]
        cc = {"doors": {}, "axleboxes": {}, "bogies": {}, "apu": None, "ac": {}, "label": None, "body": None}
        cc["body"] = {"x": px(car["x0"]), "w": round(px(car["x1"]) - px(car["x0"]), 1),
                      "y": py(R.HEIGHT + 0.285), "h": round(py(0.205) - py(R.HEIGHT + 0.285), 1)}
        cc["label"] = {"x": round((px(car["x0"]) + px(car["x1"])) / 2, 1), "y": py(-LABEL_DROP)}
        for cid, x in car["doors"].items():
            if cid[5] == "L":   # near side: the door opening on the body
                cc["doors"][cid] = {"x": round(px(x) - m(DOOR_W) / 2, 1), "y": py(DOOR_TOP), "w": m(DOOR_W),
                                    "h": round(py(DOOR_BOTTOM) - py(DOOR_TOP), 1), "near": True,
                                    "lx": px(x), "ly": py(0.6)}
            else:               # far side: a plate in the roof band (as in the design mock)
                cc["doors"][cid] = {"x": round(px(x) - m(DOOR_W) / 2, 1), "y": round(py(R.HEIGHT + 0.285) - m(FAR_PLATE_DROP), 1),
                                    "w": m(DOOR_W), "h": m(FAR_PLATE_H), "near": False,
                                    "lx": px(x), "ly": round(py(R.HEIGHT + 0.285) - m(FAR_PLATE_DROP + 0.15), 1)}
        for cid, x in car["axleboxes"].items():
            s = cid[-1]
            cc["axleboxes"][cid] = {"x": round(px(x) + (-1 if s == "L" else 1) * m(BOX_OFF) - m(BOX_W) / 2, 1),
                                    "y": round(py(R.WHEEL_R + 0.02) - m(BOX_H) / 2, 1),
                                    "w": m(BOX_W), "h": m(BOX_H), "near": s == "L"}
        for cid, x in car["bogies"].items():
            cc["bogies"][cid] = {"x": round(px(x) - m(BOGIE_HALF_W), 1), "y": py(BOGIE_TOP),
                                 "w": m(2 * BOGIE_HALF_W), "h": round(py(BOGIE_BOTTOM) - py(BOGIE_TOP), 1)}
        cc["apu"] = {"x": round(px(car["apu"]) - m(APU_W) / 2, 1), "y": round(py(0.73) - 1, 1),
                     "w": m(APU_W), "h": m(APU_H)}
        for unit, x in car["ac"].items():
            cc["ac"][unit] = {"x": round(px(x) - m(AC_W) / 2, 1), "y": py(AC_TOP), "w": m(AC_W),
                              "h": round(py(AC_BOTTOM) - py(AC_TOP), 1)}
        out["cars"][str(n)] = cc
    x0, x1 = lay["rails"]["x0"], lay["rails"]["x1"]
    for cid, top in (("rail_I", RAIL_I_TOP), ("rail_II", RAIL_II_TOP)):
        out["rails"][cid] = {"x": px(x0), "y": py(top), "w": round(px(x1) - px(x0), 1), "h": m(RAIL_PLATE_H)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-cars", type=int, default=8)
    ap.add_argument("--png", default="docs/design/r151_elevation.png")
    ap.add_argument("--out", default="web/src/components/viewport/elevationMarkers.json")
    ap.add_argument("--image-url", default="/r151_elevation.png", help="src the fallback loads")
    args = ap.parse_args(argv)

    lay = R.layout(args.n_cars)
    guess = R.elevation_transform(lay["total_len"])
    fit = measure(args.png, lay, guess)
    print(f"[fit] analytic a={guess['a']:.2f} b={guess['b']:.4f} c={guess['c']:.2f}  "
          f"({guess['width']}x{guess['height']})")
    print(f"[fit] measured a={fit['a']:.2f} b={fit['b']:.4f} c={fit['c']:.2f}  "
          f"({fit['width']}x{fit['height']}), max edge residual {fit['edge_residual_px']:.2f} px, "
          f"row {fit['row']}, rail column {fit['col']}")
    if abs(fit["b"] - guess["b"]) > 0.02 * guess["b"] or abs(fit["c"] - guess["c"]) > 4:
        raise SystemExit("fit disagrees with the camera setup by more than 2 % / 4 px; check the render")
    out = markers(lay, fit, args.image_url)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[fit] wrote {args.out}: {len(out['cars'])} cars")


if __name__ == "__main__":
    main()
