"""The 3D model contract: part layout, naming, the exported glb and the fallback markers.

No Blender needed: ``build_r151`` imports without bpy for its pure-python geometry table, the glb
is parsed as raw glTF JSON, and the markers file is checked against the layout.
"""

from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "assets3d"))

import build_r151 as R  # noqa: E402

GLB = REPO / "web" / "public" / "r151.glb"
MARKERS = REPO / "web" / "src" / "components" / "viewport" / "elevationMarkers.json"


def glb_json(path: Path) -> dict:
    with open(path, "rb") as f:
        magic, _version, _length = struct.unpack("<III", f.read(12))
        assert magic == 0x46546C67, "not a glb"
        chunk_len, chunk_type = struct.unpack("<II", f.read(8))
        assert chunk_type == 0x4E4F534A, "first chunk is not JSON"
        return json.loads(f.read(chunk_len))


# ------------------------------------------------------------------ pure python layout


def test_formation_is_dt_m_dt():
    assert R.formation(6) == ["DT", "M", "M", "M", "M", "DT"]
    assert R.formation(8) == ["DT"] + ["M"] * 6 + ["DT"]
    with pytest.raises(ValueError):
        R.formation(1)


@pytest.mark.parametrize("n", [6, 8])
def test_layout_lengths_and_counts(n):
    lay = R.layout(n)
    assert lay["n_cars"] == n
    expect = 2 * R.LEN_DT + (n - 2) * R.LEN_M + (n - 1) * R.GAP
    assert lay["total_len"] == pytest.approx(expect)
    assert lay["rails"] == {"x0": -R.RAIL_OVERHANG, "x1": pytest.approx(expect + R.RAIL_OVERHANG)}
    for car in lay["cars"]:
        assert len(car["doors"]) == 8 and len(car["axleboxes"]) == 8
        assert len(car["bogies"]) == 2 and len(car["ac"]) == 2
        assert car["x0"] <= min(car["doors"].values()) and max(car["doors"].values()) <= car["x1"]
        assert car["x0"] < car["ac"]["ac1"] < car["ac"]["ac2"] < car["x1"]
    # cars do not overlap and are in order
    for a, b in zip(lay["cars"], lay["cars"][1:]):
        assert b["x0"] == pytest.approx(a["x1"] + R.GAP)


def test_cab_doors_shift_away_from_the_cab():
    lay = R.layout(8)
    first, last = lay["cars"][0], lay["cars"][-1]
    # car 1 has its cab at -X: doors sit towards +X of the car centre on average; the last car mirrors
    assert sum(first["doors"].values()) / 8 > first["xc"]
    assert sum(last["doors"].values()) / 8 < last["xc"]


def test_selectable_names_cover_ac_and_scale_with_cars():
    R.set_formation(8)
    try:
        names = set(R.selectable_names())
    finally:
        R.set_formation(R.DEFAULT_N_CARS)
    assert {f"car{i}_ac{k}" for i in range(1, 9) for k in (1, 2)} <= names
    assert "car8_door_R4" in names and "car9_door_L1" not in names
    assert "car8_axlebox_4R" in names


def test_elevation_transform_keeps_aspect():
    t6 = R.elevation_transform(R.layout(6)["total_len"])
    t8 = R.elevation_transform(R.layout(8)["total_len"])
    assert t6["width"] == t8["width"] == R.ELEVATION_PX_W
    # same world height in both renders (height / px-per-metre), so the train keeps its aspect
    assert t8["height"] / t8["b"] == pytest.approx(t6["height"] / t6["b"], rel=0.01)
    assert t8["b"] < t6["b"]  # a longer set at the same width is drawn smaller
    assert t8["height"] == 198


# ------------------------------------------------------------------ exported glb


@pytest.fixture(scope="module")
def glb():
    if not GLB.exists():
        pytest.skip("web/public/r151.glb not built")
    return glb_json(GLB)


def test_glb_size_budget():
    if not GLB.exists():
        pytest.skip("web/public/r151.glb not built")
    assert GLB.stat().st_size < 5_000_000


def test_glb_node_names(glb):
    names = {n["name"] for n in glb["nodes"]}
    mesh_names = {n["name"] for n in glb["nodes"] if "mesh" in n}
    cars = {int(m.group(1)) for n in names if (m := re.match(r"^car(\d+)_", n))}
    assert cars == set(range(1, 9)), "eight cars"
    # every selectable part of the 8-car set is a node
    R.set_formation(8)
    try:
        missing = [n for n in R.selectable_names() if n not in names]
    finally:
        R.set_formation(R.DEFAULT_N_CARS)
    assert not missing, missing[:10]
    # counts per kind, as the viewport header computes them
    assert len([n for n in mesh_names if re.match(r"^car\d+_ac[12]$", n)]) == 16
    assert len([n for n in names if re.match(r"^car\d+_door_[LR][1-4]$", n)]) == 64
    assert len([n for n in mesh_names if re.match(r"^car\d+_axlebox_[1-4][LR]$", n)]) == 64
    assert len([n for n in names if re.match(r"^car\d+_bogie[12]$", n)]) == 16
    assert len([n for n in names if re.match(r"^car\d+_apu$", n)]) == 8
    assert {"rail", "rail001"} <= mesh_names
    # the AC units are single-material meshes (one primitive), so their node name is the mesh name
    ac_nodes = [n for n in glb["nodes"] if re.match(r"^car\d+_ac[12]$", n["name"])]
    mats = {m["name"]: i for i, m in enumerate(glb["materials"])}
    for n in ac_nodes:
        prims = glb["meshes"][n["mesh"]]["primitives"]
        assert len(prims) == 1 and prims[0]["material"] == mats["ac_default"]


def test_glb_rails_span_the_set(glb):
    lay = R.layout(8)
    acc = glb["accessors"]
    for node in glb["nodes"]:
        if node["name"] in ("rail", "rail001"):
            prim = glb["meshes"][node["mesh"]]["primitives"][0]
            pos = acc[prim["attributes"]["POSITION"]]
            # vertices are local to the node: add its translation to get world x extents
            tx = node.get("translation", [0, 0, 0])[0]
            assert tx + pos["min"][0] == pytest.approx(lay["rails"]["x0"], abs=0.01)
            assert tx + pos["max"][0] == pytest.approx(lay["rails"]["x1"], abs=0.01)


# ------------------------------------------------------------------ fallback markers


def test_markers_match_layout():
    d = json.loads(MARKERS.read_text())
    lay = R.layout(len(d["cars"]))
    assert len(d["cars"]) == 8
    fit = d["fit"]
    px = lambda x: fit["a"] + fit["b"] * x  # noqa: E731
    for car in lay["cars"]:
        c = d["cars"][str(car["n"])]
        assert set(c["doors"]) == set(car["doors"])
        assert set(c["axleboxes"]) == set(car["axleboxes"])
        assert set(c["bogies"]) == {"bogie_1", "bogie_2"}
        assert set(c["ac"]) == {"ac1", "ac2"}
        for cid, x in car["doors"].items():
            box = c["doors"][cid]
            assert box["x"] + box["w"] / 2 == pytest.approx(px(x), abs=0.6)
        for unit, x in car["ac"].items():
            box = c["ac"][unit]
            assert box["x"] + box["w"] / 2 == pytest.approx(px(x), abs=0.6)
        assert 0 <= c["body"]["x"] and c["body"]["x"] + c["body"]["w"] <= d["width"]
    assert set(d["rails"]) == {"rail_I", "rail_II"}
    assert all(0 < v["y"] < d["height"] for v in d["rails"].values())
    assert d["width"] == 4000 and d["height"] == 198
