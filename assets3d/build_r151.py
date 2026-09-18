"""Procedural, stylised R151-style MRT train (Alstom Movia R151 dimensions), 6 or 8 cars.

Run headless:
  blender -b -P assets3d/build_r151.py -- --n-cars 8 --glb web/public/r151.glb --png docs/design/

Shape only, from public dimensions. No logos, no exact livery.
Formation: DT + (N-2) x M + DT (``--n-cars N``, default 6; the organisers' PS3 trains have 8).
Every selectable subsystem part is its own mesh, named by component id:
  car{n}_door_L{1-4} / car{n}_door_R{1-4}
  car{n}_bogie{1,2}
  car{n}_axlebox_{1-4}{L,R}      (axle 1..4 along +X, L = -Y side, R = +Y side)
  car{n}_apu
  car{n}_ac1 / car{n}_ac2        (roof air-conditioning units, material ac_default)
  rail / rail001                 (the two running rails: -Y = L side first, +Y = R side second)
Everything else is joined into car{n}_body.

Axes: X along the train, Y across, Z up, rail top at z = 0.  Units: metres.

``layout(n_cars)`` (pure python, no bpy) returns the world positions of every selectable part;
``assets3d/fit_markers.py`` uses it to place the 2D fallback markers on the elevation render.
"""
import argparse
import math
import os
import sys

try:  # the geometry table below is importable without Blender (fit_markers.py, tests)
    import bpy
    import bmesh
    from mathutils import Matrix, Vector
except ImportError:  # pragma: no cover - only outside Blender
    bpy = bmesh = Matrix = Vector = None

# ----------------------------------------------------------------------------- dimensions
LEN_DT = 23.83      # driving trailer car
LEN_M = 22.80       # motor car
WIDTH = 3.20
HEIGHT = 3.70       # rail top to roof
FLOOR_Z = 1.10      # rail top to floor / body sill
DOOR_W = 1.45
DOOR_H = 1.95
WHEEL_R = 0.425     # 850 mm wheel
GAUGE_HALF = 0.7175 # 1435 mm gauge / 2
GAP = 0.65          # gap between car bodies (gangway)
DOOR_CENTRES_M = [-9.45, -3.15, 3.15, 9.45]          # motor car, from car centre
DOOR_CENTRES_DT_REAR = [-9.20, -2.90, 3.40, 9.70]    # DT with cab at +X end: shift away from cab
BOGIE_HALF_SPACING = 7.85

DEFAULT_N_CARS = 6
FORMATION = ["DT", "M", "M", "M", "M", "DT"]   # replaced by set_formation(n)

AC_CENTRES = (-5.2, 5.2)                             # roof AC units, from car centre
AC_SIZE = (2.2, 2.3, 0.33)
RAIL_OVERHANG = 5.0                                  # rails run this far past both cab ends
# Orthographic elevation render: 4000 px wide; the vertical world extent is fixed so the train
# keeps its aspect whatever the number of cars (6 cars: 4000 x 260, 8 cars: 4000 x 198).
ELEVATION_PX_W = 4000
ELEVATION_MARGIN = 1.04                              # ortho width = train length x this
ELEVATION_WORLD_H = 9.725                            # metres of world per render height
ELEVATION_CAM_Z = 2.0                                # camera height (image centre row)


def formation(n_cars):
    """DT + (n-2) x M + DT."""
    n = int(n_cars)
    if n < 2:
        raise ValueError("a set needs at least two driving trailers")
    return ["DT"] + ["M"] * (n - 2) + ["DT"]


def set_formation(n_cars):
    global FORMATION
    FORMATION = formation(n_cars)
    return FORMATION


def car_length(kind):
    return LEN_DT if kind == "DT" else LEN_M


def door_centres(kind, n, n_cars):
    """Door centres from the car centre (cab cars shift the doors away from the cab)."""
    if kind == "DT":
        # cab at -X on car 1 -> doors shift +X; cab at +X on the last car -> doors shift -X
        return list(DOOR_CENTRES_DT_REAR) if n == 1 else [-c for c in DOOR_CENTRES_DT_REAR][::-1]
    return list(DOOR_CENTRES_M)


def layout(n_cars=DEFAULT_N_CARS):
    """World coordinates (Blender axes: X along, Z up, rail top z = 0) of every selectable part.

    Returns {"n_cars", "total_len", "rails": {"x0", "x1"}, "cars": [{n, kind, x0, x1, xc,
    doors: {door_L1: x, ...}, axleboxes: {axlebox_1L: x, ...}, bogies: {bogie_1: x, ...},
    apu: x, ac: {ac1: x, ac2: x}}]}.  Mirrors build_train() exactly.
    """
    form = formation(n_cars)
    x = 0.0
    cars = []
    for i, kind in enumerate(form):
        n = i + 1
        length = car_length(kind)
        xc = x + length / 2
        car = {"n": n, "kind": kind, "x0": x, "x1": x + length, "xc": xc,
               "doors": {}, "axleboxes": {}, "bogies": {}, "apu": xc + 1.5,
               "ac": {f"ac{k + 1}": xc + dx for k, dx in enumerate(AC_CENTRES)}}
        for side in ("L", "R"):
            for d, dc in enumerate(door_centres(kind, n, len(form)), start=1):
                car["doors"][f"door_{side}{d}"] = xc + dc
        for b, bx in enumerate((xc - BOGIE_HALF_SPACING, xc + BOGIE_HALF_SPACING), start=1):
            car["bogies"][f"bogie_{b}"] = bx
            for ai, ax in enumerate((bx - 1.05, bx + 1.05)):
                axle_no = (b - 1) * 2 + ai + 1
                for side in ("L", "R"):
                    car["axleboxes"][f"axlebox_{axle_no}{side}"] = ax
        cars.append(car)
        x += length + GAP
    total_len = x - GAP
    return {"n_cars": len(form), "total_len": total_len,
            "rails": {"x0": -RAIL_OVERHANG, "x1": total_len + RAIL_OVERHANG}, "cars": cars}


def elevation_resolution(total_len):
    """(width, height) of the elevation render for a train of ``total_len`` metres."""
    world_w = total_len * ELEVATION_MARGIN
    return ELEVATION_PX_W, int(round(ELEVATION_PX_W * ELEVATION_WORLD_H / world_w))


def elevation_transform(total_len):
    """Analytic pixel transform of the orthographic elevation: x_px = a + b*x, y_px = c - b*y.

    Blender's orthographic camera fits ``ortho_scale`` across the wider image axis and centres
    the view on the camera; the elevation camera sits at (total_len/2, -120, ELEVATION_CAM_Z).
    """
    w, h = elevation_resolution(total_len)
    b = w / (total_len * ELEVATION_MARGIN)
    a = w / 2 - b * (total_len / 2)
    c = h / 2 + b * ELEVATION_CAM_Z
    return {"a": a, "b": b, "c": c, "width": w, "height": h}

# ----------------------------------------------------------------------------- materials
def mat(name, rgb, rough=0.5, metal=0.0):
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    return m

MATS = {}

def init_materials():
    MATS["body"] = mat("body_white", (0.90, 0.91, 0.92), 0.35)
    MATS["roof"] = mat("roof_grey", (0.62, 0.64, 0.66), 0.6)
    MATS["band"] = mat("window_band", (0.05, 0.06, 0.08), 0.15)
    MATS["glass"] = mat("glass_dark", (0.03, 0.04, 0.06), 0.05)
    MATS["door"] = mat("door_default", (0.72, 0.74, 0.76), 0.4)
    MATS["under"] = mat("undercarriage", (0.18, 0.19, 0.21), 0.7)
    MATS["bogie"] = mat("bogie_frame", (0.22, 0.23, 0.25), 0.6)
    MATS["wheel"] = mat("wheel_steel", (0.35, 0.36, 0.38), 0.4, 0.8)
    MATS["axlebox"] = mat("axlebox_default", (0.45, 0.47, 0.50), 0.5)
    MATS["apu"] = mat("apu_default", (0.40, 0.42, 0.45), 0.5)
    MATS["ac"] = mat("ac_default", (0.58, 0.60, 0.63), 0.55)
    MATS["gangway"] = mat("gangway", (0.12, 0.12, 0.13), 0.9)
    MATS["accent"] = mat("accent_stripe", (0.10, 0.22, 0.45), 0.4)
    MATS["mech"] = mat("mechanism", (0.50, 0.52, 0.55), 0.5, 0.3)
    MATS["sensor"] = mat("sensor_edge", (0.07, 0.07, 0.08), 0.8)
    MATS["lamp"] = mat("lamp_off", (0.25, 0.26, 0.28), 0.4)
    MATS["brake"] = mat("brake_block", (0.30, 0.28, 0.27), 0.7)
    MATS["rubber"] = mat("rubber_black", (0.06, 0.06, 0.06), 0.9)
    MATS["spring"] = mat("spring_steel", (0.40, 0.42, 0.45), 0.5, 0.6)

# ----------------------------------------------------------------------------- helpers
def _link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj

def box(name, size, loc, material, bevel=0.0, segments=4):
    """Axis-aligned box with centre `loc` and full `size` (sx, sy, sz)."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    obj.location = Vector(loc)
    me.materials.append(material)
    _link(obj)
    if bevel > 0:
        mod = obj.modifiers.new("bevel", "BEVEL")
        mod.width = bevel
        mod.segments = segments
        mod.limit_method = "ANGLE"
    return obj

def cylinder_y(name, radius, depth, loc, material, verts=32):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=verts,
                          radius1=radius, radius2=radius, depth=depth)
    # created along Z; rotate to Y
    bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(math.radians(90), 3, "X"), verts=bm.verts)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    obj.location = Vector(loc)
    me.materials.append(material)
    return _link(obj)

def nose_block(name, xf, sx, nose_len, body_h, zc):
    """Cab nose: a block in front of the body end whose outer face is narrower,
    lower and raked back, giving the rounded-off R151-style front."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector((nose_len, WIDTH, body_h)), verts=bm.verts)
    for v in bm.verts:
        if v.co.x * sx > 0:  # outer face
            v.co.y *= 0.84
            zf = (v.co.z + body_h / 2) / body_h        # 0 bottom .. 1 top
            v.co.z = v.co.z * 0.90 - 0.05
            v.co.x -= sx * (0.10 + 0.45 * zf)          # rake: top set further back
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    obj.location = Vector((xf + sx * (nose_len / 2 - 0.3), 0, zc))
    me.materials.append(MATS["body"])
    _link(obj)
    mod = obj.modifiers.new("bevel", "BEVEL")
    mod.width = 0.28
    mod.segments = 6
    mod.limit_method = "ANGLE"
    apply_modifiers(obj)
    return obj

def join(objs, name):
    """Join objects into one mesh named `name` (first object survives)."""
    objs = [o for o in objs if o is not None]
    if not objs:
        return None
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    obj.data.name = name
    return obj

def apply_modifiers(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    for m in list(obj.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)

# ----------------------------------------------------------------------------- car parts
def car_body(n, kind, x0, length):
    """Body shell + roof + window band + underframe details, joined as car{n}_body."""
    parts = []
    cab_at_plus = (kind == "DT" and n == len(FORMATION))   # last DT: cab faces +X
    cab_at_minus = (kind == "DT" and n == 1)               # first DT: cab faces -X
    body_h = HEIGHT - FLOOR_Z
    zc = FLOOR_Z + body_h / 2

    # main shell (rounded), slightly narrower at the top via roof piece
    shell = box(f"shell{n}", (length, WIDTH, body_h), (x0, 0, zc), MATS["body"], bevel=0.32, segments=6)
    apply_modifiers(shell)
    parts.append(shell)

    # roof cap (grey) sitting on the top 0.12 m
    roof = box(f"roof{n}", (length - 0.6, WIDTH - 0.9, 0.16), (x0, 0, HEIGHT - 0.02), MATS["roof"], bevel=0.06, segments=3)
    apply_modifiers(roof)
    parts.append(roof)

    # continuous window band on both sides (doors sit on top of it)
    band_z0, band_z1 = FLOOR_Z + 0.95, FLOOR_Z + 1.95
    for sy in (-1, 1):
        band = box(f"band{n}_{sy}", (length - 1.2, 0.06, band_z1 - band_z0),
                   (x0, sy * (WIDTH / 2 - 0.01), (band_z0 + band_z1) / 2), MATS["band"])
        parts.append(band)
        # thin accent stripe under the windows
        stripe = box(f"stripe{n}_{sy}", (length - 1.2, 0.05, 0.10),
                     (x0, sy * (WIDTH / 2 - 0.005), band_z0 - 0.12), MATS["accent"])
        parts.append(stripe)

    # cab: tapered nose block + raked windscreen for DT cars
    if cab_at_plus or cab_at_minus:
        sx = 1 if cab_at_plus else -1
        xf = x0 + sx * (length / 2)
        nose_len = 1.6
        nose = nose_block(f"nosecap{n}", xf, sx, nose_len, body_h, zc)
        parts.append(nose)
        # raked windscreen: thin slab tilted back ~12 degrees, sitting on the nose face
        screen = box(f"screen{n}", (0.06, WIDTH - 1.0, 1.2), (0, 0, 0), MATS["glass"])
        screen.location = Vector((xf + sx * 0.93, 0, FLOOR_Z + 1.75))
        screen.rotation_euler = (0, sx * math.radians(-10), 0)
        parts.append(screen)
        # lower front panel (dark) and headlight strip
        front = box(f"front{n}", (0.06, WIDTH - 1.4, 0.75), (xf + sx * 1.13, 0, FLOOR_Z + 0.55), MATS["under"])
        parts.append(front)
        lamp = box(f"lamp{n}", (0.05, WIDTH - 1.2, 0.12), (xf + sx * 1.06, 0, FLOOR_Z + 1.05), MATS["door"])
        parts.append(lamp)
        # cab side windows
        for sy in (-1, 1):
            cw = box(f"cabwin{n}_{sy}", (0.9, 0.06, 0.95), (xf - sx * 1.0, sy * (WIDTH / 2 - 0.005), FLOOR_Z + 1.55), MATS["glass"])
            parts.append(cw)

    # underframe skirt and equipment boxes (not selectable)
    skirt = box(f"skirt{n}", (length - 1.0, WIDTH - 0.5, 0.35), (x0, 0, FLOOR_Z - 0.18), MATS["under"])
    parts.append(skirt)
    for i, (dx, sx_, sy_) in enumerate([(-4.5, 2.2, 1.1), (4.8, 1.6, 1.0), (-1.2, 1.2, 0.9)]):
        eq = box(f"eq{n}_{i}", (sx_, sy_, 0.55), (x0 + dx, (-1) ** i * 0.55, FLOOR_Z - 0.62), MATS["under"])
        parts.append(eq)

    # gangway toward +X neighbour
    if n < len(FORMATION):
        gw = box(f"gangway{n}", (GAP + 0.2, 1.7, 2.15), (x0 + length / 2 + GAP / 2, 0, FLOOR_Z + 1.075), MATS["gangway"])
        parts.append(gw)

    return join(parts, f"car{n}_body")

def roof_ac(n, x0):
    """Two roof-mounted air-conditioning units per car, own meshes car{n}_ac1 / car{n}_ac2
    (material ac_default) so the ACV task can tint the unit of the faulty car."""
    units = []
    for k, dx in enumerate(AC_CENTRES, start=1):
        # single material on purpose: the glTF exporter then keeps one mesh named car{n}_ac{k}
        # (a multi-material object would export as a group of car{n}_ac{k}_1, _2 primitives)
        ac = box(f"car{n}_ac{k}", AC_SIZE, (x0 + dx, 0, HEIGHT + 0.12), MATS["ac"], bevel=0.05, segments=2)
        apply_modifiers(ac)
        units.append(ac)
    return units


def empty(name, loc):
    e = bpy.data.objects.new(name, None)
    e.empty_display_size = 0.2
    e.location = Vector(loc)
    return _link(e)

def parent_to(children, parent):
    for c in children:
        if c is None:
            continue
        c.parent = parent
        c.matrix_parent_inverse = parent.matrix_world.inverted()

def cylinder(name, axis, radius, depth, loc, material, verts=24):
    """Cylinder along 'x', 'y' or 'z' with origin at its centre."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=verts,
                          radius1=radius, radius2=radius, depth=depth)
    if axis == "y":
        bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(math.radians(90), 3, "X"), verts=bm.verts)
    elif axis == "x":
        bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(math.radians(90), 3, "Y"), verts=bm.verts)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    obj.location = Vector(loc)
    me.materials.append(material)
    return _link(obj)

def door(n, side, idx, x):
    """Bi-parting pocket door. Node car{n}_door_{S}{d} (empty) with children:
    _leafA (slides -X to open), _leafB (slides +X), _drive (motor, belt, rail inside the wall),
    _pulley (spins), _lsOpen / _lsClosed (limit switches)."""
    sy = -1 if side == "L" else 1
    name = f"car{n}_door_{side}{idx}"
    y = sy * (WIDTH / 2 + 0.015)
    zc = FLOOR_Z + DOOR_H / 2
    root = empty(name, (x, y, zc))
    half = DOOR_W / 2
    leaves = []
    for tag, sgn in (("A", -1), ("B", 1)):
        lx = x + sgn * half / 2
        panel = box(f"{name}_leaf{tag}", (half - 0.012, 0.05, DOOR_H), (lx, y, zc), MATS["door"])
        glass = box(f"{name}_glass{tag}", (half - 0.30, 0.06, 0.9), (lx, y + sy * 0.005, FLOOR_Z + 1.35), MATS["glass"])
        edge = box(f"{name}_edge{tag}", (0.035, 0.07, DOOR_H - 0.06), (x + sgn * 0.02, y + sy * 0.005, zc), MATS["sensor"])
        leaves.append(join([panel, glass, edge], f"{name}_leaf{tag}"))
    ztop = FLOOR_Z + DOOR_H + 0.22
    yin = y - sy * 0.20          # inside the wall
    drive = box(f"{name}_motor", (0.55, 0.22, 0.20), (x + DOOR_W * 0.95, yin, ztop), MATS["mech"])
    belt = box(f"{name}_belt", (DOOR_W * 1.9, 0.015, 0.03), (x, yin - sy * 0.02, ztop), MATS["sensor"])
    rail = box(f"{name}_rail", (DOOR_W * 2.15, 0.04, 0.05), (x, y - sy * 0.09, FLOOR_Z + DOOR_H + 0.06), MATS["mech"])
    hangers = [box(f"{name}_hanger{t}", (0.05, 0.04, 0.16), (x + sgn * half / 2, y - sy * 0.07, FLOOR_Z + DOOR_H + 0.06), MATS["mech"])
               for t, sgn in (("A", -1), ("B", 1))]
    drive = join([drive, belt, rail] + hangers, f"{name}_drive")
    pulley = cylinder(f"{name}_pulley", "y", 0.10, 0.05, (x - DOOR_W * 0.95, yin - sy * 0.02, ztop), MATS["mech"], 20)
    spoke = box(f"{name}_spoke", (0.17, 0.06, 0.03), (x - DOOR_W * 0.95, yin - sy * 0.02, ztop), MATS["sensor"])
    pulley = join([pulley, spoke], f"{name}_pulley")
    ls_open = box(f"{name}_lsOpen", (0.07, 0.07, 0.07), (x - DOOR_W * 1.02, yin, ztop - 0.18), MATS["lamp"])
    ls_closed = box(f"{name}_lsClosed", (0.07, 0.07, 0.07), (x + 0.08, yin, ztop - 0.18), MATS["lamp"])
    parent_to(leaves + [drive, pulley, ls_open, ls_closed], root)
    return [root] + leaves + [drive, pulley, ls_open, ls_closed]

def bogie(n, idx, x, motored):
    """Bogie frame, springs, dampers, traction gear as car{n}_bogie{idx} (mesh).
    Separate: car{n}_wheel_{a}{S} (spin), car{n}_brake_{a}{S} (tread brake block),
    car{n}_axlebox_{a}{S}."""
    frame_z = 0.62
    parts = []
    frame = box(f"bframe{n}{idx}", (2.9, 2.35, 0.32), (x, 0, frame_z + 0.30), MATS["bogie"])
    parts.append(frame)
    for sy in (-1, 1):
        parts.append(box(f"bside{n}{idx}{sy}", (3.1, 0.18, 0.45), (x, sy * 1.05, frame_z + 0.28), MATS["bogie"]))
        parts.append(box(f"shoe{n}{idx}{sy}", (0.6, 0.25, 0.10), (x, sy * 1.45, 0.20), MATS["bogie"]))
        # secondary air spring (bolsterless)
        parts.append(cylinder(f"airspring{n}{idx}{sy}", "z", 0.30, 0.22, (x, sy * 0.85, FLOOR_Z - 0.12), MATS["rubber"], 14))
    axle_x = [x - 1.05, x + 1.05]
    for ai, ax in enumerate(axle_x):
        parts.append(cylinder(f"axle{n}{idx}{ai}", "y", 0.09, 2.4, (ax, 0, WHEEL_R), MATS["wheel"], 12))
        for sy in (-1, 1):
            # primary coil spring over the axle box + damper
            parts.append(cylinder(f"pspring{n}{idx}{ai}{sy}", "z", 0.12, 0.30, (ax, sy * 1.12, WHEEL_R + 0.40), MATS["spring"], 8))
            parts.append(cylinder(f"damper{n}{idx}{ai}{sy}", "z", 0.035, 0.36, (ax + 0.28, sy * 1.12, WHEEL_R + 0.40), MATS["mech"], 6))
        if motored:
            parts.append(box(f"tmotor{n}{idx}{ai}", (0.62, 0.95, 0.52), (ax + (0.55 if ai == 0 else -0.55), 0.15, 0.62), MATS["mech"]))
            parts.append(box(f"gearbox{n}{idx}{ai}", (0.40, 0.28, 0.44), (ax, 0.55, WHEEL_R + 0.02), MATS["bogie"]))
        else:
            parts.append(cylinder(f"bdisc{n}{idx}{ai}", "y", 0.30, 0.08, (ax, 0.30, WHEEL_R), MATS["brake"], 16))
    bog = join(parts, f"car{n}_bogie{idx}")
    extras = []
    for ai, ax in enumerate(axle_x):
        axle_no = (idx - 1) * 2 + ai + 1
        for side, sy in (("L", -1), ("R", 1)):
            extras.append(cylinder(f"car{n}_wheel_{axle_no}{side}", "y", WHEEL_R, 0.14, (ax, sy * GAUGE_HALF, WHEEL_R), MATS["wheel"], 24))
            # tread brake block sits ahead of and above the tread, presses radially inward
            extras.append(box(f"car{n}_brake_{axle_no}{side}", (0.26, 0.11, 0.20), (ax + 0.36, sy * GAUGE_HALF, WHEEL_R + 0.36), MATS["brake"]))
            ab = box(f"car{n}_axlebox_{axle_no}{side}", (0.42, 0.28, 0.40), (ax, sy * 1.12, WHEEL_R + 0.02), MATS["axlebox"], bevel=0.04, segments=2)
            apply_modifiers(ab)
            extras.append(ab)
    return [bog] + extras

def apu(n, x):
    """Air production unit near the -Y skirt so it is visible from the side:
    node car{n}_apu (empty) with _block, _fan (spins when loaded), _tower1/_tower2 (dryer), _tank, _pipes."""
    name = f"car{n}_apu"
    yc, zc = -0.72, 0.42
    root = empty(name, (x, yc, zc))
    block = box(f"{name}_block", (1.5, 0.9, 0.62), (x, yc, zc), MATS["apu"], bevel=0.04, segments=2)
    apply_modifiers(block)
    yf = yc - 0.47
    grille = box(f"{name}_grille", (0.62, 0.02, 0.62), (x - 0.30, yf, zc), MATS["under"])
    block = join([block, grille], f"{name}_block")
    hub = cylinder(f"{name}_fanhub", "y", 0.06, 0.06, (x - 0.30, yf - 0.02, zc), MATS["mech"], 12)
    blades = [box(f"{name}_blade{k}", (0.08, 0.02, 0.5) if k == 0 else (0.5, 0.02, 0.08), (x - 0.30, yf - 0.02, zc), MATS["mech"]) for k in range(2)]
    fan = join([hub] + blades, f"{name}_fan")
    towers = [cylinder(f"{name}_tower{k}", "z", 0.11, 0.50, (x + 0.55, yc - 0.28 + 0.36 * (k - 1), zc + 0.02), MATS["apu"], 12) for k in (1, 2)]
    tank = cylinder(f"{name}_tank", "x", 0.22, 1.3, (x + 1.75, yc + 0.25, zc), MATS["apu"], 14)
    pipes = [box(f"{name}_p1", (0.9, 0.04, 0.04), (x + 1.15, yc - 0.10, zc + 0.20), MATS["mech"]),
             box(f"{name}_p2", (0.04, 0.04, 0.5), (x + 0.75, yc - 0.10, zc + 0.40), MATS["mech"]),
             box(f"{name}_p3", (2.6, 0.04, 0.04), (x + 3.2, yc + 0.45, zc + 0.55), MATS["mech"])]
    pipes = join(pipes, f"{name}_pipes")
    children = [block, fan] + towers + [tank, pipes]
    parent_to(children, root)
    return [root] + children

# ----------------------------------------------------------------------------- train
def build_train():
    init_materials()
    x = 0.0
    objects = []
    for i, kind in enumerate(FORMATION):
        n = i + 1
        length = car_length(kind)
        xc = x + length / 2
        objects.append(car_body(n, kind, xc, length))
        objects.extend(roof_ac(n, xc))
        centres = door_centres(kind, n, len(FORMATION))
        for side in ("L", "R"):
            for d, dc in enumerate(centres, start=1):
                objects.extend(door(n, side, d, xc + dc))
        for b, bx in enumerate((xc - BOGIE_HALF_SPACING, xc + BOGIE_HALF_SPACING), start=1):
            objects.extend(bogie(n, b, bx, motored=(kind == "M")))
        objects.extend(apu(n, xc + 1.5))
        x += length + GAP
    total_len = x - GAP
    # rails: span the whole set plus RAIL_OVERHANG at both ends; "rail" (-Y, the L side) and
    # "rail001" (+Y, the R side). Tintable by the rail-corrugation task, context only in the twin.
    for sy in (-1, 1):
        # named explicitly (Blender's automatic "rail.001" would be sanitised to "rail001" by
        # three's GLTFLoader anyway; this keeps the glb and the viewport spelling identical)
        rail = box("rail" if sy < 0 else "rail001", (total_len + 2 * RAIL_OVERHANG, 0.07, 0.15),
                   (total_len / 2, sy * GAUGE_HALF, -0.075), MATS["wheel"])
        objects.append(rail)
    for o in objects:
        if o is not None and o.type == "MESH":
            for p in o.data.polygons:
                p.use_smooth = False
    return total_len

def selectable_names():
    names = []
    for i in range(1, len(FORMATION) + 1):
        for side in "LR":
            names += [f"car{i}_door_{side}{d}" for d in range(1, 5)]
        names += [f"car{i}_bogie1", f"car{i}_bogie2", f"car{i}_apu", f"car{i}_ac1", f"car{i}_ac2"]
        names += [f"car{i}_axlebox_{a}{s}" for a in range(1, 5) for s in "LR"]
        names += [f"car{i}_wheel_{a}{s}" for a in range(1, 5) for s in "LR"]
        names += [f"car{i}_brake_{a}{s}" for a in range(1, 5) for s in "LR"]
        for side in "LR":
            for d in range(1, 5):
                names += [f"car{i}_door_{side}{d}_{c}" for c in ("leafA", "leafB", "drive", "pulley", "lsOpen", "lsClosed")]
        names += [f"car{i}_apu_{c}" for c in ("block", "fan", "tower1", "tower2", "tank", "pipes")]
    return names

# ----------------------------------------------------------------------------- render
def setup_render(gpu=True):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 96
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    if gpu:
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for dev_type in ("OPTIX", "CUDA"):
                try:
                    prefs.compute_device_type = dev_type
                    prefs.get_devices()
                    if any(d.type == dev_type for d in prefs.devices):
                        for d in prefs.devices:
                            d.use = d.type in (dev_type, "CPU")
                        scene.cycles.device = "GPU"
                        print(f"[render] using {dev_type}")
                        break
                except Exception as e:  # noqa: BLE001
                    print(f"[render] {dev_type} unavailable: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"[render] GPU setup failed, CPU: {e}")
    # lighting: key sun + fill sun + soft world
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.85, 0.87, 0.9, 1)
    bg.inputs[1].default_value = 0.6
    key = bpy.data.lights.new("key", "SUN"); key.energy = 3.5; key.angle = math.radians(6)
    ko = bpy.data.objects.new("key", key); ko.rotation_euler = (math.radians(50), math.radians(-20), math.radians(35)); _link(ko)
    fill = bpy.data.lights.new("fill", "SUN"); fill.energy = 1.2; fill.angle = math.radians(20)
    fo = bpy.data.objects.new("fill", fill); fo.rotation_euler = (math.radians(65), math.radians(30), math.radians(-120)); _link(fo)

def render_view(name, cam_loc, cam_rot_deg, ortho_scale, res, out_dir, perspective=False, focal=35):
    scene = bpy.context.scene
    cam = bpy.data.cameras.new(name)
    cam.type = "PERSP" if perspective else "ORTHO"
    cam.ortho_scale = ortho_scale
    cam.lens = focal
    cam.clip_end = 2000
    co = bpy.data.objects.new(name, cam)
    co.location = Vector(cam_loc)
    co.rotation_euler = tuple(math.radians(a) for a in cam_rot_deg)
    _link(co)
    scene.camera = co
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.filepath = os.path.join(out_dir, f"{name}.png")
    bpy.ops.render.render(write_still=True)
    print(f"[render] wrote {scene.render.filepath}")

def look_at(cam_loc, target):
    d = Vector(target) - Vector(cam_loc)
    rot = d.to_track_quat("-Z", "Y").to_euler()
    return tuple(math.degrees(a) for a in rot)

# ----------------------------------------------------------------------------- main
def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cars", type=int, default=DEFAULT_N_CARS, help="cars in the set (DT + M... + DT)")
    ap.add_argument("--glb", default="web/public/r151.glb")
    ap.add_argument("--png", default="docs/design")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--blend", default="")
    args = ap.parse_args(argv)

    set_formation(args.n_cars)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    total_len = build_train()
    assert abs(layout(args.n_cars)["total_len"] - total_len) < 1e-6, "layout() drifted from build_train()"

    # naming contract check
    names = set(o.name for o in bpy.data.objects)
    missing = [n for n in selectable_names() if n not in names]
    assert not missing, f"missing selectable meshes: {missing[:10]}"
    tris = sum(len(p.vertices) - 2 for o in bpy.data.objects if o.type == "MESH" for p in o.data.polygons)
    print(f"[build] {len(FORMATION)} cars, {total_len:.1f} m, {len(names)} objects, ~{tris} triangles")

    os.makedirs(os.path.dirname(os.path.abspath(args.glb)), exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=os.path.abspath(args.glb), export_format="GLB",
                              export_apply=True, export_yup=True, use_selection=True,
                              export_materials="EXPORT", export_normals=True, export_texcoords=False)
    print(f"[export] {args.glb} {os.path.getsize(args.glb)/1e6:.2f} MB")

    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.blend))

    if not args.no_render:
        os.makedirs(args.png, exist_ok=True)
        setup_render()
        cx = total_len / 2
        # full side elevation, orthographic, from -Y looking +Y; see elevation_transform()
        render_view("r151_elevation", (cx, -120, ELEVATION_CAM_Z), (90, 0, 0), total_len * ELEVATION_MARGIN,
                    elevation_resolution(total_len), args.png)
        # two-car close-up elevation (leading DT + first M)
        render_view("r151_closeup", (23.0, -60, 2.0), (90, 0, 0), 52, (2400, 620), args.png)
        # three-quarter perspective on the leading cars
        loc = (-22.0, -34.0, 12.0)
        render_view("r151_iso", loc, look_at(loc, (14.0, 0.0, 1.8)), 1, (1800, 1000), args.png, perspective=True, focal=40)

if __name__ == "__main__":
    main()
