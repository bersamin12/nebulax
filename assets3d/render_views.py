"""Render subsystem close-ups from assets3d/r151.blend for design checks.
  blender -b -P assets3d/render_views.py -- docs/design
"""
import math
import os
import sys

import bpy
from mathutils import Vector

out_dir = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "docs/design"
bpy.ops.wm.open_mainfile(filepath="assets3d/r151.blend")
sys.path.insert(0, "assets3d")
import build_r151 as b  # noqa: E402

b.setup_render()
os.makedirs(out_dir, exist_ok=True)

def look_at(cam_loc, target):
    d = Vector(target) - Vector(cam_loc)
    return tuple(math.degrees(a) for a in d.to_track_quat("-Z", "Y").to_euler())

# car 4 geometry (see build_r151.FORMATION)
xc4 = 23.83 + 0.65 + 22.8 + 0.65 + 22.8 + 0.65 + 22.8 / 2
bog1 = xc4 - b.BOGIE_HALF_SPACING
apu_x = xc4 + 1.5
door_x = xc4 + 3.15

loc = (bog1 - 3.5, -6.5, 1.6)
b.render_view("view_bogie", loc, look_at(loc, (bog1, 0, 0.7)), 1, (1600, 900), out_dir, perspective=True, focal=45)
loc = (apu_x - 1.5, -5.5, 0.9)
b.render_view("view_apu", loc, look_at(loc, (apu_x + 0.6, -0.5, 0.45)), 1, (1600, 900), out_dir, perspective=True, focal=45)

# door mechanism: hide car 4 body shell so the drive is visible
body = bpy.data.objects.get("car4_body")
body.hide_render = True
loc = (door_x - 1.2, -6.0, 2.6)
b.render_view("view_door", loc, look_at(loc, (door_x, -1.6, 2.2)), 1, (1600, 900), out_dir, perspective=True, focal=50)
body.hide_render = False
