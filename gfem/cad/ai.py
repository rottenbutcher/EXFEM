"""Text-to-CAD: describe a part (or attach a drawing) and an LLM builds it as a gfem feature tree.

The model never writes free-form CAD code. It calls one tool, ``build_cad``, with a JSON feature
list in the same schema the GUI's feature tree uses (box, cylinder, extrude of sketches, boolean,
fillet, pattern ...). Every call is

  1. validated statically (types, numbers, sketch loops, feature references) -> precise errors,
  2. regenerated on OpenCASCADE -> kernel errors,
  3. measured (bodies, volumes, bounding box, faces) and rendered to a 4-view PNG,

and the result goes back to the model as the tool result, so it can check its own work against
the request ("the plate should be 120 mm long but the bbox says 100") and correct it. The loop
ends when the model stops calling the tool; the last *successful* build is the answer.

Backends
  anthropic : official ``anthropic`` SDK (Messages API, streaming, adaptive thinking, tool use)
  openai    : official ``openai`` SDK (Chat Completions, function calling); also any
              OpenAI-compatible server (Ollama, vLLM, LM Studio ...) through ``base_url``
  scripted  : replays prepared tool calls (tests, demos, offline)

API keys come from the environment (ANTHROPIC_API_KEY / OPENAI_API_KEY) or from
~/.gfem/ai.json (written with permissions 0600 by ``save_settings`` / ``gfem ai --set-key``).
Keys are never sent to the browser.
"""
from __future__ import annotations

import base64
import copy
import json
import math
import os
import struct
import time
import zlib

import numpy as np

from .geometry import CadModel, GeometryError, GMSH_LOCK, plane_frame, dxf_loops, loop_polygon, _loop_area

# ================================================================== settings
PROVIDERS = {
    "anthropic": dict(label="Anthropic Claude", env="ANTHROPIC_API_KEY", model="claude-opus-5-5",
                      models=["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-2"]),
    "openai": dict(label="OpenAI GPT", env="OPENAI_API_KEY", model="gpt-5",
                   models=["gpt-5", "gpt-5-mini"]),
    "compatible": dict(label="OpenAI-compatible (Ollama, vLLM, LM Studio)", env="GFEM_AI_COMPAT_KEY", model="",
                       models=[]),
}
EFFORTS = ["low", "medium", "high", "xhigh", "max"]
DEFAULTS = dict(provider="claude-cli", model="", effort="high", base_url="", max_turns=10, visual=True)


def settings_path():
    return os.environ.get("GFEM_AI_CONFIG") or os.path.join(os.path.expanduser("~"), ".gfem", "ai.json")


def load_settings():
    s = dict(DEFAULTS, keys={})
    try:
        with open(settings_path(), encoding="utf-8") as f:
            d = json.load(f)
        s.update({k: v for k, v in d.items() if k in DEFAULTS or k == "keys"})
    except (OSError, ValueError):
        pass
    s["keys"] = dict(s.get("keys") or {})
    return s


def save_settings(**kw):
    """Update settings; keys=dict(provider=key) merges (an empty string deletes a key)."""
    s = load_settings()
    keys = kw.pop("keys", None)
    for k, v in kw.items():
        if k in DEFAULTS and v is not None:
            s[k] = v
    if keys:
        for p, v in keys.items():
            if v:
                s["keys"][p] = v.strip()
            else:
                s["keys"].pop(p, None)
    path = settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return s


def api_key(provider, settings=None):
    env = PROVIDERS.get(provider, {}).get("env")
    if env and os.environ.get(env):
        return os.environ[env]
    return (settings or load_settings())["keys"].get(provider) or None


def public_settings():
    """Settings for the GUI: never contains a key, only whether one is available and where from."""
    s = load_settings()
    out = {k: s[k] for k in DEFAULTS}
    out["providers"] = {}
    for p, d in PROVIDERS.items():
        env = d["env"]
        src = "env" if os.environ.get(env) else ("file" if s["keys"].get(p) else None)
        out["providers"][p] = dict(label=d["label"], default_model=d["model"], models=d["models"],
                                   has_key=src is not None, key_source=src, env=env, sdk=_sdk_ok(p))
    out["efforts"] = EFFORTS
    from .ai_cli import CLIS
    out["cli"] = {k: dict(label=v["label"], models=v["models"], account=v["account"]) for k, v in CLIS.items()}
    return out


def _sdk_ok(provider):
    try:
        __import__("anthropic" if provider == "anthropic" else "openai")
        return True
    except ImportError:
        return False


# ================================================================== the feature schema (prompt)
SYSTEM_PROMPT = r"""You are the CAD engine of gfem, a finite-element program. You turn a description (or an attached
drawing) of a mechanical part into a parametric feature tree and build it with the `build_cad` tool.

# How to work
1. Read the request and any attached drawing. Decide the overall size, the base body, and the
   features (holes, pockets, bosses, ribs, slots, fillets, patterns). Use realistic engineering
   proportions for anything not specified, and state your assumptions briefly in your final message.
2. Call `build_cad` with the complete feature list. Every call replaces the previous model.
3. Read the tool result: errors, the measured bodies / volumes / bounding box, and the rendered
   views. Check them against the request (overall dimensions, number of holes, positions,
   number of bodies). If anything is wrong or a feature failed, fix it and call `build_cad` again.
4. When the model is right, stop calling the tool and reply with a short summary of the part and
   its main dimensions. Do not paste the JSON in the reply.

# Conventions
- Units are millimetres unless the user says otherwise; the `units` field is "mm", "m", "cm" or "in".
- Right-handed axes, Z is up. Put the part in the first octant or centred on the origin; keep the
  base face on z = 0 when there is a natural base.
- One connected solid is expected unless the user asks for an assembly. Fuse features that should
  be one part; for assemblies leave touching parts separate (the mesher shares their interfaces).
- Feature ids are unique strings ("F1", "F2", ...). A feature can only reference ids defined
  BEFORE it. Booleans, fillets, chamfers, a non-copy transform and a fused pattern CONSUME their
  target/tool bodies: afterwards only the new feature's id refers to the result. Example: after
  {"id":"F3","type":"boolean","op":"cut","target":["F1"],"tool":["F2"]} use "F3", not "F1".
- Apply fillets and chamfers late but before cutting many small holes when possible; keep fillet
  radii smaller than half of the adjacent wall thickness. A fillet that fails aborts the build:
  prefer selected edges over "all".
- Holes: a cylinder longer than the material (start 1 mm below, end 1 mm above) cut from the body.
  Many holes: cylinder + pattern (copies, not fused) and one cut with all ids as tool.

# Feature types (fields not listed are ignored; numbers are plain JSON numbers)
Primitives (create a new body):
- box:      x, y, z (min corner), dx, dy, dz (sizes, may be negative)
- cylinder: x, y, z (centre of the start face), dx, dy, dz (axis vector = height), r, angle (deg, default 360)
- cone:     x, y, z, dx, dy, dz (axis vector), r1 (start radius), r2 (end radius, may be 0)
- sphere:   x, y, z, r
- torus:    x, y, z, r1 (major), r2 (minor); axis is Z
- wedge:    x, y, z, dx, dy, dz, ltx (top length along x; 0 = sharp ridge)
Sketch based:
- extrude:  plane, offset, loops, distance, symmetric (bool)
- revolve:  plane, loops, axis [[u0,v0],[u1,v1]] (two points in sketch coordinates), angle (deg)
- surface:  plane, offset, loops                -> planar sheet body (meshed with shells)
- line:     points [[x,y,z],...], closed (bool)  -> wire body (meshed with beams)
  plane: the sketch frame (u, v) and the extrusion direction (normal):
     "XY": u = +x, v = +y, normal +z          "YZ": u = +y, v = +z, normal +x
     "ZX": u = +z, v = +x, normal +y          (note: in "ZX" the FIRST coordinate is z)
     {"origin":[x,y,z], "normal":[nx,ny,nz], "xdir":[ux,uy,uz]}: u = xdir, v = normal x xdir.
     Front-view profile (u = +x, v = +z), extruded toward +y: {"origin":[0,D,0],"normal":[0,-1,0],
     "xdir":[1,0,0]} with distance D extrudes from y = D back to y = 0 (or use offset/negative distance).
     Check v with the right-hand rule: normal [0,1,0] with xdir [1,0,0] gives v = -z.
  offset: distance of the sketch plane along its normal (from origin).
  loops: closed 2-D profiles in sketch coordinates (u, v). Each loop is ONE of
     {"rect": [u0, v0, u1, v1]}
     {"circle": [uc, vc, r]}
     {"slot": [u0, v0, u1, v1, r]}         (obround between two centres)
     {"points": [[u,v], ...], "bulges": [b0, b1, ...]}   (polygon; optional bulge per segment
        i -> i+1: 0 straight, tan(theta/4) arc, +1/-1 semicircle; last segment closes the loop)
  Nesting is automatic: a loop inside another is a hole; a loop inside a hole is an island.
  Separate outer loops give separate bodies (set "fuse_regions": true to fuse them).
  Extrude goes along +normal by `distance` (negative = opposite), from `offset`.
Operations:
- boolean:  op "fuse"|"cut"|"intersect", target [ids], tool [ids], keep_tool (bool)
- fillet:   target [ids], r, edges
- chamfer:  target [ids], d, edges
  edges: "all", or a filter {"parallel":"x"|"y"|"z", "box":[xmin,ymin,zmin,xmax,ymax,zmax],
         "type":"Line"|"Circle", "min_length":L, "max_length":L}  (all criteria must hold),
         or a list of edges [{"com":[x,y,z], "size":length}, ...] (edge midpoint, length) or filters.
- transform: target id or [ids], op "translate" (vector) | "rotate" (point, axis, angle deg) |
             "mirror" (point, normal) | "scale" (point, factor), copy (bool: keep the original)
- pattern:  target id or [ids], kind "linear" (direction, spacing) | "circular" (point, axis,
            angle deg total, default 360), count (including the original), fuse (bool: fuse the
            copies with the original into one body)
Give every feature a short "name" ("Base plate", "Bolt hole", "Bolt pattern", ...): it is the label
the user sees in the feature tree. Give the part a "name" in every build_cad call.

# Example: 100 x 60 x 10 plate, 4 corner holes d8, a central boss d30 x 20 with a d16 bore
{"units":"mm","name":"Mounting plate","features":[
 {"id":"F1","type":"box","name":"Plate","x":0,"y":0,"z":0,"dx":100,"dy":60,"dz":10},
 {"id":"F2","type":"cylinder","name":"Boss","x":50,"y":30,"z":10,"dx":0,"dy":0,"dz":20,"r":15},
 {"id":"F3","type":"boolean","op":"fuse","target":["F1"],"tool":["F2"]},
 {"id":"F4","type":"fillet","target":["F3"],"r":3,"edges":[{"com":[50,30,10],"size":94.25}]},
 {"id":"F5","type":"cylinder","name":"Bore","x":50,"y":30,"z":-1,"dx":0,"dy":0,"dz":32,"r":8},
 {"id":"F6","type":"cylinder","name":"Hole","x":10,"y":10,"z":-1,"dx":0,"dy":0,"dz":12,"r":4},
 {"id":"F7","type":"pattern","target":"F6","kind":"linear","direction":[1,0,0],"spacing":80,"count":2},
 {"id":"F8","type":"pattern","target":["F6","F7"],"kind":"linear","direction":[0,1,0],"spacing":40,"count":2},
 {"id":"F9","type":"boolean","op":"cut","target":["F4"],"tool":["F5","F6","F7","F8"]}]}
"""

TOOL_NAME = "build_cad"
TOOL_DESCRIPTION = (
    "Build (or rebuild) the complete CAD model from a feature list and return validation/kernel "
    "errors, measured bodies (volume, bounding box), and rendered views. Each call replaces the "
    "whole model, so always send every feature.")
TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Short name of the part"},
        "units": {"type": "string", "enum": ["mm", "m", "cm", "in"]},
        "features": {"type": "array", "description": "Ordered feature tree (see system prompt)",
                     "items": {"type": "object"}},
    },
    "required": ["name", "features"],
}

# ================================================================== static validation
_NUM_FIELDS = {
    "box": dict(x=0, y=0, z=0, dx=1, dy=1, dz=1),
    "cylinder": dict(x=0, y=0, z=0, dx=0, dy=0, dz=1, r=1, angle=360),
    "cone": dict(x=0, y=0, z=0, dx=0, dy=0, dz=1, r1=1, r2=0),
    "sphere": dict(x=0, y=0, z=0, r=1),
    "torus": dict(x=0, y=0, z=0, r1=2, r2=0.5),
    "wedge": dict(x=0, y=0, z=0, dx=1, dy=1, dz=1, ltx=0),
    "extrude": dict(distance=10, offset=0),
    "revolve": dict(angle=360, offset=0),
    "surface": dict(offset=0),
    "fillet": dict(r=1),
    "chamfer": dict(d=1),
    "pattern": dict(count=2, spacing=1, angle=360),
    "transform": dict(angle=90, factor=1),
}
TYPES = set(_NUM_FIELDS) | {"line", "boolean", "dxf_extrude", "import"}
_CONSUMES = {"boolean", "fillet", "chamfer"}


class ValidationResult:
    def __init__(self):
        self.errors, self.warnings = [], []

    def err(self, where, msg):
        self.errors.append(f"{where}: {msg}")

    def warn(self, where, msg):
        self.warnings.append(f"{where}: {msg}")


def _isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _vec(v, n=3):
    return isinstance(v, (list, tuple)) and len(v) == n and all(_isnum(x) for x in v)


def _seg_intersect(p1, p2, p3, p4):
    d = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2, d3, d4 = d(p3, p4, p1), d(p3, p4, p2), d(p1, p2, p3), d(p1, p2, p4)
    return (d1 * d2 < -1e-12) and (d3 * d4 < -1e-12)


def _check_loop(vr, where, lp):
    if not isinstance(lp, dict):
        return vr.err(where, "a loop must be an object like {\"rect\": [...]}, {\"circle\": [...]}, {\"slot\": [...]} or {\"points\": [...]}")
    kinds = [k for k in ("rect", "circle", "slot", "points") if k in lp]
    if len(kinds) != 1:
        return vr.err(where, f"a loop needs exactly one of rect / circle / slot / points (got {sorted(lp)})")
    k = kinds[0]
    v = lp[k]
    if k == "rect":
        if not _vec(v, 4):
            return vr.err(where, "rect must be [u0, v0, u1, v1]")
        if abs(v[2] - v[0]) < 1e-9 or abs(v[3] - v[1]) < 1e-9:
            return vr.err(where, f"rect {v} has zero width or height")
    elif k == "circle":
        if not _vec(v, 3):
            return vr.err(where, "circle must be [uc, vc, r]")
        if v[2] <= 0:
            return vr.err(where, f"circle radius must be > 0 (got {v[2]})")
    elif k == "slot":
        if not _vec(v, 5):
            return vr.err(where, "slot must be [u0, v0, u1, v1, r]")
        if v[4] <= 0 or math.hypot(v[2] - v[0], v[3] - v[1]) < 1e-9:
            return vr.err(where, "slot needs r > 0 and two distinct centres")
    else:
        if not isinstance(v, list) or len(v) < 3 or not all(_vec(p, 2) for p in v):
            return vr.err(where, "points must be a list of at least 3 [u, v] pairs")
        pts = [tuple(map(float, p)) for p in v]
        if len(pts) > 3 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        for i in range(len(pts)):
            if np.allclose(pts[i], pts[(i + 1) % len(pts)], atol=1e-9):
                return vr.err(where, f"points {i} and {(i + 1) % len(pts)} coincide")
        b = lp.get("bulges")
        if b is not None:
            if not isinstance(b, list) or not all(_isnum(x) for x in b):
                return vr.err(where, "bulges must be a list of numbers")
            if len(b) > len(v):
                vr.warn(where, f"{len(b)} bulges for {len(v)} points: extra values ignored")
            if any(abs(x) > 100 for x in b):
                return vr.err(where, "bulge magnitude > 100 (use tan(theta/4); 1 = semicircle)")
        if not b or not any(b):
            n = len(pts)
            for i in range(n):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    if _seg_intersect(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n]):
                        return vr.err(where, f"profile is self-intersecting (segment {i} crosses segment {j})")
    try:
        if _loop_area(lp) < 1e-12:
            vr.err(where, "loop has zero area")
    except Exception as e:  # noqa: BLE001
        vr.err(where, f"invalid loop: {e}")


def _check_plane(vr, where, plane):
    if plane is None or isinstance(plane, str):
        if str(plane or "XY").upper() not in ("XY", "YZ", "ZX", "XZ"):
            vr.err(where, f"plane must be XY, YZ, ZX or an object (got {plane!r})")
        return
    if not isinstance(plane, dict):
        return vr.err(where, "plane must be a string or {origin, normal, xdir}")
    for k in ("origin", "normal", "xdir"):
        if k in plane and not _vec(plane[k]):
            return vr.err(where, f"plane.{k} must be [x, y, z]")
    n = np.asarray(plane.get("normal", (0, 0, 1)), float)
    if np.linalg.norm(n) < 1e-12:
        return vr.err(where, "plane.normal is zero")
    if "xdir" in plane:
        x = np.asarray(plane["xdir"], float)
        if np.linalg.norm(np.cross(x, n)) < 1e-9 * max(np.linalg.norm(x), 1e-30):
            vr.err(where, "plane.xdir is parallel to the normal")


def _ids(v):
    if v is None:
        return []
    return [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]


def validate_features(features, max_features=400):
    """Static check of a feature list. Returns (normalised features, ValidationResult).

    Normalisation: ids become strings (missing ids are assigned), numeric strings become numbers,
    "tool"/"target" lists of ids become string lists. Nothing is modified in place."""
    vr = ValidationResult()
    if not isinstance(features, list):
        vr.err("features", "must be a list")
        return [], vr
    if not features:
        vr.err("features", "is empty")
        return [], vr
    if len(features) > max_features:
        vr.err("features", f"too many features ({len(features)} > {max_features})")
        return [], vr
    out = []
    alive, seen = {}, set()       # id -> kind ('solid', 'sheet', 'wire')
    for k, f0 in enumerate(features):
        if not isinstance(f0, dict):
            vr.err(f"feature #{k + 1}", "must be an object")
            continue
        f = copy.deepcopy(f0)
        fid = str(f.get("id") or f"F{k + 1}")
        while fid in seen and not f0.get("id"):
            fid += "_"
        f["id"] = fid
        where = f"feature {fid}" + (f" ({f.get('name')})" if f.get("name") else "")
        t = f.get("type")
        if fid in seen:
            vr.err(where, "duplicate id")
        seen.add(fid)
        if t not in TYPES:
            vr.err(where, f"unknown type {t!r}; use one of {sorted(TYPES - {'import', 'dxf_extrude'})}")
            out.append(f)
            continue
        if f.get("suppressed"):
            out.append(f)
            continue
        for key, dv in _NUM_FIELDS.get(t, {}).items():
            if key in f:
                v = f[key]
                if isinstance(v, str):
                    try:
                        v = float(v)
                        f[key] = v
                    except ValueError:
                        pass
                if not _isnum(v):
                    vr.err(where, f"{key} must be a number (got {f[key]!r})")
        for key in ("vector", "point", "axis", "direction", "normal"):
            if key in f and not (t == "revolve" and key == "axis") and not _vec(f[key]):
                vr.err(where, f"{key} must be [x, y, z]")
        num = lambda key: float(f.get(key, _NUM_FIELDS.get(t, {}).get(key, 0)) if _isnum(f.get(key, 0)) else 0)
        # --- per type
        if t == "box":
            for key in ("dx", "dy", "dz"):
                if abs(num(key)) < 1e-9:
                    vr.err(where, f"{key} must be non-zero")
        elif t in ("cylinder", "cone"):
            if math.sqrt(num("dx") ** 2 + num("dy") ** 2 + num("dz") ** 2) < 1e-9:
                vr.err(where, "axis vector (dx, dy, dz) must be non-zero (it is the height)")
            if t == "cylinder" and num("r") <= 0:
                vr.err(where, "r must be > 0")
            if t == "cylinder" and not 0 < num("angle") <= 360:
                vr.err(where, "angle must be in (0, 360]")
            if t == "cone" and (num("r1") < 0 or num("r2") < 0 or num("r1") + num("r2") <= 0 or abs(num("r1") - num("r2")) < 1e-12):
                vr.err(where, "cone needs r1, r2 >= 0, not both zero and r1 != r2 (use a cylinder)")
        elif t == "sphere":
            if num("r") <= 0:
                vr.err(where, "r must be > 0")
        elif t == "torus":
            if num("r2") <= 0 or num("r1") <= num("r2"):
                vr.err(where, "torus needs r1 > r2 > 0")
        elif t == "wedge":
            for key in ("dx", "dy", "dz"):
                if abs(num(key)) < 1e-9:
                    vr.err(where, f"{key} must be non-zero")
        elif t in ("extrude", "revolve", "surface"):
            _check_plane(vr, where, f.get("plane"))
            loops = f.get("loops")
            if not isinstance(loops, list) or not loops:
                vr.err(where, "loops must be a non-empty list of closed profiles")
            else:
                for i, lp in enumerate(loops):
                    _check_loop(vr, f"{where} loop {i + 1}", lp)
            if t == "extrude" and abs(num("distance")) < 1e-9:
                vr.err(where, "distance must be non-zero")
            if t == "revolve":
                ax = f.get("axis", [[0, 0], [0, 1]])
                if not (isinstance(ax, list) and len(ax) == 2 and all(_vec(p, 2) for p in ax)):
                    vr.err(where, "revolve axis must be [[u0, v0], [u1, v1]] in sketch coordinates")
                elif np.allclose(ax[0], ax[1]):
                    vr.err(where, "revolve axis points coincide")
                elif isinstance(loops, list) and all(isinstance(lp, dict) for lp in loops):
                    a, b = np.asarray(ax[0], float), np.asarray(ax[1], float)
                    d = (b - a) / np.linalg.norm(b - a)
                    nrm = np.array([-d[1], d[0]])
                    for i, lp in enumerate(loops):
                        try:
                            side = (loop_polygon(lp) - a) @ nrm
                        except Exception:  # noqa: BLE001
                            continue
                        if side.min() < -1e-6 * (1 + abs(side).max()) and side.max() > 1e-6 * (1 + abs(side).max()):
                            vr.err(where, f"loop {i + 1} crosses the revolve axis (profile must lie on one side)")
                if not 0 < num("angle") <= 360:
                    vr.err(where, "angle must be in (0, 360]")
        elif t == "line":
            pts = f.get("points")
            if not isinstance(pts, list) or len(pts) < 2 or not all(_vec(p) for p in pts):
                vr.err(where, "points must be a list of at least 2 [x, y, z]")
        elif t == "fillet" and num("r") <= 0:
            vr.err(where, "r must be > 0")
        elif t == "chamfer" and num("d") <= 0:
            vr.err(where, "d must be > 0")
        elif t == "pattern":
            if int(num("count")) < 2 or abs(num("count") - round(num("count"))) > 1e-9:
                vr.err(where, "count must be an integer >= 2 (it includes the original)")
            if int(num("count")) > 500:
                vr.err(where, "count > 500")
            if f.get("kind", "linear") not in ("linear", "circular"):
                vr.err(where, "kind must be linear or circular")
            if f.get("kind", "linear") == "linear" and "direction" in f and _vec(f["direction"]) and np.linalg.norm(f["direction"]) < 1e-12:
                vr.err(where, "direction is zero")
        elif t == "transform":
            op = f.get("op", "translate")
            if op not in ("translate", "rotate", "mirror", "scale"):
                vr.err(where, "op must be translate, rotate, mirror or scale")
            if op == "scale" and num("factor") <= 0:
                vr.err(where, "factor must be > 0")
        elif t == "boolean":
            if f.get("op", "fuse") not in ("fuse", "cut", "intersect"):
                vr.err(where, "op must be fuse, cut or intersect")
            if not _ids(f.get("tool")) and f.get("op", "fuse") == "fuse" and len(_ids(f.get("target"))) > 1:
                tg = _ids(f.get("target"))
                f["target"], f["tool"] = tg[:1], tg[1:]              # "fuse these bodies"
            elif not _ids(f.get("tool")):
                vr.err(where, "tool must list at least one id (to fuse several bodies, list them all in target)")
            if set(_ids(f.get("target"))) & set(_ids(f.get("tool"))):
                vr.err(where, "an id cannot be both target and tool")
        if t in ("fillet", "chamfer"):
            e = f.get("edges", "all")
            if not (e == "all" or isinstance(e, (dict, list))):
                vr.err(where, "edges must be \"all\", a filter object or a list")
        # --- references
        refs = []
        if t in ("boolean", "fillet", "chamfer", "transform", "pattern"):
            tg = _ids(f.get("target"))
            if not tg:
                vr.err(where, "target must list at least one id")
            refs = tg + (_ids(f.get("tool")) if t == "boolean" else [])
            if t == "boolean":
                f["target"], f["tool"] = _ids(f.get("target")), _ids(f.get("tool"))
            elif t in ("fillet", "chamfer"):
                f["target"] = _ids(f.get("target"))
        bad = [r for r in refs if r not in alive]
        if bad:
            consumed = [r for r in bad if r in seen and r != fid]
            hint = f"; alive bodies are {sorted(alive)}" if alive else ""
            if consumed:
                vr.err(where, f"references {consumed}, which an earlier operation already consumed "
                              f"(use the id of the operation that produced the current body){hint}")
            else:
                vr.err(where, f"references unknown id(s) {bad} (only earlier features can be referenced){hint}")
        kinds = {r: alive.get(r) for r in refs if r in alive}
        if t in ("fillet", "chamfer") and any(v != "solid" for v in kinds.values()):
            vr.err(where, "fillet/chamfer need solid targets")
        # --- update alive set
        if t in _CONSUMES:
            for r in _ids(f.get("target")):
                alive.pop(r, None)
            if t != "boolean" or not f.get("keep_tool"):
                for r in _ids(f.get("tool")):
                    alive.pop(r, None)
            alive[fid] = "solid"
        elif t == "transform":
            if not f.get("copy"):
                for r in _ids(f.get("target")):
                    alive.pop(r, None)
            alive[fid] = next(iter(kinds.values()), "solid")
        elif t == "pattern":
            if f.get("fuse"):
                for r in _ids(f.get("target")):
                    alive.pop(r, None)
            alive[fid] = next(iter(kinds.values()), "solid")
        else:
            alive[fid] = {"surface": "sheet", "line": "wire"}.get(t, "solid")
        out.append(f)
    return out, vr


# ================================================================== evaluation
def evaluate(features, units="mm", render=True, size=320):
    """Validate + regenerate + measure (+ render). Returns a JSON-able report dict with keys
    ok, errors, warnings, bodies, sheets, lines, bbox, size, volume, png (bytes or None)."""
    feats, vr = validate_features(features)
    rep = dict(ok=False, errors=list(vr.errors), warnings=list(vr.warnings), features=feats, png=None)
    if vr.errors:
        return rep
    cad = CadModel(feats, units=units if units in ("mm", "m", "cm", "in") else "mm")
    try:
        with GMSH_LOCK:
            info = cad.regenerate()
            tess = cad.tessellate(resolution=60) if render else None
    except GeometryError as e:
        rep["errors"].append(f"kernel: {e}")
        return rep
    except Exception as e:  # noqa: BLE001
        rep["errors"].append(f"kernel: {type(e).__name__}: {e}")
        return rep
    bodies = [dict(feature=b.get("feature"), volume=round(float(b["volume"]), 4),
                   bbox=[round(float(x), 4) for x in b["bbox"]], faces=len(b["faces"])) for b in info["bodies"]]
    bb = np.asarray(info.get("bbox", [0, 0, 0, 0, 0, 0]), float)
    rep.update(ok=True, units=cad.units, bodies=bodies,
               sheets=[dict(feature=s.get("feature"), area=round(s["area"], 4)) for s in info.get("sheets", [])],
               lines=[dict(feature=s.get("feature"), length=round(s["length"], 4)) for s in info.get("lines", [])],
               bbox=[round(float(x), 4) for x in bb], size=[round(float(x), 4) for x in (bb[3:] - bb[:3])],
               volume=round(float(sum(b["volume"] for b in info["bodies"])), 4),
               n_faces=len(info["faces"]), n_edges=len(info["edges"]),
               holes=_count_holes(info))
    rep["warnings"] += list(info.get("messages") or [])
    if not info["bodies"] and not info.get("sheets") and not info.get("lines"):
        rep["ok"] = False
        rep["errors"].append("the model is empty (all bodies were consumed or a boolean produced nothing)")
    for b in info["bodies"]:
        if b["volume"] <= 1e-9 * max(1.0, info.get("diag", 1.0) ** 3):
            rep["warnings"].append(f"body from feature {b.get('feature')} has (almost) zero volume")
    if len(info["bodies"]) > 1:
        rep["warnings"].append(f"{len(info['bodies'])} separate solid bodies: fuse them if the part should be one piece")
    if render and tess:
        try:
            rep["png"] = render_views(tess[0], tess[1], size=size)
        except Exception as e:  # noqa: BLE001
            rep["warnings"].append(f"preview render failed: {e}")
    rep["features"] = feats
    rep["info"] = info
    return rep


def _count_holes(info):
    """Cylindrical faces grouped by axis line: a rough 'number of round holes/bosses' for the LLM."""
    n = sum(1 for f in info["faces"] if f.get("type") == "Cylinder")
    return n


def report_text(rep):
    """Compact JSON summary for the model (no png, no raw info)."""
    keys = ("ok", "errors", "warnings", "units", "size", "bbox", "volume", "bodies", "sheets", "lines",
            "n_faces", "n_edges", "holes")
    d = {k: rep[k] for k in keys if k in rep and rep[k] not in (None, [], {})}
    if "holes" in d:
        d["cylindrical_faces"] = d.pop("holes")
    if d.get("ok"):
        d["views"] = "image: top-left isometric, top-right front (XZ, looking +Y), bottom-left top (XY, looking -Z), bottom-right right side (YZ, looking -X); axes X red, Y green, Z blue"
    return json.dumps(d, separators=(",", ":"))


# ================================================================== tiny software renderer (PNG)
def _png(img):
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _view_basis(name):
    if name == "iso":                                # camera at (+1, -1, +0.8), looking at the centre
        d = -np.array([1.0, -1.0, 0.8])
        d /= np.linalg.norm(d)
        up = np.array([0, 0, 1.0])
    else:
        d, up = {"front": (np.array([0, 1.0, 0]), np.array([0, 0, 1.0])),
                 "top": (np.array([0, 0, -1.0]), np.array([0, 1.0, 0])),
                 "right": (np.array([-1.0, 0, 0]), np.array([0, 0, 1.0]))}[name]
    r = np.cross(d, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, d)
    return r, u, d


def _raster(tris, edges, name, size, center, scale):
    r, u, d = _view_basis(name)
    img = np.full((size, size, 3), 250, np.uint8)
    zb = np.full((size, size), np.inf)
    light = -d + 0.6 * u + 0.3 * r
    light /= np.linalg.norm(light)
    half = size / 2

    def proj(P):
        Q = P - center
        return np.stack([half + (Q @ r) * scale, half - (Q @ u) * scale, Q @ d], -1)

    if len(tris):
        T = proj(tris.reshape(-1, 3)).reshape(-1, 3, 3)
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        nn = np.linalg.norm(n, axis=1, keepdims=True)
        n = n / np.where(nn > 0, nn, 1)
        n = np.where((n @ d)[:, None] > 0, -n, n)            # two-sided
        shade = 0.25 + 0.75 * np.clip(n @ light, 0, 1)
        base = np.array([120, 160, 205], float)
        for k in range(len(T)):
            (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = T[k]
            xmin, xmax = int(max(min(x0, x1, x2), 0)), int(min(max(x0, x1, x2) + 1, size))
            ymin, ymax = int(max(min(y0, y1, y2), 0)), int(min(max(y0, y1, y2) + 1, size))
            if xmin >= xmax or ymin >= ymax:
                continue
            den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(den) < 1e-12:
                continue
            xs, ys = np.meshgrid(np.arange(xmin, xmax) + 0.5, np.arange(ymin, ymax) + 0.5)
            a = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / den
            b = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / den
            c = 1 - a - b
            m = (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6)
            if not m.any():
                continue
            z = a * z0 + b * z1 + c * z2
            sub = zb[ymin:ymax, xmin:xmax]
            m &= z < sub
            sub[m] = z[m]
            img[ymin:ymax, xmin:xmax][m] = (base * shade[k]).astype(np.uint8)
    for e in edges:
        P = proj(np.asarray(e, float))
        for (xa, ya, za), (xb, yb, zb_) in zip(P[:-1], P[1:]):
            n = int(max(abs(xb - xa), abs(yb - ya)) * 1.5) + 2
            t = np.linspace(0, 1, n)
            xs = (xa + (xb - xa) * t).astype(int)
            ys = (ya + (yb - ya) * t).astype(int)
            zs = za + (zb_ - za) * t
            ok = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
            xs, ys, zs = xs[ok], ys[ok], zs[ok]
            vis = zs <= zb[ys, xs] + 1e-3 * (1 + abs(zs)) + 2.0 / scale
            img[ys[vis], xs[vis]] = (30, 30, 40)
    # axis triad (bottom-left corner)
    o = np.array([28.0, size - 28.0])
    for vec, col in ((np.array([1.0, 0, 0]), (220, 40, 40)), (np.array([0, 1.0, 0]), (30, 160, 50)),
                     (np.array([0, 0, 1.0]), (40, 70, 220))):
        e2 = np.array([vec @ r, -(vec @ u)]) * 20
        for t in np.linspace(0, 1, 40):
            x, y = (o + e2 * t).astype(int)
            img[max(y - 1, 0):y + 1, max(x - 1, 0):x + 1] = col
    img[:, [0, -1]] = 180
    img[[0, -1], :] = 180
    return img


def render_views(faces, edges, size=320):
    """4-view PNG (iso, front, top, right) of a tessellation from CadModel.tessellate()."""
    tris = [f["xyz"][f["tri"]] for f in faces if len(f["tri"])]
    tris = np.concatenate(tris).astype(float) if tris else np.zeros((0, 3, 3))
    eds = [np.asarray(e["xyz"], float) for e in edges]
    pts = np.concatenate([tris.reshape(-1, 3)] + eds) if (len(tris) or eds) else np.zeros((1, 3))
    lo, hi = pts.min(0), pts.max(0)
    center = (lo + hi) / 2
    rad = max(np.linalg.norm(hi - lo) / 2, 1e-9)
    scale = size * 0.44 / rad
    views = [_raster(tris, eds, v, size, center, scale) for v in ("iso", "front", "top", "right")]
    img = np.concatenate([np.concatenate(views[:2], 1), np.concatenate(views[2:], 1)], 0)
    return _png(img)


# ================================================================== attachments
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}


def load_attachment(path):
    """-> dict(kind='image'|'pdf'|'text', media_type, data(bytes)|text, name)."""
    ext = os.path.splitext(path)[1].lower()
    name = os.path.basename(path)
    if ext in IMAGE_TYPES:
        with open(path, "rb") as f:
            return dict(kind="image", media_type=IMAGE_TYPES[ext], data=f.read(), name=name)
    if ext == ".pdf":
        with open(path, "rb") as f:
            return dict(kind="pdf", media_type="application/pdf", data=f.read(), name=name)
    if ext == ".dxf":
        loops, sc = dxf_loops(path)
        lp = []
        for L in loops:
            if "circle" in L:
                lp.append(dict(circle=[round(v * sc, 4) for v in L["circle"]]))
            else:
                d = dict(points=[[round(p[0] * sc, 4), round(p[1] * sc, 4)] for p in L["points"]])
                if L.get("bulges") and any(L["bulges"]):
                    d["bulges"] = [round(b, 6) for b in L["bulges"]]
                lp.append(d)
        text = (f"DXF drawing '{name}' converted to closed sketch loops in mm (u = drawing X, v = drawing Y). "
                f"You can use them directly as `loops` of an extrude/revolve:\n" + json.dumps(lp))
        return dict(kind="text", text=text, name=name)
    if ext in (".step", ".stp", ".iges", ".igs", ".brep"):
        cad = CadModel([dict(id="F1", type="import", file=path)])
        info = cad.regenerate()
        text = (f"Existing CAD file '{name}' (it can be referenced with a feature "
                f"{{\"type\":\"import\",\"file\":{json.dumps(path)}}}). Measured: "
                + json.dumps(dict(bbox=[round(x, 3) for x in info["bbox"]],
                                  bodies=[dict(volume=round(b["volume"], 3), bbox=[round(x, 3) for x in b["bbox"]]) for b in info["bodies"]])))
        return dict(kind="text", text=text, name=name)
    with open(path, encoding="utf-8", errors="replace") as f:
        return dict(kind="text", text=f"Attached file '{name}':\n" + f.read()[:100000], name=name)


# ================================================================== backends
class ToolCall:
    def __init__(self, id, name, args, error=None):
        self.id, self.name, self.args, self.error = id, name, args, error


class AIError(Exception):
    pass


class _Backend:
    usage = None

    def step(self, on_text=None):
        """-> (texts, [ToolCall], stop_reason)"""
        raise NotImplementedError

    def add_tool_results(self, results):
        """results: [(ToolCall, text, png_bytes|None, is_error)]"""
        raise NotImplementedError

    def add_user(self, text):
        raise NotImplementedError


class AnthropicBackend(_Backend):
    """Claude through the official SDK: streaming Messages API with one client tool."""
    _MODERN = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-5", "claude-fable-5", "claude-mythos-5",
               "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-4-6")
    _FALLBACK_OK = ("claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5")

    def __init__(self, system, user_content, model, api_key=None, effort="high", base_url=None,
                 max_tokens=32000, client=None):
        if client is None:
            try:
                import anthropic
            except ImportError as e:
                raise AIError("the 'anthropic' package is not installed: pip install anthropic") from e
            kw = dict(api_key=api_key) if api_key else {}
            if base_url:
                kw["base_url"] = base_url
            client = anthropic.Anthropic(**kw)
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        self.tools = [{"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "input_schema": TOOL_SCHEMA,
                       "eager_input_streaming": True}]
        self.messages = [{"role": "user", "content": user_content}]
        self.usage = dict(input=0, output=0, cache_read=0)
        self._eager = True
        self._fallbacks = model in self._FALLBACK_OK

    @staticmethod
    def content(text, attachments=()):
        parts = []
        for a in attachments:
            if a["kind"] == "image":
                parts.append({"type": "image", "source": {"type": "base64", "media_type": a["media_type"],
                                                          "data": base64.standard_b64encode(a["data"]).decode()}})
            elif a["kind"] == "pdf":
                parts.append({"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                                             "data": base64.standard_b64encode(a["data"]).decode()}})
            else:
                parts.append({"type": "text", "text": a["text"]})
        parts.append({"type": "text", "text": text})
        return parts

    def _request(self):
        kw = dict(model=self.model, max_tokens=self.max_tokens, system=self.system, tools=self.tools,
                  messages=self.messages)
        if self.model.startswith(self._MODERN):
            kw["thinking"] = {"type": "adaptive"}
            if self.effort:
                kw["output_config"] = {"effort": self.effort}
        if not self._eager:
            kw["tools"] = [{k: v for k, v in t.items() if k != "eager_input_streaming"} for t in self.tools]
        return kw

    def step(self, on_text=None):
        import anthropic
        kw = self._request()
        try:
            if self._fallbacks:
                stream = self.client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"],
                                                          fallbacks="default", **kw)
            else:
                stream = self.client.messages.stream(**kw)
            with stream as st:
                for ev in st:
                    if on_text and getattr(ev, "type", "") == "content_block_delta" and \
                            getattr(ev.delta, "type", "") == "text_delta":
                        on_text(ev.delta.text)
                msg = st.get_final_message()
        except anthropic.BadRequestError as e:
            text = str(e)
            if self._eager and "eager_input_streaming" in text:
                self._eager = False
                return self.step(on_text)
            if self._fallbacks and ("fallback" in text.lower()):
                self._fallbacks = False
                return self.step(on_text)
            raise AIError(f"Anthropic API rejected the request: {getattr(e, 'message', text)}") from e
        except anthropic.AuthenticationError as e:
            raise AIError("Anthropic API key is invalid (authentication failed)") from e
        except anthropic.PermissionDeniedError as e:
            raise AIError(f"Anthropic API key lacks permission: {getattr(e, 'message', e)}") from e
        except anthropic.NotFoundError as e:
            raise AIError(f"model '{self.model}' not found: {getattr(e, 'message', e)}") from e
        except anthropic.RateLimitError as e:
            raise AIError("Anthropic rate limit reached; wait a moment and try again") from e
        except anthropic.APIStatusError as e:
            raise AIError(f"Anthropic API error {e.status_code}: {getattr(e, 'message', e)}") from e
        except anthropic.APIConnectionError as e:
            raise AIError(f"cannot reach the Anthropic API: {e}") from e
        u = msg.usage
        self.usage["input"] += (u.input_tokens or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0)
        self.usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
        self.usage["output"] += u.output_tokens or 0
        self.messages.append({"role": "assistant", "content": msg.content})
        if msg.stop_reason == "refusal":
            det = getattr(msg, "stop_details", None)
            raise AIError("the model declined the request" + (f": {det.explanation}" if det and getattr(det, "explanation", None) else ""))
        texts, calls = [], []
        for b in msg.content:
            if b.type == "text" and b.text.strip():
                texts.append(b.text)
            elif b.type == "tool_use":
                args, err = b.input, None
                if not isinstance(args, dict):
                    try:
                        args = json.loads(args)
                    except (TypeError, ValueError):
                        args, err = {}, "INVALID_JSON: the tool input was not valid JSON; send it again"
                if msg.stop_reason == "max_tokens":
                    err = ("the tool input was cut off by the output limit; send a more compact feature list "
                           "(fewer features, patterns instead of repeated features)")
                calls.append(ToolCall(b.id, b.name, args, err))
        return texts, calls, msg.stop_reason

    def add_tool_results(self, results):
        content = []
        for call, text, png, is_error in results:
            c = [{"type": "text", "text": text}]
            if png:
                c.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                      "data": base64.standard_b64encode(png).decode()}})
            content.append({"type": "tool_result", "tool_use_id": call.id, "content": c, "is_error": bool(is_error)})
        self.messages.append({"role": "user", "content": content})

    def add_user(self, text):
        self.messages.append({"role": "user", "content": [{"type": "text", "text": text}]})


class OpenAIBackend(_Backend):
    """GPT (or any OpenAI-compatible server) through the official SDK: Chat Completions + function tool."""

    def __init__(self, system, user_content, model, api_key=None, effort="high", base_url=None, client=None):
        if client is None:
            try:
                import openai
            except ImportError as e:
                raise AIError("the 'openai' package is not installed: pip install openai") from e
            kw = dict(api_key=api_key or ("not-needed" if base_url else None))
            if base_url:
                kw["base_url"] = base_url
            client = openai.OpenAI(**kw)
        self.client = client
        self.model = model
        self.effort = effort if effort in ("low", "medium", "high") else ("high" if effort else None)
        self.tools = [{"type": "function", "function": {"name": TOOL_NAME, "description": TOOL_DESCRIPTION,
                                                        "parameters": TOOL_SCHEMA}}]
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": user_content}]
        self.usage = dict(input=0, output=0, cache_read=0)
        self._images = True
        self._reason = bool(self.effort) and not base_url

    @staticmethod
    def content(text, attachments=()):
        parts = []
        for a in attachments:
            if a["kind"] == "image":
                parts.append({"type": "image_url", "image_url": {
                    "url": f"data:{a['media_type']};base64," + base64.standard_b64encode(a["data"]).decode()}})
            elif a["kind"] == "pdf":
                parts.append({"type": "file", "file": {"filename": a["name"], "file_data":
                              "data:application/pdf;base64," + base64.standard_b64encode(a["data"]).decode()}})
            else:
                parts.append({"type": "text", "text": a["text"]})
        parts.append({"type": "text", "text": text})
        return parts

    def step(self, on_text=None):
        import openai
        kw = dict(model=self.model, messages=self.messages, tools=self.tools, tool_choice="auto")
        if self._reason:
            kw["reasoning_effort"] = self.effort
        try:
            r = self.client.chat.completions.create(**kw)
        except openai.BadRequestError as e:
            text = str(e)
            if self._reason and "reasoning" in text.lower():
                self._reason = False
                return self.step(on_text)
            if self._images and ("image" in text.lower() or "content" in text.lower()):
                self._images = False
                self._strip_images()
                return self.step(on_text)
            raise AIError(f"OpenAI API rejected the request: {getattr(e, 'message', text)}") from e
        except openai.AuthenticationError as e:
            raise AIError("OpenAI API key is invalid (authentication failed)") from e
        except openai.NotFoundError as e:
            raise AIError(f"model '{self.model}' not found: {getattr(e, 'message', e)}") from e
        except openai.RateLimitError as e:
            raise AIError("OpenAI rate limit / quota reached") from e
        except openai.APIStatusError as e:
            raise AIError(f"OpenAI API error {e.status_code}: {getattr(e, 'message', e)}") from e
        except openai.APIConnectionError as e:
            raise AIError(f"cannot reach the API server: {e}") from e
        if getattr(r, "usage", None):
            self.usage["input"] += r.usage.prompt_tokens or 0
            self.usage["output"] += r.usage.completion_tokens or 0
        ch = r.choices[0]
        m = ch.message
        entry = {"role": "assistant", "content": m.content or ""}
        calls = []
        if m.tool_calls:
            entry["tool_calls"] = [{"id": tc.id, "type": "function",
                                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                   for tc in m.tool_calls]
            for tc in m.tool_calls:
                try:
                    args, err = json.loads(tc.function.arguments or "{}"), None
                except ValueError:
                    args, err = {}, "INVALID_JSON: the arguments were not valid JSON" + (
                        " (cut off by the output limit; send a more compact feature list)" if ch.finish_reason == "length" else "")
                calls.append(ToolCall(tc.id, tc.function.name, args, err))
        self.messages.append(entry)
        if m.content and on_text:
            on_text(m.content)
        if getattr(m, "refusal", None):
            raise AIError(f"the model declined the request: {m.refusal}")
        return ([m.content] if m.content and m.content.strip() else []), calls, ch.finish_reason

    def _strip_images(self):
        for msg in self.messages:
            if isinstance(msg.get("content"), list):
                msg["content"] = [p for p in msg["content"] if p.get("type") == "text"] or ""

    def add_tool_results(self, results):
        imgs = []
        for call, text, png, is_error in results:
            self.messages.append({"role": "tool", "tool_call_id": call.id,
                                  "content": ("ERROR: " if is_error else "") + text})
            if png and self._images:
                imgs.append(png)
        if imgs:          # tool messages cannot carry images: send the views as a user message
            self.messages.append({"role": "user", "content": [
                {"type": "text", "text": "Rendered views of the last build_cad result (iso / front / top / right):"}] + [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.standard_b64encode(p).decode()}}
                for p in imgs]})

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})


class ScriptedBackend(_Backend):
    """Replays a list of turns, each a feature list (one build_cad call) or a str (final text).
    Used by the tests and for offline demos; records the tool results it receives."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.results = []
        self.usage = dict(input=0, output=0, cache_read=0)
        self._n = 0

    def step(self, on_text=None):
        if not self.turns:
            return ["done"], [], "end_turn"
        t = self.turns.pop(0)
        if isinstance(t, str):
            if on_text:
                on_text(t)
            return [t], [], "end_turn"
        self._n += 1
        args = t if isinstance(t, dict) and "features" in t else dict(features=t)
        return [], [ToolCall(f"call{self._n}", TOOL_NAME, args)], "tool_use"

    def add_tool_results(self, results):
        self.results.append(results)

    def add_user(self, text):
        self.results.append(text)


# ================================================================== the agent loop
class AIResult(dict):
    """features, name, units, summary, report (last successful evaluate), png, turns, builds,
    usage, log, ok"""


def describe_features(features):
    return json.dumps(features, separators=(",", ":"))


def make_backend(provider, user_content_text, attachments=(), model=None, api_key=None, effort="high",
                 base_url=None, client=None, system=SYSTEM_PROMPT):
    provider = provider or "anthropic"
    if provider not in PROVIDERS:
        raise AIError(f"unknown provider {provider!r}; use one of claude-cli, codex-cli, {', '.join(sorted(PROVIDERS))}")
    model = model or PROVIDERS[provider]["model"]
    if not model:
        raise AIError("a model name is required for an OpenAI-compatible server")
    if client is None and provider != "compatible" and not api_key:
        raise AIError(f"no API key for {PROVIDERS[provider]['label']}: set {PROVIDERS[provider]['env']} "
                      f"or save one (GUI: AI CAD > Settings, CLI: gfem ai --set-key {provider})")
    if provider == "anthropic":
        return AnthropicBackend(system, AnthropicBackend.content(user_content_text, attachments), model,
                                api_key=api_key, effort=effort, base_url=base_url or None, client=client)
    if provider == "compatible" and not base_url:
        raise AIError("an OpenAI-compatible server needs a base URL (e.g. http://localhost:11434/v1 for Ollama)")
    return OpenAIBackend(system, OpenAIBackend.content(user_content_text, attachments), model,
                         api_key=api_key, effort=effort, base_url=base_url or None, client=client)


def build_request(prompt, base_features=None, units="mm"):
    text = prompt.strip()
    if base_features:
        text = ("Modify the existing model. Its current feature tree is below; keep everything that the "
                "request does not change (same ids where possible) and send the complete new tree.\n"
                f"Current model (units {units}):\n{describe_features(base_features)}\n\nRequest: {text}")
    else:
        text = f"Units: {units}.\nRequest: {text}"
    return text


def run_agent(backend, units="mm", max_turns=10, visual=True, log=None, on_text=None, on_build=None,
              should_stop=None, base_features=None):
    """Drive the tool loop. log(str) receives progress lines, on_build(report) every build."""
    log = log or (lambda s: None)
    best, builds, texts = None, 0, []
    t0 = time.time()
    nudged = False
    for turn in range(1, max_turns + 1):
        if should_stop and should_stop():
            log("AI: stopped by the user")
            break
        log(f"AI: turn {turn} (waiting for the model…)")
        txt, calls, stop = backend.step(on_text=on_text)
        for t in txt:
            texts.append(t)
            log("AI: " + t.strip().replace("\n", "\n    "))
        if not calls:
            if best is None and not nudged and stop != "refusal":
                nudged = True
                backend.add_user("You have not built anything yet. Call the build_cad tool with the feature list.")
                continue
            break
        results = []
        for c in calls:
            if c.name != TOOL_NAME:
                results.append((c, f"unknown tool {c.name}; the only tool is {TOOL_NAME}", None, True))
                continue
            if c.error:
                results.append((c, c.error, None, True))
                log("AI: tool input rejected: " + c.error)
                continue
            feats = c.args.get("features")
            u = c.args.get("units") or units
            builds += 1
            rep = evaluate(feats, units=u, render=visual)
            rep["name"] = c.args.get("name") or (best or {}).get("name") or "AI part"
            nf = len(feats) if isinstance(feats, list) else 0
            if rep["ok"]:
                best = rep
                log(f"AI: build {builds}: {nf} features -> {len(rep['bodies'])} body(ies), "
                    f"size {' x '.join(f'{v:g}' for v in rep['size'])} {rep['units']}, volume {rep['volume']:.6g}"
                    + (f"  [{len(rep['warnings'])} warning(s)]" if rep["warnings"] else ""))
            else:
                log(f"AI: build {builds}: {nf} features -> {len(rep['errors'])} error(s): " + "; ".join(rep["errors"][:3]))
            if on_build:
                on_build(rep)
            results.append((c, report_text(rep), rep.get("png") if visual else None, not rep["ok"]))
        backend.add_tool_results(results)
    else:
        log(f"AI: reached the limit of {max_turns} turns")
    res = AIResult(ok=best is not None, features=best["features"] if best else [], name=(best or {}).get("name"),
                   units=(best or {}).get("units", units), summary="\n".join(texts[-1:]).strip(),
                   report=best, png=(best or {}).get("png"), builds=builds,
                   usage=getattr(backend, "usage", None), seconds=round(time.time() - t0, 2))
    if best is None:
        res["error"] = "the model did not produce a valid geometry"
    return res


def generate(prompt, provider=None, model=None, api_key=None, base_url=None, effort=None, attachments=(),
             base_features=None, units="mm", max_turns=None, visual=None, log=None, on_text=None,
             on_build=None, should_stop=None, client=None, backend=None, run_dir=None):
    """One-call text-to-CAD. Missing options come from the saved settings.
    provider: 'claude-cli' / 'codex-cli' (signed-in account, no key), 'anthropic' / 'openai' /
    'compatible' (API key or local server)."""
    s = load_settings()
    provider = provider or s["provider"]
    if provider in ("claude-cli", "codex-cli") and backend is None and client is None:   # signed-in CLI
        from .ai_cli import run_cli
        if model is None and provider == s.get("provider"):
            model = s.get("model") or None
        return run_cli(provider, prompt, attachments=attachments, base_features=base_features, units=units,
                       model=model, effort=effort or s.get("effort") or None,
                       visual=s.get("visual", True) if visual is None else visual, log=log, on_build=on_build,
                       should_stop=should_stop, run_dir=run_dir)
    if model is None and provider == s.get("provider"):
        model = s.get("model") or None             # the saved model belongs to the saved provider
    effort = effort or s.get("effort") or "high"
    if base_url is None:
        base_url = s.get("base_url") if provider == "compatible" else None
    visual = s.get("visual", True) if visual is None else visual
    max_turns = int(max_turns or s.get("max_turns") or 10)
    atts = [a if isinstance(a, dict) else load_attachment(a) for a in attachments]
    text = build_request(prompt, base_features, units)
    if backend is None:
        if provider in ("claude-cli", "codex-cli"):        # a client object was given: use the API path
            provider = "anthropic" if provider == "claude-cli" else "openai"
        backend = make_backend(provider, text, atts, model=model, api_key=api_key or api_key_for(provider, s),
                               effort=effort, base_url=base_url or None, client=client)
    return run_agent(backend, units=units, max_turns=max_turns, visual=visual, log=log, on_text=on_text,
                     on_build=on_build, should_stop=should_stop, base_features=base_features)


def api_key_for(provider, settings=None):
    return api_key(provider, settings)
