"""Parametric CAD on OpenCASCADE (through the gmsh API).

A geometry is an ordered *feature tree* (plain JSON-serialisable dicts), re-evaluated from scratch
on every change, exactly like the history tree of a commercial CAD system:

  primitives : box, cylinder, sphere, cone, torus, wedge
  sketches   : extrude / revolve of 2D profiles (polylines with arcs/bulges, circles, rectangles,
               slots) on any plane; DXF profiles (dxf_extrude)
  operations : boolean (fuse / cut / intersect), fillet, chamfer, transform (translate, rotate,
               mirror, scale, copy), linear / circular pattern
  import     : STEP / IGES / BREP (OpenCASCADE, unit conversion, healing), STL (faceted bodies)

Topological naming: boundary conditions refer to faces/edges/bodies by tag *and* by a geometric
signature (centroid, area/length, type). After a regeneration that renumbers entities the
signatures are used to re-attach every condition to the right face (see `match`).
"""
import math
import os
import threading
import numpy as np

GMSH_LOCK = threading.RLock()
_gmsh_ready = False
_CURRENT = [None]          # the CadModel whose shapes are currently loaded in gmsh


def gmsh_module():
    global _gmsh_ready
    import gmsh
    if not _gmsh_ready:
        try:
            gmsh.initialize(readConfigFiles=False, interruptible=False)
        except TypeError:                       # older gmsh without 'interruptible'
            gmsh.initialize(readConfigFiles=False)
        try:                                   # keep Python's SIGPIPE behaviour (BrokenPipeError)
            import signal
            if hasattr(signal, "SIGPIPE") and threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGPIPE, signal.SIG_IGN)
        except Exception:  # noqa: BLE001
            pass
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 2)
        _gmsh_ready = True
    return gmsh


UNIT_SCALE = {"mm": 1.0, "m": 1000.0, "cm": 10.0, "in": 25.4, "um": 1e-3}
_OCC_UNIT = {"mm": "MM", "m": "M", "cm": "CM", "in": "INCH", "um": "UM"}


class GeometryError(Exception):
    pass


# ------------------------------------------------------------------ sketch helpers
def plane_frame(plane):
    """plane: 'XY'|'YZ'|'ZX' (+ offset) or dict(origin, normal, xdir). Returns (o, ex, ey, ez)."""
    if isinstance(plane, str) or plane is None:
        name = (plane or "XY").upper()
        o = np.zeros(3)
        if name == "XY":
            ex, ey = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        elif name == "YZ":
            ex, ey = np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
        elif name in ("ZX", "XZ"):
            ex, ey = np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
        else:
            raise GeometryError(f"unknown plane {plane}")
        return o, ex, ey, np.cross(ex, ey)
    o = np.asarray(plane.get("origin", (0, 0, 0)), float)
    ez = np.asarray(plane.get("normal", (0, 0, 1)), float)
    ez = ez / np.linalg.norm(ez)
    ex = np.asarray(plane.get("xdir", (1, 0, 0) if abs(ez[0]) < 0.9 else (0, 1, 0)), float)
    ex = ex - ez * ex.dot(ez)
    ex /= np.linalg.norm(ex)
    return o, ex, np.cross(ez, ex), ez


def _arc_center(p0, p1, bulge):
    """Centre of the arc from p0 to p1 with DXF bulge (tan(theta/4))."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    theta = 4 * math.atan(bulge)
    chord = p1 - p0
    c = np.linalg.norm(chord)
    r = c / (2 * math.sin(abs(theta) / 2))
    mid = (p0 + p1) / 2
    n = np.array([-chord[1], chord[0]]) / c
    h = math.sqrt(max(r * r - (c / 2) ** 2, 0))
    sgn = 1 if (bulge > 0) == (abs(theta) < math.pi) else -1
    return mid + sgn * h * n, r, theta


class _Builder:
    """Adds OCC entities for sketch loops."""

    def __init__(self, gmsh, frame):
        self.g = gmsh
        self.o, self.ex, self.ey, self.ez = frame
        self.occ = gmsh.model.occ

    def P(self, uv):
        x = self.o + uv[0] * self.ex + uv[1] * self.ey
        return self.occ.addPoint(*x)

    def loop(self, lp):
        occ = self.occ
        if "circle" in lp:
            cx, cy, r = lp["circle"]
            c = self.o + cx * self.ex + cy * self.ey
            tag = occ.addCircle(*c, r, zAxis=list(self.ez), xAxis=list(self.ex))
            return occ.addCurveLoop([tag])
        if "rect" in lp:
            x0, y0, x1, y1 = lp["rect"]
            pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            return self.loop(dict(points=pts))
        if "slot" in lp:                     # obround: centres (x0,y0)-(x1,y1), radius r
            x0, y0, x1, y1, r = lp["slot"]
            d = np.array([x1 - x0, y1 - y0], float)
            L = np.linalg.norm(d)
            t = d / L
            n = np.array([-t[1], t[0]])
            a, b = np.array([x0, y0]), np.array([x1, y1])
            pts = [a + r * n, b + r * n, b - r * n, a - r * n]
            # bulge 1 = semicircle (arcs end -> end)
            return self.loop(dict(points=[list(p) for p in pts], bulges=[0, -1, 0, -1]))
        pts = [np.asarray(p, float) for p in lp["points"]]
        if np.allclose(pts[0], pts[-1]) and len(pts) > 2:
            pts = pts[:-1]
        bul = list(lp.get("bulges", [0] * len(pts))) + [0] * len(pts)
        ptags = [self.P(p) for p in pts]
        curves = []
        n = len(pts)
        for i in range(n):
            j = (i + 1) % n
            b = bul[i]
            if abs(b) < 1e-12:
                curves.append(occ.addLine(ptags[i], ptags[j]))
            else:
                c, r, th = _arc_center(pts[i], pts[j], b)
                if abs(th) > math.pi * 0.999:          # split large arcs
                    v0 = pts[i] - c
                    ang = math.atan2(v0[1], v0[0]) + th / 2
                    pm = c + r * np.array([math.cos(ang), math.sin(ang)])
                    tm = self.P(pm)
                    tc = self.P(c)
                    curves.append(occ.addCircleArc(ptags[i], tc, tm))
                    tc2 = self.P(c)
                    curves.append(occ.addCircleArc(tm, tc2, ptags[j]))
                else:
                    tc = self.P(c)
                    curves.append(occ.addCircleArc(ptags[i], tc, ptags[j]))
        return occ.addCurveLoop(curves)

    def face(self, loops):
        tags = [self.loop(lp) for lp in loops]
        return self.occ.addPlaneSurface(tags)


def _loop_area(lp):
    if "circle" in lp:
        return math.pi * lp["circle"][2] ** 2
    if "rect" in lp:
        x0, y0, x1, y1 = lp["rect"]
        return abs((x1 - x0) * (y1 - y0))
    if "slot" in lp:
        x0, y0, x1, y1, r = lp["slot"]
        return 2 * r * math.hypot(x1 - x0, y1 - y0) + math.pi * r * r
    p = np.asarray(lp["points"], float)
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def sort_loops(loops):
    """Largest loop first (outer boundary), the rest are holes."""
    return sorted(loops, key=_loop_area, reverse=True)


# ------------------------------------------------------------------ DXF
def dxf_loops(path, layer=None, tol=1e-6):
    """Closed profiles from a DXF file: LWPOLYLINE/POLYLINE (bulges), CIRCLE, and chains of
    LINE/ARC entities. Returns (loops, scale_to_mm)."""
    import ezdxf
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    units = doc.header.get("$INSUNITS", 4)
    scale = {1: 25.4, 2: 304.8, 4: 1.0, 5: 10.0, 6: 1000.0}.get(units, 1.0)
    loops, segs = [], []
    for e in msp:
        if layer and e.dxf.layer != layer:
            continue
        t = e.dxftype()
        if t == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in e.get_points("xy")]
            bul = [p[4] for p in e.get_points("xyseb")]
            if e.closed or (len(pts) > 2 and np.allclose(pts[0], pts[-1])):
                loops.append(dict(points=[list(p) for p in pts], bulges=bul))
            else:
                for i in range(len(pts) - 1):
                    segs.append(("L" if abs(bul[i]) < 1e-12 else "B", pts[i], pts[i + 1], bul[i]))
        elif t == "POLYLINE":
            pts = [(v.dxf.location[0], v.dxf.location[1]) for v in e.vertices]
            bul = [v.dxf.bulge for v in e.vertices]
            if e.is_closed:
                loops.append(dict(points=[list(p) for p in pts], bulges=bul))
        elif t == "CIRCLE":
            c = e.dxf.center
            loops.append(dict(circle=[c[0], c[1], e.dxf.radius]))
        elif t == "LINE":
            segs.append(("L", tuple(e.dxf.start)[:2], tuple(e.dxf.end)[:2], 0.0))
        elif t == "ARC":
            c, r = e.dxf.center, e.dxf.radius
            a0, a1 = math.radians(e.dxf.start_angle), math.radians(e.dxf.end_angle)
            if a1 < a0:
                a1 += 2 * math.pi
            p0 = (c[0] + r * math.cos(a0), c[1] + r * math.sin(a0))
            p1 = (c[0] + r * math.cos(a1), c[1] + r * math.sin(a1))
            segs.append(("B", p0, p1, math.tan((a1 - a0) / 4)))
    # chain loose segments into loops
    segs = list(segs)
    while segs:
        k, a, b, bu = segs.pop(0)
        pts, bul = [a, b], [bu]
        changed = True
        while changed and not np.allclose(pts[0], pts[-1], atol=tol * 1e3):
            changed = False
            for i, (k2, c, d, bu2) in enumerate(segs):
                if np.allclose(c, pts[-1], atol=1e-6):
                    pts.append(d)
                    bul.append(bu2)
                elif np.allclose(d, pts[-1], atol=1e-6):
                    pts.append(c)
                    bul.append(-bu2)
                else:
                    continue
                segs.pop(i)
                changed = True
                break
        if np.allclose(pts[0], pts[-1], atol=1e-6) and len(pts) > 2:
            loops.append(dict(points=[list(p) for p in pts[:-1]], bulges=bul))
    return loops, scale


# ------------------------------------------------------------------ the model
class CadModel:
    def __init__(self, features=None, units="mm", share_topology=True):
        self.features = list(features or [])
        self.units = units
        self.share_topology = share_topology
        self.info = None
        self.stl_mode = False
        self.messages = []

    # ---- evaluation
    def regenerate(self):
        with GMSH_LOCK:
            g = gmsh_module()
            g.clear()
            g.model.add("cad")
            g.option.setString("Geometry.OCCTargetUnit", _OCC_UNIT.get(self.units, "MM"))
            occ = g.model.occ
            self.messages = []
            bodies = {}                # feature id -> [(3, tag)]
            owner = {}                 # tag -> feature id
            self.stl_mode = any(f.get("type") == "import" and str(f.get("file", "")).lower().endswith(".stl")
                                for f in self.features if not f.get("suppressed"))
            for f in self.features:
                if f.get("suppressed"):
                    continue
                fid = f.get("id") or f"F{len(bodies) + 1}"
                try:
                    out = self._feature(g, f, bodies)
                except Exception as e:      # noqa: BLE001
                    raise GeometryError(f"feature '{f.get('name', fid)}' ({f.get('type')}): {e}")  # noqa
                if out is not None:
                    bodies[fid] = out
            owner = {(d, tag): fid for fid, lst in bodies.items() for d, tag in lst}
            if not self.stl_mode:
                occ.synchronize()
                # top-level entities: volumes, free sheets (faces not bounding a volume), free curves
                vols = list(g.model.getEntities(3))
                bounded = {abs(t) for v in vols for _, t in g.model.getBoundary([v], oriented=False)}
                sheets = [(2, t) for _, t in g.model.getEntities(2) if t not in bounded]
                on_face = {abs(t) for _, f in g.model.getEntities(2) for _, t in g.model.getBoundary([(2, f)], oriented=False)}
                curves = [(1, t) for _, t in g.model.getEntities(1)
                          if t not in on_face and not len(g.model.getAdjacencies(1, t)[0])]
                vols = vols + sheets + curves
                if self.share_topology and len(vols) > 1:
                    try:
                        _, omap = occ.fragment(vols[:1], vols[1:])
                        new_owner = {}
                        for (d, old), outs in zip(vols, omap):
                            for d2, t2 in outs:
                                if d2 == d and (d, old) in owner:
                                    new_owner.setdefault((d2, t2), owner[(d, old)])
                        owner = new_owner
                        occ.synchronize()
                    except Exception as e:   # noqa: BLE001
                        self.messages.append(f"share topology failed: {e}")
                occ.synchronize()
            self._owner = owner
            self.info = self._collect(g)
            _CURRENT[0] = self
            return self.info

    def ensure_loaded(self):
        """Re-evaluate the feature tree if gmsh currently holds another model."""
        with GMSH_LOCK:
            if _CURRENT[0] is not self or self.info is None:
                self.regenerate()

    def _ids_to_tags(self, bodies, ids):
        out = []
        for i in ids if isinstance(ids, (list, tuple)) else [ids]:
            if i not in bodies:
                raise GeometryError(f"unknown or consumed feature {i}")
            out += bodies.pop(i)
        return out

    def _feature(self, g, f, bodies):
        occ = g.model.occ
        t = f["type"]
        num = lambda k, d=0.0: float(f.get(k, d))
        if t == "box":
            return [(3, occ.addBox(num("x"), num("y"), num("z"), num("dx", 1), num("dy", 1), num("dz", 1)))]
        if t == "cylinder":
            ang = math.radians(num("angle", 360.0))
            return [(3, occ.addCylinder(num("x"), num("y"), num("z"), num("dx"), num("dy"), num("dz", 1),
                                        num("r", 1), angle=ang))]
        if t == "sphere":
            return [(3, occ.addSphere(num("x"), num("y"), num("z"), num("r", 1)))]
        if t == "cone":
            return [(3, occ.addCone(num("x"), num("y"), num("z"), num("dx"), num("dy"), num("dz", 1),
                                    num("r1", 1), num("r2", 0)))]
        if t == "torus":
            return [(3, occ.addTorus(num("x"), num("y"), num("z"), num("r1", 2), num("r2", 0.5)))]
        if t == "wedge":
            return [(3, occ.addWedge(num("x"), num("y"), num("z"), num("dx", 1), num("dy", 1), num("dz", 1),
                                     ltx=num("ltx", 0)))]
        if t in ("extrude", "revolve", "dxf_extrude"):
            frame = plane_frame(f.get("plane", "XY"))
            o, ex, ey, ez = frame
            if f.get("offset"):
                o = o + ez * float(f["offset"])
                frame = (o, ex, ey, ez)
            loops = f.get("loops", [])
            if t == "dxf_extrude":
                lp, sc = dxf_loops(f["file"], f.get("layer"))
                s = float(f.get("scale", sc))
                loops = []
                for L in lp:
                    if "circle" in L:
                        c = L["circle"]
                        loops.append(dict(circle=[c[0] * s, c[1] * s, c[2] * s]))
                    else:
                        loops.append(dict(points=[[p[0] * s, p[1] * s] for p in L["points"]],
                                          bulges=L.get("bulges")))
            if not loops:
                raise GeometryError("empty sketch")
            loops = sort_loops(loops)
            # separate disjoint outer regions: a loop is a hole if inside the first loop's bbox area
            b = _Builder(g, frame)
            outer, holes = loops[0], loops[1:]
            surf = b.face([outer] + holes)
            if t == "revolve":
                ax = f.get("axis", [[0, 0], [0, 1]])
                p0 = o + ax[0][0] * ex + ax[0][1] * ey
                p1 = o + ax[1][0] * ex + ax[1][1] * ey
                d = p1 - p0
                out = occ.revolve([(2, surf)], *p0, *d, math.radians(num("angle", 360.0)))
            else:
                dist = num("distance", 10.0)
                if f.get("symmetric"):
                    occ.translate([(2, surf)], *(-ez * dist / 2))
                out = occ.extrude([(2, surf)], *(ez * dist))
            return [x for x in out if x[0] == 3]
        if t == "surface":                     # planar sheet body from a sketch (shell mesh)
            frame = plane_frame(f.get("plane", "XY"))
            if f.get("offset"):
                o, ex, ey, ez = frame
                frame = (o + ez * float(f["offset"]), ex, ey, ez)
            return [(2, _Builder(g, frame).face(sort_loops(f["loops"])))]
        if t == "line":                        # polyline wire body (beam mesh)
            pts = [occ.addPoint(*map(float, p)) for p in f["points"]]
            if f.get("closed"):
                pts.append(pts[0])
            return [(1, occ.addLine(a, b)) for a, b in zip(pts[:-1], pts[1:])]
        if t == "boolean":
            op = f.get("op", "fuse")
            obj = self._ids_to_tags(bodies, f["target"])
            tool = self._ids_to_tags(bodies, f["tool"]) if not f.get("keep_tool") else \
                [x for i in f["tool"] for x in bodies[i]]
            fn = {"fuse": occ.fuse, "cut": occ.cut, "intersect": occ.intersect}[op]
            out, _ = fn(obj, tool, removeTool=not f.get("keep_tool", False))
            return [x for x in out if x[0] == 3]
        if t in ("fillet", "chamfer"):
            vols = self._ids_to_tags(bodies, f["target"])
            occ.synchronize()
            edges = self._edges_for(g, vols, f.get("edges", "all"))
            if not edges:
                raise GeometryError("no edges selected")
            if t == "fillet":
                out = occ.fillet([v[1] for v in vols], edges, [num("r", 1.0)], removeVolume=True)
            else:
                surfs = []
                for e in edges:
                    up, _ = g.model.getAdjacencies(1, e)
                    surfs.append(up[0])
                out = occ.chamfer([v[1] for v in vols], edges, surfs, [num("d", 1.0)], removeVolume=True)
            return [x for x in out if x[0] == 3]
        if t == "transform":
            vols = list(bodies.get(f["target"][0] if isinstance(f["target"], list) else f["target"], []))
            for i in (f["target"] if isinstance(f["target"], list) else [f["target"]])[1:]:
                vols += bodies.get(i, [])
            if f.get("copy"):
                vols = occ.copy(vols)
            else:
                for i in (f["target"] if isinstance(f["target"], list) else [f["target"]]):
                    bodies.pop(i, None)
            op = f.get("op", "translate")
            if op == "translate":
                occ.translate(vols, *f.get("vector", (0, 0, 0)))
            elif op == "rotate":
                occ.rotate(vols, *f.get("point", (0, 0, 0)), *f.get("axis", (0, 0, 1)),
                           math.radians(float(f.get("angle", 90))))
            elif op == "mirror":
                n = np.asarray(f.get("normal", (1, 0, 0)), float)
                p = np.asarray(f.get("point", (0, 0, 0)), float)
                occ.mirror(vols, *n, -float(n @ p))
            elif op == "scale":
                s = float(f.get("factor", 1.0))
                occ.dilate(vols, *f.get("point", (0, 0, 0)), s, s, s)
            return vols
        if t == "pattern":
            src = [x for i in (f["target"] if isinstance(f["target"], list) else [f["target"]])
                   for x in bodies.get(i, [])]
            out = []
            n = int(f.get("count", 2))
            for k in range(1, n):
                cp = occ.copy(src)
                if f.get("kind", "linear") == "linear":
                    d = np.asarray(f.get("direction", (1, 0, 0)), float)
                    d = d / np.linalg.norm(d) * float(f.get("spacing", 1.0)) * k
                    occ.translate(cp, *d)
                else:
                    ang = math.radians(float(f.get("angle", 360.0))) * k / (n if float(f.get("angle", 360.0)) >= 360 else n - 1)
                    occ.rotate(cp, *f.get("point", (0, 0, 0)), *f.get("axis", (0, 0, 1)), ang)
                out += cp
            if f.get("fuse"):
                res, _ = occ.fuse(src, out)
                for i in (f["target"] if isinstance(f["target"], list) else [f["target"]]):
                    bodies.pop(i, None)
                return [x for x in res if x[0] == 3]
            return out
        if t == "import":
            return self._import(g, f)
        raise GeometryError(f"unknown feature type {t}")

    def _edges_for(self, g, vols, spec):
        edges = set()
        for v in vols:
            for _, s in g.model.getBoundary([v], oriented=False, recursive=False):
                for _, e in g.model.getBoundary([(2, s)], oriented=False, recursive=False):
                    edges.add(abs(e))
        edges = sorted(edges)
        if spec == "all":
            return edges
        sigs = spec if isinstance(spec, list) else [spec]
        cand = [(e, entity_signature(g, 1, e)) for e in edges]
        out = []
        for s in sigs:
            if isinstance(s, (int, float)):
                if int(s) in edges:
                    out.append(int(s))
                continue
            best = match_one(s, cand)
            if best is not None:
                out.append(best)
        return out

    def _import(self, g, f):
        path = f["file"]
        if not os.path.exists(path):
            raise GeometryError(f"file not found: {path}")
        ext = os.path.splitext(path)[1].lower()
        occ = g.model.occ
        if ext == ".stl":
            g.merge(path)
            angle = float(f.get("angle", 40.0))
            g.model.mesh.classifySurfaces(math.radians(angle), True, True, math.pi)
            g.model.mesh.createGeometry()
            surfs = [s[1] for s in g.model.getEntities(2)]
            loop = g.model.geo.addSurfaceLoop(surfs)
            vol = g.model.geo.addVolume([loop])
            g.model.geo.synchronize()
            return [(3, vol)]
        heal = f.get("heal", True)
        for opt in ("OCCFixDegenerated", "OCCFixSmallEdges", "OCCFixSmallFaces", "OCCSewFaces", "OCCMakeSolids"):
            g.option.setNumber(f"Geometry.{opt}", 1 if heal else 0)
        out = occ.importShapes(path, highestDimOnly=True)
        vols = [x for x in out if x[0] == 3]
        if not vols:
            surf = [x for x in out if x[0] == 2]
            if surf and f.get("sew", False):     # optionally close sheets into a solid
                sl = occ.addSurfaceLoop([s[1] for s in surf], sewing=True)
                vols = [(3, occ.addVolume([sl]))]
            else:                                # keep as sheet bodies (shell mesh) / wires (beams)
                vols = surf or [x for x in out if x[0] == 1]
                if not vols:
                    raise GeometryError("file contains no solids, surfaces or curves")
        if f.get("scale") and float(f["scale"]) != 1.0:
            s = float(f["scale"])
            occ.dilate(vols, 0, 0, 0, s, s, s)
        if f.get("translate"):
            occ.translate(vols, *f["translate"])
        return vols

    # ---- information for the GUI
    def _collect(self, g):
        info = dict(bodies=[], faces=[], edges=[], vertices=[], units=self.units, messages=self.messages)
        occ_ok = not self.stl_mode
        bbmin, bbmax = np.full(3, np.inf), np.full(3, -np.inf)
        for dim, tag in g.model.getEntities(3):
            vol = g.model.occ.getMass(3, tag) if occ_ok else 0.0
            com = list(g.model.occ.getCenterOfMass(3, tag)) if occ_ok else [0, 0, 0]
            bb = g.model.getBoundingBox(3, tag)
            bbmin, bbmax = np.minimum(bbmin, bb[:3]), np.maximum(bbmax, bb[3:])
            faces = [abs(s) for _, s in g.model.getBoundary([(3, tag)], oriented=False)]
            info["bodies"].append(dict(id=tag, volume=vol, com=com, bbox=list(bb), faces=faces,
                                       sig=dict(com=com, size=vol),
                                       feature=getattr(self, "_owner", {}).get((3, tag))))
        for dim, tag in g.model.getEntities(2):
            area = g.model.occ.getMass(2, tag) if occ_ok else 0.0
            com = list(g.model.occ.getCenterOfMass(2, tag)) if occ_ok else list(np.mean(
                np.reshape(g.model.getBoundingBox(2, tag), (2, 3)), 0))
            typ = g.model.getType(2, tag)
            up, _ = g.model.getAdjacencies(2, tag)
            normal = None
            try:
                pr = g.model.getParametrizationBounds(2, tag)
                uv = [(pr[0][0] + pr[1][0]) / 2, (pr[0][1] + pr[1][1]) / 2]
                normal = list(g.model.getNormal(tag, uv))
            except Exception:  # noqa: BLE001
                pass
            info["faces"].append(dict(id=tag, type=typ, area=area, com=com, bodies=list(up), normal=normal,
                                      sig=dict(com=com, size=area, type=typ)))
        for dim, tag in g.model.getEntities(1):
            length = g.model.occ.getMass(1, tag) if occ_ok else 0.0
            com = list(g.model.occ.getCenterOfMass(1, tag)) if occ_ok else [0, 0, 0]
            up, _ = g.model.getAdjacencies(1, tag)
            info["edges"].append(dict(id=tag, type=g.model.getType(1, tag), length=length, com=com,
                                      faces=[int(x) for x in up], sig=dict(com=com, size=length)))
        for dim, tag in g.model.getEntities(0):
            xyz = g.model.getValue(0, tag, [])
            info["vertices"].append(dict(id=tag, xyz=list(xyz), sig=dict(com=list(xyz), size=0.0)))
        vol_faces = {abs(t) for b in info["bodies"] for t in b["faces"]}
        face_edges = set()
        for fdict in info["faces"]:
            if fdict["id"] not in vol_faces:
                fdict["sheet"] = True
                info.setdefault("sheets", []).append(dict(id=fdict["id"], area=fdict["area"], com=fdict["com"],
                                                          sig=fdict["sig"], feature=getattr(self, "_owner", {}).get((2, fdict["id"]))))
            for _, e in g.model.getBoundary([(2, fdict["id"])], oriented=False):
                face_edges.add(abs(e))
        line_owned = {t for (d, t) in getattr(self, "_owner", {}) if d == 1}
        for e in info["edges"]:
            # seam edges of periodic faces are missing from getBoundary: use upward adjacency too
            if (e["id"] not in face_edges and not e["faces"]) or e["id"] in line_owned:
                e["free"] = True
                info.setdefault("lines", []).append(dict(id=e["id"], length=e["length"], com=e["com"], sig=e["sig"],
                                                         feature=getattr(self, "_owner", {}).get((1, e["id"]))))
        info.setdefault("sheets", [])
        info.setdefault("lines", [])
        for dd, tag in [(2, x["id"]) for x in info["sheets"]] + [(1, x["id"]) for x in info["lines"]]:
            bb = g.model.getBoundingBox(dd, tag)
            bbmin, bbmax = np.minimum(bbmin, bb[:3]), np.maximum(bbmax, bb[3:])
        if np.all(np.isfinite(bbmin)):
            info["bbox"] = list(bbmin) + list(bbmax)
            info["diag"] = float(np.linalg.norm(bbmax - bbmin))
        else:
            info["bbox"], info["diag"] = [0, 0, 0, 1, 1, 1], 1.0
        return info

    def tessellate(self, resolution=40):
        """Display triangulation per face and polyline per edge (coordinates float32).
        Must be called right after regenerate() (same gmsh model)."""
        with GMSH_LOCK:
            self.ensure_loaded()
            g = gmsh_module()
            diag = self.info.get("diag", 1.0) if self.info else 1.0
            if not self.stl_mode:
                g.option.setNumber("Mesh.MeshSizeMax", diag / resolution)
                g.option.setNumber("Mesh.MeshSizeMin", diag / (resolution * 20))
                g.option.setNumber("Mesh.MeshSizeFromCurvature", 16)
                g.option.setNumber("Mesh.Algorithm", 6)
                g.option.setNumber("Mesh.ElementOrder", 1)
                g.model.mesh.clear()
                try:
                    g.model.mesh.generate(2)
                except Exception as e:  # noqa: BLE001
                    self.messages.append(f"display tessellation incomplete: {e}")
            faces, edges = [], []
            for _, tag in g.model.getEntities(2):
                try:
                    nt, nc, _ = g.model.mesh.getNodes(2, tag, includeBoundary=True)
                    et, _, en = g.model.mesh.getElements(2, tag)
                except Exception:  # noqa: BLE001
                    continue
                if not len(nt):
                    continue
                loc = {int(n): i for i, n in enumerate(nt)}
                tris = []
                for t_, conn in zip(et, en):
                    if t_ == 2:
                        tris.append(np.asarray(conn, np.int64).reshape(-1, 3))
                if not tris:
                    continue
                tris = np.concatenate(tris)
                idx = np.vectorize(loc.get)(tris)
                faces.append(dict(id=tag, xyz=np.asarray(nc, np.float32).reshape(-1, 3), tri=idx.astype(np.int32)))
            for _, tag in g.model.getEntities(1):
                try:
                    nt, nc, par = g.model.mesh.getNodes(1, tag, includeBoundary=True)
                except Exception:  # noqa: BLE001
                    continue
                if len(nt) < 2:
                    continue
                xyz = np.asarray(nc).reshape(-1, 3)
                if len(par) == len(nt):
                    xyz = xyz[np.argsort(par)]
                edges.append(dict(id=tag, xyz=xyz.astype(np.float32)))
            if not self.stl_mode:
                g.model.mesh.clear()
            return faces, edges

    def export(self, path):
        with GMSH_LOCK:
            self.ensure_loaded()
            g = gmsh_module()
            g.write(path)


# ------------------------------------------------------------------ signatures / matching
def entity_signature(g, dim, tag):
    if dim == 0:
        return dict(com=list(g.model.getValue(0, tag, [])), size=0.0)
    return dict(com=list(g.model.occ.getCenterOfMass(dim, tag)), size=g.model.occ.getMass(dim, tag))


def _sig_dist(a, b, scale):
    d = np.linalg.norm(np.asarray(a["com"]) - np.asarray(b["com"])) / max(scale, 1e-30)
    sa, sb = abs(a.get("size", 0)), abs(b.get("size", 0))
    if sa > 0 and sb > 0:
        d += abs(math.log(sa / sb))
    if a.get("type") and b.get("type") and a["type"] != b["type"]:
        d += 1.0
    return d


def match_one(sig, candidates, scale=None, tol=0.25):
    if not candidates:
        return None
    if scale is None:
        coms = np.array([c[1]["com"] for c in candidates])
        scale = float(np.linalg.norm(coms.max(0) - coms.min(0))) or 1.0
    d = [_sig_dist(sig, c[1], scale) for c in candidates]
    k = int(np.argmin(d))
    return candidates[k][0] if d[k] < tol else None


def match(info, dim, refs):
    """Resolve references [{id, sig}] to current entity ids: keep the id if its signature still
    matches, otherwise pick the closest entity by signature (topological naming repair)."""
    key = {3: "bodies", 2: "faces", 1: "edges", 0: "vertices"}[dim]
    ents = info[key]
    cands = [(e["id"], e["sig"]) for e in ents]
    by_id = {e["id"]: e for e in ents}
    scale = info.get("diag", 1.0)
    out, repaired = [], 0
    for r in refs:
        rid, sig = (r.get("id"), r.get("sig")) if isinstance(r, dict) else (r, None)
        if sig is None:
            if rid in by_id:
                out.append(rid)
            continue
        if rid in by_id and _sig_dist(sig, by_id[rid]["sig"], scale) < 0.05:
            out.append(rid)
            continue
        m = match_one(sig, cands, scale)
        if m is not None:
            out.append(m)
            repaired += 1
    return out, repaired
