"""gfem GUI back-end: a dependency-free HTTP server (stdlib) serving the browser front-end and a
JSON API. Long operations (meshing, solving) run in a worker thread; progress, residuals and
log messages are streamed to the browser with Server-Sent Events.

Because the GUI is a web page, the solver can run on a remote GPU workstation or cloud node
while the engineer works from a laptop:   python -m gfem gui --host 0.0.0.0 --port 8765
"""
import base64
import copy
import json
import mimetypes
import os
import queue
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .. import project as P
from ..cad.geometry import CadModel, GeometryError, GMSH_LOCK
from ..elements import ELEMENTS, FACETS
from ..structural import STRUCT_ELEMENTS

STATIC = os.path.join(os.path.dirname(__file__), "static")


def b64(a, dtype=np.float32):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


class Stop(Exception):
    pass


class GuiMonitor:
    """Solver monitor that pushes events to the session queue."""

    def __init__(self, session):
        self.s = session
        self.cum_iter = 0
        self.verbose = False

    def log(self, msg):
        self.s.emit("log", msg=str(msg))

    def iteration(self, **kw):
        self.cum_iter += 1
        r = kw.get("residual")
        self.s.emit("iter", n=self.cum_iter, inc=kw.get("inc"), it=kw.get("it"),
                    residual=float(r) if r is not None else None, t=kw.get("t"))

    def increment(self, **kw):
        self.s.emit("inc", **{k: (float(v) if isinstance(v, (np.floating, float)) else v)
                              for k, v in kw.items() if isinstance(v, (int, float, str, bool, np.floating))})

    def frame(self, t, u, states, **kw):
        pass

    def frame_ready(self, fr, i):
        self.s.emit("frame", i=i, time=float(fr.get("time", i)), step=fr.get("step"))

    def should_stop(self):
        return self.s.stop_flag.is_set()


class Session:
    def __init__(self, workdir):
        self.workdir = os.path.abspath(workdir)
        os.makedirs(os.path.join(self.workdir, "uploads"), exist_ok=True)
        os.makedirs(os.path.join(self.workdir, "results"), exist_ok=True)
        self.project = P.new_project()
        self.cad = None
        self.info = None
        self.tess = None
        self.mesh_spec = None
        self.mesh_stats = None
        self.mesh_key = None
        self.display = None
        self.model = None
        self.results = []
        self.frames = []
        self.result_meta = {}
        self.events = []
        self.cv = threading.Condition()
        self.lock = threading.RLock()
        self.job = None
        self.job_name = None
        self.stop_flag = threading.Event()
        self.status = "idle"
        self.result_project = None

    # ---------------------------------------------------------- events
    def emit(self, kind, **data):
        with self.cv:
            data["type"] = kind
            data["seq"] = len(self.events)
            data["ts"] = time.time()
            self.events.append(data)
            if len(self.events) > 20000:
                self.events = self.events[-10000:]
                for i, e in enumerate(self.events):
                    e["seq"] = i
            self.cv.notify_all()

    def log(self, msg):
        self.emit("log", msg=msg)

    # ---------------------------------------------------------- geometry
    def regenerate(self):
        g = self.project["geometry"]
        self.cad = CadModel(g.get("features", []), units=self.project.get("units", "mm"),
                            share_topology=g.get("share_topology", True))
        if not g.get("features"):
            self.info = None
            self.tess = None
            return None
        with GMSH_LOCK:
            self.info = self.cad.regenerate()
            faces, edges = self.cad.tessellate()
        self.tess = (faces, edges)
        self._sync_bodies()
        self.invalidate_mesh()
        return self.info

    def _sync_bodies(self):
        """Keep material assignments attached to bodies across regenerations (signatures)."""
        from ..cad.geometry import match
        old = self.project.get("bodies", {})
        new = {}
        refs = [(k, v) for k, v in old.items() if v.get("sig") and k.isdigit()]
        cand_ids = [b["id"] for b in self.info["bodies"]]
        for k, v in refs:
            ids, _ = match(self.info, 3, [dict(id=int(k), sig=v["sig"])])
            if ids and str(ids[0]) not in new:
                nv = dict(v)
                nv["sig"] = next(b["sig"] for b in self.info["bodies"] if b["id"] == ids[0])
                new[str(ids[0])] = nv
        mat0 = self.project["materials"][0]["id"] if self.project["materials"] else None
        for i, b in enumerate(self.info["bodies"]):
            if str(b["id"]) not in new:
                new[str(b["id"])] = dict(name=f"Body {b['id']}", material=mat0, sig=b["sig"])
        # sheet (shell) and line (beam) bodies keep their section data by signature
        diag = self.info.get("diag", 1.0)
        for pre, key, dim, defaults in (("S", "sheets", 2, dict(thickness=max(diag / 100, 1e-3))),
                                        ("L", "lines", 1, dict(profile="RECT", dims=[max(diag / 50, 1e-3)] * 2))):
            olds = {k: v for k, v in old.items() if k.startswith(pre) and v.get("sig")}
            for b in self.info.get(key, []):
                k = pre + str(b["id"])
                hit = None
                for ok, ov in olds.items():
                    ids, _ = match(self.info, dim, [dict(id=int(ok[1:]), sig=ov["sig"])])
                    if ids and ids[0] == b["id"]:
                        hit = ov
                        break
                nv = dict(hit) if hit else dict(name=f"{'Surface' if pre == 'S' else 'Line'} body {b['id']}",
                                                material=mat0, **defaults)
                nv["sig"] = b["sig"]
                new[k] = nv
        self.project["bodies"] = new

    def geometry_payload(self):
        if not self.tess:
            return dict(info=None, faces=[], edges=[])
        faces, edges = self.tess
        return dict(info=self.info,
                    faces=[dict(id=f["id"], xyz=b64(f["xyz"]), tri=b64(f["tri"], np.int32)) for f in faces],
                    edges=[dict(id=e["id"], xyz=b64(e["xyz"])) for e in edges])

    # ---------------------------------------------------------- mesh
    def invalidate_mesh(self):
        self.mesh_spec = None
        self.mesh_stats = None
        self.display = None
        self.mesh_key = None

    def _mesh_key(self):
        p = self.project
        return json.dumps([p["geometry"], p["mesh"], p.get("mesh_source")], sort_keys=True, default=str)

    def ensure_mesh(self):
        key = self._mesh_key()
        if self.mesh_spec is not None and self.mesh_key == key:
            return
        p = self.project
        if p.get("mesh_source"):
            self._load_deck(p["mesh_source"]["file"])
            self.mesh_key = key
            return
        if self.cad is None or self.info is None:
            self.regenerate()
        if self.info is None:
            raise P.ProjectError("no geometry")
        from ..cad.mesher import mesh_cad
        m = p["mesh"]
        local = []
        for loc in m.get("local", []):
            if loc.get("suppressed"):
                continue
            dim, ids = P.resolve_scope(p, loc.get("scope"), self.info)
            if ids:
                local.append((dim, ids, float(loc.get("size", 1.0))))
        t0 = time.time()
        self.log(f"meshing: size={m.get('size') or 'auto'}, order={m.get('order', 2)}, method={m.get('kind', 'tet')}")
        with GMSH_LOCK:
            # the gmsh model must be the current CAD: regenerate if another model was loaded
            self.cad.regenerate()
            sp, stats = mesh_cad(self.cad, size=m.get("size") or None, order=int(m.get("order", 2)),
                                 kind=m.get("kind", "tet"), curvature=float(m.get("curvature", 8)),
                                 local=local)
            faces, edges = self.cad.tessellate()
            self.tess = (faces, edges)
        stats["time"] = time.time() - t0
        self.mesh_spec, self.mesh_stats = sp, stats
        self.mesh_key = key
        self.display = self._display_mesh(sp)
        self.log(f"mesh: {stats['nodes']} nodes, {stats['elements']} elements {stats.get('types')} "
                 f"(min quality {stats.get('q_min', 0):.3f}) in {stats['time']:.2f}s")

    def _load_deck(self, path):
        from ..io import read_any
        from ..surfaces import Surface
        sp = read_any(path)
        self.deck_spec = sp
        # pseudo CAD faces by feature angle on the skin -> FACE_k selectable sets
        X = sp.nodes
        types, conns = sp.element_table()
        pairs = []
        keys = {}
        for e, (t, c) in enumerate(zip(types, conns)):
            et = ELEMENTS.get(t)
            if et is None:
                continue
            for k, loc in enumerate(et.faces):
                nc = 3 if len(loc) in (3, 6) else 4
                key = tuple(sorted(c[i] for i in loc[:nc]))
                keys.setdefault(key, []).append((e, k))
        bnd = [v[0] for v in keys.values() if len(v) == 1]
        # facet normals
        fac_nodes = []
        for e, k in bnd:
            et = ELEMENTS[types[e]]
            fac_nodes.append([conns[e][i] for i in et.faces[k][:3]])
        fac_nodes = np.array(fac_nodes)
        cen = np.array([X[np.asarray(conns[e])[:ELEMENTS[types[e]].ncorner]].mean(0) for e, _ in bnd])
        n = np.cross(X[fac_nodes[:, 1]] - X[fac_nodes[:, 0]], X[fac_nodes[:, 2]] - X[fac_nodes[:, 0]])
        fc = X[fac_nodes].mean(1)
        n *= np.sign(np.einsum("ij,ij->i", n, fc - cen))[:, None]
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
        # adjacency through shared corner edges
        edge_map = {}
        for i, (e, k) in enumerate(bnd):
            et = ELEMENTS[types[e]]
            loc = et.faces[k]
            nc = 3 if len(loc) in (3, 6) else 4
            cs = [conns[e][j] for j in loc[:nc]]
            for a in range(nc):
                ed = tuple(sorted((cs[a], cs[(a + 1) % nc])))
                edge_map.setdefault(ed, []).append(i)
        nbr = [[] for _ in bnd]
        for ed, fs in edge_map.items():
            if len(fs) == 2:
                nbr[fs[0]].append(fs[1])
                nbr[fs[1]].append(fs[0])
        label = -np.ones(len(bnd), int)
        cos_t = np.cos(np.radians(35))
        cur = 0
        for s0 in range(len(bnd)):
            if label[s0] >= 0:
                continue
            cur += 1
            stack = [s0]
            label[s0] = cur
            while stack:
                a = stack.pop()
                for b in nbr[a]:
                    if label[b] < 0 and n[a] @ n[b] > cos_t:
                        label[b] = cur
                        stack.append(b)
        for f in range(1, cur + 1):
            sel = np.nonzero(label == f)[0]
            sp.surfaces[f"FACE_{f}"] = dict(elems=np.array([bnd[i][0] for i in sel]),
                                            faces=np.array([bnd[i][1] for i in sel]))
            nodes = set()
            for i in sel:
                e, k = bnd[i]
                nodes.update(conns[e][j] for j in ELEMENTS[types[e]].faces[k])
            sp.nsets[f"FACE_{f}"] = np.array(sorted(nodes))
        # bodies = elsets of sections
        bodies = []
        for i, sec in enumerate(sp.sections):
            sp.elsets[f"BODY_{i + 1}"] = sp.elsets[sec["elset"]]
            bodies.append(dict(id=i + 1, name=sec["elset"], material=sec["material"]))
        self.mesh_spec = sp
        self.mesh_stats = dict(nodes=len(X), elements=sp.nelem, types={}, source=os.path.basename(path))
        from collections import Counter
        cnt = Counter()
        for g in sp.groups:
            cnt[g.etype] += len(g.conn)
        self.mesh_stats["types"] = dict(cnt)
        diag = float(np.linalg.norm(X.max(0) - X.min(0)))
        self.info = dict(bodies=[dict(id=b["id"], volume=0, com=list(X.mean(0)), sig=None, name=b["name"])
                                 for b in bodies],
                         faces=[dict(id=f, type="mesh", area=0, com=list(fc[label == f].mean(0)),
                                     normal=list(n[label == f].mean(0)), sig=None, bodies=[])
                                for f in range(1, cur + 1)],
                         edges=[], vertices=[], bbox=list(X.min(0)) + list(X.max(0)), diag=diag,
                         units=self.project.get("units"), mesh_mode=True,
                         sets=dict(nsets=sorted(sp.nsets), elsets=sorted(sp.elsets), surfaces=sorted(sp.surfaces)))
        self.display = self._display_mesh(sp)
        # fake tessellation from the display skin for selection in geometry mode
        d = self.display
        faces = []
        for f in range(1, cur + 1):
            sel = d["face_of_tri"] == f
            tri = d["tri"][sel]
            u, inv = np.unique(tri, return_inverse=True)
            faces.append(dict(id=f, xyz=d["xyz"][u].astype(np.float32), tri=inv.reshape(-1, 3).astype(np.int32)))
        self.tess = (faces, [])
        self.log(f"imported deck {os.path.basename(path)}: {sp.summary()}; {cur} faces detected by feature angle")
        for w in sp.warnings:
            self.log(f"  warning: {w}")

    def _display_mesh(self, sp):
        """Skin triangles (linear) with CAD face id per triangle + element edges."""
        types, conns = sp.element_table()
        X = sp.nodes
        tris, fid, edges = [], [], set()
        internal = set()
        if self.info and not self.info.get("mesh_mode"):
            for f in self.info["faces"]:
                if len(f.get("bodies", [])) > 1:
                    internal.add(f["id"])
        for name, s in sp.surfaces.items():
            if not name.startswith("FACE_") or "elems" not in s:
                continue
            f = int(name.split("_")[1])
            if f in internal:
                continue
            for e, k in zip(s["elems"], s["faces"]):
                if types[e] in STRUCT_ELEMENTS:                 # shell element: the element itself
                    loc = list(range(len(conns[e])))
                else:
                    loc = ELEMENTS[types[e]].faces[k]
                ft = FACETS[len(loc)]
                nodes = np.asarray([conns[e][i] for i in loc])[None]
                t = ft.triangles(nodes)
                tris.append(t)
                fid += [f] * len(t)
                ring = list(nodes[0, :ft.ncorner])
                if ft.nnode > ft.ncorner:
                    mids = list(nodes[0, ft.ncorner:])
                    ring2 = []
                    for a in range(ft.ncorner):
                        ring2 += [ring[a], mids[a]]
                    ring = ring2
                for a in range(len(ring)):
                    edges.add(tuple(sorted((ring[a], ring[(a + 1) % len(ring)]))))
        segs, seg_edge = [], []
        for name, es in sp.elsets.items():
            if name.startswith("LINE_"):
                eid = int(name.split("_")[1])
                for e in es:
                    segs.append(conns[e][:2])
                    seg_edge.append(eid)
        if not tris and not segs:
            return None
        tri = np.concatenate(tris) if tris else np.zeros((0, 3), np.int64)
        segs = np.asarray(segs, np.int64).reshape(-1, 2)
        used = np.unique(np.concatenate([tri.ravel(), segs.ravel()]))
        lut = {int(n): i for i, n in enumerate(used)}
        remap = np.vectorize(lambda n: lut[int(n)], otypes=[np.int64])
        ed = np.array([[lut[a], lut[b]] for a, b in edges if a in lut and b in lut], np.int32).reshape(-1, 2)
        return dict(nodes=used, xyz=X[used], tri=(remap(tri) if len(tri) else tri).reshape(-1, 3).astype(np.int32),
                    face_of_tri=np.asarray(fid, np.int32), edges=ed,
                    lines=(remap(segs) if len(segs) else segs).astype(np.int32).reshape(-1, 2),
                    line_edge=np.asarray(seg_edge, np.int32))

    def mesh_payload(self):
        d = self.display
        if d is None:
            return dict(stats=self.mesh_stats)
        return dict(stats=self.mesh_stats, xyz=b64(d["xyz"]), tri=b64(d["tri"], np.int32),
                    face_of_tri=b64(d["face_of_tri"], np.int32), edges=b64(d["edges"], np.int32),
                    lines=b64(d.get("lines", np.zeros((0, 2))), np.int32),
                    line_edge=b64(d.get("line_edge", np.zeros(0)), np.int32))

    # ---------------------------------------------------------- solve
    def start(self, fn, name):
        if self.job is not None and self.job.is_alive():
            raise P.ProjectError("a job is already running")
        self.stop_flag.clear()

        def run():
            self.status = "running"
            self.job_name = name
            self.emit("status", status="running", job=name)
            try:
                fn()
                self.status = "stopped" if self.stop_flag.is_set() else "done"
            except Exception as e:  # noqa: BLE001
                self.status = "failed"
                self.log(f"ERROR: {e}")
                self.log(traceback.format_exc())
            self.emit("status", status=self.status, job=name)
        self.job = threading.Thread(target=run, daemon=True)
        self.job.start()

    def solve(self):
        from ..analysis import run_spec
        p = copy.deepcopy(self.project)
        self.ensure_mesh()
        self.emit("mesh", stats=self.mesh_stats)
        sp = P.build_spec(p, self.mesh_spec, self.info, log=self.log)
        for w in sp.warnings:
            self.log(f"warning: {w}")
        mon = GuiMonitor(self)
        out = os.path.join(self.workdir, "results")
        opts = dict(linear_solver=p["analysis"].get("linear_solver", "auto"),
                    tangent=p["analysis"].get("tangent", "ebe"))
        self.results, self.frames = [], []
        self.emit("solve_start", analysis=p["analysis"]["type"])
        model, results = run_spec(sp, out, "gfem", monitor=mon, write=bool(p["analysis"].get("write_vtu", False)),
                                  options=opts)
        self.model = model
        self.results = results
        frames = []
        for r in results:
            for fr in r["frames"]:
                fr["procedure"] = r["procedure"]
                frames.append(fr)
        self.frames = frames
        self.result_project = p
        self.solved_spec = sp
        self.result_meta = dict(analysis=p["analysis"]["type"], nframes=len(frames),
                                info={r["step"]: _jsonable(r["info"]) for r in results})
        self.emit("results", nframes=len(frames))

    def frames_payload(self):
        out = []
        for i, f in enumerate(self.frames):
            d = dict(i=i, time=f.get("time"), step=f.get("step"), procedure=f.get("procedure"))
            for k in ("freq", "mode", "eigenvalue", "lpf", "compliance", "label"):
                if k in f:
                    d[k] = f[k]
            if "U" in f:
                d["umax"] = float(np.linalg.norm(f["U"], axis=1).max())
            if "MISES" in f:
                d["smax"] = float(f["MISES"].max())
            if "T" in f:
                d["tmin"], d["tmax"] = float(f["T"].min()), float(f["T"].max())
            out.append(d)
        return dict(frames=out, meta=self.result_meta)

    def result_payload(self, i, rtype):
        if not self.frames:
            raise P.ProjectError("no results")
        i = max(0, min(int(i), len(self.frames) - 1))
        fr = self.frames[i]
        d = self.display
        if d is None:
            raise P.ProjectError("no mesh display")
        val = P.result_field(fr, rtype, self.result_project)
        res = dict(i=i, time=fr.get("time"), type=rtype)
        nodes = d["nodes"]
        if val is not None:
            v = np.asarray(val)[nodes]
            res.update(values=b64(v), min=float(v.min()), max=float(v.max()),
                       imin=int(np.argmin(v)), imax=int(np.argmax(v)))
        else:
            res["values"] = None
        if "U" in fr:
            U = fr["U"][nodes]
            res["U"] = b64(U)
            res["umax"] = float(np.linalg.norm(fr["U"], axis=1).max())
        for k in ("freq", "mode", "eigenvalue", "lpf", "compliance", "step", "procedure"):
            if k in fr:
                res[k] = fr[k] if not isinstance(fr[k], np.generic) else float(fr[k])
        if rtype == "density" and "density" in fr:
            res["density_elem_min"] = float(np.min(fr["density"]))
        return res


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items() if not k.startswith("_") and k not in ("state", "modes", "u", "base_u")}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size <= 2000 else None
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return None


# ================================================================== HTTP
class Handler(BaseHTTPRequestHandler):
    session: Session = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json", headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, default=_default).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b""
        return json.loads(raw or b"{}")

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        s = self.session
        try:
            if u.path in ("/", "/index.html"):
                return self._file(os.path.join(STATIC, "index.html"))
            if u.path.startswith("/static/"):
                fp = os.path.normpath(os.path.join(STATIC, u.path[len("/static/"):]))
                if not fp.startswith(STATIC):
                    return self._send(403, {"error": "forbidden"})
                return self._file(fp)
            if u.path == "/api/state":
                return self._send(200, self._state())
            if u.path == "/api/geometry":
                return self._send(200, s.geometry_payload())
            if u.path == "/api/mesh":
                return self._send(200, s.mesh_payload())
            if u.path == "/api/events":
                return self._events(int(q.get("since", 0)))
            if u.path == "/api/frames":
                return self._send(200, s.frames_payload())
            if u.path == "/api/result":
                return self._send(200, s.result_payload(q.get("frame", -1), q.get("type", "total_deformation")))
            if u.path == "/api/export":
                return self._export(q.get("fmt", "inp"))
            if u.path == "/api/examples":
                from .examples import EXAMPLES
                return self._send(200, [dict(key=k, title=v[0], text=v[1]) for k, v in EXAMPLES.items()])
            if u.path == "/api/system":
                return self._send(200, _system())
            return self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._send(400, {"error": str(e)})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        s = self.session
        try:
            if u.path == "/api/upload":
                name = os.path.basename(q.get("name", "upload.bin"))
                n = int(self.headers.get("Content-Length", 0))
                data = self.rfile.read(n)
                path = os.path.join(s.workdir, "uploads", name)
                with open(path, "wb") as f:
                    f.write(data)
                return self._send(200, dict(path=path, size=len(data)))
            body = self._json()
            if u.path == "/api/project":
                with s.lock:
                    s.project = body["project"]
                return self._send(200, dict(ok=True))
            if u.path == "/api/regenerate":
                with s.lock:
                    if "project" in body:
                        s.project = body["project"]
                    s.project["mesh_source"] = None if s.project["geometry"]["features"] else s.project.get("mesh_source")
                    try:
                        s.regenerate()
                    except GeometryError as e:
                        return self._send(200, dict(error=str(e), project=s.project))
                    return self._send(200, dict(project=s.project, **s.geometry_payload()))
            if u.path == "/api/import":
                return self._send(200, self._import(body))
            if u.path == "/api/mesh":
                with s.lock:
                    if "project" in body:
                        s.project = body["project"]
                    s.ensure_mesh()
                return self._send(200, s.mesh_payload())
            if u.path == "/api/solve":
                with s.lock:
                    if "project" in body:
                        s.project = body["project"]
                s.start(s.solve, "solve")
                return self._send(200, dict(ok=True))
            if u.path == "/api/stop":
                s.stop_flag.set()
                return self._send(200, dict(ok=True))
            if u.path == "/api/new":
                with s.lock:
                    s.project = P.new_project(body.get("units", "mm"))
                    s.cad, s.info, s.tess = None, None, None
                    s.invalidate_mesh()
                    s.frames, s.results = [], []
                return self._send(200, self._state())
            if u.path == "/api/open":
                with s.lock:
                    s.project = body["project"] if "project" in body else P.load(body["path"])
                    s.frames, s.results = [], []
                    s.invalidate_mesh()
                    if s.project.get("mesh_source"):
                        s.ensure_mesh()
                    else:
                        s.regenerate()
                return self._send(200, dict(state=self._state(), **s.geometry_payload()))
            if u.path == "/api/example":
                from .examples import build_example
                with s.lock:
                    s.project = build_example(body["key"], s.workdir)
                    s.frames, s.results = [], []
                    s.invalidate_mesh()
                    s.regenerate()
                return self._send(200, dict(state=self._state(), **s.geometry_payload()))
            if u.path == "/api/save":
                path = body.get("path") or os.path.join(s.workdir, (s.project.get("name") or "project") + ".gfem.json")
                P.save(s.project, path)
                return self._send(200, dict(path=path))
            if u.path == "/api/detect_contacts":
                return self._send(200, dict(pairs=_detect_contacts(s, float(body.get("tol", 0)))))
            if u.path == "/api/dxf":
                from ..cad.geometry import dxf_loops
                loops, sc = dxf_loops(body["path"])
                return self._send(200, dict(loops=loops, scale=sc))
            return self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._send(400, {"error": str(e)})

    # ---------------------------------------------------------- helpers
    def _state(self):
        s = self.session
        return dict(project=s.project, info=s.info, mesh_stats=s.mesh_stats, status=s.status,
                    has_results=bool(s.frames), nframes=len(s.frames), seq=len(s.events),
                    library=[P.si_to_units(m, s.project.get("units", "mm")) for m in P.LIBRARY],
                    units=P.UNITS[s.project.get("units", "mm")], analyses=P.ANALYSES,
                    conditions=P.CONDITIONS, result_types=P.RESULTS, system=_system(),
                    workdir=s.workdir)

    def _import(self, body):
        s = self.session
        path = body["path"]
        ext = os.path.splitext(path)[1].lower()
        with s.lock:
            if ext in (".step", ".stp", ".iges", ".igs", ".brep", ".stl", ".dxf"):
                feats = s.project["geometry"]["features"]
                fid = f"F{max([int(f['id'][1:]) for f in feats if f['id'][1:].isdigit()] + [0]) + 1}"
                if ext == ".dxf":
                    feats.append(dict(id=fid, type="dxf_extrude", name=f"DXF {os.path.basename(path)}",
                                      file=path, distance=float(body.get("distance", 10.0)), plane="XY"))
                else:
                    feats.append(dict(id=fid, type="import", name=f"Import {os.path.basename(path)}",
                                      file=path, heal=True))
                s.project["mesh_source"] = None
                try:
                    s.regenerate()
                except GeometryError as e:
                    feats.pop()
                    return dict(error=str(e), project=s.project)
                return dict(project=s.project, **s.geometry_payload())
            # solver deck / mesh
            s.project["geometry"]["features"] = []
            s.project["mesh_source"] = dict(file=path)
            s.invalidate_mesh()
            s.ensure_mesh()
            sp = s.mesh_spec
            mats = []
            for k, m in sp.materials.items():
                mm = dict(id=k, name=k, model="linear", E=m.get("E", 200e3), nu=m.get("nu", 0.3),
                          density=m.get("density", 7.85e-9), alpha=m.get("alpha", 0.0), k=m.get("k", 50.0),
                          cp=m.get("cp", 4.3e8), color="#8aa9c9")
                if "plastic" in m:
                    mm["model"], mm["table"] = "plastic", m["plastic"]
                mats.append(mm)
            if mats:
                s.project["materials"] = mats
            s.project["bodies"] = {str(b["id"]): dict(name=b.get("name"), material=next(
                (sec["material"] for sec in sp.sections if sec["elset"] == b.get("name")), mats[0]["id"] if mats else "M1"))
                for b in s.info["bodies"]}
            # deck steps -> project conditions are kept in the deck; offer a direct run
            s.project["deck_has_steps"] = bool(sp.steps)
            s.project["conditions"] = []
            s.project["connections"] = []
            s.project["mesh"]["local"] = []
            if sp.steps:
                s.project["analysis"]["type"] = "deck"
            return dict(project=s.project, **s.geometry_payload(), mesh=s.mesh_payload())

    def _export(self, fmt):
        s = self.session
        from ..io import write_any
        out = os.path.join(s.workdir, "results")
        name = (s.project.get("name") or "model").replace(" ", "_")
        if fmt == "json":
            path = os.path.join(out, name + ".gfem.json")
            P.save(s.project, path)
        elif fmt in ("step", "stp", "brep", "stl"):
            with GMSH_LOCK:
                s.cad.regenerate()
                if fmt == "stl":
                    s.cad.tessellate()
                path = os.path.join(out, f"{name}.{fmt}")
                s.cad.export(path)
        elif fmt in ("inp", "bdf"):
            s.ensure_mesh()
            sp = P.build_spec(s.project, s.mesh_spec, s.info)
            path = os.path.join(out, f"{name}.{fmt}")
            write_any(sp, path)
        elif fmt == "vtu":
            if not s.frames:
                raise P.ProjectError("no results")
            from ..io.vtk import write_vtu
            path = os.path.join(out, f"{name}_result.vtu")
            write_vtu(path, s.model, s.frames[-1])
        elif fmt == "csv":
            if not s.frames:
                raise P.ProjectError("no results")
            fr = s.frames[-1]
            path = os.path.join(out, f"{name}_nodal.csv")
            cols = [("x", s.model.X[:, 0]), ("y", s.model.X[:, 1]), ("z", s.model.X[:, 2])]
            if "U" in fr:
                cols += [("ux", fr["U"][:, 0]), ("uy", fr["U"][:, 1]), ("uz", fr["U"][:, 2])]
            if "MISES" in fr:
                cols.append(("mises", fr["MISES"]))
            if "T" in fr:
                cols.append(("temp", fr["T"]))
            arr = np.stack([np.asarray(c) for _, c in cols], 1)
            np.savetxt(path, arr, delimiter=",", header=",".join(c for c, _ in cols), comments="")
        else:
            raise P.ProjectError(f"unknown export format {fmt}")
        with open(path, "rb") as f:
            data = f.read()
        self._send(200, data, "application/octet-stream",
                   {"Content-Disposition": f'attachment; filename="{os.path.basename(path)}"'})

    def _file(self, path):
        if not os.path.isfile(path):
            return self._send(404, {"error": "not found"})
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if path.endswith(".js"):
            ctype = "text/javascript"
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def _events(self, since):
        s = self.session
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        i = since
        try:
            while True:
                with s.cv:
                    if i >= len(s.events):
                        s.cv.wait(timeout=15)
                    batch = s.events[i:]
                    i = len(s.events)
                if batch:
                    payload = "".join(f"data: {json.dumps(e, default=_default)}\n\n" for e in batch)
                else:
                    payload = ": keep-alive\n\n"
                self.wfile.write(payload.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def _system():
    import jax
    try:
        devs = [str(d) for d in jax.devices()]
    except Exception:  # noqa: BLE001
        devs = []
    from .. import __version__
    return dict(backend=jax.default_backend(), devices=devs, version=__version__, jax=jax.__version__)


def _detect_contacts(s, tol):
    """Face pairs of different bodies closer than tol (bonded contact candidates)."""
    from scipy.spatial import cKDTree
    if not s.tess or not s.info:
        return []
    faces, _ = s.tess
    fb = {f["id"]: f.get("bodies", []) for f in s.info["faces"]}
    tol = tol or 1e-3 * s.info.get("diag", 1.0)
    pts, own = [], []
    for f in faces:
        pts.append(f["xyz"])
        own += [f["id"]] * len(f["xyz"])
    if not pts:
        return []
    P_ = np.concatenate(pts).astype(float)
    own = np.array(own)
    tree = cKDTree(P_)
    pairs = tree.query_pairs(tol, output_type="ndarray")
    out = set()
    for a, b in pairs:
        fa, fb_ = own[a], own[b]
        if fa == fb_:
            continue
        ba, bb = fb.get(fa, []), fb.get(fb_, [])
        if ba and bb and set(ba) != set(bb) and len(ba) == 1 and len(bb) == 1:
            out.add(tuple(sorted((int(fa), int(fb_)))))
    return sorted(out)


def serve(host="127.0.0.1", port=8765, workdir="gfem_work", open_browser=True):
    session = Session(workdir)
    Handler.session = session
    try:                                         # gmsh must be initialised from the main thread
        from ..cad.geometry import gmsh_module
        gmsh_module()
    except Exception as e:  # noqa: BLE001
        print(f"warning: CAD kernel unavailable ({e}); install gmsh for CAD import/meshing")
    import signal                                # native libraries may reset SIGPIPE to default:
    if hasattr(signal, "SIGPIPE"):               # a closed browser socket must not kill the server
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}/"
    print(f"gfem GUI running at {url}  (workdir {session.workdir}); Ctrl+C to quit")
    if open_browser:
        try:
            import webbrowser
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        except Exception:  # noqa: BLE001
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return httpd
