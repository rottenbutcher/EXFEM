"""AI simulation assistant: tools that let the model set up and run a complete analysis.

Used through the MCP server (``--project``) by a signed-in Claude Code / Codex CLI, or by any
caller that wants a scriptable, LLM-friendly front end to a gfem project. The tools operate on a
normal gfem project (the same JSON the GUI saves) through a headless GUI `Session`, so geometry,
mesh, boundary conditions and results are exactly what the GUI would produce.

Tools: get_model, build_cad (geometry), set_material, add_condition, remove_condition,
set_analysis, set_mesh, solve. Faces are selected by id or by geometric filters (normal, box,
type, area), which the model can derive from the get_model listing or the rendered views.
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np

from .. import project as P

SIM_GUIDE = """# Simulation workflow (gfem)
1. get_model: bodies, faces (id, type, centre com, outward normal, area), materials, conditions.
2. Assign materials (set_material) - default is Structural Steel. Library: {library}.
3. Supports and loads with add_condition. Select faces with "faces": either a list of face ids from
   get_model, or a filter object; all criteria must hold:
     {{"normal":[0,0,-1]}} faces whose outward normal is within 5 deg of the vector
     {{"box":[xmin,ymin,zmin,xmax,ymax,zmax]}} faces whose centre lies inside the box
     {{"type":"Plane"|"Cylinder"|"Cone"|"Sphere"|"Torus"|"BSplineSurface"}}
     {{"min_area":A, "max_area":A}}, {{"near":[x,y,z], "count":n}} (the n faces closest to a point)
   The result tells you which faces were selected: check it.
   Types and fields:
     fixed                               (faces)
     displacement   components [ux,uy,uz] (null = free)
     frictionless                        (normal constraint, symmetry planes)
     force          vector [Fx,Fy,Fz]  total force N, spread over the faces
     pressure       value (MPa, positive pushes INTO the surface)
     moment         vector [Mx,My,Mz] N*mm about the faces' centroid
     gravity        vector (direction, magnitude 9806.65 mm/s^2 if only a direction is given)
     acceleration   vector [ax,ay,az] mm/s^2
     rotational_velocity value (rad/s), axis [x,y,z], point [x,y,z]
     remote_force   vector, point [x,y,z]   (force applied at a point, rigidly coupled)
     bolt_pretension value (N) on a body/faces with axis [x,y,z]
     temperature / heat_flux (value) / convection (h, ambient) / heat_generation (value)  (thermal)
4. set_analysis: static (default), modal (nmodes), buckling (nmodes), thermal, thermal_stress,
   topopt (volfrac), transient, harmonic, nonlinear_buckling; nlgeom for large deflection.
5. set_mesh: size (mm; default automatic), order 1|2.
6. solve: returns maximum deformation, maximum von Mises stress and where, minimum safety factor,
   reaction forces, frequencies / buckling factors. Judge the result (units: mm, N, MPa, s, t).
   If the user gave a requirement (e.g. safety factor >= 2, deflection < 0.5 mm), check it and
   change the geometry (build_cad) or setup, then solve again. Report the final numbers.
"""

SIM_TOOLS = [
    dict(name="get_model", description="Current project: geometry (bodies, faces with id/type/com/normal/area), "
         "materials, conditions, analysis and mesh settings. Call after every geometry change.",
         inputSchema={"type": "object", "properties": {
             "max_faces": {"type": "integer", "description": "maximum number of faces to list (default 120)"}}}),
    dict(name="set_material", description="Assign a library material or a custom one to bodies.",
         inputSchema={"type": "object", "properties": {
             "material": {"type": "string", "description": "library name, e.g. 'Structural Steel', 'Aluminum Alloy'"},
             "custom": {"type": "object", "description": "custom properties in mm-N-t-MPa: E, nu, density, yield, alpha, k, cp"},
             "bodies": {"description": "'all' or a list of body ids", "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "integer"}}]}},
             "required": []}),
    dict(name="add_condition", description="Add a support or load. See the simulation guide (get_model returns it).",
         inputSchema={"type": "object", "properties": {
             "type": {"type": "string"}, "name": {"type": "string"},
             "faces": {"description": "face ids or a filter object", "anyOf": [{"type": "array", "items": {"type": "integer"}}, {"type": "object"}]},
             "bodies": {"description": "body ids (for body loads / bolt pretension)", "type": "array", "items": {"type": "integer"}},
             "vector": {"type": "array", "items": {"type": "number"}},
             "components": {"type": "array", "items": {"type": ["number", "null"]}},
             "value": {"type": "number"}, "axis": {"type": "array", "items": {"type": "number"}},
             "point": {"type": "array", "items": {"type": "number"}},
             "h": {"type": "number"}, "ambient": {"type": "number"}},
             "required": ["type"]}),
    dict(name="remove_condition", description="Remove conditions by id, or all of them with 'all'.",
         inputSchema={"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}),
    dict(name="set_analysis", description="Analysis type and settings.",
         inputSchema={"type": "object", "properties": {
             "type": {"type": "string"}, "nmodes": {"type": "integer"}, "nlgeom": {"type": "boolean"},
             "substeps": {"type": "integer"}, "volfrac": {"type": "number"}}, "required": ["type"]}),
    dict(name="set_mesh", description="Global mesh size (mm, null = automatic) and element order (1 or 2).",
         inputSchema={"type": "object", "properties": {"size": {"type": ["number", "null"]}, "order": {"type": "integer"}}}),
    dict(name="solve", description="Mesh and solve the project; returns a result summary.",
         inputSchema={"type": "object", "properties": {}}),
]


class SimTools:
    def __init__(self, project_path, workdir=None, log=None):
        from ..gui.server import Session
        self.path = project_path
        self.workdir = workdir or os.path.join(os.path.dirname(os.path.abspath(project_path)), "sim_work")
        self.s = Session(self.workdir)
        if os.path.exists(project_path):
            self.s.project = P.load(project_path)
        self.log = log or (lambda m: None)
        self.solved = False
        self._regen()

    # ------------------------------------------------------------ helpers
    @property
    def p(self):
        return self.s.project

    def save(self):
        tmp = self.path + ".tmp"
        P.save(self.p, tmp)
        os.replace(tmp, self.path)

    def _regen(self):
        if self.p["geometry"]["features"]:
            self.s.regenerate()

    def guide(self):
        return SIM_GUIDE.format(library=", ".join(m["name"] for m in P.LIBRARY))

    def set_geometry(self, features, units=None, name=None):
        """Replace the geometry and re-attach the conditions: conditions created with a face
        filter are re-selected with the same filter; id-based ones are repaired through the
        geometric signatures. Returns the ids of conditions that no longer select anything."""
        self.p["geometry"]["features"] = features
        if name and self.p.get("name") in (None, "", "Untitled", "AI part"):
            self.p["name"] = name
        self.s.invalidate_mesh()
        self._regen()
        self.solved = False
        stale = self.reattach()
        self.save()
        return stale

    def reattach(self):
        stale = []
        if self.s.info is None:
            return [c["id"] for c in self.p["conditions"] if c.get("scope")]
        for c in self.p["conditions"]:
            sc = c.get("scope")
            if not sc:
                continue
            if isinstance(c.get("ai_faces"), dict):
                try:
                    sel = self._faces(c["ai_faces"])
                    c["scope"] = dict(dim=2, ids=[f["id"] for f in sel], sigs=[f["sig"] for f in sel])
                    continue
                except P.ProjectError:
                    stale.append(c["id"])
                    continue
            dim, ids = P.resolve_scope(self.p, sc, self.s.info)
            if len(ids) < len(sc.get("ids") or []) and c.get("ai_face_keys") and dim == 2:
                # signature repair failed (the face grew / moved): keep a face id that still has the
                # same type and outward normal (the common "make it thicker" edit)
                cur = {f["id"]: f for f in self.s.info["faces"]}
                for i in sc.get("ids") or []:
                    k, f = c["ai_face_keys"].get(str(i)), cur.get(i)
                    if i in ids or not k or not f or f["type"] != k["type"] or f.get("normal") is None or k.get("normal") is None:
                        continue
                    if float(np.dot(f["normal"], k["normal"])) > 0.999:
                        ids = list(ids) + [i]
            if not ids:
                stale.append(c["id"])
            elif dim in (2, 3):
                key = "faces" if dim == 2 else "bodies"
                ents = {e["id"]: e for e in self.s.info[key]}
                c["scope"] = dict(dim=dim, ids=list(ids), sigs=[ents[i]["sig"] for i in ids if i in ents])
        return stale

    def _faces(self, sel):
        info = self.s.info
        if info is None:
            raise P.ProjectError("there is no geometry yet: call build_cad first")
        faces = [f for f in info["faces"]]
        if isinstance(sel, (int, float)):
            sel = [int(sel)]
        if isinstance(sel, list):
            ids = {f["id"] for f in faces}
            bad = [i for i in sel if i not in ids]
            if bad:
                raise P.ProjectError(f"unknown face ids {bad}")
            return [f for f in faces if f["id"] in set(sel)]
        if not isinstance(sel, dict):
            raise P.ProjectError("faces must be a list of ids or a filter object")
        out = faces
        if "normal" in sel:
            n = np.asarray(sel["normal"], float)
            n /= np.linalg.norm(n)
            tol = np.cos(np.radians(float(sel.get("angle_tol", 5.0))))
            out = [f for f in out if f.get("normal") is not None and np.dot(f["normal"], n) / max(np.linalg.norm(f["normal"]), 1e-30) >= tol]
        if "box" in sel:
            b = np.asarray(sel["box"], float)
            out = [f for f in out if np.all(np.asarray(f["com"]) >= b[:3] - 1e-6) and np.all(np.asarray(f["com"]) <= b[3:] + 1e-6)]
        if "type" in sel:
            out = [f for f in out if f["type"].lower() == str(sel["type"]).lower()]
        if "min_area" in sel:
            out = [f for f in out if f["area"] >= float(sel["min_area"])]
        if "max_area" in sel:
            out = [f for f in out if f["area"] <= float(sel["max_area"])]
        if "near" in sel:
            q = np.asarray(sel["near"], float)
            out = sorted(out, key=lambda f: np.linalg.norm(np.asarray(f["com"]) - q))[:int(sel.get("count", 1))]
        if not out:
            raise P.ProjectError(f"no face matches {json.dumps(sel)}; use get_model to see the faces")
        return out

    @staticmethod
    def _fdesc(f):
        n = f.get("normal")
        return dict(id=f["id"], type=f["type"], com=[round(float(x), 3) for x in f["com"]],
                    normal=[round(float(x), 3) for x in n] if n is not None else None, area=round(float(f["area"]), 3))

    # ------------------------------------------------------------ tools
    def get_model(self, max_faces=120):
        info = self.s.info
        d = dict(name=self.p.get("name"), units=self.p.get("units"), analysis=self.p["analysis"]["type"],
                 nlgeom=self.p["analysis"].get("nlgeom", False), mesh=dict(size=self.p["mesh"]["size"], order=self.p["mesh"]["order"]),
                 materials=[dict(id=m["id"], name=m.get("name"), E=m.get("E"), nu=m.get("nu"), yield_=m.get("yield"))
                            for m in self.p["materials"]],
                 conditions=[{k: v for k, v in c.items() if k not in ("scope",)} | {"faces": (c.get("scope") or {}).get("ids")}
                             for c in self.p["conditions"]],
                 features=len(self.p["geometry"]["features"]))
        if info:
            d["bbox"] = [round(float(x), 3) for x in info["bbox"]]
            d["bodies"] = [dict(id=b["id"], volume=round(float(b["volume"]), 3), bbox=[round(float(x), 3) for x in b["bbox"]],
                                material=(self.p["bodies"].get(str(b["id"])) or {}).get("material", "M1")) for b in info["bodies"]]
            fs = sorted(info["faces"], key=lambda f: -f["area"])
            d["faces"] = [self._fdesc(f) for f in fs[:max_faces]]
            if len(fs) > max_faces:
                d["faces_omitted"] = len(fs) - max_faces
        d["guide"] = self.guide()
        return d

    def set_material(self, material=None, custom=None, bodies="all"):
        units = self.p.get("units", "mm")
        if custom:
            m = dict(name=custom.get("name", "Custom"), E=float(custom["E"]), nu=float(custom.get("nu", 0.3)),
                     density=float(custom.get("density", 7.85e-9)), model="linear_elastic")
            for k in ("yield", "alpha", "k", "cp", "ultimate"):
                if k in custom:
                    m[k] = float(custom[k])
        else:
            names = {x["name"].lower(): x["name"] for x in P.LIBRARY}
            key = (material or "Structural Steel").lower()
            hit = names.get(key) or next((v for k, v in names.items() if key in k), None)
            if not hit:
                raise P.ProjectError(f"unknown material {material!r}; library: {sorted(names.values())}")
            m = P.library_material(hit, units)
        existing = next((x for x in self.p["materials"] if x.get("name") == m["name"]), None)
        if existing is None:
            m["id"] = f"M{len(self.p['materials']) + 1}"
            while any(x["id"] == m["id"] for x in self.p["materials"]):
                m["id"] += "_"
            self.p["materials"].append(m)
        else:
            m = existing
        ids = [b["id"] for b in (self.s.info or {}).get("bodies", [])] if bodies in (None, "all") else list(bodies)
        cur = {b["id"]: b for b in (self.s.info or {}).get("bodies", [])}
        for b in ids:
            e = self.p["bodies"].setdefault(str(b), dict(name=f"Body {b}"))
            e["material"] = m["id"]
            if b in cur:
                e["sig"], e["feature"] = cur[b]["sig"], cur[b].get("feature")
        self.solved = False
        self.save()
        return dict(ok=True, material=m["name"], id=m["id"], bodies=ids, E=m.get("E"), yield_=m.get("yield"))

    def add_condition(self, type, faces=None, bodies=None, name=None, **kw):
        if type not in P.CONDITIONS:
            raise P.ProjectError(f"unknown condition type {type!r}; use one of {sorted(P.CONDITIONS)}")
        cid = f"C{len(self.p['conditions']) + 1}"
        while any(c["id"] == cid for c in self.p["conditions"]):
            cid += "_"
        c = dict(id=cid, type=type, name=name or P.CONDITIONS[type]["label"])
        sel = None
        if faces is not None:
            sel = self._faces(faces)
            c["scope"] = dict(dim=2, ids=[f["id"] for f in sel], sigs=[f["sig"] for f in sel])
            if isinstance(faces, dict):
                c["ai_faces"] = faces                    # re-applied after geometry changes
            else:                                        # id-based: remember what the faces looked like
                c["ai_face_keys"] = {str(f["id"]): dict(type=f["type"], normal=f.get("normal")) for f in sel}
        elif bodies is not None:
            info = self.s.info or {}
            bb = [b for b in info.get("bodies", []) if b["id"] in set(bodies)]
            c["scope"] = dict(dim=3, ids=[b["id"] for b in bb], sigs=[b["sig"] for b in bb])
        elif P.CONDITIONS[type]["dims"] and type not in ("gravity", "acceleration", "rotational_velocity"):
            raise P.ProjectError(f"{type} needs 'faces' (ids or a filter)")
        if type == "gravity":
            v = np.asarray(kw.get("vector") or [0, 0, -1], float)
            if np.linalg.norm(v) < 100:                 # a direction: standard gravity in mm/s^2
                v = v / max(np.linalg.norm(v), 1e-30) * 9806.65
            kw["vector"] = v.tolist()
        for k in ("vector", "components", "value", "axis", "point", "h", "ambient", "emissivity", "moment", "rotations"):
            if kw.get(k) is not None:
                c[k] = kw[k]
        if type in ("force", "moment", "acceleration", "remote_force") and "vector" not in c:
            raise P.ProjectError(f"{type} needs 'vector'")
        if type in ("pressure", "heat_flux", "temperature", "heat_generation", "bolt_pretension", "rotational_velocity") and "value" not in c:
            raise P.ProjectError(f"{type} needs 'value'")
        if type == "displacement" and "components" not in c:
            c["components"] = [0, 0, 0]
        self.p["conditions"].append(c)
        self.solved = False
        self.save()
        out = dict(ok=True, id=cid, type=type)
        if sel is not None:
            out["faces"] = [self._fdesc(f) for f in sel]
            out["total_area"] = round(sum(float(f["area"]) for f in sel), 3)
        return out

    def remove_condition(self, id):
        n = len(self.p["conditions"])
        self.p["conditions"] = [] if id == "all" else [c for c in self.p["conditions"] if c["id"] != id]
        self.save()
        return dict(ok=True, removed=n - len(self.p["conditions"]))

    def set_analysis(self, type, nmodes=None, nlgeom=None, substeps=None, volfrac=None):
        if type not in P.ANALYSES or type == "deck":
            raise P.ProjectError(f"unknown analysis {type!r}; use one of {[k for k in P.ANALYSES if k != 'deck']}")
        a = self.p["analysis"]
        a["type"] = type
        if nmodes:
            a["nmodes"] = int(nmodes)
        if nlgeom is not None:
            a["nlgeom"] = bool(nlgeom)
        if substeps:
            a["substeps"] = int(substeps)
        if volfrac:
            a["topopt"]["volfrac"] = float(volfrac)
        res = {"modal": ["total_deformation"], "buckling": ["total_deformation"], "thermal": ["temperature"],
               "thermal_transient": ["temperature"], "topopt": ["density", "von_mises"]}.get(type, ["total_deformation", "von_mises", "safety_factor"])
        self.p["results"] = [dict(id=f"R{i + 1}", type=r, name=P.RESULTS.get(r, r)) for i, r in enumerate(res)]
        self.solved = False
        self.save()
        return dict(ok=True, analysis=type, settings={k: a[k] for k in ("nmodes", "nlgeom", "substeps") if k in a})

    def set_mesh(self, size=None, order=None):
        if size is not None:
            self.p["mesh"]["size"] = float(size) if size else None
        if order in (1, 2):
            self.p["mesh"]["order"] = int(order)
        self.s.invalidate_mesh()
        self.solved = False
        self.save()
        return dict(ok=True, mesh=dict(size=self.p["mesh"]["size"], order=self.p["mesh"]["order"]))

    def solve(self):
        if not self.p["geometry"]["features"]:
            raise P.ProjectError("no geometry")
        stale = self.reattach()
        if stale:
            raise P.ProjectError(f"conditions {stale} no longer select any face of the current geometry; "
                                 "remove_condition them and add them again (get_model lists the faces)")
        from ..gui.server import setup_warnings
        problems = [w for w in setup_warnings(self.p) if w.startswith(("no supports", "no temperature"))]
        if problems:
            raise P.ProjectError(" ".join(problems))
        self._ev0 = len(self.s.events)
        self.s.solve()
        self.solved = True
        self.save()
        out = self.summary()
        png = self.result_image()
        if png:
            out["_png"] = png
            out["image"] = getattr(self, "_caption", "")
        return out

    def result_image(self):
        """Contour picture of the main result on the deformed shape (iso/front/top/right)."""
        from .ai import render_field
        s, d = self.s, self.s.display
        if not s.frames or d is None or not len(d.get("tri", [])):
            return None
        kind = self.p["analysis"]["type"]
        fr = s.frames[0] if kind in ("modal", "buckling") else s.frames[-1]
        rtype = {"thermal": "temperature", "thermal_transient": "temperature", "topopt": "density",
                 "modal": "total_deformation", "buckling": "total_deformation"}.get(kind, "von_mises")
        val = P.result_field(fr, rtype, self.p)
        if val is None:
            rtype, val = "total_deformation", P.result_field(fr, "total_deformation", self.p)
        if val is None:
            return None
        nodes = d["nodes"]
        try:
            U = np.asarray(fr["U"])[nodes] if "U" in fr and rtype != "temperature" else None
            png = render_field(d["xyz"], d["tri"], np.asarray(val)[nodes], disp=U)
        except Exception:  # noqa: BLE001
            return None
        self._caption = (f"image: {P.RESULTS.get(rtype, rtype)} contour on the deformed shape (exaggerated), "
                         f"blue = min {float(np.min(val)):.4g}, red = max {float(np.max(val)):.4g}; views iso / front / top / right")
        return png

    def summary(self):
        s, p = self.s, self.p
        out = dict(analysis=p["analysis"]["type"], mesh=s.mesh_stats and {k: s.mesh_stats.get(k) for k in ("nodes", "elements", "q_min")})
        fr = s.frames
        if not fr:
            out["error"] = "no result frames"
            return out
        warnings = [e["msg"] for e in s.events[getattr(self, "_ev0", 0):]
                    if e.get("type") == "log" and "warning" in e.get("msg", "").lower()][-5:]
        if warnings:
            out["warnings"] = warnings
        kind = p["analysis"]["type"]
        if kind in ("modal",):
            out["frequencies_hz"] = [round(float(f["freq"]), 4) for f in fr if "freq" in f]
            return out
        if kind in ("buckling",):
            out["load_factors"] = [round(float(f.get("eigenvalue", f.get("lpf", 0))), 5) for f in fr]
            return out
        last = fr[-1]
        X = None
        try:
            X = np.asarray(s.model.X)[:len(last["U"])] if s.model is not None and "U" in last else None
        except Exception:  # noqa: BLE001
            X = None
        if "U" in last:
            um = np.linalg.norm(last["U"], axis=1)
            k = int(np.argmax(um))
            out["max_deformation_mm"] = round(float(um[k]), 6)
            out["max_deformation_vector"] = [round(float(x), 6) for x in last["U"][k]]
            if X is not None:
                out["max_deformation_at"] = [round(float(x), 3) for x in X[k]]
        if "MISES" in last:
            m = np.asarray(last["MISES"])
            k = int(np.argmax(m))
            out["max_von_mises_mpa"] = round(float(m[k]), 4)
            if X is not None and k < len(X):
                out["max_von_mises_at"] = [round(float(x), 3) for x in X[k]]
            sf = P.result_field(last, "safety_factor", p)
            if sf is not None:
                out["min_safety_factor"] = round(float(np.min(sf)), 4)
                ys = [m.get("yield") for m in P.materials_in_use(p, s.info) if m.get("yield")]
                out["yield_mpa"] = min(ys) if ys else None
        if "T" in last and kind in ("thermal", "thermal_transient", "thermal_stress"):
            out["temperature_range"] = [round(float(np.min(last["T"])), 4), round(float(np.max(last["T"])), 4)]
        if "RF" in last:
            out["total_reaction_n"] = [round(float(x), 4) for x in np.asarray(last["RF"]).reshape(-1, 3).sum(0)]
        if kind == "topopt" and "compliance" in last:
            out["compliance"] = float(last["compliance"])
        return out

    # ------------------------------------------------------------ dispatch
    def call(self, name, args):
        fn = {"get_model": self.get_model, "set_material": self.set_material, "add_condition": self.add_condition,
              "remove_condition": self.remove_condition, "set_analysis": self.set_analysis, "set_mesh": self.set_mesh,
              "solve": self.solve}[name]
        return fn(**{k: v for k, v in (args or {}).items() if v is not None})
