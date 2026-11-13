"""GUI project model: an ANSYS-Mechanical-like description (geometry feature tree, materials,
body assignments, mesh controls, connections, analysis settings, loads/supports scoped to CAD
entities, result objects) and its translation into the neutral solver `Spec`.

Unit systems: "mm" = (mm, N, s, MPa, t, mJ, K) and "m" = SI (m, N, s, Pa, kg, J, K).
The material library is stored in SI and converted.
"""
import copy
import json
import numpy as np

from .spec import Spec, new_step

UNITS = {
    "mm": dict(length="mm", force="N", stress="MPa", density="t/mm³", time="s", temp="°C",
               energy="mJ", conductivity="W/m·°C", cp="mJ/t·°C", film="W/mm²·°C", flux="W/mm²",
               gen="W/mm³", accel="mm/s²", mass="t"),
    "m": dict(length="m", force="N", stress="Pa", density="kg/m³", time="s", temp="°C", energy="J",
              conductivity="W/m·°C", cp="J/kg·°C", film="W/m²·°C", flux="W/m²", gen="W/m³",
              accel="m/s²", mass="kg"),
}
# SI -> unit system factors
SI_TO = {"mm": dict(stress=1e-6, density=1e-12, conductivity=1.0, cp=1e6, film=1e-3, flux=1e-3,
                    gen=1e-6, accel=1e3, length=1e3),
         "m": dict(stress=1.0, density=1.0, conductivity=1.0, cp=1.0, film=1.0, flux=1.0, gen=1.0,
                   accel=1.0, length=1.0)}

# ANSYS-like engineering data (SI)
LIBRARY = [
    dict(name="Structural Steel", model="linear", E=200e9, nu=0.3, density=7850, alpha=1.2e-5, k=60.5,
         cp=434, yield_=250e6, tangent=1450e6, color="#8aa9c9"),
    dict(name="Structural Steel NL", model="plastic", E=200e9, nu=0.3, density=7850, alpha=1.2e-5,
         k=60.5, cp=434, yield_=250e6, tangent=1450e6, color="#7d97b3"),
    dict(name="Stainless Steel", model="linear", E=193e9, nu=0.31, density=7750, alpha=1.7e-5, k=15.1,
         cp=480, yield_=207e6, tangent=1000e6, color="#a9b4c0"),
    dict(name="Aluminum Alloy", model="linear", E=71e9, nu=0.33, density=2770, alpha=2.3e-5, k=144,
         cp=875, yield_=280e6, tangent=500e6, color="#c4cbd4"),
    dict(name="Aluminum 6061-T6", model="linear", E=68.9e9, nu=0.33, density=2700, alpha=2.36e-5, k=167,
         cp=896, yield_=276e6, tangent=600e6, color="#d0d6dd"),
    dict(name="Titanium Ti-6Al-4V", model="linear", E=113.8e9, nu=0.342, density=4430, alpha=8.6e-6, k=6.7,
         cp=526, yield_=880e6, tangent=1000e6, color="#9c9fb5"),
    dict(name="Copper Alloy", model="linear", E=110e9, nu=0.34, density=8300, alpha=1.8e-5, k=401,
         cp=385, yield_=280e6, tangent=1150e6, color="#d39a6a"),
    dict(name="Gray Cast Iron", model="linear", E=110e9, nu=0.28, density=7200, alpha=1.1e-5, k=52,
         cp=447, yield_=0, tangent=0, color="#7f8287"),
    dict(name="Concrete", model="linear", E=30e9, nu=0.18, density=2300, alpha=1.4e-5, k=1.4, cp=780,
         yield_=0, tangent=0, color="#b9b6aa"),
    dict(name="Polyethylene", model="linear", E=1.1e9, nu=0.42, density=950, alpha=2.3e-4, k=0.28,
         cp=2300, yield_=25e6, tangent=0, color="#e6e2c3"),
    dict(name="ABS", model="linear", E=2.3e9, nu=0.35, density=1050, alpha=9e-5, k=0.17, cp=1400,
         yield_=40e6, tangent=0, color="#e0d6b8"),
    dict(name="Rubber (Neo-Hookean)", model="neohooke", E=3e6, nu=0.49, density=1100, alpha=2e-4, k=0.15,
         cp=1900, yield_=0, tangent=0, color="#444444"),
]


def library_material(name, units="mm"):
    for m in LIBRARY:
        if m["name"] == name:
            return si_to_units(m, units)
    raise KeyError(name)


def si_to_units(m, units):
    f = SI_TO[units]
    out = dict(name=m["name"], model=m["model"], color=m.get("color", "#8aa9c9"))
    out["E"] = m["E"] * f["stress"]
    out["nu"] = m["nu"]
    out["density"] = m["density"] * f["density"]
    out["alpha"] = m["alpha"]
    out["k"] = m["k"] * f["conductivity"]
    out["cp"] = m["cp"] * f["cp"]
    out["yield"] = m.get("yield_", 0) * f["stress"]
    out["tangent"] = m.get("tangent", 0) * f["stress"]
    out["tref"] = 22.0
    return out


def new_project(units="mm"):
    steel = library_material("Structural Steel", units)
    steel["id"] = "M1"
    return dict(version=2, name="Untitled", units=units,
                geometry=dict(features=[], share_topology=True), mesh_source=None,
                materials=[steel], bodies={},
                mesh=dict(size=None, order=2, kind="tet", curvature=12, local=[]),
                connections=[], named_selections=[],
                analysis=dict(type="static", nlgeom=False, substeps=1, max_substeps=200, time=1.0,
                              nmodes=6, linear_solver="auto", frames=20, damping=0.0, dt=None,
                              topopt=dict(volfrac=0.4, penal=3.0, iters=40, rmin=None, keep_bc=True),
                              transient=False, initial_temperature=22.0),
                conditions=[], results=[
                    dict(id="R1", type="total_deformation", name="Total Deformation"),
                    dict(id="R2", type="von_mises", name="Equivalent Stress")])


ANALYSES = {
    "static": "Static Structural", "modal": "Modal", "buckling": "Eigenvalue Buckling",
    "transient": "Transient Structural", "explicit": "Explicit Dynamics",
    "thermal": "Steady-State Thermal", "thermal_transient": "Transient Thermal",
    "thermal_stress": "Thermal-Stress (coupled)", "topopt": "Topology Optimization",
    "nonlinear_buckling": "Nonlinear Buckling (imperfection + Riks)",
    "harmonic": "Harmonic Response (MSUP)",
    "response_spectrum": "Response Spectrum (SRSS / CQC)",
    "random_vibration": "Random Vibration (PSD)",
    "deck": "Solver deck steps (as imported)",
}

CONDITIONS = {
    # structural
    "fixed": dict(label="Fixed Support", group="support", dims=[0, 1, 2]),
    "displacement": dict(label="Displacement", group="support", dims=[0, 1, 2]),
    "frictionless": dict(label="Frictionless Support", group="support", dims=[2]),
    "force": dict(label="Force", group="load", dims=[0, 1, 2]),
    "pressure": dict(label="Pressure", group="load", dims=[2]),
    "moment": dict(label="Moment", group="load", dims=[2]),
    "gravity": dict(label="Standard Earth Gravity", group="load", dims=[]),
    "acceleration": dict(label="Acceleration", group="load", dims=[]),
    "rotational_velocity": dict(label="Rotational Velocity", group="load", dims=[]),
    "body_temperature": dict(label="Thermal Condition", group="load", dims=[3]),
    "rigid_wall": dict(label="Rigid Wall", group="support", dims=[2]),
    "simply_supported": dict(label="Simply Supported", group="support", dims=[0, 1]),
    "remote_displacement": dict(label="Remote Displacement", group="support", dims=[1, 2]),
    "remote_force": dict(label="Remote Force", group="load", dims=[2]),
    "bolt_pretension": dict(label="Bolt Pretension", group="load", dims=[3]),
    "point_mass": dict(label="Point Mass", group="load", dims=[0, 1, 2]),
    "elastic_support": dict(label="Elastic Support", group="support", dims=[2]),
    "radiation": dict(label="Radiation", group="thermal", dims=[2]),
    # thermal
    "temperature": dict(label="Temperature", group="thermal", dims=[0, 1, 2]),
    "heat_flux": dict(label="Heat Flux", group="thermal", dims=[2]),
    "convection": dict(label="Convection", group="thermal", dims=[2]),
    "heat_generation": dict(label="Internal Heat Generation", group="thermal", dims=[3]),
}

RESULTS = {
    "total_deformation": "Total Deformation", "deformation_x": "Directional Deformation X",
    "deformation_y": "Directional Deformation Y", "deformation_z": "Directional Deformation Z",
    "von_mises": "Equivalent (von-Mises) Stress", "principal_max": "Maximum Principal Stress",
    "principal_min": "Minimum Principal Stress", "stress_xx": "Normal Stress X",
    "stress_yy": "Normal Stress Y", "stress_zz": "Normal Stress Z", "shear_xy": "Shear Stress XY",
    "tresca": "Stress Intensity (Tresca)", "peeq": "Equivalent Plastic Strain",
    "strain_eq": "Equivalent Total Strain", "temperature": "Temperature",
    "density": "Topology Density", "safety_factor": "Safety Factor",
    "fatigue_life": "Fatigue Life (cycles)", "fatigue_damage": "Fatigue Damage", "fatigue_sf": "Fatigue Safety Factor",
}


class ProjectError(Exception):
    pass


def resolve_scope(project, scope, info, dims=None):
    """Return (dim, [entity ids]) for a scope, repairing topological naming via signatures."""
    from .cad.geometry import match
    if scope is None:
        return None, []
    if scope.get("named"):
        ns = next((n for n in project.get("named_selections", []) if n["id"] == scope["named"]), None)
        if ns is None:
            raise ProjectError(f"named selection {scope['named']} not found")
        return resolve_scope(project, ns["scope"], info, dims)
    if scope.get("set"):
        return "set", list(scope["set"]) if isinstance(scope["set"], list) else [scope["set"]]
    dim = scope.get("dim", 2)
    ids = scope.get("ids", [])
    sigs = scope.get("sigs") or [None] * len(ids)
    if info is None:
        return dim, ids
    refs = [dict(id=i, sig=s) for i, s in zip(ids, sigs)]
    out, _ = match(info, dim, refs)
    return dim, out


def _key(dim):
    return {3: "BODY", 2: "FACE", 1: "EDGE", 0: "VERTEX"}[dim]


def scope_nodes(sp, dim, ids):
    if dim == "set":
        out = []
        for n in ids:
            n = n.upper()
            if n in sp.nsets:
                out.append(sp.nsets[n])
            elif n in sp.surfaces:
                s = sp.surfaces[n]
                if "nodes" in s:
                    out.append(np.asarray(s["nodes"]))
                else:
                    _, conns = sp.element_table()
                    from .elements import ELEMENTS
                    types, _ = sp.element_table()
                    out.append(np.unique([conns[e][i] for e, f in zip(s["elems"], s["faces"])
                                          for i in ELEMENTS[types[e]].faces[f]]))
            elif n in sp.elsets:
                _, conns = sp.element_table()
                out.append(np.unique(np.concatenate([conns[e] for e in sp.elsets[n]])))
        return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)
    out = [sp.nsets.get(f"{_key(dim)}_{i}") for i in ids]
    out = [x for x in out if x is not None]
    return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)


def scope_surface(sp, dim, ids, name):
    """Create/merge an element-face surface for face scopes; returns the surface name."""
    if dim == "set":
        if len(ids) == 1 and ids[0].upper() in sp.surfaces:
            return ids[0].upper()
        el, fc = [], []
        for n in ids:
            s = sp.surfaces.get(n.upper())
            if s and "elems" in s:
                el.append(s["elems"])
                fc.append(s["faces"])
        if not el:
            raise ProjectError(f"scope {ids} is not a surface")
        sp.surfaces[name] = dict(elems=np.concatenate(el), faces=np.concatenate(fc))
        return name
    if dim != 2:
        raise ProjectError("this condition requires faces")
    el, fc = [], []
    for i in ids:
        s = sp.surfaces.get(f"FACE_{i}")
        if s:
            el.append(s["elems"])
            fc.append(s["faces"])
    if not el:
        raise ProjectError("no meshed faces in scope")
    if len(ids) == 1:
        return f"FACE_{ids[0]}"
    sp.surfaces[name] = dict(elems=np.concatenate(el), faces=np.concatenate(fc))
    return name


def build_spec(project, mesh_spec, info=None, log=print):
    """Combine the meshed geometry (Spec with entity sets) and the project definitions."""
    p = project
    sp = copy.copy(mesh_spec)
    sp.nsets = dict(mesh_spec.nsets)
    sp.surfaces = dict(mesh_spec.surfaces)
    sp.elsets = dict(mesh_spec.elsets)
    sp.materials, sp.sections, sp.steps = {}, [], []
    sp.constraints, sp.contacts = [], []
    sp.units = p.get("units", "mm")
    sp.warnings = list(getattr(mesh_spec, "warnings", []))
    mats = {m["id"]: m for m in p["materials"]}
    if not mats:
        raise ProjectError("no material defined")
    default = p["materials"][0]["id"]
    for mid, m in mats.items():
        props = {k: m[k] for k in ("E", "nu", "density", "alpha", "k", "cp", "tref") if k in m}
        model = m.get("model", "linear")
        props["model"] = model
        if model == "plastic":
            if m.get("table"):
                props["plastic"] = m["table"]
            else:
                props["yield"] = m.get("yield", 250.0)
                props["tangent"] = m.get("tangent", 0.0)
        if model == "neohooke" and m.get("C10"):
            props["C10"], props["D1"] = m["C10"], m.get("D1")
        if model in ("arruda", "gent"):
            props.update(mu=m.get("mu", m.get("E", 3.0) / 3), lam_m=m.get("lam_m", 7.0), Jm=m.get("Jm", 50.0))
        if model == "mooney":
            props["mooney"] = [m.get("C10", 0.5), m.get("C01", 0.1), m.get("D1", 0.0)]
        sp.materials[mid] = props
    if p.get("mesh_source"):
        # imported FE deck: keep its sections unless the project overrides materials per set
        sp.sections = list(getattr(mesh_spec, "sections", []))
        for es, mid in p.get("set_materials", {}).items():
            sp.sections = [s for s in sp.sections if s["elset"] != es]
            sp.sections.append(dict(elset=es, material=mid))
        sp.materials.update({k: v for k, v in mesh_spec.materials.items() if k not in sp.materials})
        if not sp.sections:
            for es in sp.elsets:
                sp.sections.append(dict(elset=es, material=default))
    else:
        for es in [k for k in sp.elsets if k.startswith("BODY_")]:
            bid = es.split("_", 1)[1]
            b = p.get("bodies", {}).get(str(bid), {})
            if b.get("suppressed"):
                continue
            sp.sections.append(dict(elset=es, material=b.get("material", default)))
        for es in [k for k in sp.elsets if k.startswith("SHEET_")]:          # shells
            b = p.get("bodies", {}).get("S" + es.split("_", 1)[1], {})
            if not b.get("suppressed"):
                sp.sections.append(dict(elset=es, material=b.get("material", default), kind="shell",
                                        thickness=float(b.get("thickness", 1.0))))
        for es in [k for k in sp.elsets if k.startswith("LINE_")]:           # beams
            b = p.get("bodies", {}).get("L" + es.split("_", 1)[1], {})
            if not b.get("suppressed"):
                sp.sections.append(dict(elset=es, material=b.get("material", default), kind="beam",
                                        profile=b.get("profile", "RECT"), dims=b.get("dims", [10.0, 10.0]),
                                        n1=b.get("n1")))
    a = p["analysis"]
    kind = a["type"]
    if kind == "deck":
        if not mesh_spec.steps:
            raise ProjectError("the imported deck has no analysis steps")
        sp.steps = list(mesh_spec.steps)
        sp.constraints = list(mesh_spec.constraints)
        sp.contacts = list(mesh_spec.contacts)
        sp.initial_temperature = mesh_spec.initial_temperature
        sp.amplitudes = dict(mesh_spec.amplitudes)
        return sp
    T = float(a.get("time", 1.0))
    nlgeom = bool(a.get("nlgeom"))
    sub = max(int(a.get("substeps", 1)), 1)
    st = new_step("Step-1", "static", nlgeom=nlgeom, time=T, dt_init=T / sub,
                  dt_max=T / sub if nlgeom or sub > 1 else T, dt_min=T / max(int(a.get("max_substeps", 200)), sub) / 4,
                  frames=int(a.get("frames", 20)), nmodes=int(a.get("nmodes", 6)),
                  eigensolver=a.get("eigensolver", "auto"),
                  damping=float(a.get("damping", 0.0)))
    heat = dict(fixed=[], flux=[], film=[], gen=[])
    temps = []
    Tinit = float(a.get("initial_temperature", 22.0))
    for c in p.get("conditions", []):
        if c.get("suppressed"):
            continue
        typ = c["type"]
        name = f"C_{c['id']}"
        try:
            dim, ids = resolve_scope(p, c.get("scope"), info)
        except ProjectError as e:
            sp.warnings.append(str(e))
            continue
        if CONDITIONS.get(typ, {}).get("dims") and not ids:
            sp.warnings.append(f"{c.get('name', typ)}: empty scope (ignored)")
            continue
        nodes = scope_nodes(sp, dim, ids) if ids else np.zeros(0, np.int64)
        vec = np.asarray(c.get("vector", (0, 0, 0)), float)
        if typ == "fixed":                      # translations + rotations (shell/beam nodes)
            for k in range(6):
                st["bcs"].append(dict(nodes=nodes, dof=k, value=0.0))
        elif typ == "simply_supported":         # translations only
            for k in range(3):
                st["bcs"].append(dict(nodes=nodes, dof=k, value=0.0))
        elif typ == "displacement":
            comps = list(c.get("components", [0, 0, 0])) + list(c.get("rotations", [None, None, None]))
            for k, v in enumerate(comps):
                if v is not None and v != "":
                    st["bcs"].append(dict(nodes=nodes, dof=k, value=float(v)))
        elif typ == "remote_displacement":
            rp = f"RP_{c['id']}"
            sp.constraints.append(dict(type="rigid", nodes=nodes, point=c.get("point"), name=rp))
            comps = list(c.get("components", [0, 0, 0])) + list(c.get("rotations", [0, 0, 0]))
            for k, v in enumerate(comps):
                if v is not None and v != "":
                    st["bcs"].append(dict(remote=rp, dof=k, value=float(v)))
        elif typ == "remote_force":
            surf = scope_surface(sp, dim, ids, name)
            pt = c.get("point")
            if pt is None:
                pt = sp.nodes[nodes].mean(0).tolist()
            st["dsloads"].append(dict(surface=surf, kind="REMOTE", vector=list(vec), point=list(pt),
                                      moment=c.get("moment")))
        elif typ == "point_mass":
            sp.point_masses = getattr(sp, "point_masses", []) + [(nodes, float(c.get("value", 0.0)))]
        elif typ == "elastic_support":                 # foundation stiffness per unit area
            surf = scope_surface(sp, dim, ids, name)
            sp.elastic_supports = getattr(sp, "elastic_supports", []) + [(surf, float(c.get("value", 0.0)))]
        elif typ == "radiation":
            heat.setdefault("radiation", []).append((scope_surface(sp, dim, ids, name), float(c.get("emissivity", 0.8)),
                                                     float(c.get("ambient", 22.0))))
        elif typ == "bolt_pretension":
            for b in ids:
                es = f"BODY_{b}"
                axis = c.get("axis")
                if not axis:                     # longest bounding-box direction of the bolt
                    _, conns = sp.element_table()
                    nn = np.unique(np.concatenate([conns[e] for e in sp.elsets[es]]))
                    ext = sp.nodes[nn].max(0) - sp.nodes[nn].min(0)
                    axis = np.eye(3)[int(np.argmax(ext))].tolist()
                st.setdefault("bolts", []).append(dict(elset=es, axis=axis, force=float(c.get("value", 0.0)),
                                                       name=c.get("name", c["id"])))
        elif typ == "frictionless":
            fi = {f["id"]: f for f in (info or {}).get("faces", [])}
            for fid in ids:
                n = fi.get(fid, {}).get("normal")
                fn = scope_nodes(sp, 2, [fid])
                if n is None:
                    sp.warnings.append(f"frictionless support on non-planar face {fid} ignored")
                    continue
                n = np.asarray(n, float)
                ax = int(np.argmax(np.abs(n)))
                if abs(abs(n[ax]) - 1) < 1e-8:
                    st["bcs"].append(dict(nodes=fn, dof=ax, value=0.0))
                else:
                    sp.constraints.append(dict(type="normal", nodes=fn, normal=n, name=name))
        elif typ == "force":
            if dim == 2:
                surf = scope_surface(sp, dim, ids, name)
                st["dsloads"].append(dict(surface=surf, kind="FORCE", vector=list(vec)))
            else:
                for k in range(3):
                    if vec[k]:
                        st["cloads"].append(dict(nodes=nodes, dof=k, value=vec[k] / max(len(nodes), 1)))
        elif typ == "pressure":
            surf = scope_surface(sp, dim, ids, name)
            st["dsloads"].append(dict(surface=surf, kind="P", value=float(c.get("value", 0.0)),
                                      follower=nlgeom))
        elif typ == "moment":
            surf = scope_surface(sp, dim, ids, name)
            st["dsloads"].append(dict(surface=surf, kind="MOMENT", vector=list(vec)))
        elif typ in ("gravity", "acceleration"):
            g = vec if typ == "acceleration" else np.asarray(c.get("vector", (0, -1, 0)), float)
            mag = float(np.linalg.norm(g))
            if typ == "gravity":
                mag = 9.80665 * SI_TO[sp.units]["accel"]
                g = g / max(np.linalg.norm(g), 1e-300)
            else:
                g = -g / max(mag, 1e-300)   # ANSYS: acceleration -> inertial load opposite
            st["dloads"].append(dict(kind="GRAV", value=mag, vector=list(g / max(np.linalg.norm(g), 1e-300))))
        elif typ == "rotational_velocity":
            w = float(c.get("value", 0.0))       # rad/s
            st["dloads"].append(dict(kind="CENTRIF", value=w * w, point=c.get("point", [0, 0, 0]),
                                     axis=c.get("axis", [0, 0, 1])))
        elif typ == "body_temperature":
            bodies_nodes = scope_nodes(sp, 3, ids) if ids else np.arange(len(sp.nodes))
            temps.append((bodies_nodes, float(c.get("value", Tinit))))
        elif typ == "rigid_wall":
            sp.contacts.append(dict(type="wall", slave=nodes, point=c.get("point", [0, 0, 0]),
                                    normal=c.get("normal", [0, 1, 0]), motion=c.get("motion", [0, 0, 0]),
                                    fkn=c.get("fkn", 10.0), name=name))
        elif typ == "temperature":
            heat["fixed"].append((nodes, float(c.get("value", Tinit))))
        elif typ == "heat_flux":
            heat["flux"].append((scope_surface(sp, dim, ids, name), float(c.get("value", 0.0))))
        elif typ == "convection":
            heat["film"].append((scope_surface(sp, dim, ids, name), float(c.get("h", 0.0)),
                                 float(c.get("ambient", 22.0))))
        elif typ == "heat_generation":
            for b in ids:
                heat["gen"].append((f"BODY_{b}", float(c.get("value", 0.0))))
    for c in p.get("connections", []):
        if c.get("suppressed"):
            continue
        try:
            dm, im = resolve_scope(p, c.get("master"), info)
            ds, is_ = resolve_scope(p, c.get("slave"), info)
            ms = scope_surface(sp, dm, im, f"CM_{c['id']}")
            ss = scope_surface(sp, ds, is_, f"CS_{c['id']}")
        except ProjectError as e:
            sp.warnings.append(f"connection {c.get('name', c['id'])}: {e}")
            continue
        if c["type"] == "bonded":
            sp.constraints.append(dict(type="tie", slave=ss, master=ms, name=c.get("name", c["id"]),
                                       tol=c.get("tol")))
        else:
            sp.contacts.append(dict(type="frictionless", master=ms, slave=ss, fkn=float(c.get("fkn", 1.0)),
                                    mu=float(c.get("mu", 0.0)) if c["type"] == "frictional" else 0.0,
                                    name=c.get("name", c["id"])))
    st["temperatures"] = temps
    if temps or kind == "thermal_stress":
        sp.initial_temperature = np.full(len(sp.nodes), Tinit)
        for m in sp.materials.values():
            m.setdefault("tref", Tinit)
    if kind == "static":
        sp.steps.append(st)
    elif kind == "modal":
        st["procedure"] = "frequency"
        sp.steps.append(st)
    elif kind == "buckling":
        st["procedure"] = "buckle"
        sp.steps.append(st)
    elif kind == "transient":
        st["procedure"] = "dynamic"
        st["dt"] = float(a.get("dt") or T / max(sub, 20))
        st["alpha"] = -0.05
        sp.steps.append(st)
    elif kind == "explicit":
        st["procedure"] = "explicit"
        sp.steps.append(st)
    elif kind in ("thermal", "thermal_transient"):
        h = new_step("Thermal", "heat", time=T, heat=heat, transient=kind == "thermal_transient",
                     dt_init=float(a.get("dt") or T / 20), frames=int(a.get("frames", 20)))
        sp.steps.append(h)
        if kind == "thermal_transient":
            sp.initial_temperature = np.full(len(sp.nodes), Tinit)
    elif kind == "thermal_stress":
        h = new_step("Thermal", "heat", time=1.0, heat=heat)
        sp.steps.append(h)
        st["name"] = "Structural"
        sp.steps.append(st)
    elif kind == "nonlinear_buckling":
        st["procedure"] = "postbuckling"
        st["nlgeom"] = True
        st["imperfection"] = dict(modes=[int(x) for x in a.get("imperfection_modes", [1])],
                                  scale=a.get("imperfection_scale"))
        sp.steps.append(st)
    elif kind in ("response_spectrum", "random_vibration"):
        st["procedure"] = kind
        tab = a.get("spectrum") if kind == "response_spectrum" else a.get("psd")
        st["spectrum"] = dict(table=tab or ([[0.1, 9806.65], [100.0, 9806.65]] if kind == "response_spectrum"
                                            else [[20.0, 1e4], [2000.0, 1e4]]),
                              direction=a.get("direction", [0, 0, 1]), combination=a.get("combination", "SRSS"),
                              damping=float(a.get("damping_ratio", 0.02 if kind == "random_vibration" else 0.05)),
                              nmodes=int(a.get("nmodes", 20)))
        sp.steps.append(st)
    elif kind == "harmonic":
        st["procedure"] = "harmonic"
        st["harmonic"] = dict(fmin=float(a.get("fmin", 1.0)), fmax=float(a.get("fmax", 1000.0)),
                              nfreq=int(a.get("nfreq", 200)), damping=float(a.get("damping_ratio", 0.02)))
        sp.steps.append(st)
    elif kind == "topopt":
        st["procedure"] = "topopt"
        to = dict(a.get("topopt", {}))
        if to.get("keep_bc", True):
            keep = [b["nodes"] for b in st["bcs"]] + [c["nodes"] for c in st["cloads"]]
            for d in st["dsloads"]:
                s = sp.surfaces[d["surface"]]
                keep.append(scope_nodes(sp, "set", [d["surface"]]))
            to["frozen_nodes"] = np.unique(np.concatenate(keep)) if keep else None
        st["topopt"] = to
        sp.steps.append(st)
    else:
        raise ProjectError(f"unknown analysis type {kind}")
    return sp


def result_field(frame, rtype, project=None):
    """Nodal scalar for a result object type from a post-processed frame (or None)."""
    from .analysis import principal
    U = frame.get("U")
    S = frame.get("S")
    if rtype == "total_deformation" and U is not None:
        return np.linalg.norm(U, axis=1)
    if rtype.startswith("deformation_") and U is not None:
        return U[:, "xyz".index(rtype[-1])]
    if rtype == "von_mises" and "MISES" in frame:
        return frame["MISES"]
    if rtype in ("principal_max", "principal_min", "tresca") and S is not None:
        P = principal(S)
        if rtype == "principal_max":
            return P[:, 0]
        if rtype == "principal_min":
            return P[:, 2]
        return P[:, 0] - P[:, 2]
    comp = {"stress_xx": 0, "stress_yy": 1, "stress_zz": 2, "shear_xy": 3}
    if rtype in comp and S is not None:
        return S[:, comp[rtype]]
    if rtype == "peeq" and "PEEQ" in frame:
        return frame["PEEQ"]
    if rtype == "strain_eq" and "E" in frame:
        e = frame["E"]
        dev = e[:, :3] - e[:, :3].mean(1, keepdims=True)
        return np.sqrt(2 / 3 * (np.sum(dev ** 2, 1) + 0.5 * np.sum(e[:, 3:] ** 2, 1)))
    if rtype == "temperature" and "T" in frame:
        return frame["T"]
    if rtype == "density" and "density_nodal" in frame:
        return frame["density_nodal"]
    if rtype.startswith("fatigue_") and S is not None:
        from .fatigue import stress_life
        mats = (project or {}).get("materials", [])
        mf = next((m for m in mats if m.get("Sf")), mats[0] if mats else {})
        Su = mf.get("ultimate", mf.get("yield", 400.0) * 1.5 if mf.get("yield") else 400.0)
        r = stress_life(S, Su=Su, Sf=mf.get("Sf", 1.5 * Su), b=mf.get("b", -0.09), Se=mf.get("Se", 0.0),
                        Sy=mf.get("yield"), loading=(project or {}).get("fatigue", {}).get("loading", "fully_reversed"),
                        R=(project or {}).get("fatigue", {}).get("R", -1.0),
                        mean=(project or {}).get("fatigue", {}).get("mean", "goodman"),
                        scale=(project or {}).get("fatigue", {}).get("scale", 1.0),
                        design_life=(project or {}).get("fatigue", {}).get("design_life", 1e7))
        return {"fatigue_life": r["life"], "fatigue_damage": r["damage"], "fatigue_sf": r["safety_factor"]}[rtype]
    if rtype == "safety_factor" and "MISES" in frame:
        sy = None
        if project:
            ys = [m.get("yield", 0) for m in project.get("materials", []) if m.get("yield")]
            sy = min(ys) if ys else None
        if not sy:
            return None
        return np.minimum(sy / np.maximum(frame["MISES"], 1e-300), 15.0)
    return None


def save(project, path):
    with open(path, "w") as f:
        json.dump(project, f, indent=1)


def load(path):
    with open(path) as f:
        return json.load(f)
