"""Ready-made example projects for the GUI (also used as end-to-end tests)."""
import copy
import numpy as np

from .. import project as P
from ..cad.geometry import CadModel


def _faces(info, pred):
    out = []
    for f in info["faces"]:
        c = np.asarray(f["com"])
        n = np.asarray(f["normal"]) if f.get("normal") is not None else np.zeros(3)
        if pred(c, n, f):
            out.append(f)
    return out


def _scope(faces, dim=2):
    return dict(dim=dim, ids=[f["id"] for f in faces], sigs=[f["sig"] for f in faces])


def _base(name, feats, analysis="static", material="Structural Steel", mesh_size=None, share=True):
    p = P.new_project("mm")
    p["name"] = name
    p["geometry"]["features"] = feats
    p["geometry"]["share_topology"] = share
    m = P.library_material(material, "mm")
    m["id"] = "M1"
    p["materials"] = [m]
    p["analysis"]["type"] = analysis
    p["mesh"]["size"] = mesh_size
    cad = CadModel(feats, share_topology=share)
    info = cad.regenerate()
    p["bodies"] = {str(b["id"]): dict(name=f"Body {b['id']}", material="M1", sig=b["sig"]) for b in info["bodies"]}
    return p, info


def ex_bracket():
    feats = [
        dict(id="F1", type="extrude", name="L profile", plane="XY",
             loops=[dict(points=[[0, 0], [120, 0], [120, 12], [12, 12], [12, 80], [0, 80]])], distance=40),
        dict(id="F2", type="cylinder", name="Hole 1", x=40, y=-1, z=20, dx=0, dy=14, dz=0, r=6),
        dict(id="F3", type="cylinder", name="Hole 2", x=90, y=-1, z=20, dx=0, dy=14, dz=0, r=6),
        dict(id="F4", type="boolean", name="Bolt holes", op="cut", target=["F1"], tool=["F2", "F3"]),
        dict(id="F5", type="cylinder", name="Pin hole", x=-1, y=60, z=20, dx=14, dy=0, dz=0, r=8),
        dict(id="F6", type="boolean", name="Cut pin hole", op="cut", target=["F4"], tool=["F5"]),
    ]
    p, info = _base("L-Bracket", feats, mesh_size=4.0)
    holes = _faces(info, lambda c, n, f: f["type"] == "Cylinder" and c[1] < 12.5 and abs(c[2] - 20) < 1 and c[0] > 20)
    pin = _faces(info, lambda c, n, f: f["type"] == "Cylinder" and c[0] < 12.5 and abs(c[1] - 60) < 1)
    p["conditions"] = [
        dict(id="B1", type="fixed", name="Fixed Support", scope=_scope(holes)),
        dict(id="L1", type="force", name="Force", scope=_scope(pin), vector=[2000.0, -3000.0, 0.0]),
    ]
    p["results"] += [dict(id="R3", type="safety_factor", name="Safety Factor")]
    return p


def ex_plate_hole():
    feats = [dict(id="F1", type="box", name="Plate", x=0, y=0, z=0, dx=200, dy=100, dz=10),
             dict(id="F2", type="cylinder", name="Hole", x=100, y=50, z=-1, dx=0, dy=0, dz=12, r=20),
             dict(id="F3", type="boolean", name="Cut", op="cut", target=["F1"], tool=["F2"])]
    p, info = _base("Plate with hole (Kt)", feats, mesh_size=5.0)
    left = _faces(info, lambda c, n, f: abs(c[0]) < 1e-6)
    right = _faces(info, lambda c, n, f: abs(c[0] - 200) < 1e-6)
    hole = _faces(info, lambda c, n, f: f["type"] == "Cylinder")
    p["mesh"]["local"] = [dict(id="S1", name="Hole sizing", scope=_scope(hole), size=1.5)]
    p["conditions"] = [
        dict(id="B1", type="frictionless", name="Symmetry X", scope=_scope(left)),
        dict(id="B2", type="displacement", name="Rigid body", scope=_scope(_faces(info, lambda c, n, f: abs(c[1]) < 1e-6)),
             components=[None, 0.0, None]),
        dict(id="B3", type="displacement", name="Z fix", scope=_scope(_faces(info, lambda c, n, f: abs(c[2]) < 1e-6)),
             components=[None, None, 0.0]),
        dict(id="L1", type="pressure", name="Tension 100 MPa", scope=_scope(right), value=-100.0),
    ]
    p["results"] += [dict(id="R3", type="stress_xx", name="Normal Stress X")]
    return p


def ex_modal():
    feats = [dict(id="F1", type="box", name="Beam", x=0, y=0, z=0, dx=300, dy=20, dz=10)]
    p, info = _base("Cantilever modal", feats, analysis="modal", material="Aluminum Alloy", mesh_size=5.0)
    root = _faces(info, lambda c, n, f: abs(c[0]) < 1e-6)
    p["conditions"] = [dict(id="B1", type="fixed", name="Fixed Support", scope=_scope(root))]
    p["analysis"]["nmodes"] = 6
    return p


def ex_buckling():
    feats = [dict(id="F1", type="box", name="Column", x=0, y=0, z=0, dx=20, dy=20, dz=400)]
    p, info = _base("Column buckling", feats, analysis="buckling", mesh_size=6.0)
    base = _faces(info, lambda c, n, f: abs(c[2]) < 1e-6)
    top = _faces(info, lambda c, n, f: abs(c[2] - 400) < 1e-6)
    p["conditions"] = [dict(id="B1", type="fixed", name="Fixed Support", scope=_scope(base)),
                       dict(id="L1", type="force", name="Unit load", scope=_scope(top), vector=[0, 0, -1.0])]
    p["analysis"]["nmodes"] = 4
    return p


def ex_heatsink():
    feats = [dict(id="F1", type="box", name="Base", x=0, y=0, z=0, dx=80, dy=60, dz=6),
             dict(id="F2", type="box", name="Fin", x=4, y=0, z=6, dx=3, dy=60, dz=30),
             dict(id="F3", type="pattern", name="Fin pattern", target=["F2"], kind="linear", direction=[1, 0, 0],
                  count=8, spacing=10, fuse=False),
             dict(id="F4", type="boolean", name="Fuse fins", op="fuse", target=["F1"], tool=["F2", "F3"])]
    p, info = _base("Heat sink (thermal-stress)", feats, analysis="thermal_stress", material="Aluminum Alloy",
                    mesh_size=3.0)
    bottom = _faces(info, lambda c, n, f: abs(c[2]) < 1e-6)
    others = [f for f in info["faces"] if f not in bottom]
    p["conditions"] = [
        dict(id="T1", type="heat_flux", name="Heat flux (chip)", scope=_scope(bottom), value=0.02),
        dict(id="T2", type="convection", name="Convection", scope=_scope(others), h=2e-5, ambient=22.0),
        dict(id="B1", type="displacement", name="Support", scope=_scope(bottom), components=[None, None, 0.0]),
        dict(id="B2", type="displacement", name="Pin", scope=_scope(_faces(info, lambda c, n, f: abs(c[0]) < 1e-6)),
             components=[0.0, None, None]),
        dict(id="B3", type="displacement", name="Pin y", scope=_scope(_faces(info, lambda c, n, f: abs(c[1]) < 1e-6)),
             components=[None, 0.0, None]),
    ]
    p["results"] = [dict(id="R1", type="temperature", name="Temperature"),
                    dict(id="R2", type="total_deformation", name="Total Deformation"),
                    dict(id="R3", type="von_mises", name="Equivalent Stress")]
    return p


def ex_topopt():
    feats = [dict(id="F1", type="box", name="Design space", x=0, y=0, z=0, dx=160, dy=60, dz=20)]
    p, info = _base("Topology optimisation (cantilever)", feats, analysis="topopt", mesh_size=5.0)
    p["mesh"]["order"] = 1
    left = _faces(info, lambda c, n, f: abs(c[0]) < 1e-6)
    right = _faces(info, lambda c, n, f: abs(c[0] - 160) < 1e-6)
    p["conditions"] = [dict(id="B1", type="fixed", name="Fixed Support", scope=_scope(left)),
                       dict(id="L1", type="force", name="Tip load", scope=_scope(right), vector=[0, -5000.0, 0])]
    p["analysis"]["topopt"] = dict(volfrac=0.35, penal=3.0, iters=30, rmin=9.0, keep_bc=True)
    p["results"] = [dict(id="R1", type="density", name="Topology Density"),
                    dict(id="R2", type="total_deformation", name="Total Deformation")]
    return p


def ex_contact():
    feats = [dict(id="F1", type="box", name="Base block", x=0, y=0, z=0, dx=40, dy=40, dz=20),
             dict(id="F2", type="cylinder", name="Punch", x=20, y=20, z=20.0, dx=0, dy=0, dz=20, r=10)]
    p, info = _base("Punch contact (nonlinear)", feats, mesh_size=2.5, share=False)
    p["analysis"]["nlgeom"] = True
    p["analysis"]["substeps"] = 4
    bottom = _faces(info, lambda c, n, f: abs(c[2]) < 1e-6)
    top = _faces(info, lambda c, n, f: abs(c[2] - 40) < 1e-6)
    base_top = _faces(info, lambda c, n, f: abs(c[2] - 20) < 1e-6 and f["area"] > 1000)
    punch_bot = _faces(info, lambda c, n, f: abs(c[2] - 20) < 1e-6 and f["area"] < 1000)
    p["connections"] = [dict(id="C1", type="frictionless", name="Frictionless - Punch/Base",
                             master=_scope(base_top), slave=_scope(punch_bot), fkn=1.0)]
    p["conditions"] = [dict(id="B1", type="fixed", name="Fixed Support", scope=_scope(bottom)),
                       dict(id="B2", type="displacement", name="Press 0.05 mm", scope=_scope(top),
                            components=[0.0, 0.0, -0.05])]
    return p


def ex_rubber():
    feats = [dict(id="F1", type="cylinder", name="Rubber pad", x=0, y=0, z=0, dx=0, dy=0, dz=20, r=15)]
    p, info = _base("Rubber pad 30% compression (NLGEOM)", feats, material="Rubber (Neo-Hookean)", mesh_size=3.0)
    p["analysis"]["nlgeom"] = True
    p["analysis"]["substeps"] = 5
    bottom = _faces(info, lambda c, n, f: abs(c[2]) < 1e-6)
    top = _faces(info, lambda c, n, f: abs(c[2] - 20) < 1e-6)
    p["conditions"] = [dict(id="B1", type="fixed", name="Fixed Support", scope=_scope(bottom)),
                       dict(id="B2", type="displacement", name="Compression", scope=_scope(top),
                            components=[0.0, 0.0, -6.0])]
    return p


def ex_stiffened_panel():
    feats = [dict(id="F1", type="surface", name="Skin", plane="XY", loops=[dict(rect=[0, 0, 300, 200])]),
             dict(id="F2", type="line", name="Stringer 1", points=[[0, 66.67, 0], [300, 66.67, 0]]),
             dict(id="F3", type="line", name="Stringer 2", points=[[0, 133.33, 0], [300, 133.33, 0]])]
    p, info = _base("Stiffened panel buckling (shell + beam)", feats, analysis="buckling",
                    material="Aluminum Alloy", mesh_size=10.0)
    for b in info.get("sheets", []):
        p["bodies"]["S" + str(b["id"])] = dict(name="Skin", material="M1", thickness=2.0, sig=b["sig"])
    for b in info.get("lines", []):
        p["bodies"]["L" + str(b["id"])] = dict(name="Stringer", material="M1", profile="RECT", dims=[3.0, 20.0],
                                                n1=[0, 1, 0], sig=b["sig"])
    E_ = lambda pred: [e for e in info["edges"] if pred(np.asarray(e["com"])) and not e.get("free")]
    left = E_(lambda c: abs(c[0]) < 1e-6)
    right = E_(lambda c: abs(c[0] - 300) < 1e-6)
    sides = E_(lambda c: abs(c[1]) < 1e-6 or abs(c[1] - 200) < 1e-6)
    sc = lambda es: dict(dim=1, ids=[e["id"] for e in es], sigs=[e["sig"] for e in es])
    p["conditions"] = [
        dict(id="B1", type="simply_supported", name="Loaded edge support", scope=sc(left)),
        dict(id="B2", type="displacement", name="Edges w = 0", scope=sc(right + sides), components=[None, None, 0.0]),
        dict(id="L1", type="force", name="Compression", scope=sc(right), vector=[-1000.0, 0, 0]),
    ]
    p["analysis"]["nmodes"] = 4
    return p


def ex_frame():
    pts = lambda *a: [list(map(float, x)) for x in a]
    feats = [dict(id="F1", type="line", name="Columns + beams",
                  points=pts([0, 0, 0], [0, 0, 3000], [4000, 0, 3000], [4000, 0, 0])),
             dict(id="F2", type="line", name="Frame 2", points=pts([0, 3000, 0], [0, 3000, 3000], [4000, 3000, 3000], [4000, 3000, 0])),
             dict(id="F3", type="line", name="Tie 1", points=pts([0, 0, 3000], [0, 3000, 3000])),
             dict(id="F4", type="line", name="Tie 2", points=pts([4000, 0, 3000], [4000, 3000, 3000]))]
    p, info = _base("3D steel frame (beams)", feats, analysis="modal", mesh_size=250.0)
    for b in info.get("lines", []):
        p["bodies"]["L" + str(b["id"])] = dict(name=f"Member {b['id']}", material="M1", profile="I",
                                                dims=[300.0, 150.0, 10.0, 7.0], sig=b["sig"])
    feet = [v for v in info["vertices"] if abs(v["xyz"][2]) < 1e-6]
    p["conditions"] = [dict(id="B1", type="fixed", name="Column bases", scope=dict(dim=0, ids=[v["id"] for v in feet],
                                                                              sigs=[v["sig"] for v in feet]))]
    p["analysis"]["nmodes"] = 6
    p["results"] = [dict(id="R1", type="total_deformation", name="Mode shape")]
    return p


def ex_seismic_frame():
    p = ex_frame()
    p["name"] = "3D frame - seismic response spectrum"
    p["analysis"].update(type="response_spectrum", nmodes=12, direction=[1, 0, 0], combination="CQC",
                         damping_ratio=0.05,
                         spectrum=[[0.1, 980.0], [0.5, 2450.0], [2.0, 6130.0], [10.0, 6130.0], [33.0, 2450.0], [100.0, 1960.0]])
    p["results"] = [dict(id="R1", type="total_deformation", name="Total Deformation (CQC)"),
                    dict(id="R2", type="von_mises", name="Equivalent Stress (CQC)")]
    return p


def ex_random_bracket():
    p = ex_bracket()
    p["name"] = "L-bracket - random vibration + fatigue"
    p["conditions"] = [c for c in p["conditions"] if c["type"] == "fixed"]
    p["analysis"].update(type="random_vibration", nmodes=10, direction=[0, 1, 0], damping_ratio=0.02,
                         psd=[[20.0, 1e4], [80.0, 4e5], [500.0, 4e5], [2000.0, 1e4]])
    p["materials"][0].update(ultimate=460.0, Sf=750.0, b=-0.09, Se=0.0)
    p["results"] = [dict(id="R1", type="total_deformation", name="1-sigma Deformation"),
                    dict(id="R2", type="von_mises", name="1-sigma Equivalent Stress"),
                    dict(id="R3", type="fatigue_sf", name="Fatigue Safety Factor (1-sigma)")]
    return p


EXAMPLES = {
    "seismic": ("Seismic frame (response spectrum)", "CQC response-spectrum analysis of the beam frame.", ex_seismic_frame),
    "random": ("Random vibration + fatigue", "Base PSD on the bracket, 1-sigma stress and fatigue safety factor.", ex_random_bracket),
    "panel": ("Stiffened panel (shell + beam)", "MITC4 shell skin with Timoshenko stringers, eigen buckling.", ex_stiffened_panel),
    "frame": ("3D steel frame (beams)", "I-section beam frame, natural frequencies.", ex_frame),
    "bracket": ("L-Bracket (static)", "Sketch-extruded bracket with bolt holes; fixed holes, pin load.", ex_bracket),
    "plate_hole": ("Plate with hole", "Stress concentration with symmetry, local mesh sizing.", ex_plate_hole),
    "modal": ("Cantilever modal", "Natural frequencies and effective mass of an aluminium beam.", ex_modal),
    "buckling": ("Column buckling", "Eigenvalue buckling of a slender steel column.", ex_buckling),
    "heatsink": ("Heat sink thermal-stress", "Patterned fins, heat flux + convection, thermal stress.", ex_heatsink),
    "topopt": ("Topology optimisation", "SIMP compliance minimisation of a cantilever design space.", ex_topopt),
    "contact": ("Punch contact", "Two bodies with frictionless (augmented-Lagrange) contact.", ex_contact),
    "rubber": ("Rubber pad (hyperelastic)", "Neo-Hookean, large deformation, 30% compression.", ex_rubber),
}


def build_example(key, workdir=None):
    return EXAMPLES[key][2]()
