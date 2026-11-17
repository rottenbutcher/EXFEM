"""CAD -> mesh -> solve -> ParaView, entirely from a script (no GUI).

A bracket is modelled with the parametric feature tree (sketch extrude, holes, fillet),
meshed with quadratic tets, the bolt holes are fixed and a pin load is applied on the
cylindrical face of the pin hole, then the results are written to ParaView and the
deck is exported to Abaqus and Nastran for cross-checking in a commercial code."""
import numpy as np
from gfem.cad import CadModel, mesh_cad
from gfem.spec import new_step
from gfem.analysis import run_spec
from gfem.io import write_any

feats = [
    dict(id="F1", type="extrude", plane="XY", distance=40,
         loops=[dict(points=[[0, 0], [120, 0], [120, 12], [12, 12], [12, 80], [0, 80]])]),
    dict(id="F2", type="cylinder", x=40, y=-1, z=20, dx=0, dy=14, dz=0, r=6),
    dict(id="F3", type="cylinder", x=90, y=-1, z=20, dx=0, dy=14, dz=0, r=6),
    dict(id="F4", type="boolean", op="cut", target=["F1"], tool=["F2", "F3"]),
    dict(id="F5", type="cylinder", x=-1, y=60, z=20, dx=14, dy=0, dz=0, r=8),
    dict(id="F6", type="boolean", op="cut", target=["F4"], tool=["F5"]),
]
cad = CadModel(feats)
info = cad.regenerate()
holes = [f["id"] for f in info["faces"] if f["type"] == "Cylinder" and f["com"][1] < 12.5 and f["com"][0] > 20]
pin = [f["id"] for f in info["faces"] if f["type"] == "Cylinder" and f["com"][0] < 12.5]
sp, stats = mesh_cad(cad, size=4.0, order=2)
print(f"mesh: {stats['nodes']} nodes, {stats['elements']} elements, min quality {stats['q_min']:.3f}")
sp.materials["STEEL"] = dict(E=200e3, nu=0.3, density=7.85e-9)
for k in sp.elsets:
    sp.sections.append(dict(elset=k, material="STEEL"))
st = new_step("PinLoad")
fixed = np.unique(np.concatenate([sp.nsets[f"FACE_{h}"] for h in holes]))
st["bcs"] = [dict(nodes=fixed, dof=d, value=0.0) for d in range(3)]
st["dsloads"] = [dict(surface=f"FACE_{pin[0]}", kind="FORCE", vector=[2000.0, -3000.0, 0.0])]
sp.steps.append(st)
write_any(sp, "bracket.inp")
write_any(sp, "bracket.bdf")
model, res = run_spec(sp, "out_bracket", "bracket")
fr = res[0]["frames"][-1]
print(f"max displacement {np.linalg.norm(fr['U'], axis=1).max():.4f} mm, max von Mises {fr['MISES'].max():.1f} MPa")
