"""Coulomb friction (block dragged on a base), remote force and bolt pretension."""
import numpy as np
import gfem
from gfem import mesh, elements as EL
from gfem.surfaces import sub_surface
from gfem.spec import Spec, new_step
from gfem.analysis import run_spec

# friction: Rx / Rz -> mu once the block slides
X1, c1 = mesh.box_hex(2, 2, 1, 4, 4, 2)
X2, c2 = mesh.box_hex(1, 1, 1, 2, 2, 2, origin=(0.5, 0.5, 1.0))
X = np.vstack([X1, X2])
m = gfem.Model(X)
m.add_block(c1, EL.HEX8_PLAIN, gfem.LinearElastic(200e3, 0.3))
m.add_block(c2 + len(X1), EL.HEX8_PLAIN, gfem.LinearElastic(200e3, 0.3))
top1 = np.nonzero(np.abs(X1[:, 2] - 1) < 1e-9)[0]
bot2 = len(X1) + np.nonzero(np.abs(X2[:, 2] - 1) < 1e-9)[0]
sk = m.skin()
m.add_interaction(gfem.contact.PenaltyContact(X, sub_surface(sk, set(top1.tolist())), sub_surface(sk, set(bot2.tolist())),
                                              E=200e3, fkn=10.0, mu=0.3))
base = mesh.nodes_where(X, lambda x, y, z: z < 1e-9)
top = len(X1) + np.nonzero(np.abs(X2[:, 2] - 2) < 1e-9)[0]
fix = [(base, d, 0.0) for d in range(3)]
solver, st = gfem.StaticSolver(m, verbose=False), gfem.State(m)
solver.solve(gfem.Step("press", bcs=fix + [(top, 0, 0.0), (top, 1, 0.0), (top, 2, -2e-3)], dt_init=0.5), state=st)
u, s, info = solver.solve(gfem.Step("slide", bcs=fix + [(top, 0, 0.02), (top, 1, 0.0), (top, 2, -2e-3)],
                                    dt_init=0.1, dt_max=0.1), state=st)
R = np.asarray(info["reactions"]).reshape(-1, 3)
print(f"friction: Rx/Rz = {abs(R[top, 0].sum()) / abs(R[top, 2].sum()):.4f}  (mu = 0.3)")

# bolt pretension: bolt in series with a clamped member, target 1000 N
Xb, cb = mesh.box_hex(10, 2, 2, 5, 2, 2)
Xm, cm = mesh.box_hex(10, 4, 4, 5, 2, 2, origin=(10, -1, -1))
XX = np.vstack([Xb, Xm])
sp = Spec("bolt")
sp.add_nodes(XX)
sp.add_elements("C3D8", cb, elset="BOLT")
sp.add_elements("C3D8", cm + len(Xb), elset="MEMBER")
sp.materials["S"] = dict(E=200e3, nu=0.0)
sp.sections += [dict(elset="BOLT", material="S"), dict(elset="MEMBER", material="S")]
joint = np.concatenate([np.nonzero(Xb[:, 0] > 10 - 1e-9)[0], len(Xb) + np.nonzero(Xm[:, 0] < 10 + 1e-9)[0]])
sp.constraints.append(dict(type="rigid", nodes=joint, name="J"))
step = new_step("pre")
ends = np.concatenate([np.nonzero(XX[:, 0] < 1e-9)[0], len(Xb) + np.nonzero(Xm[:, 0] > 20 - 1e-9)[0]])
step["bcs"] = [dict(nodes=ends, dof=d, value=0.0) for d in range(3)]
step["bolts"] = [dict(elset="BOLT", axis=[1, 0, 0], force=1000.0)]
sp.steps.append(step)
_, res = run_spec(sp, "out_bolt", "bolt", verbose=False, write=False)
print(f"bolt pretension: section force = {res[0]['info']['bolt_forces']['bolt1']:.2f} N (target 1000)")
