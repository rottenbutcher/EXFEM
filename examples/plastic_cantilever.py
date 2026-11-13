"""Large-deflection elasto-plastic cantilever (NLGEOM + J2, hex8 F-bar), load-unload in 2 steps."""
import numpy as np
import gfem
from gfem import mesh
from gfem.io import write_vtu

L, h = 100.0, 10.0
X, c = mesh.box_hex(L, h, h, 40, 4, 4)
m = gfem.Model(X)
steel = gfem.J2FiniteStrain(210e3, 0.3, [[250, 0.0], [400, 0.2]])
m.add_block(c, gfem.HEX8, steel)

fix = mesh.nodes_where(X, lambda x, y, z: x < 1e-9)
tip = mesh.nodes_where(X, lambda x, y, z: x > L - 1e-9)
fixbc = [(fix, d, 0.0) for d in range(3)]
solver = gfem.StaticSolver(m)

load = gfem.Step("Load", bcs=fixbc + [(tip, 1, -30.0)], dt_init=0.1, dt_max=0.2)
u, s, info = solver.solve(load)
print("tip uy after loading :", np.asarray(u).reshape(-1, 3)[tip, 1].mean())
st = m.stresses(u, s)
print("max Mises:", float(st[0]["MISES"].max()), " max PEEQ:", float(st[0]["PEEQ"].max()))
write_vtu("cantilever_load.vtu", m, u, st)

# unload: release the prescribed tip displacement -> elastic springback, residual plastic set
unload = gfem.Step("Unload", bcs=fixbc, dt_init=0.25, dt_max=0.5)
fint, _ = m.internal_force(u, s)
u2, s2, _ = solver.solve(unload, u, s, fext0=fint * (1 - np.isin(np.arange(m.ndof) // 3, fix)))
print("residual tip uy      :", np.asarray(u2).reshape(-1, 3)[tip, 1].mean())
write_vtu("cantilever_unload.vtu", m, u2, m.stresses(u2, s2))
