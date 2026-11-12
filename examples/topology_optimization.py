"""SIMP topology optimisation of a cantilever design space on an unstructured mesh."""
import numpy as np
import gfem
from gfem import mesh
from gfem.optimize import topology_optimization
from gfem.io.vtk import write_vtu

X, c = mesh.box_hex(120, 40, 10, 48, 16, 4)
m = gfem.Model(X)
m.add_block(c, gfem.HEX8_PLAIN, gfem.LinearElastic(200e3, 0.3))
fix = mesh.nodes_where(X, lambda x, y, z: x < 1e-9)
load = mesh.nodes_where(X, lambda x, y, z: (x > 120 - 1e-9) & (np.abs(y - 20) < 3))
step = gfem.Step(bcs=[(fix, d, 0.0) for d in range(3)], loads=[(load, 1, -1000.0 / len(load))])
res = topology_optimization(m, step, volfrac=0.3, penal=3.0, rmin=4.0, iters=40)
write_vtu("topology.vtu", m, dict(density=res["density"]))
print("final compliance", res["history"][-1][1])
