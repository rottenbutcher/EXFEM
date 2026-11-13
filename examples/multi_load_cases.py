"""Many linear load cases in one batched (vmapped) PCG solve on the device: a load envelope
for 64 tip-load directions in a single call."""
import time
import numpy as np
import gfem
from gfem import mesh
from gfem.optimize import multi_load_cases

X, c = mesh.box_hex(100, 10, 10, 40, 4, 4)
m = gfem.Model(X)
m.add_block(c, gfem.HEX8, gfem.LinearElastic(200e3, 0.3))
fix = mesh.nodes_where(X, lambda x, y, z: x < 1e-9)
tip = mesh.nodes_where(X, lambda x, y, z: x > 100 - 1e-9)
step = gfem.Step(bcs=[(fix, d, 0.0) for d in range(3)])
ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
F = np.zeros((len(ang), m.ndof))
for i, a in enumerate(ang):
    F[i, tip * 3 + 1] = 100 * np.cos(a) / len(tip)
    F[i, tip * 3 + 2] = 100 * np.sin(a) / len(tip)
t = time.time()
U = np.asarray(multi_load_cases(m, step, F))
print(f"{len(ang)} load cases solved in {time.time() - t:.2f}s")
tipd = np.linalg.norm(U.reshape(len(ang), -1, 3)[:, tip, 1:].mean(1), axis=1)
print("tip deflection envelope: min %.4f  max %.4f mm" % (tipd.min(), tipd.max()))
