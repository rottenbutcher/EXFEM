"""Large-model workloads that are slow in commercial codes, run on the GPU.

    pip install "jax[cuda12]"
    python examples/gpu_showcase.py --n 60          # ~ 2-3 M dof on a 24-80 GB NVIDIA GPU

1. linear static, mixed-precision AMG-PCG (FP32 inner, FP64 refinement)
2. modal, 20 modes, block LOBPCG + AMG (no factorisation, memory linear in model size)
3. eigen buckling, LOBPCG
4. 256 load cases in one batched solve
Prints timings so the same script can be compared on CPU and GPU.
"""
import argparse
import time
import numpy as np
import jax
import gfem
from gfem import mesh, elements as EL
from gfem.optimize import multi_load_cases

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20, help="elements across the width (model size ~ 5 n^3 hex20)")
a = ap.parse_args()
n = a.n
X, c = mesh.hex8_to_hex20(*mesh.box_hex(10.0 * n, n, n, 10 * n, n, n))
m = gfem.Model(X)
m.add_block(c, EL.HEX20R, gfem.LinearElastic(210e3, 0.3, density=7.85e-9))
fix = mesh.nodes_where(X, lambda x, y, z: x < 1e-9)
tip = mesh.nodes_where(X, lambda x, y, z: x > 10.0 * n - 1e-9)
bcs = [(fix, d, 0.0) for d in range(3)]
print(f"device {jax.devices()}  model: {m.ndof:,} dof, {m.nelem:,} C3D20R")


def timed(label, fn):
    t = time.perf_counter()
    out = fn()
    jax.block_until_ready(out) if not isinstance(out, dict) else None
    print(f"{label:<45s} {time.perf_counter() - t:8.2f} s")
    return out


st = gfem.Step(bcs=bcs, loads=[(tip, 1, -1000.0 / len(tip))])
timed("static, mixed FP32/FP64 AMG-PCG", lambda: gfem.StaticSolver(m, verbose=False, linear_solver="mixed").solve(st)[0])
r = timed("modal 20 modes, LOBPCG + AMG", lambda: gfem.modal(m, gfem.Step(kind="frequency", bcs=bcs, nmodes=20, eigensolver="lobpcg")))
print("   f1..f3 =", np.round(r["freq"][:3], 3))
st_b = gfem.Step(kind="buckle", bcs=bcs, loads=[(tip, 0, -1.0 / len(tip))], nmodes=4, eigensolver="lobpcg", )
r = timed("buckling 4 modes, LOBPCG + AMG", lambda: gfem.buckling(m, st_b))
print("   load factors =", np.round(r["load_factors"], 2))
F = np.zeros((256, m.ndof))
ang = np.linspace(0, 2 * np.pi, 256, endpoint=False)
F[:, tip * 3 + 1] = (np.cos(ang)[:, None] / len(tip))
F[:, tip * 3 + 2] = (np.sin(ang)[:, None] / len(tip))
timed("256 load cases, batched PCG", lambda: multi_load_cases(m, gfem.Step(bcs=bcs), F))
