"""Throughput benchmark of the device kernels: internal force, element tangents, EbE K*v,
matrix-free AD-JVP K*v, and one AMG-PCG solve. Run once on CPU and once after
`pip install "jax[cuda12]"` to see the GPU speed-up:   python examples/benchmark.py 10 20 30"""
import sys
import time
import jax
import jax.numpy as jnp
import gfem, os
ET = os.environ.get("ET", "HEX8")
from gfem import mesh


def timeit(fn, n=3):
    jax.block_until_ready(fn())
    t = time.perf_counter()
    for _ in range(n):
        jax.block_until_ready(fn())
    return (time.perf_counter() - t) / n


def bench(n):
    X, c = mesh.box_hex(n, n, n, n, n, n)
    m = gfem.Model(X)
    m.add_block(c, getattr(gfem, ET), gfem.J2Plasticity(210e3, 0.3, [[250, 0], [400, 0.2]]))
    s = m.init_state()
    u = jax.random.normal(jax.random.PRNGKey(0), (m.ndof,)) * 1e-3
    f = jax.jit(lambda u: m.internal_force(u, s)[0])
    k = jax.jit(lambda u: m.element_stiffness(u, s))
    Kes = k(u)
    mv = jax.jit(lambda v: m.matvec(Kes, v))
    jv = jax.jit(lambda v: m.tangent_jvp(u, s, v))
    r = dict(fint=timeit(lambda: f(u)), Ke=timeit(lambda: k(u)), Kv=timeit(lambda: mv(u), 10), jvp=timeit(lambda: jv(u), 10))
    print(f"{c.shape[0]:>8d} hex8 {m.ndof:>9d} dofs | fint {r['fint']*1e3:8.2f} ms | Ke {r['Ke']*1e3:8.2f} ms | "
          f"EbE K*v {r['Kv']*1e3:7.2f} ms | JVP K*v {r['jvp']*1e3:7.2f} ms", flush=True)
    amg_solve(X, m, s)


def amg_solve(X, m, s):
    """One AMG-preconditioned CG solve (linear elastic tangent, one face fixed)."""
    import numpy as np
    from gfem import linalg
    from gfem.model import ops_assemble
    ops = m.stiffness_ops(jnp.zeros(m.ndof), s)
    free = np.ones(m.ndof)
    for d in range(3):
        free[np.asarray(mesh.nodes_where(X, lambda x, y, z: x < 1e-9)) * 3 + d] = 0
    A = linalg.bc_matrix(ops_assemble(ops, m.ndof), free)
    t0 = time.perf_counter()
    amg = linalg.AMG(A, np.asarray(X))
    setup = time.perf_counter() - t0
    pre, prm, Ad = amg.make(), amg.params(), linalg.csr_to_device(A)
    n = m.ndof
    solve = jax.jit(lambda Ad, p, b: linalg.pcg(lambda v: linalg.csr_mv(Ad, v, n), b, lambda r: pre(p, r), tol=1e-8))
    b = jnp.asarray(free * np.random.default_rng(0).normal(size=n))
    x, k, res = solve(Ad, prm, b)
    t = timeit(lambda: solve(Ad, prm, b)[0])
    print(f"{'':>8s} AMG-PCG: setup {setup*1e3:7.0f} ms (host) | solve {t*1e3:8.1f} ms | {int(k)} its | "
          f"{t / max(int(k), 1) * 1e3:6.2f} ms/it", flush=True)


if __name__ == "__main__":
    print("device:", jax.devices())
    for n in map(int, sys.argv[1:] or [10, 20, 30]):
        bench(n)
