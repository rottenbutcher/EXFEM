"""Throughput benchmark of the device kernels: internal force, element tangents, EbE K*v,
matrix-free AD-JVP K*v, and one AMG-PCG solve. Run once on CPU and once after
`pip install "jax[cuda12]"` to see the GPU speed-up:   python examples/benchmark.py 10 20 30"""
import sys
import time
import jax
import jax.numpy as jnp
import gfem
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
    m.add_block(c, gfem.HEX8, gfem.J2Plasticity(210e3, 0.3, [[250, 0], [400, 0.2]]))
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


if __name__ == "__main__":
    print("device:", jax.devices())
    for n in map(int, sys.argv[1:] or [10, 20, 30]):
        bench(n)
