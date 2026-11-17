"""Device code paths: LOBPCG eigensolver, mixed-precision AMG-PCG, device direct solve,
multi-device sharding (emulated on CPU)."""
import os
import subprocess
import sys
import numpy as np
import pytest
import gfem
from gfem import mesh, elements as EL


def _beam(n=20):
    X, c = mesh.hex8_to_hex20(*mesh.box_hex(10.0, 1.0, 1.0, n, 2, 2))
    m = gfem.Model(X)
    m.add_block(c, EL.HEX20R, gfem.LinearElastic(200e3, 0.3, density=7.85e-9))
    return m, X


def test_lobpcg_modal_matches_lanczos():
    m, X = _beam()
    fix = mesh.nodes_where(X, lambda x, y, z: x < 1e-9)
    f = {}
    for meth in ("lanczos", "lobpcg"):
        r = gfem.modal(m, gfem.Step(kind="frequency", bcs=[(fix, d, 0.) for d in range(3)], nmodes=5, eigensolver=meth))
        f[meth] = r["freq"]
    assert np.allclose(f["lanczos"], f["lobpcg"], rtol=1e-6)


def test_lobpcg_buckling_matches_lanczos_with_shells():
    a, t, n = 100.0, 1.0, 8
    xs = np.linspace(0, a, n + 1)
    X = np.array([[x, y, 0.] for x in xs for y in xs])
    nid = lambda i, j: i * (n + 1) + j
    c = np.array([[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)] for i in range(n) for j in range(n)])
    m = gfem.Model(X)
    m.add_shell(c, gfem.LinearElastic(200e3, 0.3), t)
    edge = np.nonzero((X[:, 0] < 1e-9) | (X[:, 0] > a - 1e-9) | (X[:, 1] < 1e-9) | (X[:, 1] > a - 1e-9))[0]
    left, right = np.nonzero(X[:, 0] < 1e-9)[0], np.nonzero(X[:, 0] > a - 1e-9)[0]
    bcs = [(edge, 2, 0.), (left, 0, 0.), ([0], 1, 0.), (np.arange(len(X)), 5, 0.)]
    loads = [(right, 0, -1.0 / len(right))]
    lam = {}
    for meth in ("lanczos", "lobpcg"):
        lam[meth] = gfem.buckling(m, gfem.Step(kind="buckle", bcs=bcs, loads=loads, nmodes=2, eigensolver=meth))["load_factors"]
    assert np.allclose(lam["lanczos"], lam["lobpcg"], rtol=1e-5)


@pytest.mark.parametrize("ls", ["mixed", "gpu_direct", "amg", "jacobi"])
def test_linear_solver_backends_agree(ls):
    m, X = _beam(10)
    fix = mesh.nodes_where(X, lambda x, y, z: x < 1e-9)
    tip = mesh.nodes_where(X, lambda x, y, z: x > 10 - 1e-9)
    st = gfem.Step(bcs=[(fix, d, 0.) for d in range(3)], loads=[(tip, 1, -1.0 / len(tip))])
    u0 = np.asarray(gfem.StaticSolver(m, verbose=False, linear_solver="direct").solve(st)[0])
    u1 = np.asarray(gfem.StaticSolver(m, verbose=False, linear_solver=ls).solve(st)[0])
    assert np.abs(u1 - u0).max() < 1e-7 * np.abs(u0).max()


def test_multi_device_sharding():
    code = r'''
import numpy as np, jax, gfem
from gfem import mesh, elements as EL
assert jax.device_count() == 4
X, c = mesh.box_hex(10, 1, 1, 20, 2, 2)
m = gfem.Model(X); m.add_block(c, EL.HEX8, gfem.LinearElastic(200e3, 0.3, density=7.85e-9))
fix = mesh.nodes_where(X, lambda x, y, z: x < 1e-9); tip = mesh.nodes_where(X, lambda x, y, z: x > 10 - 1e-9)
st = gfem.Step(bcs=[(fix, d, 0.) for d in range(3)], loads=[(tip, 1, -1.0 / len(tip))])
u0 = np.asarray(gfem.StaticSolver(m, verbose=False, linear_solver="direct").solve(st)[0])
u1 = np.asarray(gfem.StaticSolver(m, verbose=False, linear_solver="amg").solve(st)[0])
assert np.abs(u1 - u0).max() < 1e-7 * np.abs(u0).max()
r0 = gfem.modal(m, gfem.Step(kind="frequency", bcs=[(fix, d, 0.) for d in range(3)], nmodes=3, eigensolver="lanczos"))
r1 = gfem.modal(m, gfem.Step(kind="frequency", bcs=[(fix, d, 0.) for d in range(3)], nmodes=3, eigensolver="lobpcg"))
assert np.allclose(r0["freq"], r1["freq"], rtol=1e-6)
print("OK")
'''
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=4", JAX_PLATFORMS="cpu")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)
    assert "OK" in r.stdout, r.stderr[-2000:]
