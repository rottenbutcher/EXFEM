"""Verification problems against analytical solutions."""
import os
import numpy as np
import jax.numpy as jnp
import pytest
import gfem
from gfem import mesh

E, NU = 200e3, 0.3


def uniaxial_model(mat, etype=gfem.HEX8, n=2, tet=None):
    X, c = mesh.box_hex(1, 1, 1, n, n, n)
    if tet:
        X, c = mesh.hex_to_tet4(X, c)
        if tet == 10:
            X, c = mesh.tet4_to_tet10(X, c)
    m = gfem.Model(X)
    m.add_block(c, etype, mat)
    w = lambda f: mesh.nodes_where(X, f)
    bcs = [(w(lambda x, y, z: x < 1e-9), 0, 0.0), (w(lambda x, y, z: y < 1e-9), 1, 0.0),
           (w(lambda x, y, z: z < 1e-9), 2, 0.0)]
    right = w(lambda x, y, z: x > 1 - 1e-9)
    return m, bcs, right


@pytest.mark.parametrize("etype,tet", [(gfem.HEX8, None), (gfem.TET4, 4), (gfem.TET10, 10)])
def test_patch_linear(etype, tet):
    m, bcs, right = uniaxial_model(gfem.LinearElastic(E, NU), etype, tet=tet)
    st = gfem.Step(bcs=bcs + [(right, 0, 1e-3)])
    u, s, info = gfem.StaticSolver(m, verbose=False).solve(st)
    S = m.stresses(u, s)[0]["S"]
    assert np.allclose(S[:, 0, 0], E * 1e-3, rtol=1e-6)
    assert np.allclose(S[:, 1, 1], 0, atol=1e-6 * E)
    U = np.asarray(u).reshape(-1, 3)
    assert np.allclose(U[:, 1].min(), -NU * 1e-3, rtol=1e-6)


def test_j2_uniaxial_hardening():
    sy, H = 250.0, 2000.0
    mat = gfem.J2Plasticity(E, NU, [[sy, 0.0], [sy + H * 0.1, 0.1]])
    m, bcs, right = uniaxial_model(mat, n=1)
    st = gfem.Step(bcs=bcs + [(right, 0, 0.01)], dt_init=0.1, dt_max=0.1)
    u, s, info = gfem.StaticSolver(m, verbose=False).solve(st)
    sxx = float(m.stresses(u, s)[0]["S"][0, 0, 0])
    eps = 0.01
    ey = sy / E
    expected = sy + E * H / (E + H) * (eps - ey)
    assert abs(sxx - expected) / expected < 1e-4
    peeq = float(m.stresses(u, s)[0]["PEEQ"][0])
    assert abs(peeq - (eps - sxx / E)) < 1e-6


def test_neohooke_large_stretch():
    mu = 1.0
    mat = gfem.NeoHookean(C10=mu / 2, D1=2 / (2000 * mu))      # K/mu = 2000 (nearly incompressible)
    m, bcs, right = uniaxial_model(mat, n=1)
    st = gfem.Step(bcs=bcs + [(right, 0, 1.0)], dt_init=0.1, dt_max=0.2)
    u, s, _ = gfem.StaticSolver(m, verbose=False).solve(st)
    lam = 2.0
    expected = mu * (lam ** 2 - 1 / lam)                      # Cauchy, incompressible
    sxx = float(m.stresses(u, s)[0]["S"][0, 0, 0])
    assert abs(sxx - expected) / expected < 0.01


def test_cantilever_tet10_vs_beam_theory():
    L, h = 10.0, 1.0
    X, c = mesh.box_hex(L, h, h, 20, 2, 2)
    X, c = mesh.tet4_to_tet10(*mesh.hex_to_tet4(X, c))
    m = gfem.Model(X)
    m.add_block(c, gfem.TET10, gfem.LinearElastic(E, NU))
    fix = mesh.nodes_where(X, lambda x, y, z: x < 1e-9)
    tip = mesh.nodes_where(X, lambda x, y, z: x > L - 1e-9)
    P = 1.0
    st = gfem.Step(bcs=[(fix, d, 0.0) for d in range(3)], loads=[(tip, 1, -P / len(tip))])
    u, s, info = gfem.StaticSolver(m, verbose=False).solve(st)
    I = h ** 4 / 12
    d_beam = P * L ** 3 / (3 * E * I) + P * L / (5 / 6 * E / (2 * (1 + NU)) * h * h)
    d = -float(np.asarray(u).reshape(-1, 3)[tip, 1].mean())
    assert abs(d - d_beam) / d_beam < 0.03
    # reactions balance the load
    R = np.asarray(info["reactions"]).reshape(-1, 3)[:, 1].sum()
    assert abs(R - P) < 1e-6


def test_explicit_dynamic_overshoot():
    L = 1.0
    X, c = mesh.box_hex(L, 0.1, 0.1, 20, 1, 1)
    m = gfem.Model(X)
    m.add_block(c, gfem.HEX8, gfem.LinearElastic(1.0, 0.0, density=1.0))
    fix = mesh.nodes_where(X, lambda x, y, z: x < 1e-9)
    tip = mesh.nodes_where(X, lambda x, y, z: x > L - 1e-9)
    F = 1e-4
    A = 0.01
    st = gfem.Step(kind="explicit", time_period=4 * L, frames=200,
                   bcs=[(fix, d, 0.0) for d in range(3)], loads=[(tip, 0, F / len(tip))])
    peak = []
    gfem.ExplicitSolver(m, verbose=False).solve(
        st, amplitude=lambda x: 1.0,
        callback=lambda t, u, s: peak.append(float(u.reshape(-1, 3)[tip, 0].mean())))
    static = F * L / (1.0 * A)
    assert 1.8 < max(peak) / static < 2.2          # step load -> dynamic factor 2


def test_inp_reader(tmp_path):
    inp = tmp_path / "bar.inp"
    X, c = mesh.box_hex(4, 1, 1, 4, 1, 1)
    lines = ["*HEADING", "*NODE, NSET=ALL"]
    lines += [f"{i + 1}, {x[0]}, {x[1]}, {x[2]}" for i, x in enumerate(X)]
    lines += ["*ELEMENT, TYPE=C3D8, ELSET=BAR"]
    lines += [f"{e + 1}, " + ", ".join(str(n + 1) for n in r) for e, r in enumerate(c)]
    fixed = [i + 1 for i in np.nonzero(X[:, 0] < 1e-9)[0]]
    tip = [i + 1 for i in np.nonzero(X[:, 0] > 4 - 1e-9)[0]]
    lines += ["*NSET, NSET=FIX", ", ".join(map(str, fixed)), "*NSET, NSET=TIP", ", ".join(map(str, tip)),
              "*SOLID SECTION, ELSET=BAR, MATERIAL=STEEL", "*MATERIAL, NAME=STEEL",
              "*ELASTIC", f"{E}, 0.0", "*BOUNDARY", "FIX, ENCASTRE",
              "*STEP, NLGEOM=NO", "*STATIC", "1., 1.", "*CLOAD", "TIP, 1, 25.", "*END STEP"]
    inp.write_text("\n".join(lines))
    model, res = gfem.run_inp(str(inp), str(tmp_path), verbose=False)
    ux = float(res[0]["frames"][-1]["U"][np.array(tip) - 1, 0].mean())
    assert abs(ux - 100.0 * 4 / E) / (100.0 * 4 / E) < 0.02
    assert os.path.exists(res[0]["files"][-1])
