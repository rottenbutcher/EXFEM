"""Shells (MITC4, geometrically exact) and beams (Simo-Reissner) against reference solutions."""
import numpy as np
import pytest
import gfem
from gfem import mesh, elements as EL

E, NU = 200e3, 0.3


def _plate(L, W, nx, ny, z=0.0):
    xs, ys = np.linspace(0, L, nx + 1), np.linspace(0, W, ny + 1)
    X = np.array([[x, y, z] for x in xs for y in ys])
    nid = lambda i, j: i * (ny + 1) + j
    c = np.array([[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)] for i in range(nx) for j in range(ny)])
    return X, c


def _line(P0, P1, n):
    X = np.linspace(P0, P1, n + 1)
    return X, np.stack([np.arange(n), np.arange(1, n + 1)], 1)


@pytest.mark.parametrize("tri", [False, True])
def test_shell_cantilever_plate(tri):
    L, W, h = 100.0, 10.0, 1.0
    X, c = _plate(L, W, 20, 2)
    if tri:
        c = np.vstack([c[:, [0, 1, 2]], c[:, [0, 2, 3]]])
    m = gfem.Model(X)
    m.add_shell(c, gfem.LinearElastic(E, 0.0), h)
    fix = np.nonzero(X[:, 0] < 1e-9)[0]
    tip = np.nonzero(X[:, 0] > L - 1e-9)[0]
    u, _, _ = gfem.StaticSolver(m, verbose=False).solve(gfem.Step(bcs=[(fix, d, 0.0) for d in range(6)],
                                                                  loads=[(tip, 2, 1.0 / len(tip))]))
    d = np.asarray(u).reshape(-1, 3)[tip, 2].mean()
    exact = L ** 3 / (3 * E * W * h ** 3 / 12)
    assert abs(d - exact) / exact < (0.02 if not tri else 0.15)


def test_shell_curved_strip_and_beam():
    R, b, t, n = 10.0, 1.0, 0.25, 32
    ph = np.linspace(0, np.pi / 2, n + 1)
    X = np.array([[R * np.sin(p), y, R * np.cos(p)] for p in ph for y in (0.0, b)])
    c = np.array([[2 * i, 2 * (i + 1), 2 * (i + 1) + 1, 2 * i + 1] for i in range(n)])
    m = gfem.Model(X)
    m.add_shell(c, gfem.LinearElastic(E, 0.0), t)
    u, _, _ = gfem.StaticSolver(m, verbose=False).solve(gfem.Step(bcs=[([0, 1], d, 0.) for d in range(6)],
                                                                  loads=[([2 * n, 2 * n + 1], 0, 0.5)]))
    exact = R ** 3 * np.pi / 4 / (E * b * t ** 3 / 12)
    assert abs(np.asarray(u).reshape(-1, 3)[2 * n, 0] - exact) / exact < 0.01
    Xb = np.array([[R * np.sin(p), 0, R * np.cos(p)] for p in ph])
    m = gfem.Model(Xb)
    m.add_beam(np.stack([np.arange(n), np.arange(1, n + 1)], 1), gfem.LinearElastic(E, 0.0), ("RECT", (b, t)), n1=[0, 1, 0])
    u, _, _ = gfem.StaticSolver(m, verbose=False).solve(gfem.Step(bcs=[([0], d, 0.) for d in range(6)], loads=[([n], 0, 1.0)]))
    assert abs(np.asarray(u).reshape(-1, 3)[n, 0] - exact) / exact < 0.01


def test_scordelis_lo_roof():
    R, Lh, phi0, n = 25.0, 25.0, np.radians(40), 12
    xs, ph = np.linspace(0, Lh, n + 1), np.linspace(0, phi0, n + 1)
    X = np.array([[x, R * np.sin(p), R * np.cos(p)] for x in xs for p in ph])
    nid = lambda i, j: i * (n + 1) + j
    c = np.array([[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)] for i in range(n) for j in range(n)])
    m = gfem.Model(X)
    m.add_shell(c, gfem.LinearElastic(4.32e8, 0.0), 0.25)
    sx = np.nonzero(X[:, 0] < 1e-9)[0]
    dia = np.nonzero(X[:, 0] > Lh - 1e-9)[0]
    cr = np.nonzero(np.abs(X[:, 1]) < 1e-9)[0]
    bcs = [(sx, 0, 0.), (sx, 4, 0.), (sx, 5, 0.), (dia, 1, 0.), (dia, 2, 0.), (cr, 1, 0.), (cr, 3, 0.), (cr, 5, 0.)]
    w = np.zeros(len(X))
    for row in c:
        w[row] += R * phi0 / n * Lh / n / 4
    f = np.zeros(m.ndof)
    f[np.arange(len(X)) * 3 + 2] = -90 * w
    u, _, _ = gfem.StaticSolver(m, verbose=False).solve(gfem.Step(bcs=bcs, dloads=[(f,)]))
    uz = np.asarray(u).reshape(-1, 3)[nid(0, n), 2]
    assert abs(uz + 0.3024) / 0.3024 < 0.03


def test_beam_rollup_large_rotation():
    L, n = 100.0, 20
    X, c = _line([0, 0, 0], [L, 0, 0], n)
    m = gfem.Model(X)
    m.add_beam(c, gfem.StVenantKirchhoff(E, NU), ("RECT", (10.0, 1.0)), n1=[0, 1, 0])
    M = 2 * np.pi * E * 10 / 12 / L
    st = gfem.Step(bcs=[([0], d, 0.0) for d in range(6)], loads=[([n], 4, -M)], dt_init=0.05, dt_max=0.1)
    u, _, _ = gfem.StaticSolver(m, verbose=False).solve(st)
    U = np.asarray(u).reshape(-1, 3)
    assert np.linalg.norm(U[n] - [-L, 0, 0]) < 1e-3 * L          # closes into a full circle
    assert abs(U[m.rot_of[n], 1] + 2 * np.pi) < 1e-4


def test_plate_and_column_buckling():
    a, t, n = 100.0, 1.0, 10
    X, c = _plate(a, a, n, n)
    m = gfem.Model(X)
    m.add_shell(c, gfem.LinearElastic(E, NU), t)
    edge = np.nonzero((X[:, 0] < 1e-9) | (X[:, 0] > a - 1e-9) | (X[:, 1] < 1e-9) | (X[:, 1] > a - 1e-9))[0]
    left, right = np.nonzero(X[:, 0] < 1e-9)[0], np.nonzero(X[:, 0] > a - 1e-9)[0]
    w = np.full(len(right), 1.0 / n)
    w[[0, -1]] *= 0.5
    loads = [([nd], 0, -wi) for nd, wi in zip(right, w)]
    bcs = [(edge, 2, 0.), (left, 0, 0.), ([0], 1, 0.), (np.arange(len(X)), 5, 0.)]
    r = gfem.buckling(m, gfem.Step(kind="buckle", bcs=bcs, loads=loads, nmodes=2))
    D = E * t ** 3 / (12 * (1 - NU ** 2))
    assert abs(r["load_factors"][0] - 4 * np.pi ** 2 * D / a) / (4 * np.pi ** 2 * D / a) < 0.03
    L = 400.0
    X, c = _line([0, 0, 0], [0, 0, L], 20)
    m = gfem.Model(X)
    m.add_beam(c, gfem.LinearElastic(E, NU), ("RECT", (20., 20.)), n1=[1, 0, 0])
    r = gfem.buckling(m, gfem.Step(kind="buckle", bcs=[([0], d, 0.) for d in range(6)], loads=[([20], 2, -1.0)], nmodes=2))
    Pcr = np.pi ** 2 * E * 20 ** 4 / 12 / (4 * L ** 2)
    assert abs(r["load_factors"][0] - Pcr) / Pcr < 0.01


def test_shell_modal_simply_supported_plate():
    a, t, n, rho = 100.0, 1.0, 12, 7.85e-9
    X, c = _plate(a, a, n, n)
    m = gfem.Model(X)
    m.add_shell(c, gfem.LinearElastic(E, NU, density=rho), t)
    edge = np.nonzero((X[:, 0] < 1e-9) | (X[:, 0] > a - 1e-9) | (X[:, 1] < 1e-9) | (X[:, 1] > a - 1e-9))[0]
    bcs = [(edge, 2, 0.), (np.arange(len(X)), 0, 0.), (np.arange(len(X)), 1, 0.), (np.arange(len(X)), 5, 0.)]
    res = gfem.modal(m, gfem.Step(kind="frequency", bcs=bcs, nmodes=3))
    D = E * t ** 3 / (12 * (1 - NU ** 2))
    f11 = np.pi / 2 * (2 / a ** 2) * np.sqrt(D / (rho * t))
    assert abs(res["freq"][0] - f11) / f11 < 0.02


def test_shell_beam_inp_roundtrip(tmp_path):
    """Stiffened plate (S4 + B31 sharing nodes), pressure on SPOS, through the Abaqus reader/writer."""
    from gfem.io.inp import read_inp_spec, write_inp
    from gfem.analysis import run_spec
    X, c = _plate(100, 50, 10, 5)
    stiff = [i for i in range(len(X)) if abs(X[i, 1] - 20) < 1e-9]
    L = ["*NODE"] + [f"{i + 1}, {x[0]}, {x[1]}, {x[2]}" for i, x in enumerate(X)]
    L += ["*ELEMENT, TYPE=S4R, ELSET=PLATE"] + [f"{e + 1}, " + ", ".join(str(k + 1) for k in r) for e, r in enumerate(c)]
    L += ["*ELEMENT, TYPE=B31, ELSET=STIFF"] + [f"{1000 + k}, {a + 1}, {b + 1}" for k, (a, b) in enumerate(zip(stiff[:-1], stiff[1:]))]
    root = [i + 1 for i in range(len(X)) if X[i, 0] < 1e-9]
    L += ["*NSET, NSET=ROOT", ", ".join(map(str, root)), "*SURFACE, NAME=TOP", "PLATE, SPOS",
          "*MATERIAL, NAME=AL", "*ELASTIC", "70000., 0.33", "*SHELL SECTION, ELSET=PLATE, MATERIAL=AL", "2.0, 5",
          "*BEAM SECTION, ELSET=STIFF, MATERIAL=AL, SECTION=RECT", "4., 10.", "0., 1., 0.",
          "*BOUNDARY", "ROOT, ENCASTRE", "*STEP", "*STATIC", "1., 1.", "*DSLOAD", "TOP, P, 0.01", "*END STEP"]
    p = tmp_path / "sp.inp"
    p.write_text("\n".join(L))
    sp = read_inp_spec(str(p))
    _, r1 = run_spec(sp, str(tmp_path), "a", verbose=False, write=True)
    write_inp(sp, str(tmp_path / "sp2.inp"))
    _, r2 = run_spec(read_inp_spec(str(tmp_path / "sp2.inp")), str(tmp_path), "b", verbose=False, write=False)
    U1, U2 = r1[0]["frames"][-1]["U"], r2[0]["frames"][-1]["U"]
    assert U1.shape == (len(X), 3) and np.abs(U1[:, 2]).max() > 0
    assert np.abs(U1 - U2).max() < 1e-9 * np.abs(U1).max()
    # the stiffener reduces the deflection compared with the bare plate
    sp.sections = [s for s in sp.sections if s["elset"] != "STIFF"]
    sp.groups = [g for g in sp.groups if g.etype != "B31"]
    sp.elsets.pop("STIFF")
    _, r3 = run_spec(sp, str(tmp_path), "c", verbose=False, write=False)
    assert np.abs(r3[0]["frames"][-1]["U"][:, 2]).max() > 1.2 * np.abs(U1[:, 2]).max()
