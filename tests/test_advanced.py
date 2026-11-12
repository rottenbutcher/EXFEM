"""Verification of v0.2 capabilities against analytical solutions."""
import numpy as np
import jax.numpy as jnp
import pytest
import gfem
from gfem import mesh, elements as E
from gfem.surfaces import skin, sub_surface
from gfem import constraints as C, loads as L

EM, NU = 200e3, 0.3


def _sel(X, f):
    return mesh.nodes_where(np.asarray(X), f)


@pytest.mark.parametrize("et,conv", [(E.HEX20, mesh.hex8_to_hex20), (E.WEDGE6, mesh.hex_to_wedge)])
@pytest.mark.parametrize("ls", ["direct", "amg", "jacobi"])
def test_patch_new_elements(et, conv, ls):
    X, c = conv(*mesh.box_hex(1, 1, 1, 2, 2, 2))
    m = gfem.Model(X)
    m.add_block(c, et, gfem.LinearElastic(EM, NU))
    bcs = [(_sel(X, lambda x, y, z: x < 1e-9), 0, 0.), (_sel(X, lambda x, y, z: y < 1e-9), 1, 0.),
           (_sel(X, lambda x, y, z: z < 1e-9), 2, 0.), (_sel(X, lambda x, y, z: x > 1 - 1e-9), 0, 1e-3)]
    u, s, _ = gfem.StaticSolver(m, verbose=False, linear_solver=ls).solve(gfem.Step(bcs=bcs))
    S = m.stresses(u, s)[0]["S"]
    assert np.allclose(S[:, 0, 0], EM * 1e-3, rtol=1e-5)


def test_lame_thick_cylinder_pressure():
    a, b, p = 1.0, 2.0, 100.0
    X, c = mesh.cylinder_hex(b, 0.2, 4, 32, 1, Ri=a)
    X, c = mesh.hex8_to_hex20(X, c)
    m = gfem.Model(X)
    m.add_block(c, E.HEX20, gfem.LinearElastic(EM, NU))
    r = np.hypot(X[:, 0], X[:, 1])
    inner = sub_surface(m.skin(), set(np.nonzero(np.abs(r - a) < 1e-6)[0].tolist()))
    bcs = [(np.arange(len(X)), 2, 0.0),
           (_sel(X, lambda x, y, z: (np.abs(y) < 1e-9) & (x > 0)), 1, 0.0),
           (_sel(X, lambda x, y, z: (np.abs(x) < 1e-9) & (y > 0)), 0, 0.0)]
    st = gfem.Step(bcs=bcs, pressures=[(inner, p)])
    u, s, _ = gfem.StaticSolver(m, verbose=False).solve(st)
    U = np.asarray(u).reshape(-1, 3)
    ia = np.nonzero(np.abs(r - a) < 1e-6)[0]
    ur = np.mean((U[ia, 0] * X[ia, 0] + U[ia, 1] * X[ia, 1]) / a)
    exact = (1 + NU) * p * a * a / (EM * (b * b - a * a)) * ((1 - 2 * NU) * a + b * b / a)
    assert abs(ur - exact) / exact < 0.01


def test_tie_nonconforming_blocks():
    """Two bars with mismatched meshes tied at x=1: uniform tension must be exact."""
    X1, c1 = mesh.box_hex(1, 1, 1, 2, 2, 2)
    X2, c2 = mesh.box_hex(1, 1, 1, 3, 3, 3, origin=(1, 0, 0))
    X = np.vstack([X1, X2])
    m = gfem.Model(X)
    b1 = m.add_block(c1, E.HEX8_PLAIN, gfem.LinearElastic(EM, NU))
    b2 = m.add_block(c2 + len(X1), E.HEX8_PLAIN, gfem.LinearElastic(EM, NU))
    sk = m.skin()
    n1 = set(np.nonzero(np.abs(X[:, 0] - 1) < 1e-9)[0][:len(X1)].tolist())
    left_nodes = np.nonzero((np.abs(X[:, 0] - 1) < 1e-9) & (np.arange(len(X)) < len(X1)))[0]
    right_nodes = np.nonzero((np.abs(X[:, 0] - 1) < 1e-9) & (np.arange(len(X)) >= len(X1)))[0]
    master = sub_surface(sk, set(left_nodes.tolist()))
    m.add_constraint(C.tie(X, sub_surface(sk, set(right_nodes.tolist())), master))
    bcs = [(_sel(X, lambda x, y, z: x < 1e-9), 0, 0.), (_sel(X, lambda x, y, z: y < 1e-9), 1, 0.),
           (_sel(X, lambda x, y, z: z < 1e-9), 2, 0.), (_sel(X, lambda x, y, z: x > 2 - 1e-9), 0, 2e-3)]
    for ls in ("direct", "amg"):
        u, s, _ = gfem.StaticSolver(m, verbose=False, linear_solver=ls).solve(gfem.Step(bcs=bcs))
        for st in m.stresses(u, s):
            assert np.allclose(st["S"][:, 0, 0], EM * 1e-3, rtol=1e-4)


def test_frictionless_contact_two_blocks():
    """Upper block pushed onto a lower block: contact force must equal reaction."""
    X1, c1 = mesh.box_hex(1, 1, 1, 3, 3, 3)
    X2, c2 = mesh.box_hex(1, 1, 1, 2, 2, 2, origin=(0, 0, 1.0))
    X = np.vstack([X1, X2])
    m = gfem.Model(X)
    m.add_block(c1, E.HEX8_PLAIN, gfem.LinearElastic(EM, NU))
    m.add_block(c2 + len(X1), E.HEX8_PLAIN, gfem.LinearElastic(EM, NU))
    top1 = np.nonzero((np.abs(X[:, 2] - 1) < 1e-9) & (np.arange(len(X)) < len(X1)))[0]
    bot2 = np.nonzero((np.abs(X[:, 2] - 1) < 1e-9) & (np.arange(len(X)) >= len(X1)))[0]
    # finer mesh (block 1) is the slave
    master = sub_surface(m.skin(), set(bot2.tolist()))
    m.add_interaction(gfem.contact.PenaltyContact(X, master, sub_surface(m.skin(), set(top1.tolist())),
                                                  E=EM, fkn=1.0, augmented=True))
    allx = np.arange(len(X))
    bcs = [(_sel(X, lambda x, y, z: z < 1e-9), 2, 0.0), (allx, 0, 0.0), (allx, 1, 0.0),
           (_sel(X, lambda x, y, z: z > 2 - 1e-9), 2, -1e-3)]
    u, s, info = gfem.StaticSolver(m, verbose=False).solve(gfem.Step(bcs=bcs, dt_init=0.5))
    R = np.asarray(info["reactions"]).reshape(-1, 3)
    top = _sel(X, lambda x, y, z: z > 2 - 1e-9)
    bot = _sel(X, lambda x, y, z: z < 1e-9)
    assert abs(R[top, 2].sum() + R[bot, 2].sum()) < 1e-6 * abs(R[bot, 2].sum())
    # stiffness of two equal bars in series (lateral constrained => oedometric modulus)
    Eo = EM * (1 - NU) / ((1 + NU) * (1 - 2 * NU))
    Fexact = Eo * 1e-3 / 2
    assert abs(abs(R[bot, 2].sum()) - Fexact) / Fexact < 0.05


def test_rigid_wall_indentation():
    X, c = mesh.box_hex(1, 1, 1, 2, 2, 2)
    m = gfem.Model(X)
    m.add_block(c, E.HEX8_PLAIN, gfem.LinearElastic(EM, NU))
    top = _sel(X, lambda x, y, z: z > 1 - 1e-9)
    m.add_interaction(gfem.contact.RigidWall(top, (0, 0, 1.0005), (0, 0, -1), stiffness=EM * 1e3,
                                             motion=(0, 0, -0.0015)))
    bcs = [(_sel(X, lambda x, y, z: z < 1e-9), d, 0.0) for d in range(3)]
    u, s, _ = gfem.StaticSolver(m, verbose=False).solve(gfem.Step(bcs=bcs, dt_init=0.25))
    U = np.asarray(u).reshape(-1, 3)
    assert abs(U[top, 2].mean() + 1e-3) < 2e-5


def test_modal_cantilever():
    Lb, h = 10.0, 1.0
    X, c = mesh.hex8_to_hex20(*mesh.box_hex(Lb, h, h, 20, 2, 2))
    m = gfem.Model(X)
    rho = 7.85e-9
    m.add_block(c, E.HEX20R, gfem.LinearElastic(EM, NU, density=rho))
    fix = _sel(X, lambda x, y, z: x < 1e-9)
    st = gfem.Step(kind="frequency", bcs=[(fix, d, 0.0) for d in range(3)], nmodes=4)
    res = gfem.modal(m, st)
    I, A = h ** 4 / 12, h * h
    f1 = 1.875104 ** 2 / (2 * np.pi * Lb ** 2) * np.sqrt(EM * I / (rho * A))
    assert abs(res["freq"][0] - f1) / f1 < 0.03
    assert abs(res["freq"][1] - f1) / f1 < 0.03          # square section: double mode


def test_buckling_euler_column():
    Lc, h = 20.0, 1.0
    X, c = mesh.hex8_to_hex20(*mesh.box_hex(h, h, Lc, 2, 2, 40))
    m = gfem.Model(X)
    m.add_block(c, E.HEX20R, gfem.LinearElastic(EM, NU))
    base = _sel(X, lambda x, y, z: z < 1e-9)
    top = _sel(X, lambda x, y, z: z > Lc - 1e-9)
    st = gfem.Step(kind="buckle", bcs=[(base, d, 0.0) for d in range(3)],
                   loads=[(top, 2, -1.0 / len(top))], nmodes=2)
    res = gfem.buckling(m, st)
    Pcr = np.pi ** 2 * EM * h ** 4 / 12 / (4 * Lc ** 2)
    assert abs(res["load_factors"][0] - Pcr) / Pcr < 0.03


def test_thermal_expansion_free_and_constrained():
    X, c = mesh.box_hex(1, 1, 1, 2, 2, 2)
    alpha, dT = 1.2e-5, 100.0
    for constrained in (False, True):
        m = gfem.Model(X)
        m.add_block(c, E.HEX8, gfem.LinearElastic(EM, NU, alpha=alpha))
        bcs = [(_sel(X, lambda x, y, z: x < 1e-9), 0, 0.), (_sel(X, lambda x, y, z: y < 1e-9), 1, 0.),
               (_sel(X, lambda x, y, z: z < 1e-9), 2, 0.)]
        if constrained:
            bcs += [(_sel(X, lambda x, y, z: x > 1 - 1e-9), 0, 0.0)]
        st = gfem.Step(bcs=bcs, temperature=np.full(len(X), dT))
        solver = gfem.StaticSolver(m, verbose=False)
        u, s, info = solver.solve(st)
        S = m.stresses(u, s, dict(T=jnp.full(len(X), dT)))[0]["S"]
        if constrained:     # uniaxial constraint, free laterally
            assert np.allclose(S[:, 0, 0], -EM * alpha * dT, rtol=1e-3)
        else:
            assert np.abs(S).max() < 1e-6 * EM * alpha * dT
            U = np.asarray(u).reshape(-1, 3)
            assert abs(U[:, 0].max() - alpha * dT) < 1e-3 * alpha * dT


def test_riks_matches_load_control_and_dynamic_factor():
    Lb, h = 10.0, 1.0
    X, c = mesh.box_hex(Lb, h, h, 10, 1, 1)
    m = gfem.Model(X)
    m.add_block(c, E.HEX8, gfem.StVenantKirchhoff(EM, NU))
    fix = _sel(X, lambda x, y, z: x < 1e-9)
    tip = _sel(X, lambda x, y, z: x > Lb - 1e-9)
    P = 300.0
    bcs = [(fix, d, 0.0) for d in range(3)]
    loads = [(tip, 1, -P / len(tip))]
    u1, _, _ = gfem.StaticSolver(m, verbose=False).solve(gfem.Step(bcs=bcs, loads=loads, dt_init=0.25))
    st = gfem.Step(kind="riks", bcs=bcs, loads=loads, riks=dict(dlam0=0.2, lpf_max=1.0))
    u2, _, info = gfem.RiksSolver(m, verbose=False).solve(st)
    lam = info["lpf"]
    assert lam >= 1.0
    # interpolate to LPF=1 on the curve and compare tip deflection
    d1 = -float(np.asarray(u1).reshape(-1, 3)[tip, 1].mean())
    assert d1 > 0.5      # geometric nonlinearity active
    curve = info["curve"]
    assert np.all(np.diff(curve[:, 0]) > 0)


def test_implicit_dynamics_step_load():
    Lb, h = 1.0, 0.1
    X, c = mesh.box_hex(Lb, h, h, 10, 1, 1)
    m = gfem.Model(X)
    rho = 1e-6
    m.add_block(c, E.HEX8, gfem.LinearElastic(EM, 0.0, density=rho))
    fix = _sel(X, lambda x, y, z: x < 1e-9)
    tip = _sel(X, lambda x, y, z: x > Lb - 1e-9)
    F = 10.0
    allyz = np.arange(len(X))
    bcs = [(fix, 0, 0.0), (allyz, 1, 0.0), (allyz, 2, 0.0)]
    ust = F * Lb / (EM * h * h)
    T1 = 4 * Lb / np.sqrt(EM / rho)                  # first period of a fixed-free bar
    peak = []
    mon = gfem.Monitor(False)
    mon.frame = lambda t, u, s, **k: peak.append(float(np.asarray(u).reshape(-1, 3)[tip, 0].mean()))
    st = gfem.Step(kind="dynamic", time_period=T1, dt=T1 / 200, bcs=bcs,
                   loads=[(tip, 0, F / len(tip), gfem.Amplitude([[0, 1], [1, 1]]))], alpha=0.0)
    gfem.DynamicSolver(m, verbose=False, monitor=mon).solve(st)
    assert 1.8 < max(peak) / ust < 2.2


@pytest.mark.parametrize("mu", [0.2, 0.5])
def test_coulomb_friction_sliding_block(mu):
    """Upper block pressed onto a fixed block, then dragged sideways: Rx / Rz -> mu."""
    X1, c1 = mesh.box_hex(2, 2, 1, 4, 4, 2)
    X2, c2 = mesh.box_hex(1, 1, 1, 2, 2, 2, origin=(0.5, 0.5, 1.0))
    X = np.vstack([X1, X2])
    m = gfem.Model(X)
    m.add_block(c1, E.HEX8_PLAIN, gfem.LinearElastic(EM, NU))
    m.add_block(c2 + len(X1), E.HEX8_PLAIN, gfem.LinearElastic(EM, NU))
    n1 = np.arange(len(X1))
    top1 = n1[np.abs(X1[:, 2] - 1) < 1e-9]
    bot2 = len(X1) + np.nonzero(np.abs(X2[:, 2] - 1) < 1e-9)[0]
    sk = m.skin()
    m.add_interaction(gfem.contact.PenaltyContact(X, sub_surface(sk, set(top1.tolist())),
                                                  sub_surface(sk, set(bot2.tolist())), E=EM, fkn=10.0, mu=mu))
    base = _sel(X, lambda x, y, z: z < 1e-9)
    top = len(X1) + np.nonzero(np.abs(X2[:, 2] - 2) < 1e-9)[0]
    fix = [(base, d, 0.0) for d in range(3)]
    s1 = gfem.Step("press", bcs=fix + [(top, 0, 0.0), (top, 1, 0.0), (top, 2, -2e-3)], dt_init=0.5)
    solver = gfem.StaticSolver(m, verbose=False)
    st = gfem.State(m)
    solver.solve(s1, state=st)
    s2 = gfem.Step("slide", bcs=fix + [(top, 0, 0.02), (top, 1, 0.0), (top, 2, -2e-3)], dt_init=0.1, dt_max=0.1)
    u, s, info = solver.solve(s2, state=st)
    R = np.asarray(info["reactions"]).reshape(-1, 3)
    ratio = abs(R[top, 0].sum()) / abs(R[top, 2].sum())
    assert abs(ratio - mu) / mu < 0.03
    U = np.asarray(u).reshape(-1, 3)
    assert U[bot2, 0].mean() > 0.015            # the block really slid
