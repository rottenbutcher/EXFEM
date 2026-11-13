"""Springs, point masses, response spectrum, random vibration, fatigue, radiation, new materials."""
import numpy as np
import pytest
import gfem
from gfem import mesh, elements as EL

E, NU = 200e3, 0.3


def _beam(n=20, L=100.0, rho=7.85e-9):
    X = np.stack([np.linspace(0, L, n + 1), np.zeros(n + 1), np.zeros(n + 1)], 1)
    c = np.stack([np.arange(n), np.arange(1, n + 1)], 1)
    m = gfem.Model(X)
    m.add_beam(c, gfem.LinearElastic(E, NU, density=rho), ("RECT", (10.0, 1.0)), n1=[0, 1, 0])
    return m, X, n


def test_spring_series_and_ground_spring():
    X = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0.]])
    m = gfem.Model(X)
    m.add_spring([[0, 1], [1, 2]], 100.0)                         # two axial springs in series
    st = gfem.Step(bcs=[([0], d, 0.) for d in range(3)] + [([1, 2], 1, 0.), ([1, 2], 2, 0.)], loads=[([2], 0, 10.0)])
    u, _, _ = gfem.StaticSolver(m, verbose=False).solve(st)
    assert abs(np.asarray(u).reshape(-1, 3)[2, 0] - 0.2) < 1e-9
    m2 = gfem.Model(X[:1])
    m2.add_ground_spring([0], [10.0, 20.0, 40.0])
    u, _, _ = gfem.StaticSolver(m2, verbose=False).solve(gfem.Step(loads=[([0], 0, 1.0), ([0], 1, 1.0), ([0], 2, 1.0)]))
    assert np.allclose(np.asarray(u), [0.1, 0.05, 0.025])


def test_point_mass_on_spring_frequency():
    X = np.array([[0, 0, 0], [1, 0, 0.]])
    m = gfem.Model(X)
    m.add_spring([[0, 1]], 1000.0, kind="bushing")
    m.add_point_mass([1], 2.5)
    r = gfem.modal(m, gfem.Step(kind="frequency", bcs=[([0], d, 0.) for d in range(3)], nmodes=1))
    assert abs(r["freq"][0] - np.sqrt(1000 / 2.5) / (2 * np.pi)) < 1e-6


def test_response_spectrum_sdof_limit():
    """Flat spectrum: tip displacement of mode 1 equals Gamma*Sa/w^2*phi; SRSS >= mode 1."""
    m, X, n = _beam()
    st = gfem.Step(bcs=[([0], d, 0.) for d in range(6)])
    Sa = 9806.65
    r = gfem.vibration.response_spectrum(m, st, [[0.1, Sa], [1e4, Sa]], direction=(0, 0, 1), nmodes=6)
    f1, g1 = r["freq"][0], r["participation"][0]
    phi = np.asarray(r["modal"]["modes"][0]).reshape(-1, 3)[n]
    u1 = abs(g1 * Sa / (2 * np.pi * f1) ** 2 * np.abs(phi).max())
    assert r["U"][n].max() >= u1 * 0.999
    assert r["U"][n].max() < 1.2 * u1                              # dominated by mode 1
    rc = gfem.vibration.response_spectrum(m, st, [[0.1, Sa], [1e4, Sa]], direction=(0, 0, 1), nmodes=6,
                                          combination="CQC", res=r["modal"])
    assert abs(rc["U"][n].max() - r["U"][n].max()) / r["U"][n].max() < 0.05   # well separated modes


def test_random_vibration_sdof_miles():
    """Single mass-spring: RMS displacement vs Miles' equation."""
    k, mass, zeta, G = 1e4, 1.0, 0.02, 1e5
    X = np.array([[0, 0, 0], [0, 0, 1.]])
    m = gfem.Model(X)
    m.add_spring([[0, 1]], [1e9, 1e9, k], kind="bushing")
    m.add_point_mass([1], mass)
    st = gfem.Step(bcs=[([0], d, 0.) for d in range(3)])
    fn = np.sqrt(k / mass) / (2 * np.pi)
    r = gfem.vibration.random_vibration(m, st, [[1.0, G], [1e4, G]], direction=(0, 0, 1), nmodes=1,
                                        damping=zeta, nfreq=4000)
    Q = 1 / (2 * zeta)
    grms_resp = np.sqrt(np.pi / 2 * fn * Q * G)                 # Miles: response acceleration RMS
    urms = grms_resp / (2 * np.pi * fn) ** 2
    assert abs(r["U"][1, 2] - urms) / urms < 0.03


def test_fatigue_basquin_goodman():
    from gfem.fatigue import stress_life
    Sv = np.array([[300.0, 0, 0, 0, 0, 0], [-300.0, 0, 0, 0, 0, 0]])
    r = stress_life(Sv, Su=600, Sf=900, b=-0.1, Se=0, loading="fully_reversed", mean="none")
    N = 0.5 * (300 / 900) ** (1 / -0.1)
    assert np.allclose(r["life"], N)
    rz = stress_life(Sv[:1], Su=600, Sf=900, b=-0.1, loading="zero_based", mean="goodman")
    sar = 150 / (1 - 150 / 600)
    assert np.allclose(rz["life"], 0.5 * (sar / 900) ** (1 / -0.1))


def test_radiation_plate_balance():
    """Insulated bar heated at one end, radiating from the other: energy balance."""
    from gfem.thermal import HeatTransfer
    from gfem.surfaces import sub_surface
    X, c = mesh.box_hex(10, 1, 1, 10, 1, 1)
    m = gfem.Model(X)
    m.add_block(c, EL.HEX8, gfem.LinearElastic(E, NU, k=50.0))
    sk = m.skin()
    hot = sub_surface(sk, set(np.nonzero(X[:, 0] < 1e-9)[0].tolist()))
    cold = sub_surface(sk, set(np.nonzero(X[:, 0] > 10 - 1e-9)[0].tolist()))
    q = 0.05
    T = HeatTransfer(m).solve(flux=[(hot, q)], radiation=[(cold, 0.8, 20.0)])
    Tend = T[np.nonzero(X[:, 0] > 10 - 1e-9)[0]].mean()
    out = 0.8 * 5.670374419e-14 * ((Tend + 273.15) ** 4 - 293.15 ** 4) * 1.0
    assert abs(out - q * 1.0) / q < 0.01


@pytest.mark.parametrize("mat", ["arruda", "gent"])
def test_new_hyperelastic_small_strain_limit(mat):
    from gfem.materials import from_props
    mu = 1.0
    m_ = from_props(dict(model=mat, mu=mu, lam_m=7.0, Jm=50.0, D1=2 / (1000 * mu)))
    X, c = mesh.box_hex(1, 1, 1, 1, 1, 1)
    m = gfem.Model(X)
    m.add_block(c, EL.HEX8, m_)
    w = lambda f: mesh.nodes_where(X, f)
    bcs = [(w(lambda x, y, z: x < 1e-9), 0, 0.), (w(lambda x, y, z: y < 1e-9), 1, 0.), (w(lambda x, y, z: z < 1e-9), 2, 0.),
           (w(lambda x, y, z: x > 1 - 1e-9), 0, 1e-3)]
    u, s, _ = gfem.StaticSolver(m, verbose=False).solve(gfem.Step(bcs=bcs))
    sxx = float(m.stresses(u, s)[0]["S"][0, 0, 0])
    assert abs(sxx - 3 * mu * 1e-3) / (3 * mu * 1e-3) < 0.02     # E = 3 mu (incompressible)


def test_springs_masses_from_decks(tmp_path):
    from gfem.io.inp import read_inp_spec
    from gfem.io.nastran import read_bdf
    from gfem.analysis import run_spec
    inp = tmp_path / "s.inp"
    inp.write_text("\n".join(["*NODE", "1, 0, 0, 0", "2, 1, 0, 0", "*ELEMENT, TYPE=SPRINGA, ELSET=SP", "1, 1, 2",
                              "*SPRING, ELSET=SP", "", "1000.", "*ELEMENT, TYPE=MASS, ELSET=M", "2, 2", "*MASS, ELSET=M", "2.5",
                              "*BOUNDARY", "1, 1, 3", "2, 2, 3", "*STEP", "*FREQUENCY", "1,", "*END STEP"]))
    _, r = run_spec(read_inp_spec(str(inp)), str(tmp_path), "a", verbose=False, write=False)
    assert abs(r[0]["info"]["freq"][0] - np.sqrt(1000 / 2.5) / (2 * np.pi)) < 1e-6
    bdf = tmp_path / "s.bdf"
    bdf.write_text("\n".join(["SOL 101", "CEND", "SUBCASE 1", " SPC = 1", " LOAD = 2", "BEGIN BULK",
                              "GRID,1,,0.,0.,0.", "GRID,2,,1.,0.,0.", "CELAS2,10,500.,1,1,2,1", "CONM2,20,2,,3.0",
                              "SPC1,1,123,1", "SPC1,1,23,2", "FORCE,2,2,0,10.,1.,0.,0.", "ENDDATA"]))
    _, r = run_spec(read_bdf(str(bdf)), str(tmp_path), "b", verbose=False, write=False)
    assert abs(r[0]["frames"][-1]["U"][1, 0] - 10 / 500) < 1e-9
