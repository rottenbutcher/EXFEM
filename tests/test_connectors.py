"""Remote points (rigid coupling, remote force) and bolt pretension."""
import numpy as np
import gfem
from gfem import mesh, elements as EL
from gfem.spec import Spec, new_step
from gfem.analysis import run_spec

E, NU = 200e3, 0.3


def _bar_spec(L=10.0, n=10):
    X, c = mesh.box_hex(L, 1, 1, n, 2, 2)
    sp = Spec("bar")
    sp.add_nodes(X)
    sp.add_elements("C3D8", c, elset="BAR")
    sp.materials["S"] = dict(E=E, nu=NU)
    sp.sections.append(dict(elset="BAR", material="S"))
    sp.nsets["ROOT"] = mesh.nodes_where(X, lambda x, y, z: x < 1e-9)
    sp.nsets["TIP"] = mesh.nodes_where(X, lambda x, y, z: x > L - 1e-9)
    from gfem.elements import ELEMENTS
    types, conns = sp.element_table()
    tip = set(sp.nsets["TIP"].tolist())
    el, fc = [], []
    for e, row in enumerate(conns):
        for k, f in enumerate(ELEMENTS["C3D8"].faces):
            if all(row[i] in tip for i in f):
                el.append(e)
                fc.append(k)
    sp.surfaces["TIPS"] = dict(elems=np.array(el), faces=np.array(fc))
    return sp, X


def test_rigid_coupling_remote_rotation(tmp_path):
    sp, X = _bar_spec()
    sp.constraints.append(dict(type="rigid", nodes="TIP", point=[10.0, 0.5, 0.5], name="RP"))
    st = new_step("rot")
    st["bcs"] = [dict(nodes=sp.nsets["ROOT"], dof=d, value=0.0) for d in range(3)]
    th = 1e-3
    st["bcs"] += [dict(remote="RP", dof=d, value=0.0) for d in (0, 1, 2, 3, 4)] + [dict(remote="RP", dof=5, value=th)]
    sp.steps.append(st)
    _, r = run_spec(sp, str(tmp_path), "a", verbose=False, write=False)
    U = r[0]["frames"][-1]["U"]
    tip = sp.nsets["TIP"]
    d = X[tip] - [10.0, 0.5, 0.5]
    assert np.allclose(U[tip, 0], -th * d[:, 1], atol=1e-12)
    assert np.allclose(U[tip, 1], th * d[:, 0], atol=1e-12)


def test_remote_force_equilibrium(tmp_path):
    sp, X = _bar_spec()
    F, P = np.array([0.0, 0.0, -100.0]), np.array([15.0, 3.0, 0.5])
    st = new_step("rf")
    st["bcs"] = [dict(nodes=sp.nsets["ROOT"], dof=d, value=0.0) for d in range(3)]
    st["dsloads"] = [dict(surface="TIPS", kind="REMOTE", vector=list(F), point=list(P))]
    sp.steps.append(st)
    model, r = run_spec(sp, str(tmp_path), "b", verbose=False, write=False)
    R = np.asarray(r[0]["info"]["reactions"]).reshape(-1, 3)[:len(X)]
    root = sp.nsets["ROOT"]
    assert np.allclose(R[root].sum(0), -F, atol=1e-3)          # reactions = internal force at supports
    Mr = np.cross(X[root], R[root]).sum(0)
    assert np.allclose(Mr + np.cross(P, F), 0, atol=1e-5 * np.abs(np.cross(P, F)).max())


def test_bolt_pretension_series(tmp_path):
    """Bolt (block A) in series with a clamped member (block B): bolt force = target."""
    X1, c1 = mesh.box_hex(10, 2, 2, 5, 2, 2)
    X2, c2 = mesh.box_hex(10, 4, 4, 5, 2, 2, origin=(10, -1, -1))
    X = np.vstack([X1, X2])
    sp = Spec("bolt")
    sp.add_nodes(X)
    sp.add_elements("C3D8", c1, elset="BOLT")
    sp.add_elements("C3D8", c2 + len(X1), elset="MEMBER")
    sp.materials["S"] = dict(E=E, nu=0.0)
    sp.sections += [dict(elset="BOLT", material="S"), dict(elset="MEMBER", material="S")]
    i1 = np.nonzero(X1[:, 0] > 10 - 1e-9)[0]
    i2 = len(X1) + np.nonzero(X2[:, 0] < 10 + 1e-9)[0]
    sp.nsets["A"] = i1
    sp.constraints.append(dict(type="rigid", nodes=np.concatenate([i1, i2]), name="J"))
    st = new_step("pre")
    left = np.nonzero(X[:, 0] < 1e-9)[0]
    right = len(X1) + np.nonzero(X2[:, 0] > 20 - 1e-9)[0]
    st["bcs"] = [dict(nodes=np.concatenate([left, right]), dof=d, value=0.0) for d in range(3)]
    st["bolts"] = [dict(elset="BOLT", axis=[1, 0, 0], force=1000.0)]
    sp.steps.append(st)
    model, r = run_spec(sp, str(tmp_path), "c", verbose=False, write=False)
    Nb = r[0]["info"]["bolt_forces"]["bolt1"]
    assert abs(Nb - 1000.0) < 1.0
    S = r[0]["frames"][-1]["S_elem"]
    assert np.allclose(S[:len(c1), 0], 1000.0 / 4, rtol=1e-3)            # bolt in tension
    assert np.allclose(S[len(c1):, 0], 1000.0 / 16, rtol=1e-3)           # series member: same force
