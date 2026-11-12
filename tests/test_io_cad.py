"""Interchange formats (Abaqus / Nastran / ANSYS CDB) and CAD-associative workflow."""
import os
import numpy as np
import pytest
import gfem
from gfem import mesh
from gfem.spec import Spec, new_step
from gfem.analysis import run_spec

E, NU = 200e3, 0.3


def _bar_spec(etype="C3D8", order=1):
    X, c = mesh.box_hex(4, 1, 1, 8, 2, 2)
    if etype == "C3D20":
        X, c = mesh.hex8_to_hex20(X, c)
    if etype == "C3D10":
        X, c = mesh.tet4_to_tet10(*mesh.hex_to_tet4(X, c))
    sp = Spec("bar")
    sp.add_nodes(X)
    sp.add_elements(etype, c, elset="BAR")
    sp.materials["STEEL"] = dict(E=E, nu=NU, density=7.85e-9)
    sp.sections.append(dict(elset="BAR", material="STEEL"))
    sp.nsets["FIX"] = mesh.nodes_where(X, lambda x, y, z: x < 1e-9)
    sp.nsets["TIP"] = mesh.nodes_where(X, lambda x, y, z: x > 4 - 1e-9)
    # element-face surface on the tip (face x=4)
    types, conns = sp.element_table()
    from gfem.elements import ELEMENTS
    el, fc = [], []
    tip = set(sp.nsets["TIP"].tolist())
    for e, (t, row) in enumerate(zip(types, conns)):
        for k, f in enumerate(ELEMENTS[t].faces):
            if all(row[i] in tip for i in f):
                el.append(e)
                fc.append(k)
    sp.surfaces["TIPS"] = dict(elems=np.array(el), faces=np.array(fc))
    st = new_step("LOAD")
    st["bcs"] = [dict(nodes=sp.nsets["FIX"], dof=d, value=0.0) for d in range(3)]
    st["dsloads"] = [dict(surface="TIPS", kind="P", value=-50.0)]      # tension 50 MPa
    sp.steps.append(st)
    return sp


def _tip_ux(sp, res):
    U = res[-1]["frames"][-1]["U"]
    return float(U[sp.nsets["TIP"], 0].mean())


@pytest.mark.parametrize("etype", ["C3D8", "C3D20", "C3D10"])
def test_inp_roundtrip(tmp_path, etype):
    from gfem.io.inp import write_inp, read_inp_spec
    sp = _bar_spec(etype)
    _, r0 = run_spec(sp, str(tmp_path), "a", verbose=False, write=False)
    path = tmp_path / "bar.inp"
    write_inp(sp, str(path))
    sp2 = read_inp_spec(str(path))
    _, r1 = run_spec(sp2, str(tmp_path), "b", verbose=False, write=False)
    exact = 50.0 * 4 / E
    assert abs(_tip_ux(sp, r0) - exact) / exact < 0.02
    assert abs(_tip_ux(sp2, r1) - _tip_ux(sp, r0)) < 1e-9


@pytest.mark.parametrize("etype", ["C3D8", "C3D20", "C3D10"])
def test_bdf_roundtrip(tmp_path, etype):
    from gfem.io.nastran import write_bdf, read_bdf
    sp = _bar_spec(etype)
    _, r0 = run_spec(sp, str(tmp_path), "a", verbose=False, write=False)
    path = tmp_path / "bar.bdf"
    write_bdf(sp, str(path))
    sp2 = read_bdf(str(path))
    assert sp2.nelem == sp.nelem and len(sp2.nodes) == len(sp.nodes)
    _, r1 = run_spec(sp2, str(tmp_path), "b", verbose=False, write=False)
    tip2 = [i for i, n in enumerate(sp2.node_ids) if n - 1 in set(sp.nsets["TIP"].tolist())]
    U = r1[-1]["frames"][-1]["U"]
    assert abs(float(U[tip2, 0].mean()) - _tip_ux(sp, r0)) < 1e-9


def test_bdf_small_field_and_continuation(tmp_path):
    lines = ["SOL 101", "CEND", "SUBCASE 1", "  SPC = 1", "  LOAD = 2", "BEGIN BULK"]
    X, c = mesh.box_hex(2, 1, 1, 2, 1, 1)
    for i, x in enumerate(X):
        lines.append(f"GRID    {i + 1:<8d}        {x[0]:<8.3f}{x[1]:<8.3f}{x[2]:<8.3f}")
    for e, row in enumerate(c):
        n = [str(k + 1) for k in row]
        lines.append(f"CHEXA   {e + 1:<8d}1       " + "".join(f"{v:<8s}" for v in n[:6]) + "+C" + str(e))
        lines.append(f"+C{e:<6d}" + "".join(f"{v:<8s}" for v in n[6:]))
    lines += ["PSOLID  1       1", "MAT1    1       200000.         0.3     7.85-9"]
    fix = [i + 1 for i in np.nonzero(X[:, 0] < 1e-9)[0]]
    tip = [i + 1 for i in np.nonzero(X[:, 0] > 2 - 1e-9)[0]]
    lines.append("SPC1,1,123," + ",".join(map(str, fix)))
    for t in tip:
        lines.append(f"FORCE,2,{t},0,25.,1.,0.,0.")
    lines.append("ENDDATA")
    p = tmp_path / "x.bdf"
    p.write_text("\n".join(lines))
    from gfem.io.nastran import read_bdf
    sp = read_bdf(str(p))
    assert sp.nelem == 2 and abs(sp.materials["MAT_1"]["density"] - 7.85e-9) < 1e-15
    _, r = run_spec(sp, str(tmp_path), "x", verbose=False, write=False)
    ux = r[0]["frames"][-1]["U"][np.array(tip) - 1, 0].mean()
    assert abs(ux - 100 * 2 / E) / (100 * 2 / E) < 0.05      # clamped end restrains Poisson contraction


def test_cdb_reader(tmp_path):
    X, c = mesh.box_hex(2, 1, 1, 2, 1, 1)
    L = ["/PREP7", "ET,1,185", "MPDATA,R5.0, 1,EX  ,       1, 1,  200000.0    ,", "MPDATA,R5.0, 1,PRXY,       1, 1,  0.3    ,",
         f"NBLOCK,6,SOLID,{len(X)},{len(X)}", "(3i9,6e21.13e3)"]
    for i, x in enumerate(X):
        L.append(f"{i + 1:9d}{0:9d}{0:9d}" + "".join(f"{v:21.13e}" for v in x))
    L += ["N,R5.3,LOC,     -1,", f"EBLOCK,19,SOLID,{len(c)},{len(c)}", "(19i9)"]
    for e, row in enumerate(c):
        L.append("".join(f"{v:9d}" for v in [1, 1, 1, 1, 0, 0, 0, 0, 8, 0, e + 1] + [k + 1 for k in row]))
    L += ["       -1"]
    fix = [i + 1 for i in np.nonzero(X[:, 0] < 1e-9)[0]]
    tip = [i + 1 for i in np.nonzero(X[:, 0] > 2 - 1e-9)[0]]
    L += [f"D,{n},ALL,0.0" for n in fix] + [f"F,{n},FX,25.0" for n in tip]
    L += ["CMBLOCK,TIPNODES,NODE,       4", "(8i10)", "".join(f"{t:10d}" for t in tip)]
    p = tmp_path / "x.cdb"
    p.write_text("\n".join(L) + "\n")
    from gfem.io.ansys_cdb import read_cdb
    sp = read_cdb(str(p))
    assert sp.nelem == 2 and "TIPNODES" in sp.nsets and len(sp.nsets["TIPNODES"]) == 4
    _, r = run_spec(sp, str(tmp_path), "x", verbose=False, write=False)
    ux = r[0]["frames"][-1]["U"][np.array(tip) - 1, 0].mean()
    assert abs(ux - 100 * 2 / E) / (100 * 2 / E) < 0.05      # clamped end restrains Poisson contraction


def test_inp_parts_instances_and_contact(tmp_path):
    """Two instances of one part, stacked; frictionless contact pair and amplitude."""
    X, c = mesh.box_hex(1, 1, 1, 2, 2, 2)
    L = ["*HEADING", "*PART, NAME=BLOCK", "*NODE"] + [f"{i + 1}, {x[0]}, {x[1]}, {x[2]}" for i, x in enumerate(X)]
    L += ["*ELEMENT, TYPE=C3D8R, ELSET=ALL"] + [f"{e + 1}, " + ", ".join(str(k + 1) for k in r) for e, r in enumerate(c)]
    top = [e + 1 for e, r in enumerate(c) if all(X[k, 2] > 1 - 1e-9 for k in r[4:])]
    bot = [e + 1 for e, r in enumerate(c) if all(X[k, 2] < 1e-9 for k in r[:4])]
    L += ["*ELSET, ELSET=TOPE", ", ".join(map(str, top)), "*ELSET, ELSET=BOTE", ", ".join(map(str, bot))]
    L += ["*SURFACE, NAME=TOP", "TOPE, S2", "*SURFACE, NAME=BOT", "BOTE, S1"]
    L += ["*NSET, NSET=BOTN"] + [", ".join(str(i + 1) for i in np.nonzero(X[:, 2] < 1e-9)[0])]
    L += ["*NSET, NSET=TOPN"] + [", ".join(str(i + 1) for i in np.nonzero(X[:, 2] > 1 - 1e-9)[0])]
    L += ["*NSET, NSET=ALLN, GENERATE", f"1, {len(X)}, 1"]
    L += ["*SOLID SECTION, ELSET=ALL, MATERIAL=STEEL", ",", "*END PART"]
    L += ["*ASSEMBLY, NAME=A", "*INSTANCE, NAME=LOW, PART=BLOCK", "*END INSTANCE",
          "*INSTANCE, NAME=UP, PART=BLOCK", "0., 0., 1.0", "*END INSTANCE", "*END ASSEMBLY"]
    L += ["*MATERIAL, NAME=STEEL", "*ELASTIC", f"{E}, {NU}", "*AMPLITUDE, NAME=RAMP2", "0., 0., 1., 1.",
          "*SURFACE INTERACTION, NAME=SMOOTH", "1.,", "*CONTACT PAIR, INTERACTION=SMOOTH", "UP.BOT, LOW.TOP",
          "*BOUNDARY", "LOW.BOTN, 3, 3", "LOW.ALLN, 1, 2", "UP.ALLN, 1, 2",
          "*STEP, NLGEOM=NO", "*STATIC", "0.5, 1.", "*BOUNDARY, AMPLITUDE=RAMP2", "UP.TOPN, 3, 3, -0.001", "*END STEP"]
    p = tmp_path / "pair.inp"
    p.write_text("\n".join(L))
    _, res = gfem.run_inp(str(p), str(tmp_path), verbose=False)
    fr = res[0]["frames"][-1]
    Eo = E * (1 - NU) / ((1 + NU) * (1 - 2 * NU))
    szz = fr["S"][:, 2].mean()
    assert abs(abs(szz) - Eo * 0.5e-3) / (Eo * 0.5e-3) < 0.08


def test_cad_mesh_solve_and_naming(tmp_path):
    pytest.importorskip("gmsh")
    from gfem.cad import CadModel, mesh_cad, match
    feats = [dict(id="F1", type="box", x=0, y=0, z=0, dx=100, dy=10, dz=10)]
    cad = CadModel(feats)
    info = cad.regenerate()
    end = [f for f in info["faces"] if abs(f["com"][0] - 100) < 1e-6][0]
    root = [f for f in info["faces"] if abs(f["com"][0]) < 1e-6][0]
    # topological naming: add a hole far away -> face ids change, signatures re-attach
    cad2 = CadModel(feats + [dict(id="F2", type="cylinder", x=50, y=5, z=-1, dx=0, dy=0, dz=12, r=2),
                             dict(id="F3", type="boolean", op="cut", target=["F1"], tool=["F2"])])
    info2 = cad2.regenerate()
    ids, _ = match(info2, 2, [dict(id=end["id"], sig=end["sig"])])
    assert len(ids) == 1 and abs(next(f for f in info2["faces"] if f["id"] == ids[0])["com"][0] - 100) < 1e-6
    sp, st = mesh_cad(cad, size=2.5, order=2)
    assert st["q_min"] > 0.1
    sp.materials["S"] = dict(E=E, nu=NU)
    sp.sections.append(dict(elset=f"BODY_{info['bodies'][0]['id']}", material="S"))
    s = new_step("T")
    s["bcs"] = [dict(nodes=sp.nsets[f"FACE_{root['id']}"], dof=0, value=0.0)]
    s["bcs"] += [dict(nodes=sp.nsets[f"VERTEX_{v['id']}"], dof=d, value=0.0)
                 for v in info["vertices"] if v["xyz"][0] < 1e-9 for d in (1, 2) if v["xyz"][1] < 1e-9 or d == 2]
    s["dsloads"] = [dict(surface=f"FACE_{end['id']}", kind="FORCE", vector=[1000.0, 0, 0])]
    sp.steps.append(s)
    _, r = run_spec(sp, str(tmp_path), "bar", verbose=False, write=False)
    fr = r[0]["frames"][-1]
    sxx = fr["S"][:, 0]
    mid = np.abs(sp.nodes[:, 0] - 50) < 20
    assert abs(np.median(sxx[mid]) - 10.0) < 0.2         # 1000 N / 100 mm^2


def test_step_export_reimport(tmp_path):
    pytest.importorskip("gmsh")
    from gfem.cad import CadModel
    feats = [dict(id="F1", type="extrude", plane="XY", distance=5,
                  loops=[dict(points=[[0, 0], [40, 0], [40, 20], [0, 20]], bulges=[0, 0.4, 0, 0]),
                         dict(circle=[10, 10, 4])]),
             dict(id="F2", type="fillet", target=["F1"], edges="all", r=0.5)]
    cad = CadModel(feats)
    info = cad.regenerate()
    path = str(tmp_path / "part.step")
    cad.export(path)
    cad2 = CadModel([dict(id="I", type="import", file=path)])
    info2 = cad2.regenerate()
    assert len(info2["faces"]) == len(info["faces"])
    assert abs(info2["bodies"][0]["volume"] - info["bodies"][0]["volume"]) / info["bodies"][0]["volume"] < 1e-3


def test_project_build_all_analyses(tmp_path):
    pytest.importorskip("gmsh")
    from gfem.gui.examples import build_example
    from gfem.cad import CadModel, mesh_cad
    from gfem import project as P
    p = build_example("modal")
    cad = CadModel(p["geometry"]["features"])
    info = cad.regenerate()
    sp, _ = mesh_cad(cad, size=8.0, order=2)
    for a in ("static", "modal", "buckling", "thermal_stress", "topopt", "transient", "explicit"):
        p["analysis"]["type"] = a
        s = P.build_spec(p, sp, info)
        assert s.steps and s.sections


def test_skewed_frictionless_support_equation_roundtrip(tmp_path):
    from gfem.io.inp import write_inp, read_inp_spec
    X, c = mesh.box_hex(1, 1, 1, 2, 2, 2)
    sp = Spec("b")
    sp.add_nodes(X)
    sp.add_elements("C3D8", c, elset="B")
    sp.materials["S"] = dict(E=E, nu=NU)
    sp.sections.append(dict(elset="B", material="S"))
    bot = mesh.nodes_where(X, lambda x, y, z: z < 1e-9)
    top = mesh.nodes_where(X, lambda x, y, z: z > 1 - 1e-9)
    n = np.array([0, np.sin(np.pi / 6), np.cos(np.pi / 6)])
    sp.constraints.append(dict(type="normal", nodes=bot, normal=n))
    st = new_step("S")
    st["bcs"] = [dict(nodes=[bot[0]], dof=0, value=0.0), dict(nodes=[bot[0]], dof=1, value=0.0),
                 dict(nodes=[bot[1]], dof=0, value=0.0)]
    st["cloads"] = [dict(nodes=top, dof=2, value=-10.0)]
    sp.steps.append(st)
    _, r1 = run_spec(sp, str(tmp_path), "a", verbose=False, write=False)
    write_inp(sp, str(tmp_path / "eq.inp"))
    _, r2 = run_spec(read_inp_spec(str(tmp_path / "eq.inp")), str(tmp_path), "b", verbose=False, write=False)
    U1, U2 = r1[0]["frames"][-1]["U"], r2[0]["frames"][-1]["U"]
    assert np.abs(U1[bot] @ n).max() < 1e-12
    assert np.abs(U1 - U2).max() < 1e-9 * np.abs(U1).max()
