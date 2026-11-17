"""AI simulation assistant: tools on a project (materials, conditions by face filters, solve,
result summary, re-attachment after geometry edits), the scripted agent loop, and MCP project mode."""
import json
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("gmsh")
from gfem import project as P
from gfem.cad import ai as AI
from gfem.cad.ai_sim import SimTools

BAR = [dict(id="F1", type="box", name="Bar", dx=200, dy=20, dz=10)]


def test_sim_tools_cantilever(tmp_path):
    st = SimTools(str(tmp_path / "p.gfem.json"), workdir=str(tmp_path / "w"))
    st.set_geometry(BAR, name="Bar")
    m = st.get_model()
    assert len(m["faces"]) == 6 and "add_condition" in m["guide"]
    assert st.set_material("aluminum")["material"] == "Aluminum Alloy"
    r = st.add_condition("fixed", faces={"normal": [-1, 0, 0]})
    assert [f["id"] for f in r["faces"]] == [1]
    st.add_condition("force", faces={"near": [200, 10, 5], "count": 1}, vector=[0, 0, -100])
    st.set_mesh(size=5)
    out = st.solve()
    E, I = 71e3, 20 * 10 ** 3 / 12
    assert out["max_deformation_mm"] == pytest.approx(100 * 200 ** 3 / (3 * E * I), rel=0.03)
    assert out["min_safety_factor"] == pytest.approx(280 / out["max_von_mises_mpa"], rel=1e-3)   # aluminium, not the unused steel
    assert "temperature_range" not in out
    # thicker bar: both conditions re-attach (filter + id/normal fallback), deflection / 8
    assert st.set_geometry([dict(BAR[0], dz=20)]) == []
    out2 = st.solve()
    assert out2["max_deformation_mm"] == pytest.approx(out["max_deformation_mm"] / 8, rel=0.03)
    # saved project is a normal GUI project
    p = P.load(str(tmp_path / "p.gfem.json"))
    assert [c["type"] for c in p["conditions"]] == ["fixed", "force"] and p["bodies"]["1"]["material"] == "M2"


def test_sim_tools_errors_and_modal(tmp_path):
    st = SimTools(str(tmp_path / "p.gfem.json"), workdir=str(tmp_path / "w"))
    with pytest.raises(P.ProjectError, match="no geometry yet"):
        st.add_condition("fixed", faces=[1])
    st.set_geometry(BAR)
    with pytest.raises(P.ProjectError, match="no face matches"):
        st.add_condition("fixed", faces={"normal": [0, 0, 1], "type": "Cylinder"})
    with pytest.raises(P.ProjectError, match="needs 'vector'"):
        st.add_condition("force", faces=[2])
    with pytest.raises(P.ProjectError, match="unknown condition"):
        st.add_condition("glue", faces=[2])
    st.add_condition("fixed", faces={"normal": [-1, 0, 0]})
    st.set_analysis("modal", nmodes=3)
    st.set_mesh(size=6)
    out = st.solve()
    f1 = 1.875104 ** 2 / (2 * np.pi * 200 ** 2) * np.sqrt(200e3 * 20 * 10 ** 3 / 12 / (7.85e-9 * 200))
    assert len(out["frequencies_hz"]) == 3 and out["frequencies_hz"][0] == pytest.approx(f1, rel=0.05)
    # a condition whose faces disappear blocks the solve instead of returning zeros
    st.add_condition("force", faces=[2], vector=[0, 0, -1])
    st.p["conditions"][-1]["scope"] = dict(dim=2, ids=[999], sigs=[dict(com=[1e6, 0, 0], size=1)])
    st.p["conditions"][-1].pop("ai_face_keys", None)
    with pytest.raises(P.ProjectError, match="no longer select"):
        st.solve()


def test_scripted_simulation_agent(tmp_path):
    be = AI.ScriptedBackend([
        dict(name="Bar", features=BAR),
        dict(tool="get_model", args={}),
        dict(tool="add_condition", args=dict(type="fixed", faces={"normal": [-1, 0, 0]})),
        dict(tool="add_condition", args=dict(type="force", faces={"normal": [1, 0, 0]}, vector=[0, 0, -100])),
        dict(tool="add_condition", args=dict(type="force", faces={"normal": [9, 9, 9]})),          # error -> fed back
        dict(tool="set_mesh", args=dict(size=6)),
        dict(tool="solve", args={}),
        "Done: 0.8 mm.",
    ])
    sims = []
    res = AI.generate("cantilever", project=P.new_project("mm"), backend=be, run_dir=str(tmp_path), on_sim=sims.append)
    assert res["ok"] and res["project"]["conditions"][1]["vector"] == [0, 0, -100]
    assert [d["tool"] for d in sims] == ["get_model", "add_condition", "add_condition", "add_condition", "set_mesh", "solve"]
    assert sims[3]["error"] and sims[-1]["result"]["max_deformation_mm"] == pytest.approx(0.80, rel=0.05)
    err = [r for batch in be.results if isinstance(batch, list) for r in batch if r[3]]
    assert err and "no face matches" in err[0][1]


def test_mcp_project_mode(tmp_path):
    pj = tmp_path / "proj.gfem.json"
    P.save(P.new_project("mm"), str(pj))
    msgs = [dict(jsonrpc="2.0", id=1, method="tools/list"),
            dict(jsonrpc="2.0", id=2, method="tools/call", params=dict(name="build_cad", arguments=dict(name="Bar", features=BAR))),
            dict(jsonrpc="2.0", id=3, method="tools/call", params=dict(name="add_condition", arguments=dict(type="fixed", faces={"normal": [-1, 0, 0]}))),
            dict(jsonrpc="2.0", id=4, method="tools/call", params=dict(name="add_condition", arguments=dict(type="bogus")))]
    p = subprocess.run([sys.executable, "-m", "gfem.cad.mcp_server", "--run-dir", str(tmp_path / "run"), "--project", str(pj)],
                       input="\n".join(json.dumps(m) for m in msgs) + "\n", capture_output=True, text=True, timeout=300)
    out = {m["id"]: m for m in map(json.loads, p.stdout.splitlines())}
    names = {t["name"] for t in out[1]["result"]["tools"]}
    assert {"solve", "add_condition", "get_model", "build_cad"} <= names
    assert "project geometry updated" in out[2]["result"]["content"][0]["text"]
    assert json.loads(out[3]["result"]["content"][0]["text"])["faces"][0]["id"] == 1
    assert out[4]["result"]["isError"]
    assert P.load(str(pj))["conditions"][0]["type"] == "fixed"
    assert len(list((tmp_path / "run").glob("sim_*.json"))) == 2


def test_setup_warnings_and_sim_refuses_unsupported(tmp_path):
    from gfem.gui.server import setup_warnings
    p = P.new_project("mm")
    p["geometry"]["features"] = BAR
    w = setup_warnings(p)
    assert any(x.startswith("no supports") for x in w) and any(x.startswith("no loads") for x in w)
    p["conditions"] = [dict(id="B1", type="fixed", scope=dict(dim=2, ids=[1])), dict(id="L1", type="force", vector=[0, 0, -1], scope=dict(dim=2, ids=[]))]
    w = setup_warnings(p)
    assert not any(x.startswith("no supports") for x in w) and any("empty scope" in x for x in w)
    st = SimTools(str(tmp_path / "p.gfem.json"), workdir=str(tmp_path / "w"))
    st.set_geometry(BAR)
    st.add_condition("force", faces=[2], vector=[0, 0, -1])
    with pytest.raises(P.ProjectError, match="no supports"):
        st.solve()
