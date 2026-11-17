"""GUI back-end: project -> CAD -> mesh -> solve through the session, and the HTTP API."""
import json
import threading
import urllib.request
import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")


def test_session_example_modal(tmp_path):
    from gfem.gui.server import Session
    from gfem.gui.examples import build_example
    s = Session(str(tmp_path))
    s.project = build_example("modal")
    s.project["mesh"]["size"] = 10.0
    s.regenerate()
    s.solve()
    assert len(s.frames) == 6
    f1 = s.frames[0]["freq"]
    E, rho, L, b, h = 71e3, 2.77e-9, 300.0, 20.0, 10.0
    exact = 1.875104 ** 2 / (2 * np.pi * L ** 2) * np.sqrt(E * b * h ** 3 / 12 / (rho * b * h))
    assert abs(f1 - exact) / exact < 0.03
    r = s.result_payload(0, "total_deformation")
    assert r["values"] and 0.99 < r["max"] < 1.75          # modes normalised by max component


def test_session_static_bracket_and_export(tmp_path):
    from gfem.gui.server import Session
    from gfem.gui.examples import build_example
    s = Session(str(tmp_path))
    s.project = build_example("bracket")
    s.project["mesh"]["size"] = 8.0
    s.regenerate()
    s.solve()
    fr = s.frames[-1]
    assert fr["MISES"].max() > 10 and np.linalg.norm(fr["U"], axis=1).max() > 0
    # deck export of the GUI model and re-run through the Abaqus reader
    from gfem import project as P
    from gfem.io.inp import write_inp, read_inp_spec
    from gfem.analysis import run_spec
    sp = P.build_spec(s.project, s.mesh_spec, s.info)
    write_inp(sp, str(tmp_path / "bracket.inp"))
    _, res = run_spec(read_inp_spec(str(tmp_path / "bracket.inp")), str(tmp_path), "b", verbose=False, write=False)
    u2 = res[0]["frames"][-1]["U"]
    assert abs(np.linalg.norm(u2, axis=1).max() - np.linalg.norm(fr["U"], axis=1).max()) < 1e-6


def test_http_api(tmp_path):
    from http.server import ThreadingHTTPServer
    from gfem.gui.server import Session, Handler
    Handler.session = Session(str(tmp_path))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    B = f"http://127.0.0.1:{port}"

    def post(u, d):
        req = urllib.request.Request(B + u, data=json.dumps(d).encode(), headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req))
    try:
        st = json.load(urllib.request.urlopen(B + "/api/state"))
        assert "project" in st and "Structural Steel" in [m["name"] for m in st["library"]]
        html = urllib.request.urlopen(B + "/").read().decode()
        assert "gfem" in html and "importmap" in html
        p = st["project"]
        p["geometry"]["features"] = [dict(id="F1", type="box", x=0, y=0, z=0, dx=10, dy=10, dz=10)]
        r = post("/api/regenerate", {"project": p})
        assert len(r["info"]["faces"]) == 6 and r["faces"]
        m = post("/api/mesh", {"project": r["project"]})
        assert m["stats"]["nodes"] > 8 and m["xyz"]
        ex = json.load(urllib.request.urlopen(B + "/api/examples"))
        assert len(ex) >= 8
    finally:
        srv.shutdown()
