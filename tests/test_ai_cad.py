"""Text-to-CAD: validation, evaluation, the agent loop, and the real SDK request/response paths
(Anthropic and OpenAI) against a mocked HTTP transport. No network, no API key needed."""
import base64
import json

import numpy as np
import pytest

from gfem.cad import ai as AI

PLATE = [
    dict(id="F1", type="box", x=0, y=0, z=0, dx=100, dy=60, dz=10),
    dict(id="F2", type="cylinder", x=20, y=30, z=-1, dx=0, dy=0, dz=12, r=5),
    dict(id="F3", type="pattern", target="F2", kind="linear", direction=[1, 0, 0], spacing=30, count=3),
    dict(id="F4", type="boolean", op="cut", target=["F1"], tool=["F2", "F3"]),
]


# ------------------------------------------------------------------ validation
def test_validation_messages():
    bad = [dict(id="A", type="box", dx=0), dict(id="B", type="cylinder", r=-1),
           dict(id="C", type="boolean", target=["A"], tool=["Q"]),
           dict(id="D", type="fillet", target=["A"]),
           dict(id="E", type="extrude", loops=[dict(points=[[0, 0], [10, 10], [10, 0], [0, 10]])], distance=3),
           dict(id="G", type="revolve", loops=[dict(rect=[-1, 0, 5, 5])]),
           dict(id="H", type="sphere", r="abc"), dict(type="spline")]
    _, vr = AI.validate_features(bad)
    msg = "\n".join(vr.errors)
    for s in ("dx must be non-zero", "r must be > 0", "unknown id(s) ['Q']", "already consumed",
              "self-intersecting", "crosses the revolve axis", "r must be a number", "unknown type 'spline'"):
        assert s in msg, s


def test_validation_normalises():
    feats, vr = AI.validate_features([dict(type="box", dx="10", dy=5, dz=5), dict(type="box", x=20, dx=5, dy=5, dz=5),
                                      dict(type="boolean", op="fuse", target="F1", tool="F2")])
    assert not vr.errors, vr.errors
    assert [f["id"] for f in feats] == ["F1", "F2", "F3"]
    assert feats[0]["dx"] == 10.0 and feats[2]["target"] == ["F1"] and feats[2]["tool"] == ["F2"]


def test_evaluate_reports_and_renders():
    rep = AI.evaluate(PLATE)
    assert rep["ok"], rep["errors"]
    assert rep["size"] == [100.0, 60.0, 10.0]
    assert len(rep["bodies"]) == 1
    assert rep["volume"] == pytest.approx(100 * 60 * 10 - 3 * np.pi * 25 * 10, rel=1e-6)
    assert rep["png"][:8] == b"\x89PNG\r\n\x1a\n" and len(rep["png"]) > 2000
    assert not rep.get("lines")                      # no seam edges reported as wire bodies
    txt = json.loads(AI.report_text(rep))
    assert txt["ok"] and "png" not in txt and txt["cylindrical_faces"] == 3


def test_evaluate_kernel_error():
    rep = AI.evaluate([dict(id="F1", type="box", dx=10, dy=10, dz=2),
                       dict(id="F2", type="fillet", target=["F1"], r=5, edges="all")])
    assert not rep["ok"] and any("kernel" in e for e in rep["errors"])


# ------------------------------------------------------------------ geometry kernel improvements
def test_sketch_regions_and_islands():
    from gfem.cad import CadModel
    f = [dict(id="A", type="extrude", distance=5, loops=[dict(rect=[0, 0, 10, 10]), dict(rect=[20, 0, 30, 10]),
                                                         dict(circle=[5, 5, 2]), dict(circle=[5, 5, 1])])]
    vols = sorted(b["volume"] for b in CadModel(f).regenerate()["bodies"])
    assert vols == pytest.approx(sorted([500 - np.pi * 4 * 5, 500, np.pi * 5]), rel=1e-6)


def test_edge_filters():
    from gfem.cad import CadModel
    base = dict(id="B", type="box", dx=20, dy=20, dz=10)
    v = lambda e: CadModel([base, dict(id="F", type="fillet", target=["B"], r=2, edges=e)]).regenerate()["bodies"][0]["volume"]
    assert v({"parallel": "z"}) == pytest.approx(4000 - 4 * (4 - np.pi) * 10, rel=1e-6)
    assert v([{"com": [0, 0, 5], "size": 10}]) == pytest.approx(4000 - (4 - np.pi) * 10, rel=1e-6)


# ------------------------------------------------------------------ agent loop (scripted)
def test_agent_repairs_errors():
    broken = [dict(id="F1", type="box", dx=100, dy=60, dz=10), dict(id="F2", type="boolean", op="cut", target=["F1"], tool=["X"])]
    be = AI.ScriptedBackend([broken, PLATE, "A 100 x 60 x 10 plate with three 10 mm holes."])
    logs = []
    res = AI.run_agent(be, log=logs.append)
    assert res["ok"] and res["builds"] == 2
    assert res["features"][-1]["id"] == "F4"
    assert res["summary"].startswith("A 100 x 60")
    first = be.results[0][0]
    assert first[3] is True and "unknown id" in first[1]            # error sent back to the model
    second = be.results[1][0]
    assert second[3] is False and second[2][:4] == b"\x89PNG"
    assert any("build 2" in s for s in logs)


def test_agent_nudges_when_nothing_built():
    be = AI.ScriptedBackend(["I think the part is a plate.", PLATE, "done"])
    res = AI.run_agent(be)
    assert res["ok"] and "build_cad" in be.results[0]


def test_agent_no_result():
    res = AI.run_agent(AI.ScriptedBackend(["no", "still no"]))
    assert not res["ok"] and res["error"]


def test_modify_request_contains_tree():
    txt = AI.build_request("make the holes bigger", PLATE, "mm")
    assert "Modify the existing model" in txt and '"F3"' in txt


# ------------------------------------------------------------------ settings
def test_settings_roundtrip(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv("GFEM_AI_CONFIG", str(tmp_path / "ai.json"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    AI.save_settings(provider="openai", model="gpt-x", keys={"anthropic": " sk-test "})
    s = AI.load_settings()
    assert s["provider"] == "openai" and s["keys"]["anthropic"] == "sk-test"
    assert AI.api_key("anthropic") == "sk-test"
    pub = AI.public_settings()
    assert "sk-test" not in json.dumps(pub) and pub["providers"]["anthropic"]["has_key"]
    if os.name == "posix":
        assert (os.stat(tmp_path / "ai.json").st_mode & 0o777) == 0o600
    AI.save_settings(keys={"anthropic": ""})
    assert AI.api_key("anthropic") is None
    with pytest.raises(AI.AIError, match="no API key"):
        AI.make_backend("anthropic", "x")


# ------------------------------------------------------------------ real SDKs, mocked HTTP
def _sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _anthropic_tool_turn(mid, tool_input):
    s = json.dumps(tool_input)
    return [
        {"type": "message_start", "message": {"id": mid, "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                                              "content": [], "stop_reason": None, "stop_sequence": None,
                                              "usage": {"input_tokens": 100, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Building the plate."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "build_cad", "input": {}}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": s[:20]}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": s[20:]}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 50}},
        {"type": "message_stop"},
    ]


def _anthropic_text_turn(mid, text):
    return [
        {"type": "message_start", "message": {"id": mid, "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                                              "content": [], "stop_reason": None, "stop_sequence": None,
                                              "usage": {"input_tokens": 200, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 20}},
        {"type": "message_stop"},
    ]


def test_anthropic_sdk_roundtrip():
    anthropic = pytest.importorskip("anthropic")
    import httpx2
    turns = [_anthropic_tool_turn("msg_1", dict(name="Plate", units="mm", features=PLATE)),
             _anthropic_text_turn("msg_2", "Plate 100 x 60 x 10 mm with 3 holes.")]
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append((str(req.url), dict(req.headers), body))
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(turns[len(seen) - 1]))

    client = anthropic.Anthropic(api_key="sk-test", http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
                                 max_retries=0)
    streamed = []
    res = AI.generate("a plate with three holes", provider="anthropic", model="claude-opus-5-5", client=client,
                      on_text=streamed.append)
    assert res["ok"] and res["name"] == "Plate" and res["builds"] == 1
    assert res["summary"].startswith("Plate 100 x 60")
    assert res["usage"]["input"] == 300 and res["usage"]["output"] == 70
    assert "Building the plate." in streamed
    url, headers, body = seen[0]
    assert url.endswith("/v1/messages") or "/v1/messages" in url
    assert body["model"] == "claude-opus-5-5" and body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["effort"] in AI.EFFORTS
    assert body["tools"][0]["name"] == "build_cad" and "tool_choice" not in body
    assert body["fallbacks"] == "default" and "server-side-fallback" in headers.get("anthropic-beta", "")
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    # second request carries the assistant turn (unchanged) and the tool result with the PNG views
    msgs = seen[1][2]["messages"]
    assert msgs[1]["role"] == "assistant" and msgs[1]["content"][1]["type"] == "tool_use"
    tr = msgs[2]["content"][0]
    assert tr["type"] == "tool_result" and tr["tool_use_id"] == "toolu_1" and not tr["is_error"]
    assert tr["content"][1]["type"] == "image"
    assert base64.b64decode(tr["content"][1]["source"]["data"])[:4] == b"\x89PNG"
    assert json.loads(tr["content"][0]["text"])["size"] == [100.0, 60.0, 10.0]


def test_anthropic_errors_are_readable():
    anthropic = pytest.importorskip("anthropic")
    import httpx2
    client = anthropic.Anthropic(api_key="bad", max_retries=0, http_client=anthropic.DefaultHttpxClient(
        transport=httpx2.MockTransport(lambda r: httpx2.Response(401, json={"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}))))
    with pytest.raises(AI.AIError, match="invalid"):
        AI.generate("plate", provider="anthropic", client=client)


def test_openai_sdk_roundtrip():
    openai = pytest.importorskip("openai")
    import httpx2
    seen = []
    replies = [
        {"id": "c1", "object": "chat.completion", "created": 0, "model": "gpt-5",
         "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
                      "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "build_cad",
                                      "arguments": json.dumps(dict(name="Plate", features=PLATE))}}]}}],
         "usage": {"prompt_tokens": 120, "completion_tokens": 60, "total_tokens": 180}},
        {"id": "c2", "object": "chat.completion", "created": 0, "model": "gpt-5",
         "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Done: plate with 3 holes."}}],
         "usage": {"prompt_tokens": 300, "completion_tokens": 10, "total_tokens": 310}},
    ]

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx2.Response(200, json=replies[len(seen) - 1])

    client = openai.OpenAI(api_key="sk-test", max_retries=0, http_client=openai.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    res = AI.generate("a plate with three holes", provider="openai", model="gpt-5", client=client)
    assert res["ok"] and res["summary"] == "Done: plate with 3 holes." and res["usage"]["input"] == 420
    b0 = seen[0]
    assert b0["tools"][0]["function"]["name"] == "build_cad" and b0["reasoning_effort"] in ("low", "medium", "high")
    m = seen[1]["messages"]
    assert m[2]["role"] == "assistant" and m[2]["tool_calls"][0]["id"] == "call_1"
    assert m[3]["role"] == "tool" and m[3]["tool_call_id"] == "call_1" and json.loads(m[3]["content"])["ok"]
    assert m[4]["role"] == "user" and m[4]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_attachments(tmp_path):
    png = tmp_path / "d.png"
    png.write_bytes(AI.evaluate(PLATE)["png"])
    a = AI.load_attachment(str(png))
    assert a["kind"] == "image" and a["media_type"] == "image/png"
    parts = AI.AnthropicBackend.content("make this", [a])
    assert parts[0]["type"] == "image" and parts[-1]["text"] == "make this"
    ezdxf = pytest.importorskip("ezdxf")
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 4
    doc.modelspace().add_lwpolyline([(0, 0), (50, 0), (50, 20), (0, 20)], close=True)
    doc.modelspace().add_circle((10, 10), 3)
    doc.saveas(tmp_path / "p.dxf")
    t = AI.load_attachment(str(tmp_path / "p.dxf"))
    assert t["kind"] == "text" and '"circle": [10.0, 10.0, 3.0]' in t["text"]


def test_cli_scripted(tmp_path, monkeypatch):
    """gfem ai end to end with a scripted backend (writes STEP and a project)."""
    from gfem.__main__ import main
    orig = AI.generate

    def fake(prompt, **kw):
        kw.pop("backend", None)
        return orig(prompt, backend=AI.ScriptedBackend([PLATE, "ok"]), **kw)
    monkeypatch.setattr(AI, "generate", fake)
    out = tmp_path / "p.step"
    assert main(["ai", "#3 holes plate", "-o", str(out), "--preview", str(tmp_path / "v.png")]) == 0
    assert out.stat().st_size > 1000 and (tmp_path / "v.png").read_bytes()[:4] == b"\x89PNG"
    pj = tmp_path / "p.gfem.json"
    assert main(["ai", "plate", "-o", str(pj)]) == 0
    from gfem import project as P
    assert P.load(str(pj))["geometry"]["features"][-1]["id"] == "F4"


def test_gui_ai_endpoints(tmp_path, monkeypatch):
    """Settings never leak keys, cross-origin POSTs are refused, a scripted run streams events."""
    import time
    import urllib.error
    import urllib.request
    import threading
    from http.server import ThreadingHTTPServer
    from gfem.gui.server import Session, Handler
    monkeypatch.setenv("GFEM_AI_CONFIG", str(tmp_path / "ai.json"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    orig = AI.generate
    monkeypatch.setattr(AI, "generate", lambda prompt, **kw: orig(prompt, backend=AI.ScriptedBackend([PLATE, "ok"]),
                                                                 **{k: v for k, v in kw.items() if k != "backend"}))
    Handler.session = Session(str(tmp_path / "work"))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(path, body, headers=None):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    try:
        st, pub = post("/api/ai/settings", {"provider": "anthropic", "key": "sk-secret-123", "effort": "medium"})
        assert st == 200 and pub["providers"]["anthropic"]["has_key"] and pub["effort"] == "medium"
        with urllib.request.urlopen(base + "/api/ai/settings") as r:
            assert b"sk-secret" not in r.read()
        st, _ = post("/api/ai/generate", {"prompt": "x"}, {"Origin": "http://evil.example"})
        assert st == 403
        st, r = post("/api/ai/generate", {"prompt": "a plate", "mode": "new"})
        assert st == 200 and r["rid"]
        Handler.session.ai_job.join(60)
        evs = [e for e in Handler.session.events if e["type"] == "ai"]
        steps = [e["step"] for e in evs]
        assert steps[0] == "start" and "build" in steps and steps[-1] == "done"
        done = evs[-1]
        assert done["ok"] and done["features"][-1]["id"] == "F4" and base64.b64decode(done["png"])[:4] == b"\x89PNG"
        st, r = post("/api/ai/generate", {"prompt": "x", "attachments": ["/etc/passwd"]})
        assert st == 400 and "uploaded" in r["error"]
    finally:
        srv.shutdown()


def test_new_cad_features_exact_volumes():
    from gfem.cad import CadModel
    vol = lambda f: sum(b["volume"] for b in CadModel(f).regenerate()["bodies"])
    box = dict(id="B", type="box", dx=50, dy=50, dz=10)
    assert vol([dict(id="B", type="box", dx=100, dy=60, dz=40),
                dict(id="S", type="shell", target=["B"], thickness=2, open_faces=[{"normal": [0, 0, 1]}])]) == pytest.approx(100 * 60 * 40 - 96 * 56 * 38, rel=1e-9)
    assert vol([box, dict(id="H", type="hole", target=["B"], position=[25, 25, 10], d=8, cbore=[14, 4])]) == pytest.approx(
        25000 - np.pi * 16 * 10 - np.pi * (49 - 16) * 4, rel=1e-9)
    assert vol([box, dict(id="H", type="hole", target=["B"], position=[25, 25, 10], d=8, depth=5)]) == pytest.approx(25000 - np.pi * 16 * 5, rel=1e-9)
    assert vol([box, dict(id="H", type="hole", target=["B"], position=[25, 25, 10], d=6, csk=[12, 90])]) == pytest.approx(
        25000 - np.pi * 9 * 10 - np.pi * (36 + 18 + 9) + np.pi * 9 * 3, rel=1e-9)
    assert vol([dict(id="P", type="sweep", plane={"origin": [0, 0, 0], "normal": [1, 0, 0]},
                     loops=[dict(circle=[0, 0, 5]), dict(circle=[0, 0, 4])], path=[[0, 0, 0], [100, 0, 0]])]) == pytest.approx(np.pi * 9 * 100, rel=1e-6)
    spring = vol([dict(id="H", type="sweep", plane={"origin": [20, 0, 0], "normal": [0, 1, 0]}, loops=[dict(circle=[0, 0, 2])],
                       helix=dict(radius=20, pitch=10, turns=3))])
    assert spring == pytest.approx(np.pi * 4 * 2 * np.pi * np.hypot(20, 10 / (2 * np.pi)) * 3, rel=0.01)
    frustum = vol([dict(id="L", type="loft", ruled=True, sections=[dict(plane="XY", loop=dict(rect=[-20, -20, 20, 20])),
                                                                   dict(plane="XY", offset=30, loop=dict(rect=[-10, -10, 10, 10]))])])
    assert frustum == pytest.approx(30 / 3 * (1600 + 400 + np.sqrt(1600 * 400)), rel=1e-6)


def test_shell_and_loft_robustness():
    """Lofts are aligned (no twist, edge counts matched, outward orientation); shells are checked
    against a thin-wall estimate, so an invalid OpenCASCADE result is never returned."""
    from gfem.cad import CadModel
    from gfem.cad.geometry import GeometryError

    def bodies(f):
        return [b["volume"] for b in CadModel(f).regenerate()["bodies"]]
    duct = lambda l1: dict(id="L", type="loft", sections=[dict(plane="XY", loop=l1), dict(plane="XY", offset=80, loop=dict(circle=[0, 0, 15]))])
    full = bodies([duct(dict(rect=[-30, -20, 30, 20]))])[0]
    assert 100000 < full < 117600                      # smooth loft below the linear frustum, no twist collapse
    assert bodies([dict(id="L", type="loft", sections=[dict(plane="XY", loop=dict(circle=[0, 0, 30])),
                                                       dict(plane="XY", offset=80, loop=dict(circle=[0, 0, 15]))])])[0] == pytest.approx(80 / 3 * np.pi * (900 + 225 + 450), rel=1e-6)
    vols = {}
    for k, opens in (("both", [{"normal": [0, 0, -1]}, {"normal": [0, 0, 1]}]), ("one", [{"normal": [0, 0, 1]}]), ("closed", [])):
        v = bodies([duct(dict(rect=[-30, -20, 30, 20])), dict(id="S", type="shell", target=["L"], thickness=2, open_faces=opens)])
        assert len(v) == 1 and v[0] > 0
        vols[k] = v[0]
    assert vols["both"] < vols["one"] < vols["closed"] < 0.3 * full
    assert bodies([dict(id="B", type="box", dx=100, dy=60, dz=40), dict(id="S", type="shell", target=["B"], thickness=2)]) == [pytest.approx(240000 - 96 * 56 * 36)]
    assert bodies([dict(id="C", type="cylinder", r=30, dz=50),
                   dict(id="S", type="shell", target=["C"], thickness=3, open_faces=[{"normal": [0, 0, 1]}])]) == [pytest.approx(np.pi * (900 * 50 - 27 ** 2 * 47))]
    # a shape OpenCASCADE cannot offset fails cleanly instead of returning a wrong solid
    for opens in ([{"normal": [0, 0, -1]}, {"normal": [0, 0, 1]}], []):
        try:
            v = bodies([duct(dict(slot=[-15, 0, 15, 0, 20])), dict(id="S", type="shell", target=["L"], thickness=2, open_faces=opens)])
            assert len(v) == 1 and 0 < v[0] < 40000
        except GeometryError:
            pass
