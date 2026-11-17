"""AI CAD through signed-in CLIs (Claude Code / Codex): MCP server, command construction, event
parsing, login flow. Uses fake CLIs (tests/fakes) so no account or network is needed."""
import json
import os
import subprocess
import sys
import time

import pytest

from gfem.cad import ai as AI
from gfem.cad import ai_cli

FAKES = os.path.join(os.path.dirname(__file__), "fakes")
pytestmark = pytest.mark.skipif(os.name == "nt", reason="fake CLIs are POSIX scripts")


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setenv("GFEM_CLAUDE_PATH", os.path.join(FAKES, "claude"))
    monkeypatch.setenv("GFEM_CODEX_PATH", os.path.join(FAKES, "codex"))


def _rpc_session(msgs, tmp_path, *extra):
    p = subprocess.run([sys.executable, "-m", "gfem.cad.mcp_server", "--run-dir", str(tmp_path), *extra],
                       input="\n".join(json.dumps(m) for m in msgs) + "\n", capture_output=True, text=True, timeout=120)
    return [json.loads(line) for line in p.stdout.splitlines() if line.strip()]


def test_mcp_server_protocol(tmp_path):
    out = _rpc_session([
        dict(jsonrpc="2.0", id=1, method="initialize", params=dict(protocolVersion="2024-11-05")),
        dict(jsonrpc="2.0", method="notifications/initialized"),
        dict(jsonrpc="2.0", id=2, method="tools/list"),
        dict(jsonrpc="2.0", id=3, method="tools/call", params=dict(name="build_cad", arguments=dict(
            name="Box", features=[dict(id="F1", type="box", dx=10, dy=20, dz=5)]))),
        dict(jsonrpc="2.0", id=4, method="tools/call", params=dict(name="build_cad", arguments=dict(
            features=json.dumps([dict(id="F1", type="box", dx=0)])))),
        dict(jsonrpc="2.0", id=5, method="nope"),
        dict(jsonrpc="2.0", id=6, method="ping"),
    ], tmp_path)
    by = {m["id"]: m for m in out}
    assert by[1]["result"]["protocolVersion"] == "2024-11-05" and "tools" in by[1]["result"]["capabilities"]
    assert {t["name"] for t in by[2]["result"]["tools"]} == {"build_cad", "cad_guide", "current_model"}
    r3 = by[3]["result"]
    assert not r3["isError"] and json.loads(r3["content"][0]["text"])["size"] == [10.0, 20.0, 5.0]
    assert r3["content"][1]["type"] == "image" and r3["content"][1]["mimeType"] == "image/png"
    assert by[4]["result"]["isError"] and "dx must be non-zero" in by[4]["result"]["content"][0]["text"]
    assert by[5]["error"]["code"] == -32601 and by[6]["result"] == {}
    best = json.load(open(tmp_path / "best.json"))
    assert best["name"] == "Box" and (tmp_path / "build_0001.png").exists() and (tmp_path / "build_0002.json").exists()


def test_mcp_current_model(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps([dict(id="F1", type="box", dx=1, dy=1, dz=1)]))
    out = _rpc_session([dict(jsonrpc="2.0", id=1, method="tools/call", params=dict(name="current_model", arguments={})),
                        dict(jsonrpc="2.0", id=2, method="tools/call", params=dict(name="cad_guide", arguments={}))],
                       tmp_path, "--base", str(base))
    assert json.loads(out[0]["result"]["content"][0]["text"])["features"][0]["id"] == "F1"
    assert "Feature types" in out[1]["result"]["content"][0]["text"] and "Current model" in out[1]["result"]["content"][0]["text"]


def test_status_and_missing_cli(fakes, monkeypatch):
    st = ai_cli.status("claude-cli")
    assert st["installed"] and st["logged_in"] and "fake" in st["version"]
    assert ai_cli.status("codex-cli")["logged_in"]
    monkeypatch.setenv("FAKE_LOGGED_IN", "0")
    assert not ai_cli.status("claude-cli")["logged_in"]
    with pytest.raises(AI.AIError, match="not signed in"):
        ai_cli.run_cli("claude-cli", "x")
    monkeypatch.setenv("GFEM_CLAUDE_PATH", "/nonexistent")
    monkeypatch.setenv("PATH", "/nonexistent")
    st = ai_cli.status("claude-cli")
    assert not st["installed"] and "npm install -g @anthropic-ai/claude-code" in st["install"]


def test_claude_cli_run_and_modify(fakes, tmp_path):
    png = AI.evaluate([dict(id="F1", type="box", dx=1, dy=1, dz=1)])["png"]
    (tmp_path / "d.png").write_bytes(png)
    logs, builds = [], []
    r = AI.generate("a block", provider="claude-cli", attachments=[str(tmp_path / "d.png")], log=logs.append,
                    on_build=builds.append, run_dir=str(tmp_path / "run"))
    assert r["ok"] and r["name"] == "Block" and r["builds"] == 2 and r["features"][0]["dx"] == 100
    assert r["summary"] == "Built the block (images 1)." and r["usage"]["cost_usd"] == 0.001 and r["usage"]["input"] == 15
    assert [b["ok"] for b in builds] == [False, True] and builds[1]["png"][:4] == b"\x89PNG"
    assert any("model claude-fake" in s for s in logs)
    r2 = AI.generate("make it 20 thick", provider="claude-cli", base_features=r["features"], run_dir=str(tmp_path / "run2"))
    assert r2["ok"] and r2["features"][0]["dz"] == 20


def test_codex_cli_run(fakes, tmp_path):
    r = AI.generate("plate", provider="codex-cli", run_dir=str(tmp_path / "r"))
    assert r["ok"] and r["name"] == "Plate" and r["usage"] == dict(input=1000, output=300, cache_read=200)
    assert r["session"] == "t1"
    r2 = AI.generate("more", provider="codex-cli", base_features=r["features"], session="t1", run_dir=str(tmp_path / "r2"))
    assert r2["ok"] and r2["session"] == "t1"
    assert r["report"]["volume"] == pytest.approx(100 * 60 * 10 - 3.141592653589793 * 25 * 10, rel=1e-6)


def test_login_flow(fakes):
    lines, done = [], []
    lg = ai_cli.Login("claude-cli", on_line=lambda s, u: lines.append((s, u)), on_exit=done.append)
    t0 = time.time()
    while not any("Paste code" in s for s, _ in lines) and time.time() - t0 < 20:
        time.sleep(0.05)
    assert lg.urls == ["https://claude.example/oauth?x=1"]
    lg.send("good")
    assert lg.done.wait(20) and done == [True]
    lg2 = ai_cli.Login("claude-cli")
    time.sleep(0.5)
    lg2.send("bad")
    assert lg2.done.wait(20) and lg2.ok is False


def test_codex_overrides_are_toml(fakes, tmp_path):
    tomllib = pytest.importorskip("tomllib")
    argv, stdin, env = ai_cli.build_command("codex-cli", "p", str(tmp_path), effort="max")
    ov = [argv[i + 1] for i, a in enumerate(argv) if a == "-c"]
    for o in ov:
        k, v = o.split("=", 1)
        tomllib.loads("v=" + v)
    assert 'approval_policy="never"' in ov and 'model_reasoning_effort="high"' in ov
    assert "build_cad" in stdin and argv[-1] == "-" and argv[argv.index("-s") + 1] == "read-only"
    argv2, stdin2, _ = ai_cli.build_command("codex-cli", "more", str(tmp_path), session="abc")
    assert argv2[1:3] == ["exec", "resume"] and argv2[-2:] == ["abc", "-"] and stdin2 == "more"


def test_windows_cmd_shim_runs_node_entry(tmp_path, monkeypatch):
    """npm's claude.cmd must not be run through cmd.exe: gfem starts node + the package entry."""
    bin_ = tmp_path / "npm"
    (bin_ / "node_modules" / "@anthropic-ai" / "claude-code").mkdir(parents=True)
    (bin_ / "node_modules" / "@anthropic-ai" / "claude-code" / "cli.js").write_text("")
    (bin_ / "claude.cmd").write_text("@echo off")
    monkeypatch.setenv("GFEM_CLAUDE_PATH", str(bin_ / "claude.cmd"))
    c = ai_cli.cmd("claude-cli")
    if c[0].endswith("claude.cmd"):
        pytest.skip("node not found")
    assert c[1].endswith("cli.js") and "node" in os.path.basename(c[0])
