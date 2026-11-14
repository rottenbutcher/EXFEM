"""AI CAD through a signed-in Claude Code or OpenAI Codex CLI (no API key).

The user signs in once with their Claude or ChatGPT account through the official CLI login
(``claude auth login`` / ``codex login``: a browser window opens). gfem then runs the CLI
non-interactively with gfem's MCP server attached (`gfem.cad.mcp_server`), so the model's
``build_cad`` calls go straight into gfem: they are validated, regenerated, measured and rendered,
and every build appears live in the GUI. The CLI's own tools (shell, file edits, web) are disabled
or sandboxed read-only; the model can only use the gfem CAD tools.

Providers
  claude-cli : Claude Code  (`claude`, npm i -g @anthropic-ai/claude-code), Claude Pro/Max/Team/Enterprise
  codex-cli  : OpenAI Codex (`codex`,  npm i -g @openai/codex),            ChatGPT Plus/Pro/Team/Enterprise
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from .ai import SYSTEM_PROMPT, AIResult, AIError, build_request, load_attachment, IMAGE_TYPES

CLIS = {
    "claude-cli": dict(label="Claude (sign in with your Claude account)", exe="claude", npm="@anthropic-ai/claude-code",
                       models=["", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "opus", "sonnet"],
                       account="Claude Pro / Max / Team / Enterprise"),
    "codex-cli": dict(label="ChatGPT (sign in with your ChatGPT account)", exe="codex", npm="@openai/codex",
                      models=["", "gpt-5", "gpt-5-codex", "gpt-5-mini"], account="ChatGPT Plus / Pro / Team / Enterprise"),
}
URL_RE = re.compile(r"https://[^\s\"'<>]+")
_NOWIN = dict(creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else {}


def exe(provider):
    """Full path of the CLI executable (also finds npm's .cmd shims on Windows)."""
    name = CLIS[provider]["exe"]
    env = os.environ.get(f"GFEM_{name.upper()}_PATH")
    if env and os.path.exists(env):
        return env
    return shutil.which(name) or (shutil.which(name + ".cmd") if os.name == "nt" else None)


def _run(args, timeout=20, env=None, inp=None):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env, input=inp,
                           encoding="utf-8", errors="replace", **_NOWIN)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        return -1, str(e)


def status(provider):
    """-> dict(installed, version, logged_in, detail, install)"""
    d = CLIS[provider]
    x = exe(provider)
    out = dict(provider=provider, label=d["label"], account=d["account"], installed=bool(x), version=None,
               logged_in=False, detail="", install=f"npm install -g {d['npm']}", models=d["models"])
    if not x:
        out["detail"] = f"{d['exe']} is not installed"
        return out
    rc, txt = _run([x, "--version"], timeout=20)
    out["version"] = txt.strip().splitlines()[0] if rc == 0 and txt.strip() else None
    if provider == "claude-cli":
        rc, txt = _run([x, "auth", "status", "--json"], timeout=25)
        try:
            j = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
            out["logged_in"] = bool(j.get("loggedIn"))
            out["detail"] = j.get("authMethod") or ""
            if j.get("email"):
                out["detail"] = j["email"]
        except ValueError:
            out["detail"] = txt.strip()[:200]
    else:
        rc, txt = _run([x, "login", "status"], timeout=25)
        t = txt.strip()
        out["logged_in"] = rc == 0 and "not logged in" not in t.lower()
        out["detail"] = t.splitlines()[-1][:200] if t else ""
    return out


# ================================================================== login
class Login:
    """A running ``claude auth login`` / ``codex login``. Output lines (with the sign-in URL) go to
    ``on_line``; ``send(code)`` answers 'paste code' prompts; ``on_exit(ok)`` fires when it ends."""

    def __init__(self, provider, on_line=None, on_exit=None, device=False, console=False):
        x = exe(provider)
        if not x:
            raise AIError(f"{CLIS[provider]['exe']} is not installed: {status(provider)['install']}")
        if provider == "claude-cli":
            args = [x, "auth", "login", "--console" if console else "--claudeai"]
        else:
            args = [x, "login"] + (["--device-auth"] if device else [])
        self.provider = provider
        self.urls = []
        self.lines = []
        self.on_line, self.on_exit = on_line, on_exit
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1, **_NOWIN)
        self.done = threading.Event()
        self.ok = None
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        buf = ""
        while True:
            ch = self.proc.stdout.read(1)              # char-wise: prompts have no trailing newline
            if not ch:
                break
            if ch in "\r\n":
                if buf.strip():
                    self._line(buf)
                buf = ""
            else:
                buf += ch
                if buf.rstrip().endswith((">", ":")) and ("paste" in buf.lower() or "code" in buf.lower()):
                    self._line(buf)
                    buf = ""
        if buf.strip():
            self._line(buf)
        rc = self.proc.wait()
        self.ok = rc == 0 and status(self.provider)["logged_in"]
        self.done.set()
        if self.on_exit:
            self.on_exit(self.ok)

    def _line(self, s):
        s = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", s).strip()
        self.lines.append(s)
        for u in URL_RE.findall(s):
            if u not in self.urls:
                self.urls.append(u)
        if self.on_line:
            self.on_line(s, self.urls[-1] if self.urls else None)

    def send(self, code):
        if self.proc.poll() is None:
            self.proc.stdin.write(code.strip() + "\n")
            self.proc.stdin.flush()

    def cancel(self):
        if self.proc.poll() is None:
            self.proc.terminate()


def logout(provider):
    x = exe(provider)
    if not x:
        return False
    rc, _ = _run([x, "auth", "logout"] if provider == "claude-cli" else [x, "logout"], timeout=30)
    return rc == 0


# ================================================================== run
def _mcp_cmd(run_dir, units, base_file, visual):
    args = ["-m", "gfem.cad.mcp_server", "--run-dir", run_dir, "--units", units]
    if base_file:
        args += ["--base", base_file]
    if not visual:
        args.append("--no-visual")
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pp = os.pathsep.join([pkg_root] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    return sys.executable, args, {"PYTHONPATH": pp, "PYTHONIOENCODING": "utf-8"}


CLI_PROMPT = """{system}

You are running inside gfem. Use ONLY the tools of the "gfem" MCP server: `cad_guide` (reference,
call it first), `current_model` (when modifying) and `build_cad` (build). Do not write files, edit code
or run shell commands; the build_cad result (measurements + rendered views) is how you check your
work. Finish with a short summary of the part."""


def _toml_str(s):
    return json.dumps(s)                                # a JSON string is a valid TOML basic string


def build_command(provider, prompt_text, run_dir, units="mm", base_file=None, model=None, effort=None,
                  visual=True, images=()):
    """-> (argv, stdin_text|None, env)"""
    x = exe(provider)
    if not x:
        raise AIError(f"{CLIS[provider]['exe']} is not installed: {status(provider)['install']}")
    py, margs, menv = _mcp_cmd(run_dir, units, base_file, visual)
    env = dict(os.environ)
    if provider == "claude-cli":
        cfg = os.path.join(run_dir, "mcp.json")
        with open(cfg, "w", encoding="utf-8") as f:
            json.dump({"mcpServers": {"gfem": {"type": "stdio", "command": py, "args": margs, "env": menv}}}, f)
        argv = [x, "-p", "--output-format", "stream-json", "--verbose", "--input-format", "stream-json",
                "--system-prompt", CLI_PROMPT.format(system=SYSTEM_PROMPT),
                "--tools", "", "--mcp-config", cfg, "--strict-mcp-config",
                "--allowedTools", "mcp__gfem__build_cad", "mcp__gfem__cad_guide", "mcp__gfem__current_model",
                "--no-session-persistence", "--setting-sources", ""]
        if model:
            argv += ["--model", model]
        if effort:
            argv += ["--effort", effort]
        content = []
        for a in images:
            import base64
            if a["kind"] == "image":
                content.append({"type": "image", "source": {"type": "base64", "media_type": a["media_type"],
                                                            "data": base64.b64encode(a["data"]).decode()}})
            elif a["kind"] == "pdf":
                content.append({"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                                               "data": base64.b64encode(a["data"]).decode()}})
        content.append({"type": "text", "text": prompt_text})
        stdin = json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"
        env.pop("ANTHROPIC_API_KEY", None)            # use the signed-in account, not a stray key
        for k in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ID"):
            env.pop(k, None)                          # gfem started from inside a Claude Code session
        return argv, stdin, env
    # codex
    ov = [f"mcp_servers.gfem.command={_toml_str(py)}",
          "mcp_servers.gfem.args=[" + ",".join(_toml_str(a) for a in margs) + "]",
          "mcp_servers.gfem.env={" + ",".join(f"{k}={_toml_str(v)}" for k, v in menv.items()) + "}",
          "mcp_servers.gfem.default_tools_approval_mode=\"approve\"",
          "mcp_servers.gfem.startup_timeout_sec=60", "mcp_servers.gfem.tool_timeout_sec=600",
          "approval_policy=\"never\""]
    if effort:
        ov.append(f"model_reasoning_effort={_toml_str({'xhigh': 'high', 'max': 'high'}.get(effort, effort))}")
    argv = [x, "exec", "--json", "--skip-git-repo-check", "--ephemeral", "-s", "read-only", "-C", run_dir]
    for o in ov:
        argv += ["-c", o]
    if model:
        argv += ["-m", model]
    paths = []
    for i, a in enumerate(images):
        if a["kind"] == "image":
            ext = {v: k for k, v in IMAGE_TYPES.items()}.get(a["media_type"], ".png")
            p = os.path.join(run_dir, f"attach_{i}{ext}")
            with open(p, "wb") as f:
                f.write(a["data"])
            paths.append(p)
    for p in paths:
        argv += ["-i", p]
    full = CLI_PROMPT.format(system=SYSTEM_PROMPT) + "\n\n" + prompt_text
    argv += ["--", full]
    env.pop("OPENAI_API_KEY", None)
    return argv, None, env


class _Watcher(threading.Thread):
    """Polls the run directory for build_NNNN.json written by the MCP server."""

    def __init__(self, run_dir, on_build, log):
        super().__init__(daemon=True)
        self.run_dir, self.on_build, self.log = run_dir, on_build, log
        self.seen = set()
        self.stop = threading.Event()
        self.n = 0

    def poll(self):
        for p in sorted(glob.glob(os.path.join(self.run_dir, "build_*.json"))):
            if p in self.seen:
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    rep = json.load(f)
            except (OSError, ValueError):
                continue
            self.seen.add(p)
            self.n += 1
            if rep.get("png"):
                try:
                    with open(os.path.join(self.run_dir, rep["png"]), "rb") as f:
                        rep["png"] = f.read()
                except OSError:
                    rep["png"] = None
            nf = len(rep.get("features") or [])
            if rep.get("ok"):
                self.log(f"AI: build {self.n}: {nf} features -> {len(rep.get('bodies') or [])} body(ies), size "
                         + " x ".join(f"{v:g}" for v in rep.get("size") or []) + f" {rep.get('units', '')}, volume {rep.get('volume', 0):.6g}"
                         + (f"  [{len(rep['warnings'])} warning(s)]" if rep.get("warnings") else ""))
            else:
                self.log(f"AI: build {self.n}: {nf} features -> {len(rep.get('errors') or [])} error(s): "
                         + "; ".join((rep.get("errors") or [])[:3]))
            if self.on_build:
                self.on_build(rep)

    def run(self):
        while not self.stop.is_set():
            self.poll()
            self.stop.wait(0.3)
        self.poll()


def _parse_claude(e, log, texts, usage):
    t = e.get("type")
    if t == "system" and e.get("subtype") == "init":
        st = {s.get("name"): s.get("status") for s in e.get("mcp_servers") or []}
        if st.get("gfem") not in (None, "connected"):
            log(f"AI: gfem tool server status: {st.get('gfem')}")
        if e.get("model"):
            log(f"AI: model {e['model']}")
    elif t == "assistant":
        for c in e.get("message", {}).get("content") or []:
            if c.get("type") == "text" and c.get("text", "").strip():
                texts.append(c["text"])
                log("AI: " + c["text"].strip())
            elif c.get("type") == "tool_use" and not str(c.get("name", "")).endswith("build_cad"):
                log(f"AI: (reading {str(c.get('name')).split('__')[-1]})")
    elif t == "result":
        u = e.get("usage") or {}
        usage.update(input=u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0),
                     cache_read=u.get("cache_read_input_tokens", 0), output=u.get("output_tokens", 0))
        if e.get("total_cost_usd") is not None:
            usage["cost_usd"] = round(float(e["total_cost_usd"]), 4)
        if e.get("is_error") or (e.get("subtype") not in (None, "success")):
            raise AIError("Claude: " + str(e.get("result") or e.get("subtype") or "error"))


def _parse_codex(e, log, texts, usage):
    t = e.get("type", "")
    item = e.get("item") or {}
    if t == "item.completed" and item.get("type") == "agent_message" and item.get("text", "").strip():
        texts.append(item["text"])
        log("AI: " + item["text"].strip())
    elif t == "item.completed" and item.get("type") == "reasoning" and item.get("text"):
        pass
    elif t in ("item.started", "item.completed") and item.get("type") == "mcp_tool_call":
        if item.get("status") == "failed" or item.get("error"):
            log(f"AI: tool call failed: {item.get('error') or item.get('status')}")
    elif t == "item.completed" and item.get("type") == "command_execution":
        log("AI: (the model tried to run a shell command; ignored)")
    elif t == "turn.completed":
        u = e.get("usage") or {}
        usage["input"] = usage.get("input", 0) + u.get("input_tokens", 0)
        usage["cache_read"] = usage.get("cache_read", 0) + u.get("cached_input_tokens", 0)
        usage["output"] = usage.get("output", 0) + u.get("output_tokens", 0)
    elif t == "turn.failed":
        raise AIError("Codex: " + str((e.get("error") or {}).get("message") or "turn failed"))
    elif t == "error":
        msg = str(e.get("message") or e)
        if "reconnect" in msg.lower() or "retry" in msg.lower():
            log("AI: Codex: " + msg)
        else:
            raise AIError("Codex: " + msg)


def run_cli(provider, prompt, attachments=(), base_features=None, units="mm", model=None, effort=None,
            visual=True, log=None, on_build=None, should_stop=None, timeout=1800, run_dir=None):
    """Generate a part through a signed-in CLI. Same result dict as `gfem.cad.ai.run_agent`."""
    log = log or (lambda s: None)
    st = status(provider)
    if not st["installed"]:
        raise AIError(f"{CLIS[provider]['exe']} is not installed. Install it with: {st['install']}")
    if not st["logged_in"]:
        raise AIError(f"not signed in to {CLIS[provider]['label'].split(' (')[0]}: press 'Sign in' first")
    run_dir = run_dir or tempfile.mkdtemp(prefix="gfem_ai_")
    os.makedirs(run_dir, exist_ok=True)
    base_file = None
    if base_features:
        base_file = os.path.join(run_dir, "base.json")
        with open(base_file, "w", encoding="utf-8") as f:
            json.dump(base_features, f)
    atts = [a if isinstance(a, dict) else load_attachment(a) for a in attachments]
    text = build_request(prompt, base_features, units)
    extra = [a["text"] for a in atts if a["kind"] == "text"]
    if extra:
        text = "\n\n".join(extra) + "\n\n" + text
    if provider == "codex-cli" and any(a["kind"] == "pdf" for a in atts):
        log("AI: note: Codex CLI cannot read PDF attachments; convert the drawing to PNG")
    argv, stdin, env = build_command(provider, text, run_dir, units, base_file, model, effort, visual,
                                     [a for a in atts if a["kind"] in ("image", "pdf")])
    builds = []

    def ob(rep):
        builds.append(rep)
        if on_build:
            on_build(rep)
    w = _Watcher(run_dir, ob, log)
    w.start()
    texts, usage = [], dict(input=0, output=0, cache_read=0)
    t0 = time.time()
    log(f"AI: starting {CLIS[provider]['exe']} ({st.get('version') or 'unknown version'})…")
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                            errors="replace", env=env, cwd=run_dir, bufsize=1, **_NOWIN)
    err_lines = []
    threading.Thread(target=lambda: err_lines.extend(proc.stderr), daemon=True).start()
    if stdin is not None:
        try:
            proc.stdin.write(stdin)
            proc.stdin.close()
        except OSError:
            pass
    killer_stop = threading.Event()

    def killer():
        while not killer_stop.wait(0.5):
            if (should_stop and should_stop()) or time.time() - t0 > timeout:
                log("AI: stopped by the user" if should_stop and should_stop() else "AI: time limit reached")
                proc.terminate()
                return
    threading.Thread(target=killer, daemon=True).start()
    error = None
    parse = _parse_claude if provider == "claude-cli" else _parse_codex
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            try:
                parse(e, log, texts, usage)
            except AIError as ex:
                error = str(ex)
        proc.wait()
    finally:
        killer_stop.set()
        w.stop.set()
        w.join(5)
    if proc.returncode not in (0, None) and not error and not builds and not (should_stop and should_stop()):
        tail = "".join(err_lines[-15:]).strip()
        error = f"{CLIS[provider]['exe']} exited with code {proc.returncode}" + (f": {tail[-600:]}" if tail else "")
    best = next((b for b in reversed(builds) if b.get("ok")), None)
    res = AIResult(ok=best is not None, features=best["features"] if best else [], name=(best or {}).get("name"),
                   units=(best or {}).get("units", units), summary=(texts[-1].strip() if texts else ""),
                   report=best, png=(best or {}).get("png"), builds=len(builds), usage=usage,
                   seconds=round(time.time() - t0, 2), run_dir=run_dir)
    if best is None:
        res["error"] = error or "the model did not produce a valid geometry"
    elif error:
        log("AI: " + error)
    return res
