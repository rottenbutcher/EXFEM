"""gfem CAD tools as an MCP server (stdio, JSON-RPC 2.0, no dependencies).

This is how a *logged-in* Claude Code or OpenAI Codex CLI builds geometry in gfem: the CLI is
started with this server attached, the model calls ``build_cad``, and every build is validated,
regenerated on OpenCASCADE, measured and rendered exactly as in the API-key path
(`gfem.cad.ai.evaluate`). Results go back to the model (text + PNG) and are also written to a
run directory that the gfem GUI/CLI watches:

    <run_dir>/build_0001.json   report (features, errors, warnings, size, volume, bodies, ...)
    <run_dir>/build_0001.png    4-view preview
    <run_dir>/best.json         last successful build

Run:  python -m gfem.cad.mcp_server --run-dir DIR [--units mm] [--base FEATURES.json] [--no-visual]

It can also be registered permanently, e.g. ``claude mcp add gfem -- python -m gfem.cad.mcp_server``,
to use gfem CAD from any Claude Code / Codex session.
"""
import argparse
import base64
import json
import os
import sys
import threading
import time

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")


class Server:
    def __init__(self, run_dir=None, units="mm", base=None, visual=True, out=None, project=None):
        self.project = project
        self._sim = None
        self.nsim = 0
        self.run_dir = run_dir
        self.units = units
        self.base = base or []
        self.visual = visual
        self.n = 0
        self.out = out or sys.stdout
        self.lock = threading.Lock()
        if run_dir:
            os.makedirs(run_dir, exist_ok=True)

    # ------------------------------------------------------------ tools
    def tools(self):
        from .ai import TOOL_NAME, TOOL_DESCRIPTION, TOOL_SCHEMA, SYSTEM_PROMPT
        guide = SYSTEM_PROMPT.split("# Conventions", 1)[1]
        return [
            dict(name=TOOL_NAME, description=TOOL_DESCRIPTION + " Read the `cad_guide` tool output once before the first build.",
                 inputSchema=TOOL_SCHEMA),
            dict(name="cad_guide", description="Returns the gfem feature-tree reference (feature types, fields, sketch planes, "
                 "conventions) and the current model if one is being modified. Call it before the first build_cad.",
                 inputSchema={"type": "object", "properties": {}}),
            dict(name="current_model", description="Returns the feature tree of the model the user wants to modify "
                 "(empty if a new part is requested).", inputSchema={"type": "object", "properties": {}}),
        ] + (self._sim_tools() if self.project else []), "# Conventions" + guide

    def _sim_tools(self):
        from .ai_sim import SIM_TOOLS
        return SIM_TOOLS

    @property
    def sim(self):
        if self._sim is None:
            from .ai_sim import SimTools
            self._sim = SimTools(self.project, workdir=os.path.join(self.run_dir or os.path.dirname(self.project), "sim_work"))
        return self._sim

    def _log_sim(self, name, args, result, err):
        if not self.run_dir:
            return
        self.nsim += 1
        tmp = os.path.join(self.run_dir, f"sim_{self.nsim:04d}.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dict(tool=name, args=args, result=result, error=err, time=time.time()), f, default=float)
        os.replace(tmp, tmp[:-4])

    def call(self, name, args):
        from .ai import TOOL_NAME, evaluate, report_text
        from .ai_sim import SIM_TOOLS
        if self.project and name in {t["name"] for t in SIM_TOOLS}:
            try:
                res = self.sim.call(name, args)
                err = None
            except Exception as e:  # noqa: BLE001
                res, err = None, f"{type(e).__name__}: {e}"
            self._log_sim(name, args, res, err)
            if err:
                return [dict(type="text", text="ERROR: " + err)], True
            return [dict(type="text", text=json.dumps(res, default=float))], False
        if name == "cad_guide":
            extra = ""
            if self.project:
                from .ai_sim import SIM_GUIDE
                from .. import project as P
                extra = "\n\n" + SIM_GUIDE.format(library=", ".join(m["name"] for m in P.LIBRARY))
            return [dict(type="text", text=self.tools()[1] + extra + ("\n\n# Current model\n" + json.dumps(self.base) if self.base else ""))], False
        if name == "current_model":
            return [dict(type="text", text=json.dumps(dict(units=self.units, features=self.base)))], False
        if name != TOOL_NAME:
            return [dict(type="text", text=f"unknown tool {name}")], True
        feats = args.get("features")
        if isinstance(feats, str):                       # some clients send nested JSON as a string
            try:
                feats = json.loads(feats)
            except ValueError:
                return [dict(type="text", text="features must be a JSON array of feature objects")], True
        units = args.get("units") or self.units
        rep = evaluate(feats, units=units, render=self.visual)
        rep["name"] = args.get("name") or "AI part"
        with self.lock:
            self.n += 1
            k = self.n
        if self.run_dir:
            self._write(k, rep)
        if self.project and rep["ok"]:
            try:
                stale = self.sim.set_geometry(rep["features"], units=units, name=args.get("name"))
                rep["warnings"] = list(rep["warnings"]) + ["project geometry updated; call get_model for face ids before adding conditions"] + (
                    [f"conditions {stale} no longer match any face: remove and re-add them"] if stale else [])
            except Exception as e:  # noqa: BLE001
                rep["warnings"] = list(rep["warnings"]) + [f"project update failed: {e}"]
        content = [dict(type="text", text=report_text(rep))]
        if rep.get("png"):
            content.append(dict(type="image", data=base64.b64encode(rep["png"]).decode(), mimeType="image/png"))
        return content, not rep["ok"]

    def _write(self, k, rep):
        d = {key: rep.get(key) for key in ("ok", "errors", "warnings", "features", "name", "units", "size", "bbox",
                                            "volume", "bodies", "sheets", "lines", "n_faces", "n_edges")}
        d["build"], d["time"] = k, time.time()
        base = os.path.join(self.run_dir, f"build_{k:04d}")
        if rep.get("png"):
            with open(base + ".png", "wb") as f:
                f.write(rep["png"])
            d["png"] = os.path.basename(base + ".png")
        tmp = base + ".json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, default=float)
        os.replace(tmp, base + ".json")                  # atomic: the watcher never sees half a file
        if rep["ok"]:
            tmp = os.path.join(self.run_dir, "best.json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(d, f, default=float)
            os.replace(tmp, os.path.join(self.run_dir, "best.json"))

    # ------------------------------------------------------------ JSON-RPC
    def send(self, obj):
        self.out.write(json.dumps(obj, separators=(",", ":"), default=float) + "\n")
        self.out.flush()

    def handle(self, msg):
        mid, method = msg.get("id"), msg.get("method")
        if method is None:                               # a response / unknown: ignore
            return
        params = msg.get("params") or {}
        try:
            if method == "initialize":
                v = params.get("protocolVersion")
                res = dict(protocolVersion=v if v in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                           capabilities=dict(tools=dict(listChanged=False)),
                           serverInfo=dict(name="gfem-cad", version=_version()),
                           instructions="gfem CAD: build parametric parts with build_cad (call cad_guide first).")
            elif method == "ping":
                res = {}
            elif method == "tools/list":
                res = dict(tools=self.tools()[0])
            elif method == "tools/call":
                content, err = self.call(params.get("name"), params.get("arguments") or {})
                res = dict(content=content, isError=err)
            elif method in ("resources/list", "prompts/list", "resources/templates/list"):
                res = {method.split("/")[0]: [], **({"resourceTemplates": []} if "templates" in method else {})}
            elif method.startswith("notifications/"):
                return
            else:
                if mid is not None:
                    self.send(dict(jsonrpc="2.0", id=mid, error=dict(code=-32601, message=f"method not found: {method}")))
                return
        except Exception as e:  # noqa: BLE001
            if mid is not None:
                self.send(dict(jsonrpc="2.0", id=mid, result=dict(content=[dict(type="text", text=f"internal error: {e}")], isError=True)))
            return
        if mid is not None:
            self.send(dict(jsonrpc="2.0", id=mid, result=res))

    def serve(self, inp=None):
        inp = inp or sys.stdin
        for line in inp:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self.send(dict(jsonrpc="2.0", id=None, error=dict(code=-32700, message="parse error")))
                continue
            for m in (msg if isinstance(msg, list) else [msg]):
                self.handle(m)


def _version():
    try:
        from .. import __version__
        return __version__
    except Exception:  # noqa: BLE001
        return "0"


def main(argv=None):
    p = argparse.ArgumentParser(prog="gfem mcp", description="gfem CAD MCP server (stdio)")
    p.add_argument("--run-dir")
    p.add_argument("--units", default="mm")
    p.add_argument("--base", help="JSON file with the feature tree being modified")
    p.add_argument("--no-visual", action="store_true")
    p.add_argument("--project", help="gfem project (.gfem.json): enables the simulation tools")
    a = p.parse_args(argv)
    base = []
    if a.base:
        with open(a.base, encoding="utf-8") as f:
            base = json.load(f)
    # stdout is the protocol channel: anything else printed by libraries must go to stderr
    out = sys.stdout
    sys.stdout = sys.stderr
    Server(a.run_dir, a.units, base, not a.no_visual, out=out, project=a.project).serve()


if __name__ == "__main__":
    main()
