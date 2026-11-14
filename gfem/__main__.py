"""Command line interface.

  gfem run job.inp|model.bdf|model.cdb [-o outdir] [--solver auto|direct|amg|jacobi]
  gfem gui [--host 127.0.0.1] [--port 8765] [--workdir gfem_work] [--no-browser]
  gfem convert in.inp out.bdf          (Abaqus <-> Nastran via the neutral model)
  gfem mesh part.step -o part.inp [--size 5] [--order 2]
  gfem bench [n ...]                   kernel throughput (CPU/GPU)
  gfem info                            devices / versions
  gfem ai "describe the part" [-o part.step|part.gfem.json|features.json] [--attach drawing.png]
          [--provider anthropic|openai|compatible] [--model M] [--modify project.gfem.json]
  gfem ai --set-key anthropic|openai    store an API key in ~/.gfem/ai.json (0600)
"""
import argparse
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if (not argv or argv[0] != "ai") and ("#" in argv or any(a.startswith("#") for a in argv)):   # a pasted README comment (cmd/PowerShell)
        argv = argv[:next(i for i, a in enumerate(argv) if a.startswith("#"))]
    if argv and argv[0].lower().endswith((".inp", ".bdf", ".nas", ".dat", ".cdb")):
        argv = ["run"] + argv                     # backwards compatible: python -m gfem job.inp
    p = argparse.ArgumentParser(prog="gfem", description="GPU nonlinear FEM")
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="run a solver deck")
    r.add_argument("deck")
    r.add_argument("-o", "--outdir", default=".")
    r.add_argument("--solver", default="auto")
    r.add_argument("--tangent", default="ebe", choices=["ebe", "jvp"])
    g = sub.add_parser("gui", help="start the graphical user interface (browser)")
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--port", type=int, default=8765)
    g.add_argument("--workdir", default="gfem_work")
    g.add_argument("--no-browser", action="store_true")
    c = sub.add_parser("convert", help="convert solver decks (.inp/.bdf/.cdb/.msh -> .inp/.bdf)")
    c.add_argument("src")
    c.add_argument("dst")
    m = sub.add_parser("mesh", help="mesh a CAD file (STEP/IGES/BREP/STL) and write a deck")
    m.add_argument("cad")
    m.add_argument("-o", "--out", required=True)
    m.add_argument("--size", type=float, default=None)
    m.add_argument("--order", type=int, default=2)
    m.add_argument("--kind", default="tet")
    b = sub.add_parser("bench", help="kernel throughput benchmark")
    b.add_argument("n", nargs="*", type=int)
    sub.add_parser("info", help="show devices and versions")
    ai = sub.add_parser("ai", help="text-to-CAD: an LLM (Claude / GPT) builds the part you describe")
    ai.add_argument("prompt", nargs="*", help="description of the part (or of the change with --modify)")
    ai.add_argument("-o", "--out", help="output: .step/.stp/.iges/.brep/.stl (CAD), .gfem.json (project), .json (features)")
    ai.add_argument("--attach", action="append", default=[], help="drawing/photo (png/jpg/webp/gif), PDF, DXF, STEP or text")
    ai.add_argument("--provider", choices=["anthropic", "openai", "compatible"])
    ai.add_argument("--model")
    ai.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    ai.add_argument("--base-url", help="OpenAI-compatible server URL (e.g. http://localhost:11434/v1)")
    ai.add_argument("--modify", help="project (.gfem.json) or features (.json) to modify")
    ai.add_argument("--units", default=None, choices=["mm", "m", "cm", "in"])
    ai.add_argument("--max-turns", type=int)
    ai.add_argument("--no-visual", action="store_true", help="do not send rendered views back to the model")
    ai.add_argument("--preview", help="write the 4-view preview PNG here")
    ai.add_argument("--set-key", metavar="PROVIDER", choices=["anthropic", "openai", "compatible"],
                    help="store an API key (read from stdin / prompt) and exit")
    a = p.parse_args(argv)
    if a.cmd == "run":
        from . import run_file
        run_file(a.deck, a.outdir) if a.solver == "auto" else _run(a)
    elif a.cmd == "gui":
        from .gui.server import serve
        serve(a.host, a.port, a.workdir, open_browser=not a.no_browser)
    elif a.cmd == "convert":
        from .io import read_any, write_any
        sp = read_any(a.src)
        write_any(sp, a.dst)
        print(f"{a.src} -> {a.dst}: {sp.summary()}")
    elif a.cmd == "mesh":
        from .cad import CadModel, mesh_cad
        from .io import write_any
        cad = CadModel([dict(id="F1", type="import", file=a.cad)])
        cad.regenerate()
        sp, st = mesh_cad(cad, size=a.size, order=a.order, kind=a.kind)
        sp.materials["MATERIAL"] = dict(E=200e3, nu=0.3, density=7.85e-9)
        for k in sp.elsets:
            sp.sections.append(dict(elset=k, material="MATERIAL"))
        write_any(sp, a.out)
        print(f"{a.cad} -> {a.out}: {st['nodes']} nodes, {st['elements']} elements, min quality {st.get('q_min', 0):.3f}")
    elif a.cmd == "bench":
        sys.argv = ["bench"] + [str(x) for x in a.n]
        import runpy
        import os
        runpy.run_path(os.path.join(os.path.dirname(__file__), "..", "examples", "benchmark.py"), run_name="__main__")
    elif a.cmd == "info":
        import jax
        from . import __version__
        print(f"gfem {__version__}, jax {jax.__version__}, backend {jax.default_backend()}, devices {jax.devices()}")
        try:
            import gmsh  # noqa: F401
            print("CAD kernel: gmsh/OpenCASCADE available")
        except Exception:  # noqa: BLE001
            print("CAD kernel: gmsh not installed (pip install gmsh)")
    elif a.cmd == "ai":
        return _ai(a)
    else:
        p.print_help()


def _ai(a):
    import getpass
    import json
    import os
    from .cad import ai as AI
    if a.set_key:
        key = sys.stdin.readline().strip() if not sys.stdin.isatty() else getpass.getpass(f"{a.set_key} API key: ")
        AI.save_settings(keys={a.set_key: key})
        print(f"{'saved' if key else 'removed'} {a.set_key} key in {AI.settings_path()}")
        return 0
    prompt = " ".join(a.prompt).strip()
    if not prompt:
        print("describe the part, e.g.  gfem ai \"flange d120 with 6 bolt holes d10 on a d95 circle\" -o flange.step")
        return 2
    base, units, project = None, a.units or "mm", None
    if a.modify:
        with open(a.modify, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, list):
            base = d
        elif "features" in d and "geometry" not in d:
            base, units = d["features"], a.units or d.get("units", units)
        else:
            project = d
            base, units = d["geometry"]["features"], a.units or d.get("units", units)
    res = AI.generate(prompt, provider=a.provider, model=a.model, base_url=a.base_url, effort=a.effort,
                      attachments=a.attach, base_features=base, units=units, max_turns=a.max_turns,
                      visual=False if a.no_visual else None, log=print)
    if res.get("usage"):
        u = res["usage"]
        print(f"tokens: {u['input']} in (+{u.get('cache_read', 0)} cached), {u['output']} out; {res['seconds']} s, {res['builds']} build(s)")
    if not res["ok"]:
        print("ERROR:", res.get("error"))
        return 1
    if res.get("summary"):
        print("\n" + res["summary"])
    if a.preview and res.get("png"):
        with open(a.preview, "wb") as f:
            f.write(res["png"])
    out = a.out or ((res.get("name") or "ai_part").strip().replace(" ", "_") + ".gfem.json")
    ext = out.lower()
    if ext.endswith(".gfem.json"):
        from . import project as P
        pr = project or P.new_project(res["units"])
        pr["geometry"]["features"] = res["features"]
        pr["name"] = pr.get("name") if project else (res.get("name") or "AI part")
        P.save(pr, out)
    elif ext.endswith(".json"):
        with open(out, "w", encoding="utf-8") as f:
            json.dump(dict(name=res.get("name"), units=res["units"], features=res["features"]), f, indent=1)
    else:
        from .cad import CadModel
        cad = CadModel(res["features"], units=res["units"])
        cad.regenerate()
        cad.export(out)
    print(f"wrote {os.path.abspath(out)}")
    return 0


def _run(a):
    import os
    from .analysis import run_spec
    from .io import read_any
    sp = read_any(a.deck)
    run_spec(sp, a.outdir, os.path.splitext(os.path.basename(a.deck))[0],
             options=dict(linear_solver=a.solver, tangent=a.tangent))


if __name__ == "__main__":
    sys.exit(main())
