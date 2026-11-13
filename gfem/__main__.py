"""Command line interface.

  gfem run job.inp|model.bdf|model.cdb [-o outdir] [--solver auto|direct|amg|jacobi]
  gfem gui [--host 127.0.0.1] [--port 8765] [--workdir gfem_work] [--no-browser]
  gfem convert in.inp out.bdf          (Abaqus <-> Nastran via the neutral model)
  gfem mesh part.step -o part.inp [--size 5] [--order 2]
  gfem bench [n ...]                   kernel throughput (CPU/GPU)
  gfem info                            devices / versions
"""
import argparse
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "#" in argv or any(a.startswith("#") for a in argv):   # a pasted README comment (cmd/PowerShell)
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
    else:
        p.print_help()


def _run(a):
    import os
    from .analysis import run_spec
    from .io import read_any
    sp = read_any(a.deck)
    run_spec(sp, a.outdir, os.path.splitext(os.path.basename(a.deck))[0],
             options=dict(linear_solver=a.solver, tangent=a.tangent))


if __name__ == "__main__":
    main()
