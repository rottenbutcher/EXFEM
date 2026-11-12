"""gfem - GPU-accelerated nonlinear finite element analysis (JAX).

Python API quick reference::

    import gfem
    m = gfem.Model(X); m.add_block(conn, gfem.TET10, gfem.LinearElastic(210e3, 0.3))
    u, s, info = gfem.StaticSolver(m).solve(gfem.Step(bcs=..., loads=...))
    gfem.run_inp("job.inp")            # Abaqus deck
    gfem.run_file("model.bdf")         # Nastran bulk data / ANSYS .cdb / gmsh .msh
    python -m gfem gui                 # ANSYS-Mechanical-style GUI in the browser
"""
import os as _os
import jax

jax.config.update("jax_enable_x64", True)

from .elements import (ELEMENTS, HEX8, HEX8_PLAIN, HEX20, HEX20R, TET4, TET10,   # noqa: E402
                       WEDGE6)
from .materials import (LinearElastic, StVenantKirchhoff, NeoHookean, MooneyRivlin,  # noqa: E402
                        Yeoh, OrthotropicElastic, J2Plasticity, J2FiniteStrain, UserMaterial)
from .model import Model                                                    # noqa: E402
from .surfaces import Surface                                               # noqa: E402
from .solvers import (Step, State, Monitor, StaticSolver, RiksSolver,       # noqa: E402
                      DynamicSolver, ExplicitSolver, modal, buckling)
from .loads import Amplitude                                                # noqa: E402
from . import mesh, constraints, contact, loads                             # noqa: E402

__version__ = "0.2.0"


def run_inp(path, outdir=".", verbose=True, monitor=None):
    """Run an Abaqus-style input deck; writes .vtu/.pvd per step. Returns (spec, results)."""
    from .analysis import run_spec
    from .io.inp import read_inp_spec
    spec = read_inp_spec(path)
    base = _os.path.splitext(_os.path.basename(path))[0]
    return run_spec(spec, outdir, base, verbose=verbose, monitor=monitor)


def run_file(path, outdir=".", verbose=True, monitor=None):
    """Run any supported solver deck (.inp, .bdf/.nas/.dat, .cdb)."""
    from .analysis import run_spec
    from .io import read_any
    spec = read_any(path)
    base = _os.path.splitext(_os.path.basename(path))[0]
    return run_spec(spec, outdir, base, verbose=verbose, monitor=monitor)
