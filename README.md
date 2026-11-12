# gfem: GPU-accelerated nonlinear FEM with CAD and a Mechanical-style GUI

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-GPU%20%7C%20CPU-A8B9CC?logo=google&logoColor=white)](https://github.com/jax-ml/jax)
[![NVIDIA CUDA](https://img.shields.io/badge/NVIDIA-CUDA-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![AMD ROCm](https://img.shields.io/badge/AMD-ROCm-ED1C24?logo=amd&logoColor=white)](https://www.amd.com/en/products/software/rocm.html)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![PyAMG](https://img.shields.io/badge/PyAMG-AMG%20solver-4B8BBE)](https://github.com/pyamg/pyamg)
[![gmsh](https://img.shields.io/badge/gmsh-mesher-2E6DB4)](https://gmsh.info/)
[![OpenCASCADE](https://img.shields.io/badge/OpenCASCADE-CAD%20kernel-0065A9)](https://dev.opencascade.org/)
[![three.js](https://img.shields.io/badge/three.js-3D%20viewer-000000?logo=threedotjs&logoColor=white)](https://threejs.org/)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES%20modules-F7DF1E?logo=javascript&logoColor=black)](https://developer.mozilla.org/docs/Web/JavaScript)
[![ParaView](https://img.shields.io/badge/ParaView-.vtu%20output-4A90D9)](https://www.paraview.org/)
[![pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC?logo=pytest&logoColor=white)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

gfem is an open-source structural/thermal finite element package written in Python on
[JAX](https://github.com/jax-ml/jax). The same code runs on CPU, NVIDIA and AMD GPUs. It has three layers:

1. **Solver.** Batched element kernels (one `vmap` per element block), automatic-differentiation
   tangents, matrix-free or element-by-element operators, AMG-preconditioned CG on the device.
2. **CAD and interchange.** OpenCASCADE (through gmsh) for STEP/IGES/BREP/STL/DXF import and
   parametric modelling, CAD-associative meshing, and Abaqus / Nastran / ANSYS solver-deck
   readers and writers.
3. **GUI.** A browser application laid out like ANSYS Mechanical: outline tree, details pane,
   3D viewport with face/edge/vertex/body picking, a sketcher, a solution monitor and contour
   results. Because it runs in a browser, the solver can sit on a remote GPU node while you work
   from a laptop.

```bash
pip install -e ".[all]"            # CPU: jax, scipy, gmsh (OpenCASCADE), pyamg, ezdxf, meshio
pip install "jax[cuda12]"          # NVIDIA GPU (no code change); AMD: jax[rocm]
gfem gui                           # GUI at http://127.0.0.1:8765
gfem gui --host 0.0.0.0            # serve the GUI from a GPU workstation / cloud node
gfem run job.inp                   # Abaqus deck  (also .bdf/.nas Nastran, .cdb ANSYS)
gfem convert job.inp job.bdf       # Abaqus <-> Nastran through the neutral model
gfem mesh part.step -o part.inp --size 3
python -m pytest tests             # 39 verification tests against analytical solutions
```

| | |
|---|---|
| ![bracket](docs/img/result_bracket.png) | ![contact](docs/img/contact.png) |
| ![topology](docs/img/topology.png) | ![sketcher](docs/img/sketcher.png) |

## What sets it apart from other open-source FE codes

| | gfem | typical open-source FE |
|---|---|---|
| GPU | Everything (assembly, Newton, contact, explicit loop, topology optimisation) runs through JAX on GPU | CPU only, or GPU for the linear solver only |
| New material | Write one stress (or energy) function; the consistent tangent comes from AD | Hand-derived tangent (UMAT) |
| Memory | Matrix-free JVP tangent option; frozen AMG hierarchy; no global matrix needed | Assembled global matrix |
| CAD | STEP/IGES/BREP import, parametric feature tree, sketcher, booleans, fillets, DXF profiles | Usually needs a separate pre-processor |
| Boundary conditions | Applied to CAD faces; re-attached by geometric signature after edits (topological-naming repair) | Applied to mesh node/element sets |
| Bonded contact | Dual-mortar MPC elimination, passes the non-conforming interface patch test | Penalty or node-to-surface |
| Commercial-deck compatibility | Reads and writes Abaqus `.inp` and Nastran `.bdf`; reads ANSYS `.cdb` | Single native format |
| GUI | Mechanical-style workflow in the browser, can run against a remote GPU server | Separate pre/post tools |
| Optimisation | SIMP topology optimisation and batched multi-load-case solves on the GPU | Separate tools |

## Capabilities

**Elements**: C3D4, C3D10 (also M/H), C3D8 (F-bar anti-locking; R/H/I mapped), C3D8 full,
C3D20, C3D20R, C3D6. Abaqus face numbering (S1..S6) is used for surfaces.

**Materials**: linear isotropic; orthotropic (engineering constants, material axes);
St. Venant–Kirchhoff; Neo-Hookean, Mooney–Rivlin and Yeoh hyperelasticity; J2 plasticity with
tabular isotropic, linear kinematic or combined hardening, or Johnson–Cook (small strain and
large rotation); thermal expansion; Python user materials (`UserMaterial`).

**Procedures**
- Static (Newton–Raphson, tangent predictor, line search, Abaqus-style automatic incrementation with cutback, NLGEOM)
- Riks arc-length (Crisfield cylindrical) for snap-through and limit loads
- Implicit dynamics (HHT-α / Newmark) and explicit dynamics (central difference, lumped mass, on-device `lax.scan`, mass scaling)
- Modal: shift-invert Lanczos with participation factors and effective mass; pre-stress supported
- Linear buckling (eigenvalue) with initial-stress stiffness
- Steady and transient heat transfer (flux, convection, generation), with sequential thermal-stress coupling
- SIMP topology optimisation (density filter, OC update) on unstructured CAD meshes
- Batched multi-load-case linear solves (`optimize.multi_load_cases`)

**Loads and constraints**: displacement (with amplitudes), concentrated force, pressure (dead or
follower), surface traction, total force on a face, moment on a face, gravity, acceleration,
centrifugal, temperature field; fixed support, frictionless support on any plane (MPC),
bonded contact (dual-mortar tie), frictionless contact (augmented Lagrange, finer mesh
automatically chosen as slave), rigid wall with prescribed motion.

**Linear solvers**: sparse direct (SuperLU) • smoothed-aggregation AMG (PyAMG setup, V-cycle on
the device with Chebyshev smoothing), reused across Newton iterations and rebuilt when PCG slows
down • Jacobi-PCG • element-by-element stored tangents or fully matrix-free AD-JVP tangents.

**CAD**: box, cylinder, sphere, cone, torus, wedge; sketch → extrude/revolve (polylines with
arcs, circles, rectangles, slots, DXF import); boolean union/subtract/intersect; fillet and
chamfer (edges selected in the viewer); move/rotate/mirror/scale/copy; linear and circular
patterns; STEP/IGES/BREP import with unit conversion and healing; STL import; shared topology
for conformal multi-body meshes; STEP/STL export.

**Meshing**: gmsh tetrahedral (linear or quadratic), transfinite hex where possible, hex-dominant;
global, curvature and local face/edge/body sizing; quality statistics (SICN histogram).
Every CAD body, face, edge and vertex keeps its identity as a node/element set and surface.

**Interchange**
- Abaqus `.inp` read: parts and instances (translation/rotation), `*INCLUDE`, sets (GENERATE,
  nested, instance-qualified), element/node surfaces, materials, `*TIE`, `*CONTACT PAIR`,
  `*AMPLITUDE`, all procedures above, `*DSLOAD`/`*DLOAD` (P, TRVEC, GRAV, CENTRIF, Pn faces),
  `*TEMPERATURE`, heat-transfer loads. Write: flat deck.
- Nastran bulk data read/write: free, small and large field with continuations; GRID, CTETRA,
  CHEXA, CPENTA, PSOLID, MAT1, MAT4, SPC/SPC1, FORCE, PLOAD4, GRAV, LOAD, TEMP, EIGRL, SOL
  101/103/105/106.
- ANSYS `.cdb` read: NBLOCK, EBLOCK (SOLID185/186/187/285 and degenerate forms), MPDATA, CMBLOCK
  components, D, F, SFE.
- gmsh `.msh`/UNV/MED read (physical groups become sets). ParaView `.vtu`/`.pvd` output; CSV.

## The GUI

`gfem gui` opens a Mechanical-style window:

- **Ribbon**: Home / Geometry / Mesh / Environment / Solution / Results / Display.
- **Outline tree**: Geometry (feature tree and bodies), Materials, Connections, Named Selections,
  Mesh (sizing controls), the analysis (settings, supports, loads), and Solution (Solution
  Information and result objects). Each item shows a status light. Right-click to delete,
  suppress or edit; double-click to solve or mesh.
- **Details pane**: an editable property grid for every object. Scope a condition by selecting
  faces in the viewer and pressing **Apply**, or pick a named selection.
- **Viewport**: selection filters (vertex/edge/face/body; Ctrl/Shift to add), standard views,
  section plane, wireframe, edges, a triad, hover information, and drag-and-drop CAD or deck
  import.
- **Sketcher**: polyline/arc/rectangle/circle/slot with grid snap and numeric entry, DXF import,
  extrude or revolve as a new body or add/cut.
- **Solution Information**: live log, force-convergence chart (residual against cumulative
  iteration), time/LPF and compliance histories.
- **Results**: total/directional deformation, equivalent/principal/normal stress, stress
  intensity, plastic strain, total strain, temperature, safety factor, topology density.
  Banded or smooth contours with an ANSYS-style legend, Max/Min markers, a deformation scale
  (auto/true/×N), animation of load history and mode shapes, probes, and tabular and graph views.
- **Export**: Abaqus `.inp`, Nastran `.bdf`, STEP, STL, ParaView `.vtu`, CSV, PNG.
- **Examples**: L-bracket, plate with hole (Kt), cantilever modal, column buckling, heat-sink
  thermal-stress, topology optimisation, punch contact, rubber pad (hyperelastic NLGEOM).

## Python API

```python
import gfem
from gfem import mesh
X, c = mesh.hex8_to_hex20(*mesh.box_hex(100, 10, 10, 20, 2, 2))
m = gfem.Model(X)
m.add_block(c, gfem.HEX20R, gfem.J2FiniteStrain(210e3, 0.3, [[250, 0], [400, 0.2]]))
fix = mesh.nodes_where(X, lambda x, y, z: x < 1e-9)
tip = mesh.nodes_where(X, lambda x, y, z: x > 100 - 1e-9)
step = gfem.Step(bcs=[(fix, d, 0.0) for d in range(3)] + [(tip, 1, -20.0)], dt_init=0.1)
u, states, info = gfem.StaticSolver(m, linear_solver="amg").solve(step)
```

CAD to results in a script:

```python
from gfem.cad import CadModel, mesh_cad
from gfem.spec import new_step
from gfem.analysis import run_spec
cad = CadModel([dict(id="F1", type="import", file="part.step")]); info = cad.regenerate()
sp, stats = mesh_cad(cad, size=3.0, order=2)          # FACE_<id>, EDGE_<id>, BODY_<id> sets
...                                                    # materials, sections, steps
model, results = run_spec(sp, "out", "job")
```

## Verification (`tests/`)

Patch tests for every element and every linear solver; J2 uniaxial with hardening (exact);
nearly incompressible Neo-Hookean at stretch 2; cantilever against Timoshenko; Lamé thick
cylinder under pressure (< 1%); non-conforming dual-mortar tie (uniform stress to 1e-4);
frictionless contact force; rigid-wall indentation; cantilever eigenfrequency (< 3%); Euler
buckling (< 3%); free and constrained thermal expansion; Riks against load control; HHT
step-load dynamic factor; explicit dynamic factor; Abaqus and Nastran round trips; Nastran
small-field and continuation parsing; ANSYS CDB; Abaqus parts/instances/contact pair; CAD
STEP export and re-import; topological-naming repair; CAD-to-results stress; project builder
for every analysis type.

## Architecture

```
gfem/elements.py     element/facet types, quadrature, Abaqus face tables
gfem/materials.py    constitutive models (AD tangents), from_props()
gfem/model.py        Block kernels (force, tangent, geometric stiffness, mass), EbE operators
gfem/surfaces.py     skin extraction, outward facets, projection onto facets
gfem/constraints.py  MPC elimination, dual-mortar tie, symmetry, coupling
gfem/contact.py      penalty / augmented-Lagrange contact, rigid wall
gfem/loads.py        amplitudes, pressure/traction/force/moment/gravity/centrifugal, follower pressure
gfem/linalg.py       PCG, AMG (device V-cycle), direct
gfem/solvers.py      Static, Riks, Dynamic (HHT), Explicit, modal, buckling
gfem/thermal.py      heat transfer
gfem/optimize.py     topology optimisation, batched load cases
gfem/spec.py         neutral model shared by every front end
gfem/analysis.py     Spec -> Model builder, procedure runner, post-processing
gfem/io/             Abaqus, Nastran, ANSYS CDB, VTK
gfem/cad/            OpenCASCADE feature tree, DXF, CAD-associative meshing
gfem/project.py      GUI project model, units, material library
gfem/gui/            HTTP/SSE server, examples, browser front-end (three.js)
```

## Roadmap
1. Friction (Coulomb, regularised stick) and self-contact
2. Shells (S4R/S3) and beams (B31) for thin CAD parts; mid-surface extraction
3. Remote points / bolt pretension / joints
4. Finite-strain multiplicative plasticity (log strain), viscoelasticity, damage
5. Multi-GPU domain decomposition (`jax.sharding`), cuDSS direct solver, HDF5 results
6. Parasolid / CATIA / NX native import (needs commercial translators); for now export STEP or IGES from those systems

License: Apache-2.0
