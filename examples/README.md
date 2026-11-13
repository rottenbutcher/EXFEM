# Examples

## 1. GUI projects (`projects/`)
Start the GUI (`gfem gui`), then **Open** a project file, or pick the same model from the
**Examples** menu, and press **Solve (F5)**.

| File | Analysis | Shows |
|---|---|---|
| `bracket.gfem.json` | static | sketch-extruded L-bracket, fixed bolt holes, pin load, safety factor |
| `plate_hole.gfem.json` | static | stress concentration, symmetry, local mesh sizing |
| `modal.gfem.json` | modal | cantilever natural frequencies, effective mass |
| `buckling.gfem.json` | eigenvalue buckling | Euler column |
| `heatsink.gfem.json` | thermal-stress | patterned fins, heat flux + convection -> thermal stress |
| `topopt.gfem.json` | topology optimisation | SIMP cantilever design space |
| `contact.gfem.json` | static, nonlinear | punch with frictionless contact |
| `rubber.gfem.json` | static, NLGEOM | Neo-Hookean pad, 30 % compression |
| `panel.gfem.json` | eigenvalue buckling | stiffened panel: shell skin + beam stringers |
| `frame.gfem.json` | modal | 3D steel frame of I-section beams |
| `seismic.gfem.json` | response spectrum | CQC seismic analysis of the frame |
| `random.gfem.json` | random vibration | base PSD on the bracket, 1-sigma stress, fatigue safety factor |

## 2. CAD files (`cad/`)
Import with **Geometry > Import CAD** or drag-and-drop into the viewport.
`bracket.step`, `flanged_hub.step` (filleted solid), `stiffened_panel.step` (sheet + wire
bodies -> shells and beams), `gusset_profile.dxf` (2D profile with holes -> **DXF** extrude).

## 3. Solver decks (`decks/`)
Run from the command line or import with **Home > Solver deck**:

```bash
gfem run examples/decks/bracket.inp -o out
```

| File | Content |
|---|---|
| `bracket.inp`, `bracket.bdf` | C3D10 bracket, Abaqus and Nastran versions of the same model |
| `plate_hole.inp` | plate with hole, frictionless support, pressure |
| `contact.inp` | two bodies, `*CONTACT PAIR`, NLGEOM |
| `panel.inp`, `panel.bdf` | S4 shells + B31 beams, `*BUCKLE` / SOL 105 |
| `spring_mass_modal.inp` | `SPRINGA` + `MASS`, `*FREQUENCY` (f = 3.183 Hz) |

## 4. Python scripts
```bash
python examples/shells_and_beams.py
```

| Script | Content |
|---|---|
| `shells_and_beams.py` | Scordelis-Lo roof benchmark; stiffened plate with and without stringer |
| `spectrum_random_fatigue.py` | response spectrum (CQC), random vibration, 3-sigma fatigue |
| `friction_bolt_remote.py` | Coulomb friction (Rx/Rz = mu), bolt pretension to a target force |
| `postbuckling.py` | imperfect column, Riks path vs the elastica |
| `plastic_cantilever.py` | large-deflection elasto-plastic bending, load and unload |
| `cad_to_results.py` | CAD feature tree -> mesh -> solve -> ParaView, Abaqus/Nastran export |
| `topology_optimization.py` | SIMP on a structured mesh |
| `multi_load_cases.py` | 64 load cases in one batched solve |
| `gpu_showcase.py` | large static / modal / buckling / 256-load-case timings (CPU vs GPU) |
| `benchmark.py` | kernel throughput (internal force, tangent, K*v) |
| `run_all.py` | runs every deck and script above and checks the key numbers |
