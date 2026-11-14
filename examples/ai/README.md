# AI CAD / AI simulation examples

Sign in once (or set an API key), then use the GUI (**AI CAD**) or the command line.

```bash
gfem ai --login claude-cli
```

## Prompts that work well

**CAD**
- Flange: outer diameter 120 mm, thickness 15 mm, central hub d60 x 40 mm high with a d30 bore through everything, 6 bolt holes d11 equally spaced on a d95 bolt circle, 3 mm fillet where the hub meets the flange.
- L-shaped steel bracket, 5 mm plate, horizontal leg 80 x 60 mm, vertical leg 60 mm high, one triangular gusset 40 x 40 mm in the middle, two d9 holes in each leg, 3 mm inner bend fillet.
- Stepped shaft along X: d40 x 60 long, then d30 x 80, then d25 x 40, 1 mm chamfers on the ends, a 8 x 4 mm keyway 40 mm long on the d30 section.
- Model the part in the attached drawing. (attach `plate_drawing.png`: 120 x 80 x 10 plate, expected volume 74 921 mm³)

**Follow-up changes** (Modify mode, same conversation)
- Make the bolt holes M12 clearance (d13) and add a 1 mm chamfer on the outer edge.
- Add a second identical hole 25 mm to the right of the first one.

**Simulation** (Simulate mode or `--simulate`)
- Aluminium cantilever bar 200 mm long, 20 x 10 mm section, fixed at x=0, 100 N downward on the free end. Safety factor at least 3; compare the deflection with beam theory.
- Steel L-bracket: 6 mm plate, base leg 100 x 60 mm on z=0, vertical leg 80 mm high at x=0, two d10 bolt holes in the base. Bolt holes fixed, 1500 N downward on the top face of the vertical leg. Safety factor >= 2; if not met, make the plate thicker and solve again.
- First three natural frequencies of a 300 x 30 x 3 mm steel cantilever strip; the first one must stay above 50 Hz.

## Command line

```bash
gfem ai "model the part in this drawing" --attach examples/ai/plate_drawing.png -o plate.step --preview plate.png
```

```bash
gfem ai "aluminium cantilever 200 x 20 x 10, fixed at x=0, 100 N down at the tip; safety factor >= 3" --simulate -o cantilever.gfem.json
```

## Python

`ai_demo.py` generates a part, modifies it in the same conversation, and runs a simulation.
