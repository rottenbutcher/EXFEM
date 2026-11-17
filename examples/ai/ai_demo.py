"""AI CAD + AI simulation from Python (uses your signed-in Claude account by default).

    python examples/ai/ai_demo.py [claude-cli|codex-cli|anthropic|openai]
"""
import os
import sys

from gfem import project as P
from gfem.cad import ai as AI
from gfem.cad import CadModel

provider = sys.argv[1] if len(sys.argv) > 1 else "claude-cli"
here = os.path.dirname(os.path.abspath(__file__))

# 1. a part from a description
r = AI.generate("A 100 x 60 x 10 mm mounting plate with four d8 holes 10 mm from the corners and a "
                "central d30 boss 20 mm high with a d16 bore.", provider=provider, log=print)
print("size", r["report"]["size"], "volume", r["report"]["volume"])

# 2. a change in the same conversation
r2 = AI.generate("Make the boss 30 mm high and add a 1 mm chamfer on its top edge.", provider=provider,
                 base_features=r["features"], session=r.get("session"), log=print)
cad = CadModel(r2["features"])
cad.regenerate()
cad.export(os.path.join(here, "mounting_plate.step"))

# 3. a part from a drawing
r3 = AI.generate("Model the part in the attached drawing.", provider=provider,
                 attachments=[os.path.join(here, "plate_drawing.png")], log=print)
print("drawing part volume", r3["report"]["volume"], "(expected 74921)")

# 4. a complete simulation with a design requirement
r4 = AI.generate("Aluminium cantilever bar 200 mm long, 20 x 10 mm section, fixed at x=0, 100 N downward "
                 "at the free end. Safety factor at least 3.", provider=provider, project=P.new_project("mm"), log=print)
print("result", r4.get("solve_summary"))
P.save(r4["project"], os.path.join(here, "cantilever.gfem.json"))
