import glob
import os
from gfem import project as P
from gfem.io.inp import read_inp_spec

ROOT = os.path.join(os.path.dirname(__file__), "..", "examples")


def test_projects_load():
    files = glob.glob(os.path.join(ROOT, "projects", "*.gfem.json"))
    assert len(files) >= 12
    for f in files:
        p = P.load(f)
        assert p["geometry"]["features"]


def test_inp_decks_parse():
    for f in glob.glob(os.path.join(ROOT, "decks", "*.inp")):
        sp = read_inp_spec(f)
        assert len(sp.element_table()[0]) > 0 and sp.steps
