"""Neutral analysis description shared by every front end.

Abaqus .inp, Nastran .bdf, ANSYS .cdb readers, the CAD/GUI project builder, and the writers
all go through `Spec`, so a model created in the GUI from a STEP file can be exported to
Abaqus/Nastran decks for validation in a commercial code, and vice versa.

Conventions: node / element indices are 0-based internal indices; `node_ids` / element `ids`
keep the external numbers. Element types use Abaqus names (C3D4, C3D10, C3D8, C3D20, C3D6...).
"""
import numpy as np


class ElementGroup:
    def __init__(self, etype, conn, ids=None, elset=None):
        self.etype = etype.upper()
        self.conn = np.asarray(conn, dtype=np.int64)
        self.ids = np.asarray(ids if ids is not None else np.arange(len(self.conn)) + 1, dtype=np.int64)
        self.elset = elset


class Spec:
    def __init__(self, title="model"):
        self.title = title
        self.nodes = np.zeros((0, 3))
        self.node_ids = np.zeros(0, np.int64)
        self.groups = []            # [ElementGroup]
        self.nsets = {}             # name -> node idx
        self.elsets = {}            # name -> global element idx
        self.surfaces = {}          # name -> dict(elems=global idx, faces=face idx) | dict(nodes=idx)
        self.materials = {}         # name -> property dict (see materials.from_props)
        self.sections = []          # [dict(elset, material)]
        self.amplitudes = {}        # name -> (n, 2) table
        self.initial_temperature = None
        self.constraints = []       # [dict(type="tie", slave, master, tol)] (surface names)
        self.contacts = []          # [dict(type="frictionless"|"wall", master, slave, fkn, ...)]
        self.steps = []             # [dict] see analysis.build
        self.units = None
        self.warnings = []

    # ---------- construction helpers ----------
    def add_nodes(self, X, ids=None):
        X = np.asarray(X, float).reshape(-1, 3)
        start = len(self.nodes)
        self.nodes = np.vstack([self.nodes, X])
        ids = np.arange(start, start + len(X)) + 1 if ids is None else np.asarray(ids)
        self.node_ids = np.concatenate([self.node_ids, ids]).astype(np.int64)
        return np.arange(start, start + len(X))

    def add_elements(self, etype, conn, ids=None, elset=None):
        start = self.nelem
        if ids is None:
            ids = np.arange(start, start + len(conn)) + 1
        g = ElementGroup(etype, conn, ids, elset)
        self.groups.append(g)
        idx = np.arange(start, start + len(conn))
        if elset:
            self.elsets[elset] = np.concatenate([self.elsets.get(elset, np.zeros(0, np.int64)), idx])
        return idx

    @property
    def nelem(self):
        return sum(len(g.conn) for g in self.groups)

    def element_table(self):
        """(etype per element, conn list) for all elements in global order."""
        types, conns = [], []
        for g in self.groups:
            types += [g.etype] * len(g.conn)
            conns += list(g.conn)
        return types, conns

    def surface_object(self, name):
        """gfem Surface (outward facets) for an element-face surface of this spec."""
        from .elements import ELEMENTS, FACETS
        from .surfaces import Surface, orient_outward
        sd = self.surfaces[name]
        types, conns = self.element_table()
        surf = Surface(name=name)
        groups = {}
        for e, k in zip(sd["elems"], sd["faces"]):
            et = ELEMENTS[types[e]]
            loc = et.faces[k]
            row = np.asarray(conns[e])
            groups.setdefault(len(loc), ([], []))
            groups[len(loc)][0].append(row[loc])
            groups[len(loc)][1].append(self.nodes[row[:et.ncorner]].mean(0))
        for n, (f, cen) in groups.items():
            ft = FACETS[n]
            surf.add(ft, orient_outward(self.nodes, ft, np.array(f), np.array(cen)))
        return surf

    def equivalent_nodal_loads(self, step):
        """FORCE / MOMENT surface loads of a step as nodal force vectors {node: (fx, fy, fz)}
        (used by the deck writers so exported models reproduce gfem results exactly)."""
        from . import loads as L
        out = np.zeros((len(self.nodes), 3))
        for d in step["dsloads"]:
            k = d.get("kind", "P").upper()
            if k == "FORCE":
                out += L.surface_force_vector(self.nodes, self.surface_object(d["surface"]), d["vector"]).reshape(-1, 3)
            elif k == "MOMENT":
                out += L.surface_moment_vector(self.nodes, self.surface_object(d["surface"]), d["vector"],
                                               d.get("center")).reshape(-1, 3)
        nz = np.nonzero(np.abs(out).max(1) > 0)[0]
        return {int(n): out[n] for n in nz}

    def summary(self):
        from collections import Counter
        c = Counter()
        for g in self.groups:
            c[g.etype] += len(g.conn)
        return (f"{len(self.nodes)} nodes, {self.nelem} elements ("
                + ", ".join(f"{k}: {v}" for k, v in c.items()) + f"), {len(self.materials)} materials, "
                f"{len(self.steps)} steps")


def new_step(name="Step-1", procedure="static", **kw):
    s = dict(name=name, procedure=procedure, nlgeom=False, time=1.0, dt_init=None, dt_min=None,
             dt_max=None, max_inc=1000, bcs=[], cloads=[], dsloads=[], dloads=[], temperatures=[],
             heat=dict(fixed=[], flux=[], film=[], gen=[]), nmodes=10, riks={}, frames=20,
             damping=0.0, alpha=-0.05, mass_scaling=None, topopt={}, output_every=1, dt=None,
             transient=False)
    s.update(kw)
    return s
