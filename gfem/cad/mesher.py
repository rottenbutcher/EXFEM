"""CAD-associative meshing with gmsh and conversion to the solver `Spec`.

Every geometric entity keeps its identity in the mesh:
  BODY_<id>  : elset (+ nset) of the volume
  FACE_<id>  : element-face surface (outward oriented, for pressure / contact / tie) + nset
  EDGE_<id>  : nset,  VERTEX_<id> : nset
so loads and supports defined on CAD faces are mapped exactly onto the mesh.
"""
import numpy as np
from .geometry import GMSH_LOCK, gmsh_module
from ..spec import Spec
from ..elements import TET4, TET10, HEX8, HEX20, WEDGE6

# gmsh element type -> (Abaqus name, nnode, our ElementType)
GMSH_TYPES = {4: ("C3D4", 4, TET4), 11: ("C3D10", 10, TET10), 5: ("C3D8", 8, HEX8),
              17: ("C3D20", 20, HEX20), 6: ("C3D6", 6, WEDGE6), 12: ("C3D20", 27, HEX20),
              7: ("PYR5", 5, None), 18: ("C3D6", 15, WEDGE6), 13: ("C3D6", 18, WEDGE6)}


def reorder_midnodes(X, conn, etype):
    """Put mid-edge nodes in Abaqus order by geometric matching (robust to any mesher)."""
    if not etype.edges:
        return conn
    conn = np.asarray(conn).copy()
    nc = etype.ncorner
    mids = conn[:, nc:etype.nnode]
    mpos = X[mids]                                                 # (ne, nm, 3)
    out = conn.copy()
    for (i, j, k) in etype.edges:
        target = 0.5 * (X[conn[:, i]] + X[conn[:, j]])
        d = np.linalg.norm(mpos - target[:, None, :], axis=-1)
        out[:, k] = mids[np.arange(len(conn)), np.argmin(d, axis=1)]
    return out


def fix_orientation(X, conn, etype):
    """Flip elements with negative Jacobian at the centroid."""
    c = conn[:, :etype.ncorner]
    if etype.name in ("C3D4", "C3D10"):
        v = np.einsum("ij,ij->i", np.cross(X[c[:, 1]] - X[c[:, 0]], X[c[:, 2]] - X[c[:, 0]]), X[c[:, 3]] - X[c[:, 0]])
        bad = v < 0
        if bad.any():
            conn = conn.copy()
            conn[bad, 1], conn[bad, 2] = conn[bad, 2].copy(), conn[bad, 1].copy()
            if etype.nnode == 10:   # swap edge nodes accordingly: (0,1)<->(0,2), (1,3)<->(2,3)
                conn[bad, 4], conn[bad, 6] = conn[bad, 6].copy(), conn[bad, 4].copy()
                conn[bad, 8], conn[bad, 9] = conn[bad, 9].copy(), conn[bad, 8].copy()
    elif etype.ncorner == 8:
        v = np.einsum("ij,ij->i", np.cross(X[c[:, 1]] - X[c[:, 0]], X[c[:, 3]] - X[c[:, 0]]), X[c[:, 4]] - X[c[:, 0]])
        bad = v < 0
        if bad.any():
            conn = conn.copy()
            perm = [4, 5, 6, 7, 0, 1, 2, 3] + ([12, 13, 14, 15, 8, 9, 10, 11, 16, 17, 18, 19] if etype.nnode == 20 else [])
            conn[bad] = conn[bad][:, perm]
    elif etype.ncorner == 6:
        v = np.einsum("ij,ij->i", np.cross(X[c[:, 1]] - X[c[:, 0]], X[c[:, 2]] - X[c[:, 0]]), X[c[:, 3]] - X[c[:, 0]])
        bad = v < 0
        if bad.any():
            conn = conn.copy()
            conn[bad] = conn[bad][:, [3, 4, 5, 0, 1, 2]]
    return conn


def mesh_cad(cad, size=None, size_min=None, order=2, kind="tet", curvature=8, local=(),
             optimize=True, algorithm3d=1, log=None):
    """Mesh the current (regenerated) CAD model. `local`: [(dim, [ids], size)].
    kind: 'tet' | 'hex' (transfinite/recombine where possible, tets elsewhere) | 'hexdom'
    Returns (Spec, stats)."""
    with GMSH_LOCK:
        cad.ensure_loaded()
        g = gmsh_module()
        info = cad.info
        diag = info.get("diag", 1.0)
        size = size or diag / 20
        g.model.mesh.clear()
        g.option.setNumber("Mesh.MeshSizeMax", size)
        smin = size_min or min([size / 4] + [float(x[2]) * 0.5 for x in local])
        g.option.setNumber("Mesh.MeshSizeMin", smin)
        g.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
        g.option.setNumber("Mesh.ElementOrder", 1)
        g.option.setNumber("Mesh.SecondOrderIncomplete", 1)
        g.option.setNumber("Mesh.Algorithm", 6)
        g.option.setNumber("Mesh.Algorithm3D", algorithm3d)
        g.option.setNumber("Mesh.Optimize", 1 if optimize else 0)
        g.option.setNumber("Mesh.RecombineAll", 0)
        g.option.setNumber("Mesh.SubdivisionAlgorithm", 0)
        g.model.mesh.setSize(g.model.getEntities(0), size)
        for dim, ids, s in local:
            pts = set()
            for t in ids:
                for d, p in g.model.getBoundary([(dim, int(t))], recursive=True):
                    if d == 0:
                        pts.add(p)
                if dim == 0:
                    pts.add(int(t))
            if pts:
                g.model.mesh.setSize([(0, p) for p in pts], float(s))
        if kind == "hex":
            g.option.setNumber("Mesh.RecombineAll", 1)
            g.option.setNumber("Mesh.Recombine3DAll", 1)
            g.option.setNumber("Mesh.Algorithm", 8)
            for dim, tag in g.model.getEntities(3):
                try:
                    g.model.mesh.setTransfiniteAutomatic([(dim, tag)], cornerAngle=2.35, recombine=True)
                except Exception:   # noqa: BLE001
                    pass
        elif kind == "hexdom":
            g.option.setNumber("Mesh.SubdivisionAlgorithm", 2)
        vb = {abs(t) for v in g.model.getEntities(3) for _, t in g.model.getBoundary([v], oriented=False)}
        for _, ftag in g.model.getEntities(2):
            if ftag not in vb and kind != "tri_shell":         # sheets: quad-dominant shell mesh
                g.model.mesh.setRecombine(2, ftag)
        g.model.mesh.generate(3)
        if order == 2:
            g.model.mesh.setOrder(2)
        if optimize and order == 1 and kind == "tet":
            try:
                g.model.mesh.optimize("Netgen")
            except Exception:   # noqa: BLE001
                pass
        return extract_spec(g, log=log, names=None, beam_curves={l["id"] for l in info.get("lines", [])})


def extract_spec(g, log=None, names=None, physical=False, beam_curves=None):
    """Build a Spec from the current gmsh mesh with entity-based sets."""
    nt, nc, _ = g.model.mesh.getNodes()
    order = np.argsort(nt)
    nt, X = np.asarray(nt)[order], np.asarray(nc).reshape(-1, 3)[order]
    gid = {int(t): i for i, t in enumerate(nt)}
    lookup = np.full(int(nt.max()) + 1, -1, np.int64)
    lookup[nt.astype(np.int64)] = np.arange(len(nt))
    sp = Spec("cad-mesh")
    sp.add_nodes(X, np.arange(1, len(X) + 1))
    face_keys = {}           # sorted corner key -> (global elem, face idx)
    quality = []
    for dim, tag in g.model.getEntities(3):
        et_list, _, en_list = g.model.mesh.getElements(3, tag)
        elidx = []
        for et, en in zip(et_list, en_list):
            if et not in GMSH_TYPES:
                continue
            name, nn, etype = GMSH_TYPES[et]
            conn = lookup[np.asarray(en, np.int64).reshape(-1, nn)]
            if name == "PYR5":        # split pyramids into two tets
                conn = np.vstack([conn[:, [0, 1, 2, 4]], conn[:, [0, 2, 3, 4]]])
                name, etype = "C3D4", TET4
            conn = conn[:, :etype.nnode]
            conn = reorder_midnodes(X, conn, etype)
            conn = fix_orientation(X, conn, etype)
            idx = sp.add_elements(name, conn, elset=f"BODY_{tag}")
            elidx.append(idx)
            for k, loc in enumerate(etype.faces):
                nc_ = 3 if len(loc) in (3, 6) else 4
                keys = np.sort(conn[:, loc[:nc_]], axis=1)
                for e, key in zip(idx, map(tuple, keys)):
                    face_keys.setdefault(key, (e, k))
        nodes_b, _, _ = g.model.mesh.getNodes(3, tag, includeBoundary=True)
        sp.nsets[f"BODY_{tag}"] = lookup[np.asarray(nodes_b, np.int64)]
    # ---- sheets (faces not bounding a volume) -> shells, free curves -> beams
    vb = {abs(t) for v in g.model.getEntities(3) for _, t in g.model.getBoundary([v], oriented=False)}
    fe = {abs(t) for _, f in g.model.getEntities(2) for _, t in g.model.getBoundary([(2, f)], oriented=False)}
    for dim, tag in g.model.getEntities(2):
        if tag in vb:
            continue
        et_list, _, en_list = g.model.mesh.getElements(2, tag)
        idxs = []
        for et, en in zip(et_list, en_list):
            nn = {2: 3, 9: 6, 3: 4, 16: 8, 10: 9}.get(et)
            if nn is None:
                continue
            ncr = 3 if et in (2, 9) else 4
            conn = lookup[np.asarray(en, np.int64).reshape(-1, nn)[:, :ncr]]
            idxs.append(sp.add_elements("S3" if ncr == 3 else "S4", conn, elset=f"SHEET_{tag}"))
        if idxs:
            allx = np.concatenate(idxs)
            sp.surfaces[f"FACE_{tag}"] = dict(elems=allx, faces=np.zeros(len(allx), np.int64))
    for dim, tag in g.model.getEntities(1):
        if tag in fe and not (beam_curves and tag in beam_curves):
            continue
        et_list, _, en_list = g.model.mesh.getElements(1, tag)
        for et, en in zip(et_list, en_list):
            nn = {1: 2, 8: 3, 26: 4}.get(et)
            if nn:
                conn = lookup[np.asarray(en, np.int64).reshape(-1, nn)[:, :2]]
                sp.add_elements("B31", conn, elset=f"LINE_{tag}")
    for dim, tag in g.model.getEntities(2):
        nodes_f, _, _ = g.model.mesh.getNodes(2, tag, includeBoundary=True)
        sp.nsets[f"FACE_{tag}"] = np.unique(lookup[np.asarray(nodes_f, np.int64)])
        et_list, _, en_list = g.model.mesh.getElements(2, tag)
        el, fc = [], []
        for et, en in zip(et_list, en_list):
            nn = {2: 3, 9: 6, 3: 4, 16: 8, 10: 9}.get(et)
            if nn is None:
                continue
            ncr = 3 if et in (2, 9) else 4
            conn = lookup[np.asarray(en, np.int64).reshape(-1, nn)[:, :ncr]]
            for key in map(tuple, np.sort(conn, axis=1)):
                hit = face_keys.get(key)
                if hit is not None:
                    el.append(hit[0])
                    fc.append(hit[1])
        if el:
            sp.surfaces[f"FACE_{tag}"] = dict(elems=np.array(el, np.int64), faces=np.array(fc, np.int64))
    for dim, tag in g.model.getEntities(1):
        nodes_e, _, _ = g.model.mesh.getNodes(1, tag, includeBoundary=True)
        sp.nsets[f"EDGE_{tag}"] = np.unique(lookup[np.asarray(nodes_e, np.int64)])
    for dim, tag in g.model.getEntities(0):
        nodes_v, _, _ = g.model.mesh.getNodes(0, tag)
        sp.nsets[f"VERTEX_{tag}"] = lookup[np.asarray(nodes_v, np.int64)]
    if physical:
        for dim, ptag in g.model.getPhysicalGroups():
            name = g.model.getPhysicalName(dim, ptag) or f"PHYS_{dim}_{ptag}"
            ents = g.model.getEntitiesForPhysicalGroup(dim, ptag)
            key = {3: "BODY", 2: "FACE", 1: "EDGE", 0: "VERTEX"}[dim]
            ns = [sp.nsets.get(f"{key}_{e}", np.zeros(0, np.int64)) for e in ents]
            sp.nsets[name.upper()] = np.unique(np.concatenate(ns)) if ns else np.zeros(0, np.int64)
            if dim == 3:
                es = [sp.elsets.get(f"BODY_{e}", np.zeros(0, np.int64)) for e in ents]
                sp.elsets[name.upper()] = np.concatenate(es) if es else np.zeros(0, np.int64)
            if dim == 2:
                ss = [sp.surfaces[f"FACE_{e}"] for e in ents if f"FACE_{e}" in sp.surfaces]
                if ss:
                    sp.surfaces[name.upper()] = dict(elems=np.concatenate([s["elems"] for s in ss]),
                                                     faces=np.concatenate([s["faces"] for s in ss]))
    _compact(sp)
    # quality: scaled Jacobian / SICN statistics
    stats = dict(nodes=len(X), elements=sp.nelem)
    try:
        etags = []
        for dim, tag in g.model.getEntities(3):
            ets, tg, _ = g.model.mesh.getElements(3, tag)
            for t in tg:
                etags.extend(t)
        if etags:
            q = np.asarray(g.model.mesh.getElementQualities(etags, "minSICN"))
            stats.update(q_min=float(q.min()), q_avg=float(q.mean()),
                         q_hist=np.histogram(np.clip(q, 0, 1), bins=10, range=(0, 1))[0].tolist())
    except Exception:   # noqa: BLE001
        pass
    from collections import Counter
    cnt = Counter()
    for gr in sp.groups:
        cnt[gr.etype] += len(gr.conn)
    stats["types"] = dict(cnt)
    return sp, stats


def _compact(sp):
    """Drop nodes that no element references (mid-edge nodes of shells/beams in a quadratic
    mesh) and renumber everything."""
    used = np.zeros(len(sp.nodes), bool)
    for g in sp.groups:
        used[g.conn.ravel()] = True
    if used.all():
        return
    new = -np.ones(len(sp.nodes), np.int64)
    new[used] = np.arange(used.sum())
    sp.nodes = sp.nodes[used]
    sp.node_ids = np.arange(1, used.sum() + 1)
    for g in sp.groups:
        g.conn = new[g.conn]
    for k, v in list(sp.nsets.items()):
        v = new[np.asarray(v, np.int64)]
        sp.nsets[k] = v[v >= 0]


def read_mesh_file(path):
    """Read any gmsh-supported mesh (.msh, .unv, .med, .mesh...) into a Spec. Physical groups
    become named sets/surfaces."""
    with GMSH_LOCK:
        g = gmsh_module()
        g.clear()
        from . import geometry as _geo
        _geo._CURRENT[0] = None
        g.open(path)
        sp, _ = extract_spec(g, physical=True)
        sp.title = path
        if len(sp.groups) and not sp.materials:
            sp.materials["DEFAULT"] = dict(E=200e3, nu=0.3, density=7.85e-9)
        return sp
