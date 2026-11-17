"""Surfaces: element faces, skin extraction, outward orientation and surface integrals.

A `Surface` is a list of facet groups (FacetType, node array (nf, nfn)) whose node order gives an
outward normal (right-hand rule). Surfaces carry pressure, traction, convection, contact, ties.
"""
import numpy as np
from .elements import FACETS


class Surface:
    def __init__(self, groups=None, name="surface"):
        self.name = name
        self.groups = []          # [(FacetType, nodes(nf, nfn))]
        for ft, f in (groups or []):
            self.add(ft, f)

    def add(self, ft, f):
        f = np.asarray(f, dtype=np.int64).reshape(-1, ft.nnode)
        if len(f):
            for i, (g, a) in enumerate(self.groups):
                if g is ft:
                    self.groups[i] = (g, np.vstack([a, f]))
                    return
            self.groups.append((ft, f))

    def __len__(self):
        return sum(len(f) for _, f in self.groups)

    def nodes(self):
        if not self.groups:
            return np.zeros(0, np.int64)
        return np.unique(np.concatenate([f.ravel() for _, f in self.groups]))

    def triangles(self):
        if not self.groups:
            return np.zeros((0, 3), np.int64)
        return np.concatenate([ft.triangles(f) for ft, f in self.groups])

    def triangles_with_owner(self):
        """Display/search triangles and their owning facet (group index, row)."""
        tris, own = [], []
        for gi, (ft, f) in enumerate(self.groups):
            t = ft.triangles(f)
            nsub = len(t) // max(len(f), 1)
            tris.append(t)
            own.append(np.stack([np.full(len(t), gi), np.tile(np.arange(len(f)), nsub)], 1))
        if not tris:
            return np.zeros((0, 3), np.int64), np.zeros((0, 2), np.int64)
        return np.concatenate(tris), np.concatenate(own)

    def project(self, P, X, k=8):
        """Closest point on the surface using the facets' own (bi)linear/quadratic shape
        functions. Returns (nodes (n, nmax) padded with -1, weights (n, nmax), distance)."""
        import jax
        import jax.numpy as jnp
        from .constraints import project_to_triangles
        X = np.asarray(X)
        P = np.asarray(P, float).reshape(-1, 3)
        tris, own = self.triangles_with_owner()
        t, _, _ = project_to_triangles(P, X, tris, k)
        g, r = own[t, 0], own[t, 1]
        nmax = max(ft.nnode for ft, _ in self.groups)
        nodes = np.full((len(P), nmax), -1, np.int64)
        wts = np.zeros((len(P), nmax))
        dist = np.zeros(len(P))
        for gi, (ft, fac) in enumerate(self.groups):
            sel = np.nonzero(g == gi)[0]
            if not len(sel):
                continue
            f = fac[r[sel]]
            xe = X[f]
            Nf = jax.vmap(ft.Nfun)
            dNf = jax.vmap(jax.jacfwd(ft.Nfun))
            xi = np.tile([1 / 3, 1 / 3] if ft.ncorner == 3 else [0.0, 0.0], (len(sel), 1))
            for _ in range(10):
                N = np.asarray(Nf(jnp.asarray(xi)))
                dN = np.asarray(dNf(jnp.asarray(xi)))                  # (n, nfn, 2)
                x = np.einsum("na,nai->ni", N, xe)
                J = np.einsum("nai,nak->nik", xe, dN)                  # (n, 3, 2)
                rhs = np.einsum("nik,ni->nk", J, P[sel] - x)
                JTJ = np.einsum("nik,nil->nkl", J, J)
                xi = xi + np.linalg.solve(JTJ + 1e-300 * np.eye(2), rhs[..., None])[..., 0]
                if ft.ncorner == 3:
                    xi = np.clip(xi, 0, 1)
                    s_ = xi.sum(1)
                    over = s_ > 1
                    xi[over] /= s_[over, None]
                else:
                    xi = np.clip(xi, -1, 1)
            N = np.asarray(Nf(jnp.asarray(xi)))
            nodes[sel, :ft.nnode] = f
            wts[sel, :ft.nnode] = N
            dist[sel] = np.linalg.norm(P[sel] - np.einsum("na,nai->ni", N, xe), axis=1)
        return nodes, wts, dist

    def merged(self, other):
        s = Surface(self.groups, self.name)
        for ft, f in other.groups:
            s.add(ft, f)
        return s

    # ---------- reference-configuration integrals (numpy) ----------
    def integrate(self, X, field=None):
        """Return (nodal weights w_a = int N_a dA  (nn,), area, area-weighted normal sum).
        `field` optional nodal-or-facet scalar multiplier (per facet, constant)."""
        X = np.asarray(X)
        w = np.zeros(len(X))
        area, nsum = 0.0, np.zeros(3)
        for ft, f in self.groups:
            xe = X[f]                                          # (nf, nfn, 3)
            a1 = np.einsum("fai,ga->fgi", xe, ft.dN[:, :, 0])
            a2 = np.einsum("fai,ga->fgi", xe, ft.dN[:, :, 1])
            n = np.cross(a1, a2) * ft.gw[None, :, None]        # (nf, ngp, 3)
            da = np.linalg.norm(n, axis=-1)
            mult = 1.0 if field is None else np.asarray(field)[:, None]
            wf = np.einsum("fg,ga->fa", da * mult, ft.N)
            np.add.at(w, f, wf)
            area += da.sum()
            nsum += n.sum((0, 1))
        return w, area, nsum

    def nodal_normals(self, X):
        """Area-weighted outward normal at every surface node, (nn, 3) (zero elsewhere)."""
        X = np.asarray(X)
        out = np.zeros((len(X), 3))
        for ft, f in self.groups:
            xe = X[f]
            a1 = np.einsum("fai,ga->fgi", xe, ft.dN[:, :, 0])
            a2 = np.einsum("fai,ga->fgi", xe, ft.dN[:, :, 1])
            n = (np.cross(a1, a2) * ft.gw[None, :, None])
            nf = np.einsum("fgi,ga->fai", n, ft.N)
            np.add.at(out, f, nf)
        nrm = np.linalg.norm(out, axis=1, keepdims=True)
        return np.where(nrm > 0, out / np.maximum(nrm, 1e-300), 0)

    def char_length(self, X):
        tri = self.triangles()
        if not len(tri):
            return 1.0
        X = np.asarray(X)
        return float(np.mean(np.linalg.norm(X[tri[:, 1]] - X[tri[:, 0]], axis=1)))


def _elem_faces(etype, conn):
    """Yield (face_index, FacetType, nodes (ne, nfn))."""
    for k, loc in enumerate(etype.faces):
        yield k, FACETS[len(loc)], np.asarray(conn)[:, loc]


def orient_outward(X, ft, facets, elem_centroids):
    """Flip facets whose right-hand normal points toward the owning element's centroid."""
    X = np.asarray(X)
    c = X[facets[:, :ft.ncorner]]
    if ft.ncorner == 3:
        n = np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0])
    else:
        n = np.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 1])
    inward = np.einsum("fi,fi->f", n, elem_centroids - c.mean(1)) > 0
    out = facets.copy()
    out[inward] = ft.flip(facets[inward])
    return out


def element_face_surface(X, blocks, pairs, name="surface"):
    """Surface from (block index, local element, face index) triples; outward oriented."""
    s = Surface(name=name)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 3)
    for bi in np.unique(pairs[:, 0]) if len(pairs) else []:
        b = blocks[bi]
        conn = np.asarray(b.conn)
        sel = pairs[pairs[:, 0] == bi]
        for k in np.unique(sel[:, 2]):
            el = sel[sel[:, 2] == k, 1]
            loc = b.etype.faces[k]
            ft = FACETS[len(loc)]
            f = conn[el][:, loc]
            cen = np.asarray(X)[conn[el, :b.etype.ncorner]].mean(1)
            s.add(ft, orient_outward(X, ft, f, cen))
    return s


def skin(X, blocks, return_owner=False):
    """Boundary faces of a set of volume element blocks: a Surface plus, optionally, the
    owner (block, element, face) of every facet in group order."""
    X = np.asarray(X)
    recs = []                      # (key corners sorted, block, elem, face)
    for bi, b in enumerate(blocks):
        conn = np.asarray(b.conn)
        for k, ft, f in _elem_faces(b.etype, conn):
            key = np.sort(f[:, :ft.ncorner], axis=1)
            if ft.ncorner == 3:
                key = np.hstack([np.full((len(f), 1), -1), key])
            ne = len(f)
            recs.append((key, np.full(ne, bi), np.arange(ne), np.full(ne, k)))
    if not recs:
        s = Surface(name="skin")
        return (s, np.zeros((0, 3), np.int64)) if return_owner else s
    keys = np.concatenate([r[0] for r in recs])
    owner = np.stack([np.concatenate([r[i] for r in recs]) for i in (1, 2, 3)], 1)
    _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    boundary = cnt[inv] == 1
    own = owner[boundary]
    s = element_face_surface(X, blocks, own, "skin")
    if return_owner:
        # regenerate owners in the same order as the surface groups
        order = []
        for bi in np.unique(own[:, 0]):
            sel = own[own[:, 0] == bi]
            b = blocks[bi]
            for k in np.unique(sel[:, 2]):
                order.append(sel[sel[:, 2] == k])
        own_sorted = np.concatenate(order) if order else own
        # group order is by facet type; rebuild explicitly
        by_type = {}
        for o in own_sorted:
            ft = FACETS[len(blocks[o[0]].etype.faces[o[2]])]
            by_type.setdefault(ft.name, []).append(o)
        own_final = np.concatenate([np.array(by_type[ft.name]) for ft, _ in s.groups])
        return s, own_final
    return s


def facet_lookup(surface):
    """Dict: sorted corner-node tuple -> (group index, row) for matching external face lists."""
    d = {}
    for gi, (ft, f) in enumerate(surface.groups):
        keys = np.sort(f[:, :ft.ncorner], axis=1)
        for r, k in enumerate(map(tuple, keys)):
            d[k] = (gi, r)
    return d


def sub_surface(surface, node_set):
    """Facets of `surface` whose corner nodes all belong to node_set."""
    mask_nodes = np.zeros(int(max(surface.nodes().max(initial=0), max(node_set, default=0))) + 1, bool)
    mask_nodes[np.asarray(list(node_set), dtype=np.int64)] = True
    out = Surface(name="sub")
    for ft, f in surface.groups:
        keep = mask_nodes[f[:, :ft.ncorner]].all(1)
        out.add(ft, f[keep])
    return out
