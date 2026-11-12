"""Linear multi-point constraints (MPC) by master-slave elimination.

Every constraint is written as  u[slave] = sum_k w_k u[master_k]. All constraints are merged into a
single sparse transformation u = T u_bar that the solvers apply matrix-free on the device
(T v and T^T y are two gather/segment-sum kernels). This gives *exact* bonded contact, symmetry on
skewed planes, and rigid/averaging couplings without penalty ill-conditioning.
"""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial import cKDTree


class MPC:
    """Raw constraint rows: slave dofs (nc,), master dofs (nc, k), weights (nc, k)."""

    def __init__(self, slave, masters, weights, name="mpc"):
        self.slave = np.asarray(slave, dtype=np.int64).ravel()
        k = np.asarray(masters).shape[-1] if np.asarray(masters).ndim > 1 else 1
        self.masters = np.asarray(masters, dtype=np.int64).reshape(len(self.slave), k)
        self.weights = np.asarray(weights, dtype=np.float64).reshape(len(self.slave), k)
        self.name = name

    def __len__(self):
        return len(self.slave)


def _closest_on_tris(P, A, B, C):
    """Barycentric (wa, wb, wc) of the closest point on triangles ABC to points P (vectorised)."""
    ab, ac, ap = B - A, C - A, P - A
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    d20 = np.einsum("ij,ij->i", ap, ab)
    d21 = np.einsum("ij,ij->i", ap, ac)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-300)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    # clamp into the triangle (approximate projection onto the boundary)
    v, w = np.clip(v, 0, 1), np.clip(w, 0, 1)
    s = v + w
    over = s > 1
    v[over] /= s[over]
    w[over] /= s[over]
    return np.stack([1 - v - w, v, w], 1)


def project_to_triangles(P, X, tris, k=8):
    """For each point, the closest triangle among the k nearest by centroid.
    Returns (tri index, barycentric weights (n,3), distance)."""
    X = np.asarray(X)
    cen = X[tris].mean(1)
    tree = cKDTree(cen)
    k = min(k, len(tris))
    _, cand = tree.query(P, k=k)
    cand = cand.reshape(len(P), k)
    best_d = np.full(len(P), np.inf)
    best_t = np.zeros(len(P), np.int64)
    best_w = np.zeros((len(P), 3))
    for j in range(k):
        t = tris[cand[:, j]]
        A, B, C = X[t[:, 0]], X[t[:, 1]], X[t[:, 2]]
        wts = _closest_on_tris(P, A, B, C)
        Q = wts[:, :1] * A + wts[:, 1:2] * B + wts[:, 2:] * C
        d = np.linalg.norm(P - Q, axis=1)
        better = d < best_d
        best_d[better], best_t[better], best_w[better] = d[better], cand[better, j], wts[better]
    return best_t, best_w, best_d


def _facet_quadrature(ft, n=3, sub=4):
    """Composite rule on the reference facet: sub x sub cells with n x n Gauss each
    (resolves kinks of the master interpolation inside a slave facet)."""
    g, w = np.polynomial.legendre.leggauss(n)
    P, W = [], []
    if ft.ncorner == 4:
        h = 2.0 / sub
        for i in range(sub):
            for j in range(sub):
                for a, wa in zip(g, w):
                    for b, wb in zip(g, w):
                        P.append([-1 + h * (i + (a + 1) / 2), -1 + h * (j + (b + 1) / 2)])
                        W.append(wa * wb * h * h / 4)
        return np.array(P), np.array(W)
    # triangle: collapsed Gauss on sub^2 sub-triangles
    a = (g + 1) / 2
    ref = [(s_ * (1 - t_), t_, w[i] * w[j] / 4 * (1 - t_)) for i, s_ in enumerate(a)
           for j, t_ in enumerate(a)]
    h = 1.0 / sub
    for i in range(sub):
        for j in range(sub - i):
            for flip in (False, True):
                if flip and i + j >= sub - 1:
                    continue
                o = np.array([i * h, j * h])
                e1, e2 = np.array([h, 0]), np.array([0, h])
                if flip:
                    o, e1, e2 = o + e1 + e2, -e1, -e2
                for s_, t_, ww in ref:
                    P.append(o + s_ * e1 + t_ * e2)
                    W.append(ww * h * h)
    return np.array(P), np.array(W)


def _dual_mortar(X, slave_surface, master, tol):
    """Dual-mortar weights W[s, m]: u_s = sum_m W_sm u_m. Exact for linear fields and for
    the transfer of uniform tractions (passes the interface patch test)."""
    import jax
    import jax.numpy as jnp
    rows, cols, vals = [], [], []
    D = {}
    for ft, fac in slave_surface.groups:
        P, Wq = _facet_quadrature(ft)
        N = np.array([np.asarray(ft_N(ft, p)) for p in P])          # (nq, nfn)
        dN = np.array([np.asarray(jax.jacfwd(lambda q: ft_N(ft, q))(jnp.asarray(p))) for p in P])
        xe = X[fac]                                                    # (nf, nfn, 3)
        a1 = np.einsum("fai,qa->fqi", xe, dN[:, :, 0])
        a2 = np.einsum("fai,qa->fqi", xe, dN[:, :, 1])
        da = np.linalg.norm(np.cross(a1, a2), axis=-1) * Wq[None]    # (nf, nq)
        Me = np.einsum("fq,qa,qb->fab", da, N, N)
        De = np.einsum("fq,qa->fa", da, N)
        A = np.einsum("fa,fab->fab", De, np.linalg.inv(Me))           # dual basis coefficients
        psi = np.einsum("fab,qb->fqa", A, N)                          # (nf, nq, nfn)
        xq = np.einsum("qa,fai->fqi", N, xe).reshape(-1, 3)
        mnodes_, wb, d = master.project(xq, X)
        km = mnodes_.shape[1]
        ok = (d <= tol).reshape(len(fac), -1)
        mn = mnodes_.reshape(len(fac), len(P), km)
        wb = wb.reshape(len(fac), len(P), km)
        contrib = np.einsum("fq,fqa,fqk->fqak", da * ok, psi, wb)       # (nf,nq,nfn,km)
        rows.append(np.broadcast_to(fac[:, None, :, None], contrib.shape).ravel())
        cols.append(np.broadcast_to(mn[:, :, None, :], contrib.shape).ravel())
        vals.append(contrib.ravel())
        np.add.at(D.setdefault("d", np.zeros(len(X))), fac, De)
    import scipy.sparse as sp
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    v = cols >= 0
    Wm = sp.coo_matrix((vals[v], (rows[v], cols[v])), shape=(len(X), len(X))).tocsr()
    d = D["d"]
    return Wm, d


def ft_N(ft, p):
    from .elements import _tri3N, _quad4N
    import jax.numpy as jnp
    return (_tri3N if ft.ncorner == 3 else _quad4N)(jnp.asarray(p))


def tie(X, slave, master_surface, tol=None, name="tie"):
    """Bonded (tied) contact: slave nodes follow the master surface.
    `slave` may be a Surface (linear facets -> dual mortar, passes the patch test) or a node array
    (node-to-surface projection)."""
    from .surfaces import Surface
    X = np.asarray(X)
    tris = master_surface.triangles()
    mnodes = set(master_surface.nodes().tolist())
    if tol is None:
        tol = 0.5 * master_surface.char_length(X)
    empty = MPC([], np.zeros((0, 3)), np.zeros((0, 3)), name)
    if not len(tris):
        return empty
    if isinstance(slave, Surface) and all(ft.nnode == ft.ncorner for ft, _ in slave.groups) and len(slave):
        Wm, d = _dual_mortar(X, slave, master_surface, tol)
        snodes = np.array([n for n in slave.nodes() if n not in mnodes], dtype=np.int64)
        Wm = Wm[snodes]
        rs = np.asarray(Wm.sum(1)).ravel()
        good = (d[snodes] > 0) & (np.abs(rs / np.maximum(d[snodes], 1e-300) - 1) < 0.05)
        snodes, Wm, rs = snodes[good], Wm[good], rs[good]
        if not len(snodes):
            return empty
        Wm = Wm.tocoo()
        k = np.bincount(Wm.row, minlength=len(snodes)).max()
        md = np.zeros((len(snodes), k), np.int64)
        wv = np.zeros((len(snodes), k))
        fill = np.zeros(len(snodes), np.int64)
        for r, c, v in zip(Wm.row, Wm.col, Wm.data):
            md[r, fill[r]], wv[r, fill[r]] = c, v / rs[r]
            fill[r] += 1
        sd = (snodes[:, None] * 3 + np.arange(3)).ravel()
        mdd = (np.repeat(md[:, None, :] * 3, 3, axis=1) + np.arange(3)[None, :, None]).reshape(-1, k)
        wd = np.repeat(wv[:, None, :], 3, axis=1).reshape(-1, k)
        return MPC(sd, mdd, wd, name)
    slave_nodes = slave.nodes() if isinstance(slave, Surface) else slave
    slave = np.array([n for n in np.unique(slave_nodes) if n not in mnodes], dtype=np.int64)
    if not len(slave):
        return empty
    mn, w, d = master_surface.project(X[slave], X)
    ok = d <= tol
    slave, mn, w = slave[ok], mn[ok], w[ok]
    k = mn.shape[1]
    sd = (slave[:, None] * 3 + np.arange(3)).ravel()
    md = np.where(mn[:, None, :] >= 0, np.repeat(mn[:, None, :] * 3, 3, axis=1) + np.arange(3)[None, :, None], -1)
    md = md.reshape(-1, k)
    wd = np.repeat(w[:, None, :], 3, axis=1).reshape(-1, k)
    return MPC(sd, md, wd, name)


def normal_constraint(nodes, normal, name="normal"):
    """n . u = 0 for the given nodes (frictionless support / symmetry on any plane)."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    k = int(np.argmax(np.abs(n)))
    others = [j for j in range(3) if j != k]
    nodes = np.asarray(nodes, dtype=np.int64)
    sd = nodes * 3 + k
    md = np.stack([nodes * 3 + j for j in others], 1)
    w = np.tile([-n[j] / n[k] for j in others], (len(nodes), 1))
    # pure axis-aligned plane: zero weights -> master list harmless, slave forced to 0
    return MPC(sd, md, w, name)


def coupling(ref_dofs_nodes, nodes, name="coupling"):
    """Kinematic coupling (translations only): every node follows a reference node."""
    ref = int(ref_dofs_nodes)
    nodes = np.asarray([n for n in nodes if n != ref], dtype=np.int64)
    sd = (nodes[:, None] * 3 + np.arange(3)).ravel()
    md = np.tile(ref * 3 + np.arange(3), len(nodes))[:, None]
    return MPC(sd, md, np.ones((len(sd), 1)), name)


class Transformation:
    """u = T u_bar  (sparse, resolved for chained constraints)."""

    def __init__(self, ndof, mpcs, fixed=None):
        import scipy.sparse as sp
        self.ndof = ndof
        rows = [m for m in mpcs if len(m)]
        self.active = bool(rows)
        self.slave_mask = np.zeros(ndof, bool)
        self.dropped = 0
        if not self.active:
            return
        sl = np.concatenate([m.slave for m in rows])
        ms = [m.masters for m in rows]
        kmax = max(a.shape[1] for a in ms)
        md = np.concatenate([np.pad(a, ((0, 0), (0, kmax - a.shape[1])), constant_values=-1) for a in ms])
        wt = np.concatenate([np.pad(m.weights, ((0, 0), (0, kmax - m.weights.shape[1]))) for m in rows])
        # one constraint per slave dof (first wins), slaves that are prescribed are dropped
        _, first = np.unique(sl, return_index=True)
        sl, md, wt = sl[first], md[first], wt[first]
        if fixed is not None:
            ok = ~np.asarray(fixed)[sl]
            self.dropped = int((~ok).sum())
            sl, md, wt = sl[ok], md[ok], wt[ok]
        smask = np.zeros(ndof, bool)
        smask[sl] = True
        # a master that is itself a slave cannot also be a slave of its own slave: break cycles
        valid = md >= 0
        r = np.repeat(sl, valid.sum(1))
        c = md[valid]
        v = wt[valid]
        keep_diag = ~smask
        T = sp.csr_matrix((np.concatenate([v, np.ones(keep_diag.sum())]),
                           (np.concatenate([r, np.nonzero(keep_diag)[0]]),
                            np.concatenate([c, np.nonzero(keep_diag)[0]]))), shape=(ndof, ndof))
        for _ in range(8):                                    # resolve chains u_s -> u_s' -> u_m
            if not T[:, smask].nnz:
                break
            T = T @ T
        T = (T @ sp.diags((~smask).astype(float))).tocsr()     # break residual cycles
        T.eliminate_zeros()
        self.T = T
        self.slave_mask = smask
        coo = T.tocoo()
        self.rows = jnp.asarray(coo.row.astype(np.int32))
        self.cols = jnp.asarray(coo.col.astype(np.int32))
        self.vals = jnp.asarray(coo.data)

    def apply(self, v):
        if not self.active:
            return v
        return jax.ops.segment_sum(self.vals * v[self.cols], self.rows, num_segments=self.ndof)

    def apply_T(self, y):
        if not self.active:
            return y
        return jax.ops.segment_sum(self.vals * y[self.rows], self.cols, num_segments=self.ndof)

    def project_matrix(self, A):
        """T^T A T for a scipy matrix (direct solver path)."""
        if not self.active:
            return A
        return (self.T.T @ A @ self.T).tocsr()

    def params(self):
        if not self.active:
            return None
        return (self.rows, self.cols, self.vals)


def t_apply(p, v, ndof):
    if p is None:
        return v
    rows, cols, vals = p
    return jax.ops.segment_sum(vals * v[cols], rows, num_segments=ndof)


def t_apply_T(p, y, ndof):
    if p is None:
        return y
    rows, cols, vals = p
    return jax.ops.segment_sum(vals * y[rows], cols, num_segments=ndof)
