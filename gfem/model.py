"""Model: nodes + element blocks, with batched (GPU-parallel) element kernels.

Data layout is "one block = one element type + one material", so every kernel is a single
vmap over thousands of identical elements -> maps directly onto GPU threads.
Global matrices are never needed by the iterative path: element tangents are kept per element
("EbE") or not formed at all (forward-mode AD Jacobian-vector products, "jvp" mode), and the
global product K*v is evaluated element-by-element with scatter-add.

Generic linear operators are represented as a list of (dofs (ne, nd), Ke (ne, nd, nd)) pairs
("ops"); element blocks, contact and tie elements, mass, etc. all share this representation.
"""
import numpy as np
import jax
import jax.numpy as jnp

from .surfaces import Surface, skin as _skin

I3 = jnp.eye(3)


class Block:
    def __init__(self, conn, etype, material, name="block"):
        self.conn = jnp.asarray(np.asarray(conn, dtype=np.int32))
        self.etype, self.material, self.name = etype, material, name
        self.ne = int(self.conn.shape[0])
        self._build()

    def _build(self):
        et, mat = self.etype, self.material
        dN, w, N = jnp.asarray(et.dN), jnp.asarray(et.gw), jnp.asarray(et.N)
        dN0 = jnp.asarray(et.dN0) if et.fbar else None
        fs = not mat.small_strain
        alpha, tref = mat.alpha, mat.tref

        def geom(Xe):
            Jac = jnp.einsum("an,gai->gni", Xe, dN)              # (ngp,3,3) dX/dxi
            dNdX = jnp.einsum("gai,gij->gaj", dN, jnp.linalg.inv(Jac))
            return dNdX, w * jnp.linalg.det(Jac)

        def defgrad(Xe, ue, dNdX):
            F = I3 + jnp.einsum("ai,gaj->gij", ue, dNdX)
            if et.fbar:
                J0inv = jnp.linalg.inv(Xe.T @ dN0)
                F0 = I3 + ue.T @ (dN0 @ J0inv)
                if fs:          # F-bar (finite strain)
                    J = jnp.linalg.det(F)
                    F = F * ((jnp.linalg.det(F0) / J) ** (1 / 3))[:, None, None]
                else:           # B-bar (linear): replace the volumetric part by the centroid value
                    tr0 = jnp.trace(F0) - 3.0
                    tr = jnp.trace(F, axis1=1, axis2=2) - 3.0
                    F = F + ((tr0 - tr) / 3.0)[:, None, None] * I3
            return F

        def stress_gp(F, s, T, T0, eig):
            """Thermal strain alpha (T - T_initial) (Abaqus convention: no thermal strain at the
            initial temperature) and eigenstrain `eig` (bolt pretension): additive for small-strain
            models, multiplicative F = Fe Fp with Fp = (1 + th) (I + eig) for finite-strain models."""
            th = alpha * (T - T0)
            if fs:
                Fp_inv = jnp.linalg.inv(I3 + eig) / (1.0 + th)
                P, sn = mat.pk1T(F @ Fp_inv, s, T)
                return P @ Fp_inv.T * jnp.linalg.det(jnp.linalg.inv(Fp_inv)), sn
            return mat.pk1T(F - th * I3 - eig, s, T)

        def stress(F, se, Te, eig):
            """Te: (nn, 2) nodal [T, T_initial]."""
            Tg = N @ Te
            return jax.vmap(stress_gp, in_axes=(0, 0, 0, 0, None))(F, se, Tg[:, 0], Tg[:, 1], eig)

        def force(Xe, ue, se, Te, scv):
            sc, eig = scv[0], scv[1:10].reshape(3, 3)
            dNdX, vol = geom(Xe)
            if et.fbar:
                F, pullback = jax.vjp(lambda u: defgrad(Xe, u, dNdX), ue)
                P, snew = stress(F, se, Te, eig)
                fe = pullback(P * (sc * vol)[:, None, None])[0]
            else:
                F = defgrad(Xe, ue, dNdX)
                P, snew = stress(F, se, Te, eig)
                fe = jnp.einsum("gij,gaj,g->ai", P, dNdX, sc * vol)
            return fe, snew

        def stiff(Xe, ue, se, Te, scv):
            nn = ue.shape[0]
            if et.fbar:
                K = jax.jacfwd(lambda u: force(Xe, u, se, Te, scv)[0])(ue)
                return K.reshape(nn * 3, nn * 3)
            # fast path: material tangent A = dP/dF per point (9 AD directions instead of 3*nn)
            sc, eig = scv[0], scv[1:10].reshape(3, 3)
            dNdX, vol = geom(Xe)
            F = defgrad(Xe, ue, dNdX)
            Tg = N @ Te
            A = jax.vmap(jax.jacfwd(lambda f, s, t, t0: stress_gp(f, s, t, t0, eig)[0]))(F, se, Tg[:, 0], Tg[:, 1])
            K = jnp.einsum("gaj,gijkl,gbl,g->aibk", dNdX, A, dNdX, sc * vol)
            return K.reshape(nn * 3, nn * 3)

        def geo_stiff(Xe, sig):
            """Initial-stress (geometric) stiffness from a Cauchy stress per element (3,3)."""
            dNdX, vol = geom(Xe)
            k = jnp.einsum("gai,ij,gbj,g->ab", dNdX, sig, dNdX, vol)
            nn = k.shape[0]
            return jnp.einsum("ab,ij->aibj", k, I3).reshape(nn * 3, nn * 3)

        def mass(Xe):
            """HRZ diagonal lumping (positive masses also for quadratic elements)."""
            _, vol = geom(Xe)
            d = mat.density * jnp.einsum("g,ga->a", vol, N * N)
            return d * (mat.density * vol.sum()) / d.sum()

        def volume(Xe):
            return geom(Xe)[1].sum()

        def cauchy(Xe, ue, se, Te, scv):
            dNdX, vol = geom(Xe)
            F = defgrad(Xe, ue, dNdX)
            P, snew = stress(F, se, Te, scv[1:10].reshape(3, 3))
            J = jnp.linalg.det(F)
            sig = P if mat.small_strain else jnp.einsum("gij,gkj->gik", P, F) / J[:, None, None]
            wv = vol / vol.sum()
            return (jnp.einsum("g,gij->ij", wv, sig), jnp.einsum("g,g->", wv, mat.eqps(snew)),
                    jnp.einsum("g,gij->ij", wv, F))

        def charlen(Xe):
            nc = et.ncorner
            Xc = Xe[:nc]
            d = jnp.linalg.norm(Xc[:, None] - Xc[None], axis=-1) + jnp.eye(nc) * 1e30
            h = d.min()
            return h / et.order

        self.force = jax.vmap(force)
        self.stiff = jax.vmap(stiff)
        self.geo_stiff = jax.vmap(geo_stiff)
        self.mass = jax.vmap(mass)
        self.volume = jax.vmap(volume)
        self.cauchy = jax.vmap(cauchy)
        self.charlen = jax.vmap(charlen)

    def init_state(self):
        return jnp.zeros((self.ne, self.etype.ngp, self.material.nstate))

    def dofs(self):
        return (self.conn[:, :, None] * 3 + jnp.arange(3)).reshape(self.ne, -1)


# ------------------------------------------------------------------ generic EbE operators
def ops_matvec(ops, v, ndof):
    out = jnp.zeros(ndof, dtype=v.dtype)
    for d, K in ops:
        out = out.at[d].add(jnp.einsum("eij,ej->ei", K, v[d]))
    return out


def ops_diagonal(ops, ndof):
    out = jnp.zeros(ndof)
    for d, K in ops:
        out = out.at[d].add(jnp.diagonal(K, axis1=1, axis2=2))
    return out


def ops_assemble(ops, ndof):
    """Assemble a scipy CSR matrix (host) from EbE operators."""
    import scipy.sparse as sp
    rows, cols, vals = [], [], []
    for d, K in ops:
        d = np.asarray(d)
        K = np.asarray(K)
        nd = d.shape[1]
        rows.append(np.repeat(d, nd, axis=1).ravel())
        cols.append(np.tile(d, (1, nd)).ravel())
        vals.append(K.reshape(len(d), -1).ravel())
    if not rows:
        return sp.csr_matrix((ndof, ndof))
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(ndof, ndof)).tocsr()
    A.sum_duplicates()
    return A


def csr_pattern(dofs_list, ndof):
    """Static CSR pattern for element blocks: element matrix entries (all blocks, flattened)
    are permuted so that a *sorted* segment-sum produces the CSR values directly - an
    atomic-free GPU assembly. Returns device arrays perm, seg, rows, cols and nnz."""
    r_all, c_all = [], []
    for d in dofs_list:
        d = np.asarray(d, dtype=np.int64)
        nd = d.shape[1]
        r_all.append(np.repeat(d, nd, axis=1).ravel())
        c_all.append(np.tile(d, (1, nd)).ravel())
    r, c = np.concatenate(r_all), np.concatenate(c_all)
    key = r * ndof + c
    perm = np.argsort(key, kind="stable")
    ks = key[perm]
    first = np.ones(len(ks), bool)
    first[1:] = ks[1:] != ks[:-1]
    seg = np.cumsum(first) - 1
    uk = ks[first]
    rows, cols = uk // ndof, uk % ndof
    # ELLPACK layout: every row padded to the max row length -> the mat-vec is a pure gather
    # + row reduction (no scatter / atomics, coalesced on GPUs)
    cnt = np.bincount(rows, minlength=ndof)
    w = int(cnt.max()) if len(cnt) else 1
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    pos = rows * w + (np.arange(len(rows)) - start[rows])
    ell_cols = np.repeat(np.arange(ndof), w)
    ell_cols[pos] = cols
    return dict(perm=jnp.asarray(perm.astype(np.int32 if len(perm) < 2**31 else np.int64)),
                seg=jnp.asarray(seg.astype(np.int32)), rows=jnp.asarray(rows.astype(np.int32)),
                cols=jnp.asarray(cols.astype(np.int32)), ell_pos=jnp.asarray(pos.astype(np.int64)),
                ell_cols=jnp.asarray(ell_cols.astype(np.int32).reshape(ndof, w)))


def csr_values(pattern, Ks):
    """Element matrices -> ELL values (ndof, w): one sorted segment-sum per tangent."""
    flat = jnp.concatenate([K.reshape(-1) for K in Ks])
    v = jax.ops.segment_sum(flat[pattern["perm"]], pattern["seg"], num_segments=pattern["rows"].shape[0],
                            indices_are_sorted=True)
    E = pattern["ell_cols"]
    return jnp.zeros(E.size, v.dtype).at[pattern["ell_pos"]].set(v, unique_indices=True).reshape(E.shape)


def csr_matvec(pattern, vals, v, ndof):
    return jnp.sum(vals * v[pattern["ell_cols"]], axis=1)


def gather_plan(conns, nn):
    """Atomic-free assembly plan: for every node the list of (element, local node) slots that
    contribute to it, padded to the max valence. Element-level results (ne, nnode, 3) are
    summed with a pure gather + reduction instead of a scatter-add."""
    slots, nodes = [], []
    off = 0
    for c in conns:
        c = np.asarray(c)
        nodes.append(c.ravel())
        slots.append(off + np.arange(c.size))
        off += c.size
    nodes, slots = np.concatenate(nodes), np.concatenate(slots)
    order = np.argsort(nodes, kind="stable")
    nodes, slots = nodes[order], slots[order]
    cnt = np.bincount(nodes, minlength=nn)
    vmax = max(int(cnt.max()), 1)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    pos = np.arange(len(nodes)) - start[nodes]
    g = np.full((nn, vmax), off, np.int64)                 # sentinel -> zero row
    g[nodes, pos] = slots
    return jnp.asarray(g.astype(np.int32 if off < 2**31 else np.int64))


def gather_matvec(plan, ops, v, ndof):
    parts = [jnp.einsum("eij,ej->ei", K, v[d]).reshape(-1, 3) for d, K in ops]
    ys = jnp.concatenate(parts + [jnp.zeros((1, 3), v.dtype)])
    return jnp.sum(ys[plan], axis=1).reshape(-1)


class Model:
    def __init__(self, nodes):
        self.X = jnp.asarray(np.asarray(nodes, dtype=np.float64))
        self.nn = int(self.X.shape[0])
        self.ndof = 3 * self.nn
        self.blocks = []
        self.nsets = {}
        self.elsets = {}          # name -> list of (block index, local element ids)
        self.surfaces = {}        # name -> Surface
        self.constraints = []     # MPC objects (ties, symmetry, couplings)
        self.interactions = []    # contact / rigid wall contributors
        self.node_ids = None      # external (user) node numbers
        self.elem_ids = None      # per block arrays of external element numbers
        self.units = None
        self.T0 = None            # initial (stress-free) nodal temperature
        self.n_real = self.nn     # rows >= n_real are rotation nodes (shells, beams, remote points)
        self.rot_of = {}          # real node -> rotation row

    def add_rotation_nodes(self, nodes):
        """Create (or return existing) rotation rows for real nodes; returns their row indices."""
        nodes = [int(n) for n in np.asarray(nodes).ravel()]
        new = [n for n in dict.fromkeys(nodes) if n not in self.rot_of]
        if new:
            X = np.asarray(self.X)
            for k, n in enumerate(new):
                self.rot_of[n] = len(X) + k
            self.X = jnp.asarray(np.vstack([X, X[new]]))
            self.nn = int(self.X.shape[0])
            self.ndof = 3 * self.nn
        return np.array([self.rot_of[n] for n in nodes], dtype=np.int64)

    def add_point(self, xyz, rotations=True):
        """Add a free reference node (remote point); returns (node, rotation row)."""
        X = np.asarray(self.X)
        n = len(X)
        self.X = jnp.asarray(np.vstack([X, np.asarray(xyz, float)[None]]))
        self.nn += 1
        self.ndof = 3 * self.nn
        self.extra_points = getattr(self, "extra_points", []) + [n]
        r = self.add_rotation_nodes([n])[0] if rotations else None
        return n, r

    def dof(self, nodes, d):
        """Global dof indices of component d (0-2 translations, 3-5 rotations) of nodes."""
        nodes = np.asarray(nodes, dtype=np.int64).ravel()
        if d < 3:
            return nodes * 3 + d
        rows = np.array([self.rot_of.get(int(n), -1) for n in nodes], dtype=np.int64)
        rows = rows[rows >= 0]
        return rows * 3 + (d - 3)

    def rot_mask(self):
        m = np.zeros(self.nn, bool)
        m[list(self.rot_of.values())] = True
        return m

    def add_shell(self, conn, material, thickness, name=None, etype=None):
        from .structural import ShellBlock, S4, S3
        conn = np.asarray(conn, dtype=np.int64)
        r = self.add_rotation_nodes(conn.ravel()).reshape(conn.shape)
        et = etype or (S4 if conn.shape[1] == 4 else S3)
        b = ShellBlock(conn, r, np.asarray(self.X), et, material, thickness, name or f"shell{len(self.blocks)}")
        self.blocks.append(b)
        return b

    def add_beam(self, conn, material, section, n1=None, name=None):
        from .structural import BeamBlock, B31, beam_section
        conn = np.asarray(conn, dtype=np.int64)
        r = self.add_rotation_nodes(conn.ravel()).reshape(conn.shape)
        if isinstance(section, (tuple, list)):
            section = beam_section(*section)
        b = BeamBlock(conn, r, np.asarray(self.X), B31, material, section, name or f"beam{len(self.blocks)}", n1)
        self.blocks.append(b)
        return b

    def add_block(self, conn, etype, material, name=None):
        b = Block(conn, etype, material, name or f"block{len(self.blocks)}")
        self.blocks.append(b)
        return b

    def add_surface(self, name, surface):
        surface.name = name
        self.surfaces[name] = surface
        return surface

    def add_constraint(self, c):
        self.constraints.append(c)
        return c

    def add_interaction(self, c):
        self.interactions.append(c)
        return c

    def init_state(self):
        return [b.init_state() for b in self.blocks]

    # ---------- passing model data as jit *arguments* (no large baked-in constants) ----------
    def data(self):
        return dict(X=self.X, conn=[b.conn for b in self.blocks], plan=self.gather(),
                    T0=jnp.zeros(self.nn) if self.T0 is None else jnp.asarray(self.T0))

    def bound(self, data):
        model = self

        class _Bind:
            def __enter__(self_):
                self_.saved = (model.X, [b.conn for b in model.blocks], getattr(model, "_gather", None), model.T0)
                model.X = data["X"]
                for b, c in zip(model.blocks, data["conn"]):
                    b.conn = c
                model._gather = data["plan"]
                model._gather_lock = True
                model.T0 = data["T0"]

            def __exit__(self_, *exc):
                X, conns, g, T0 = self_.saved
                model.X = X
                for b, c in zip(model.blocks, conns):
                    b.conn = c
                model._gather = g
                model._gather_lock = False
                model.T0 = T0
        return _Bind()

    def jit(self, fn):
        """jit `fn(*args)` with the model arrays passed as runtime arguments."""
        model = self

        @jax.jit
        def g(data, *args):
            with model.bound(data):
                return fn(*args)
        return lambda *args: g(model.data(), *args)

    def gather(self):
        if getattr(self, "_gather_lock", False):
            return self._gather
        key = (self.nn, len(self.blocks))
        if getattr(self, "_gather_key", None) != key:
            self._gather = gather_plan([np.asarray(b.conn) for b in self.blocks], self.nn)
            self._gather_key = key
        return self._gather

    def csr(self):
        """Cached device CSR pattern of all element blocks (None for very large models,
        where the element-by-element operator is used instead)."""
        key = (self.ndof, len(self.blocks))
        if getattr(self, "_csr_key", None) != key:
            tot = sum(b.ne * (b.dofs().shape[1] ** 2) for b in self.blocks)
            self._csr = csr_pattern([b.dofs() for b in self.blocks], self.ndof) if tot < 4e8 else None
            self._csr_key = key
        return self._csr

    @property
    def nelem(self):
        return sum(b.ne for b in self.blocks)

    def default_fields(self):
        return dict(T=jnp.zeros(self.nn), scale=[jnp.ones(b.ne) for b in self.blocks])

    def _fields(self, fields):
        """Returns nodal [T, T0] (nn, 2) and per block element data (ne, 10) =
        [stiffness scale, eigenstrain (9)]."""
        f = self.default_fields() if fields is None else fields
        T0 = jnp.zeros(self.nn) if self.T0 is None else jnp.asarray(self.T0)
        T = f.get("T")
        T = T0 if T is None else T
        sc = f.get("scale") or [None] * len(self.blocks)
        eig = f.get("eig") or getattr(self, "eigenstrain", None) or [None] * len(self.blocks)
        out = []
        for b, s, e in zip(self.blocks, sc, eig):
            s = jnp.ones(b.ne) if s is None else jnp.asarray(s)
            e = jnp.zeros((b.ne, 9)) if e is None else jnp.asarray(e).reshape(b.ne, 9)
            out.append(jnp.concatenate([s[:, None], e], 1))
        return jnp.stack([T, T0], 1), out

    # ---------- global operators (all jit-able) ----------
    def internal_force(self, u, states, fields=None):
        T, sc = self._fields(fields)
        U = u.reshape(-1, 3)
        new, parts = [], []
        for b, s, c in zip(self.blocks, states, sc):
            fe, sn = b.force(self.X[b.conn], U[b.conn], s, T[b.conn], c)
            parts.append(fe.reshape(-1, 3))
            new.append(sn)
        if not parts:
            return jnp.zeros(self.ndof), new
        ys = jnp.concatenate(parts + [jnp.zeros((1, 3), parts[0].dtype)])
        return jnp.sum(ys[self.gather()], axis=1).reshape(-1), new     # atomic-free assembly

    def element_stiffness(self, u, states, fields=None):
        T, sc = self._fields(fields)
        U = u.reshape(-1, 3)
        return [b.stiff(self.X[b.conn], U[b.conn], s, T[b.conn], c)
                for b, s, c in zip(self.blocks, states, sc)]

    def stiffness_ops(self, u, states, fields=None):
        return list(zip([b.dofs() for b in self.blocks], self.element_stiffness(u, states, fields)))

    def tangent_jvp(self, u, states, v, fields=None):
        """Fully matrix-free K(u) v by forward-mode AD of the internal force."""
        return jax.jvp(lambda x: self.internal_force(x, states, fields)[0], (u,), (v,))[1]

    def _dofs(self, b):
        return b.dofs()

    def matvec(self, Kes, v):
        return ops_matvec(list(zip([b.dofs() for b in self.blocks], Kes)), v, self.ndof)

    def diagonal(self, Kes):
        return ops_diagonal(list(zip([b.dofs() for b in self.blocks], Kes)), self.ndof)

    def lumped_mass(self):
        m = jnp.zeros(self.nn)
        for b in self.blocks:
            m = m.at[b.conn].add(b.mass(self.X[b.conn]))
        return jnp.repeat(m, 3)

    def volumes(self):
        return [b.volume(self.X[b.conn]) for b in self.blocks]

    def total_mass(self):
        return float(sum(jnp.sum(v) * b.material.density for v, b in zip(self.volumes(), self.blocks)))

    def stable_dt(self):
        dts = [float(jnp.min(b.charlen(self.X[b.conn]))) / b.material.wave_speed()
               for b in self.blocks]
        return float(min(dts))

    def char_length(self):
        return float(np.mean([float(jnp.mean(b.charlen(self.X[b.conn]))) * b.etype.order
                              for b in self.blocks]))

    def modulus(self):
        return max(b.material.modulus() for b in self.blocks)

    def stresses(self, u, states, fields=None):
        T, sc = self._fields(fields)
        U = u.reshape(-1, 3)
        out = []
        for b, s, c in zip(self.blocks, states, sc):
            sig, ep, F = b.cauchy(self.X[b.conn], U[b.conn], s, T[b.conn], c)
            dev = sig - jnp.trace(sig, axis1=1, axis2=2)[:, None, None] / 3 * I3
            mises = jnp.sqrt(1.5 * jnp.sum(dev * dev, axis=(1, 2)))
            out.append(dict(S=sig, MISES=mises, PEEQ=ep, F=F))
        return out

    def geometric_ops(self, stresses, u=None, states=None):
        """Initial-stress stiffness. Solids: exact K_sigma from the stress; shells/beams:
        directional derivative of the AD tangent along the pre-buckling displacement."""
        out = []
        T, _ = self._fields(None)
        U = None if u is None else u.reshape(-1, 3)
        states = states or self.init_state()
        for b, st, s in zip(self.blocks, stresses, states):
            if hasattr(b, "geo_dir"):
                out.append((b.dofs(), b.geo_dir(self.X[b.conn], U[b.conn], s, T[b.conn])))
            else:
                out.append((b.dofs(), b.geo_stiff(self.X[b.conn], st["S"])))
        return out

    # ---------- topology helpers ----------
    def skin(self, return_owner=False):
        return _skin(np.asarray(self.X), self.blocks, return_owner)

    def nset(self, name):
        return np.asarray(self.nsets[name])

    def elset_nodes(self, name):
        nodes = [np.asarray(self.blocks[bi].conn)[le].ravel() for bi, le in self.elsets[name]]
        return np.unique(np.concatenate(nodes)) if nodes else np.zeros(0, np.int64)

    def summary(self):
        lines = [f"{self.nn} nodes, {self.nelem} elements, {self.ndof} dofs"]
        for b in self.blocks:
            lines.append(f"  {b.name}: {b.ne} x {b.etype.name}, {b.material.describe()}")
        return "\n".join(lines)


__all__ = ["Model", "Block", "Surface", "ops_matvec", "ops_diagonal", "ops_assemble"]
