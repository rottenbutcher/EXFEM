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
                J = jnp.linalg.det(F)
                F = F * ((jnp.linalg.det(F0) / J) ** (1 / 3))[:, None, None]
            return F

        def stress_gp(F, s, T, T0):
            """Thermal strain alpha (T - T_initial) (Abaqus convention: no thermal strain at the
            initial temperature): additive for small-strain models, multiplicative
            F = Fe (theta I), P = theta^2 Pe(Fe) for finite-strain models."""
            th = alpha * (T - T0)
            if fs:
                P, sn = mat.pk1T(F / (1.0 + th), s, T)
                return P * (1.0 + th) ** 2, sn
            return mat.pk1T(F - th * I3, s, T)

        def stress(F, se, Te):
            """Te: (nn, 2) nodal [T, T_initial]."""
            Tg = N @ Te
            return jax.vmap(stress_gp)(F, se, Tg[:, 0], Tg[:, 1])

        def force(Xe, ue, se, Te, sc):
            dNdX, vol = geom(Xe)
            if et.fbar:
                F, pullback = jax.vjp(lambda u: defgrad(Xe, u, dNdX), ue)
                P, snew = stress(F, se, Te)
                fe = pullback(P * (sc * vol)[:, None, None])[0]
            else:
                F = defgrad(Xe, ue, dNdX)
                P, snew = stress(F, se, Te)
                fe = jnp.einsum("gij,gaj,g->ai", P, dNdX, sc * vol)
            return fe, snew

        def stiff(Xe, ue, se, Te, sc):
            nn = ue.shape[0]
            if et.fbar:
                K = jax.jacfwd(lambda u: force(Xe, u, se, Te, sc)[0])(ue)
                return K.reshape(nn * 3, nn * 3)
            # fast path: material tangent A = dP/dF per point (9 AD directions instead of 3*nn)
            dNdX, vol = geom(Xe)
            F = defgrad(Xe, ue, dNdX)
            Tg = N @ Te
            A = jax.vmap(jax.jacfwd(lambda f, s, t, t0: stress_gp(f, s, t, t0)[0]))(F, se, Tg[:, 0], Tg[:, 1])
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

        def cauchy(Xe, ue, se, Te):
            dNdX, vol = geom(Xe)
            F = defgrad(Xe, ue, dNdX)
            P, snew = stress(F, se, Te)
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

    @property
    def nelem(self):
        return sum(b.ne for b in self.blocks)

    def default_fields(self):
        return dict(T=jnp.zeros(self.nn), scale=[jnp.ones(b.ne) for b in self.blocks])

    def _fields(self, fields):
        f = self.default_fields() if fields is None else fields
        T0 = jnp.zeros(self.nn) if self.T0 is None else jnp.asarray(self.T0)
        T = f.get("T")
        T = T0 if T is None else T
        sc = f.get("scale") or [jnp.ones(b.ne) for b in self.blocks]
        return jnp.stack([T, T0], 1), sc

    # ---------- global operators (all jit-able) ----------
    def internal_force(self, u, states, fields=None):
        T, sc = self._fields(fields)
        U = u.reshape(-1, 3)
        f = jnp.zeros_like(U, dtype=self.X.dtype)
        new = []
        for b, s, c in zip(self.blocks, states, sc):
            fe, sn = b.force(self.X[b.conn], U[b.conn], s, T[b.conn], c)
            f = f.at[b.conn].add(fe)
            new.append(sn)
        return f.reshape(-1), new

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
        T, _ = self._fields(fields)
        U = u.reshape(-1, 3)
        out = []
        for b, s in zip(self.blocks, states):
            sig, ep, F = b.cauchy(self.X[b.conn], U[b.conn], s, T[b.conn])
            dev = sig - jnp.trace(sig, axis1=1, axis2=2)[:, None, None] / 3 * I3
            mises = jnp.sqrt(1.5 * jnp.sum(dev * dev, axis=(1, 2)))
            out.append(dict(S=sig, MISES=mises, PEEQ=ep, F=F))
        return out

    def geometric_ops(self, stresses):
        return [(b.dofs(), b.geo_stiff(self.X[b.conn], st["S"]))
                for b, st in zip(self.blocks, stresses)]

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
