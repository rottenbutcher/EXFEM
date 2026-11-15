"""Heat transfer: steady-state and transient (backward Euler) conduction with fixed
temperatures, surface heat flux, convection (film) and internal heat generation.
Element matrices are batched kernels like the structural ones; the system is solved with a
sparse direct factorisation or Jacobi/AMG-PCG on the device.

The nodal temperature field can be passed directly to a structural Step (`temperature=`) for
sequentially coupled thermal-stress analysis.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp

from .model import ops_assemble, ops_matvec, ops_diagonal
from . import linalg


class HeatTransfer:
    def __init__(self, model):
        self.m = model
        self.cond, self.cap, self.gen = [], [], []
        for b in model.blocks:
            et, mat = b.etype, b.material
            dN, w, N = jnp.asarray(et.dN), jnp.asarray(et.gw), jnp.asarray(et.N)

            def geom(Xe, dN=dN, w=w):
                Jac = jnp.einsum("an,gai->gni", Xe, dN)
                dNdX = jnp.einsum("gai,gij->gaj", dN, jnp.linalg.inv(Jac))
                return dNdX, w * jnp.linalg.det(Jac)

            def ke(Xe, geom=geom, k=mat.k):
                dNdX, vol = geom(Xe)
                return k * jnp.einsum("gai,gbi,g->ab", dNdX, dNdX, vol)

            def ce(Xe, geom=geom, N=N, rc=mat.density * mat.cp):
                _, vol = geom(Xe)
                d = jnp.einsum("g,ga->a", vol, N * N)
                return rc * d * vol.sum() / d.sum()

            def qe(Xe, geom=geom, N=N):
                _, vol = geom(Xe)
                d = jnp.einsum("g,ga->a", vol, N * N)
                return d * vol.sum() / d.sum()
            self.cond.append(jax.vmap(ke))
            self.cap.append(jax.vmap(ce))
            self.gen.append(jax.vmap(qe))

    def ops(self):
        X = self.m.X
        return [(b.conn, k(X[b.conn])) for b, k in zip(self.m.blocks, self.cond)]

    def capacity(self):
        X = self.m.X
        c = jnp.zeros(self.m.nn)
        for b, k in zip(self.m.blocks, self.cap):
            c = c.at[b.conn].add(k(X[b.conn]))
        return c

    def generation(self, q_by_block):
        """q_by_block: {block index: volumetric heat generation}."""
        X = self.m.X
        f = jnp.zeros(self.m.nn)
        for i, q in q_by_block.items():
            b = self.m.blocks[i]
            f = f.at[b.conn].add(q * self.gen[i](X[b.conn]))
        return f

    @staticmethod
    def film_ops(X, surface, h):
        X = np.asarray(X)
        ops = []
        for ft, f in surface.groups:
            xe = X[f]
            a1 = np.einsum("fai,ga->fgi", xe, ft.dN[:, :, 0])
            a2 = np.einsum("fai,ga->fgi", xe, ft.dN[:, :, 1])
            da = np.linalg.norm(np.cross(a1, a2), axis=-1) * ft.gw[None]
            K = h * np.einsum("fg,ga,gb->fab", da, ft.N, ft.N)
            ops.append((jnp.asarray(f), jnp.asarray(K)))
        return ops

    def solve(self, fixed=(), flux=(), film=(), gen=None, T0=None, transient=None,
              monitor=None, method="auto", radiation=(), sigma=5.670374419e-14, offset=273.15):
        """radiation: [(Surface, emissivity, T_ambient)] surface-to-ambient radiation
        q = eps sigma (T^4 - Ta^4) on absolute temperatures (T + offset); sigma defaults to
        the Stefan-Boltzmann constant in mm-N-s units (W/mm^2 K^4); use 5.670374419e-8 for SI."""
        """fixed: [(nodes, T)], flux: [(Surface, q)] (q into body), film: [(Surface, h, Tinf)],
        gen: {block: q}. transient: dict(time=, dt=, frames=) -> backward Euler from T0.
        Returns final nodal temperatures (and stores self.history for transient)."""
        m = self.m
        X = np.asarray(m.X)
        nn = m.nn
        mon = monitor
        t0 = time.time()
        ops = self.ops()
        rhs = np.zeros(nn)
        for s, q in flux:
            w, _, _ = s.integrate(X)
            rhs += q * w
        for s, h, Tinf in film:
            ops += self.film_ops(X, s, h)
            w, _, _ = s.integrate(X)
            rhs += h * Tinf * w
        if gen:
            rhs += np.asarray(self.generation(gen))
        fixed_mask = np.zeros(nn, bool)
        Tfix = np.zeros(nn)
        for nodes, T in fixed:
            fixed_mask[np.asarray(nodes, dtype=np.int64)] = True
            Tfix[np.asarray(nodes, dtype=np.int64)] = T
        free = (~fixed_mask).astype(float)
        K = ops_assemble(ops, nn)
        if method == "auto":
            method = "direct" if nn < 300000 else "amg"

        def solve_lin(A, b):
            Ab = linalg.bc_matrix(A, free)
            if method == "direct":
                return linalg.Direct(Ab).solve(b)
            amg = linalg.AMG(Ab, None)
            pre = amg.make()
            p = amg.params()
            Ad = linalg.csr_to_device(Ab)
            x, k, rr = jax.jit(lambda b: linalg.pcg(lambda v: linalg.csr_mv(Ad, v, nn), b,
                                                    lambda r: pre(p, r), tol=1e-10))(jnp.asarray(b))
            return np.asarray(x)

        def rhs_bc(A, b):
            # move the prescribed temperatures to the right-hand side
            return np.where(fixed_mask, Tfix, b - A @ (Tfix * fixed_mask))

        wr = np.zeros(nn)
        Ta4 = np.zeros(nn)
        for srf, eps, Ta in radiation:
            w, _, _ = srf.integrate(X)
            wr += eps * w
            Ta4 += eps * w * (Ta + offset) ** 4
        Ta4 = np.where(wr > 0, Ta4 / np.maximum(wr, 1e-300), 0.0)

        def newton(Aop, b, Tg):
            """Newton on the radiation term (lumped): (A + dRad) dT = -R."""
            import scipy.sparse as sp
            T = Tg.copy()
            for it in range(40):
                Ta_ = T + offset
                R = Aop @ T - b + wr * sigma * (Ta_ ** 4 - Ta4)
                R[fixed_mask] = 0.0
                J = (Aop + sp.diags(4 * wr * sigma * Ta_ ** 3)).tocsr()
                dT = linalg.Direct(linalg.bc_matrix(J, free)).solve(np.where(fixed_mask, 0.0, -R))
                T = T + dT
                if np.abs(dT).max() < 1e-8 * max(np.abs(T).max(), 1.0):
                    break
            return T

        if transient is None:
            if radiation:                 # start from ambient (the radiation-free system may be singular)
                Tg = np.full(nn, float(np.mean([r[2] for r in radiation])))
                Tg[fixed_mask] = Tfix[fixed_mask]
                T = newton(K, rhs, Tg)
            else:
                T = solve_lin(K, rhs_bc(K, rhs))
            if mon:
                mon.log(f"  heat transfer (steady): {nn} nodes, T in [{T.min():.4g}, {T.max():.4g}], "
                        f"{time.time() - t0:.2f}s")
                mon.frame(1.0, None, None, T=jnp.asarray(T), step="thermal")
            self.history = [(1.0, T)]
            return T
        import scipy.sparse as sp
        C = np.asarray(self.capacity())
        dt = transient.get("dt", transient["time"] / 50)
        nsteps = int(np.ceil(transient["time"] / dt))
        dt = transient["time"] / nsteps
        A = (K + sp.diags(C / dt)).tocsr()
        T = np.full(nn, 0.0) if T0 is None else np.asarray(T0, float).copy()
        T[fixed_mask] = Tfix[fixed_mask]
        Ab = linalg.bc_matrix(A, free)
        lu = linalg.Direct(Ab)
        every = max(1, nsteps // transient.get("frames", 20))
        self.history = [(0.0, T.copy())]
        for i in range(1, nsteps + 1):
            b = rhs + C / dt * T
            T = lu.solve(rhs_bc(A, b))
            if radiation:
                T = newton(A, b, T)
            if i % every == 0 or i == nsteps:
                self.history.append((i * dt, T.copy()))
                if mon:
                    mon.increment(step="thermal", inc=i, t=i * dt, dt=dt, iters=1, ok=True, time_total=i * dt)
                    mon.frame(i * dt, None, None, T=jnp.asarray(T), step="thermal")
        if mon:
            mon.log(f"  heat transfer (transient): {nsteps} steps dt={dt:.3g}, T in "
                    f"[{T.min():.4g}, {T.max():.4g}], {time.time() - t0:.2f}s")
        return T
