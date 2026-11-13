"""Linear algebra back-ends.

* `pcg`      : preconditioned conjugate gradients written with lax.while_loop (runs on device,
               returns iteration count and residual for monitoring).
* `AMG`      : smoothed-aggregation algebraic multigrid. The hierarchy is set up once on the host
               (PyAMG, rigid-body near-null space, 3x3 nodal blocks) and the V-cycle is executed on
               the device (CSR mat-vecs as gather + segment_sum, Chebyshev-accelerated Jacobi
               smoothing, dense Cholesky on the coarsest grid). The tangent itself stays
               matrix-free; the hierarchy is frozen and reused across Newton iterations and
               increments ("frozen AMG"), and rebuilt only when PCG iteration counts degrade.
* `Direct`   : sparse direct factorisation (SuperLU / UMFPACK on the host) for small or very
               ill-conditioned problems.
"""
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg  # noqa: F401


# ------------------------------------------------------------------ PCG
def pcg(A, b, M=None, x0=None, tol=1e-8, maxiter=2000):
    M = M or (lambda r: r)
    x = jnp.zeros_like(b) if x0 is None else x0
    r = b - A(x)
    z = M(r)
    p = z
    rz = jnp.vdot(r, z)
    bn = jnp.maximum(jnp.linalg.norm(b), 1e-300)

    def cond(c):
        x, r, z, p, rz, k = c
        return (jnp.linalg.norm(r) > tol * bn) & (k < maxiter)

    def body(c):
        x, r, z, p, rz, k = c
        Ap = A(p)
        alpha = rz / jnp.vdot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        z = M(r)
        rz_new = jnp.vdot(r, z)
        p = z + (rz_new / rz) * p
        return x, r, z, p, rz_new, k + 1

    x, r, z, p, rz, k = jax.lax.while_loop(cond, body, (x, r, z, p, rz, 0))
    return x, k, jnp.linalg.norm(r) / bn


# ------------------------------------------------------------------ CSR helpers
def csr_to_device(A):
    A = A.tocsr()
    A.sort_indices()
    rows = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
    return dict(data=jnp.asarray(A.data), cols=jnp.asarray(A.indices.astype(np.int32)),
                rows=jnp.asarray(rows.astype(np.int32)), n=A.shape[0])


def csr_mv(M, x, n):
    return jax.ops.segment_sum(M["data"] * x[M["cols"]], M["rows"], num_segments=n,
                               indices_are_sorted=True)


def rigid_body_modes(X, rot_mask=None):
    """Near-null space for elasticity; rotation rows (shells/beams) get unit rotations."""
    X = np.asarray(X)
    n = len(X)
    t = np.ones(n) if rot_mask is None else (~np.asarray(rot_mask)).astype(float)
    c = (X - X[t > 0].mean(0)) * t[:, None]
    B = np.zeros((3 * n, 6))
    for i in range(3):
        B[i::3, i] = t
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    B[0::3, 3], B[1::3, 3] = -y, x
    B[1::3, 4], B[2::3, 4] = -z, y
    B[0::3, 5], B[2::3, 5] = z, -x
    if rot_mask is not None:
        r = 1.0 - t
        B[2::3, 3] += r          # rotation about z
        B[0::3, 4] += r          # rotation about x
        B[1::3, 5] += r          # rotation about y
    return B


# ------------------------------------------------------------------ AMG
class AMG:
    def __init__(self, A, X=None, max_coarse=300, degree=2, presmooth=1, rot_mask=None):
        import pyamg
        import scipy.sparse as sp
        A = A.tocsr()
        n = A.shape[0]
        B = rigid_body_modes(X, rot_mask) if X is not None and n == 3 * len(X) else np.ones((n, 1))
        if B.shape[1] == 6:
            Ab = sp.bsr_matrix(A, blocksize=(3, 3))
        else:
            Ab = A
        self.ml = pyamg.smoothed_aggregation_solver(
            Ab, B=B, strength=("symmetric", {"theta": 0.0}),
            smooth=("jacobi", {"omega": 4.0 / 3.0, "weighting": "local"}),
            max_coarse=max_coarse, keep=False, improve_candidates=None)
        self.levels = []
        for lv in self.ml.levels[:-1]:
            Al = lv.A.tocsr()
            d = Al.diagonal()
            dinv = np.where(np.abs(d) > 0, 1.0 / np.where(d == 0, 1, d), 0.0)
            rho = self._rho(Al, dinv)
            self.levels.append(dict(A=csr_to_device(Al), dinv=jnp.asarray(dinv), rho=float(rho),
                                    P=csr_to_device(lv.P.tocsr()), R=csr_to_device(lv.P.T.tocsr()),
                                    n=Al.shape[0], nc=lv.P.shape[1]))
        Ac = self.ml.levels[-1].A.toarray()
        Ac = 0.5 * (Ac + Ac.T)
        try:
            Lc = np.linalg.cholesky(Ac + 1e-13 * np.abs(Ac).max() * np.eye(len(Ac)))
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(Ac)
            Lc = np.linalg.cholesky(V @ np.diag(np.maximum(w, 1e-12 * w.max())) @ V.T)
        self.coarse = jnp.asarray(Lc)            # coarse solve by two triangular solves
        self.degree, self.presmooth = degree, presmooth
        self.complexity = float(self.ml.operator_complexity())
        self.nlevels = len(self.ml.levels)

    @staticmethod
    def _rho(A, dinv, it=15):
        x = np.random.default_rng(0).standard_normal(A.shape[0])
        lam = 1.0
        for _ in range(it):
            y = dinv * (A @ x)
            lam = np.linalg.norm(y) / max(np.linalg.norm(x), 1e-300)
            x = y / max(np.linalg.norm(y), 1e-300)
        return 1.1 * lam

    def info(self):
        sizes = [lv.A.shape[0] for lv in self.ml.levels]
        return f"AMG levels={self.nlevels} sizes={sizes} op.complexity={self.complexity:.2f}"

    def params(self):
        return dict(levels=[{k: v for k, v in lv.items() if k in ("A", "dinv", "P", "R")}
                            for lv in self.levels], coarse=self.coarse)

    def make(self):
        """Return a pure function precond(params, r) suitable for jit."""
        meta = [(lv["n"], lv["nc"], lv["rho"]) for lv in self.levels]
        deg = self.degree

        def cheb(A, dinv, rho, n, x, b):
            # Chebyshev smoother on D^-1 A over [rho/30, rho]
            lmax, lmin = rho, rho / 30.0
            theta, delta = 0.5 * (lmax + lmin), 0.5 * (lmax - lmin)
            sigma = theta / delta
            rh = 1.0 / sigma
            r = dinv * (b - csr_mv(A, x, n))
            d = r / theta
            for _ in range(deg):
                x = x + d
                r = r - dinv * csr_mv(A, d, n)
                rh_new = 1.0 / (2 * sigma - rh)
                d = rh_new * rh * d + 2 * rh_new / delta * r
                rh = rh_new
            return x + d

        def vcycle(p, lvl, b):
            if lvl == len(meta):
                L = p["coarse"]
                y = jax.scipy.linalg.solve_triangular(L, b, lower=True)
                return jax.scipy.linalg.solve_triangular(L.T, y, lower=False)
            n, nc, rho = meta[lvl]
            L = p["levels"][lvl]
            x = cheb(L["A"], L["dinv"], rho, n, jnp.zeros_like(b), b)
            r = b - csr_mv(L["A"], x, n)
            xc = vcycle(p, lvl + 1, csr_mv(L["R"], r, nc))
            x = x + csr_mv(L["P"], xc, n)
            return cheb(L["A"], L["dinv"], rho, n, x, b)

        return lambda p, r: vcycle(p, 0, r)


# ------------------------------------------------------------------ direct
class Direct:
    def __init__(self, A):
        import scipy.sparse.linalg as spla
        self.n = A.shape[0]
        try:
            self.lu = spla.splu(A.tocsc(), permc_spec="COLAMD")
            self._solve = self.lu.solve
        except RuntimeError:
            self._solve = spla.factorized(A.tocsc())

    def solve(self, b):
        return self._solve(np.asarray(b))


def bc_matrix(A, free):
    """D A D + (I - D) for a 0/1 free mask (fixed dofs -> identity rows)."""
    import scipy.sparse as sp
    D = sp.diags(np.asarray(free, float))
    return (D @ A @ D + sp.diags(1.0 - np.asarray(free, float))).tocsr()
