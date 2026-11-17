"""Device eigensolvers for very large modal and buckling problems.

`lobpcg` is a block LOBPCG (Knyazev) for the generalised problem  A x = lambda B x  (smallest
eigenvalues), written entirely with matrix-free operators so that on an NVIDIA GPU every
operation (EbE mat-vecs, AMG V-cycles, block Rayleigh-Ritz) runs on the device through
cuBLAS / cuSOLVER / XLA. Constraints (fixed dofs, MPC) are handled by projection.

Modal     :  K phi = w^2 M phi          A = K,   B = M (lumped), T = AMG(K)
Buckling  :  K0 phi = -lambda KG phi    A = KG,  B = K0,         T = AMG(K0)
             (smallest mu of KG phi = mu K0 phi  ->  lambda = -1/mu)

Compared with shift-invert Lanczos (one sparse factorisation of K, memory ~ fill-in), LOBPCG
needs only mat-vecs and a preconditioner: memory is linear in the model size, which is what
makes 10-100 M dof eigen analyses feasible on a single GPU.
"""
import numpy as np
import jax
import jax.numpy as jnp


def _svqb(B, S, tol=1e-10):
    """B-orthonormalise the columns of S (drop near-dependent directions)."""
    BS = B(S)
    G = S.T @ BS
    G = 0.5 * (G + G.T)
    d = jnp.sqrt(jnp.maximum(jnp.diag(G), 1e-300))
    Gs = G / d[:, None] / d[None, :]
    w, V = jnp.linalg.eigh(Gs)
    return w, V, d


def lobpcg(A, B, T, X0, k, tol=1e-6, maxiter=500, log=None):
    """A, B, T: functions on (n, m) blocks. X0: (n, m) start block with m >= k.
    Returns (eigenvalues (k,), eigenvectors (n, k), iterations)."""
    X = X0

    def orth(S):
        w, V, d = _svqb(B, S)
        keep = w > 1e-12 * w.max()
        Vk = V[:, keep] / jnp.sqrt(w[keep])[None, :]
        return (S / d[None, :]) @ Vk

    def rr(S, m):
        AS = A(S)
        Ga = S.T @ AS
        Ga = 0.5 * (Ga + Ga.T)
        lam, C = jnp.linalg.eigh(Ga)
        return lam[:m], C[:, :m], AS

    X = orth(X)
    m = X.shape[1]
    lam, C, AS = rr(X, m)
    X = X @ C
    AX = AS @ C
    BX = B(X)
    P = None
    it = 0
    for it in range(1, maxiter + 1):
        R = AX - BX * lam[None, :]
        rn = jnp.linalg.norm(R, axis=0) / jnp.maximum(jnp.abs(lam) * jnp.linalg.norm(BX, axis=0), 1e-300)
        res = float(jnp.max(rn[:k]))
        if log and (it % 10 == 1):
            log(f"    LOBPCG it {it:4d}: max rel. residual {res:.2e}")
        if res < tol:
            break
        act = rn > tol * 0.1
        W = T(R)
        W = W - X @ (BX.T @ W)                       # B-orthogonal to X
        parts = [X, W] if P is None else [X, W, P - X @ (BX.T @ P)]
        S = orth(jnp.concatenate(parts, 1))
        lam, C, AS = rr(S, m)
        Xn = S @ C
        # new search direction: component of the update outside the old X
        P = Xn - X @ (BX.T @ Xn)
        X = Xn
        AX = AS @ C
        BX = B(X)
    return np.asarray(lam[:k]), X[:, :k], it


def _setup(model, step, state):
    from .solvers import _Setup, _ramp, State
    st = state or State(model)
    return _Setup(model, step, st, _ramp), st


def _amg_precond(model, ops, setup):
    from .model import ops_assemble
    from . import linalg
    A = setup.tr.project_matrix(ops_assemble(ops, model.ndof))
    A = linalg.bc_matrix(A, setup.free_np)
    amg = linalg.AMG(A, np.asarray(model.X), rot_mask=model.rot_mask() if model.rot_of else None)
    pre = amg.make()
    p = amg.params()
    f = jax.jit(jax.vmap(lambda pp, r: pre(pp, r), in_axes=(None, 1), out_axes=1))
    return (lambda R: f(p, R)), amg


def _block_op(ops, setup, ndof, diag=None):
    from .model import ops_matvec
    from .constraints import t_apply, t_apply_T
    if ops is not None and jax.device_count() > 1:
        from .parallel import shard_ops
        ops = shard_ops(ops)
    free = setup.free
    tp = setup.tr.params()

    def one(ops_, diag_, free_, tp_, v):
        w = t_apply(tp_, free_ * v, ndof)
        y = ops_matvec(ops_, w, ndof) if ops_ is not None else diag_ * w
        return free_ * t_apply_T(tp_, y, ndof)
    f = jax.jit(jax.vmap(one, in_axes=(None, None, None, None, 1), out_axes=1))
    return lambda V: f(ops, diag, free, tp, V)


def modal_lobpcg(model, step, state=None, monitor=None, tol=1e-6):
    from .solvers import Monitor
    mon = monitor or Monitor(False)
    setup, st = _setup(model, step, state)
    nd = model.ndof
    ops = model.stiffness_ops(st.u, st.states)
    M = model.lumped_mass()
    A = _block_op(ops, setup, nd)
    Bop = _block_op(None, setup, nd, diag=M)
    T, amg = _amg_precond(model, ops, setup)
    mon.log(f"  modal (device LOBPCG): {int(setup.free_np.sum())} free dofs, {amg.info()}")
    k = step.nmodes
    m = k + max(4, k // 2)
    X0 = jax.random.normal(jax.random.PRNGKey(0), (nd, m)) * setup.free[:, None]
    lam, V, it = lobpcg(A, Bop, T, X0, k, tol=tol, log=mon.log)
    w2 = np.maximum(lam, 0.0)
    modes = np.asarray(jax.vmap(setup.tr.apply, in_axes=1, out_axes=1)(V) if setup.tr.active else V).T
    mon.log(f"  LOBPCG converged in {it} iterations")
    return w2, modes


def buckling_lobpcg(model, K0ops, KGops, setup, nmodes, monitor=None, tol=1e-6):
    from .solvers import Monitor
    mon = monitor or Monitor(False)
    nd = model.ndof
    A = _block_op(KGops, setup, nd)
    Bop = _block_op(K0ops, setup, nd)
    T, amg = _amg_precond(model, K0ops, setup)
    mon.log(f"  buckling (device LOBPCG): {int(setup.free_np.sum())} free dofs, {amg.info()}")
    m = nmodes + max(4, nmodes // 2)
    X0 = jax.random.normal(jax.random.PRNGKey(1), (nd, m)) * setup.free[:, None]
    mu, V, it = lobpcg(A, Bop, T, X0, nmodes, tol=tol, log=mon.log)
    lam = np.where(mu < 0, -1.0 / np.where(mu < 0, mu, -1.0), np.inf)
    modes = np.asarray(jax.vmap(setup.tr.apply, in_axes=1, out_axes=1)(V) if setup.tr.active else V).T
    mon.log(f"  LOBPCG converged in {it} iterations")
    return lam, modes
