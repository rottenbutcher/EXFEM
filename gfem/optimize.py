"""Design optimisation on the GPU solver.

* `topology_optimization` : SIMP compliance minimisation with a volume constraint, density
  filter and optimality-criteria update, on arbitrary unstructured (CAD) meshes. Each design
  iteration is one linear solve (direct or AMG-PCG on the device).
* `multi_load_cases`      : many linear load cases in a single batched solve (vmap over PCG),
  useful for load envelopes and design-of-experiments studies on one GPU.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial import cKDTree

from .solvers import _Setup, _LinearSystem, _ramp, State, Monitor
from .model import ops_matvec


def _element_centroids(model):
    X = np.asarray(model.X)
    out = []
    for b in model.blocks:
        c = np.asarray(b.conn)[:, :b.etype.ncorner]
        out.append(X[c].mean(1))
    return np.concatenate(out)


def topology_optimization(model, step, volfrac=0.4, penal=3.0, rmin=None, iters=50, move=0.2,
                          Emin=1e-6, tol=0.01, frozen=None, monitor=None, linear_solver="auto"):
    """frozen: boolean mask over all elements kept solid (e.g. elements touching loads/supports).
    Returns dict(density, compliance history)."""
    mon = monitor or Monitor(True)
    m = model
    ne = [b.ne for b in m.blocks]
    offs = np.cumsum([0] + ne)
    vol = np.concatenate([np.asarray(v) for v in m.volumes()])
    cen = _element_centroids(m)
    if rmin is None:
        rmin = 1.5 * np.mean(vol) ** (1 / 3) * 1.5
    tree = cKDTree(cen)
    pairs = tree.sparse_distance_matrix(tree, rmin, output_type="coo_matrix")
    import scipy.sparse as sp
    H = sp.coo_matrix((rmin - pairs.data, (pairs.row, pairs.col)), shape=(len(vol), len(vol))).tocsr()
    H = H + sp.diags(np.full(len(vol), rmin) - H.diagonal())
    Hs = np.asarray(H.sum(1)).ravel()
    frozen = np.zeros(len(vol), bool) if frozen is None else np.asarray(frozen, bool)
    x = np.full(len(vol), volfrac)
    x[frozen] = 1.0
    st = State(m)
    setup = _Setup(m, step, st, _ramp)
    lin = _LinearSystem(m, linear_solver, "ebe", monitor=mon)
    fext = setup.fext(step.T)
    up = setup.prescribed(step.T)
    zero_s = m.init_state()
    ops_fn = jax.jit(lambda sc: m.stiffness_ops(jnp.zeros(m.ndof), zero_s, dict(T=None, scale=sc)))

    @jax.jit
    def energies(u):
        U = u.reshape(-1, 3)
        out = []
        for b, s in zip(m.blocks, zero_s):
            fe, _ = b.force(m.X[b.conn], U[b.conn], s, jnp.zeros(b.conn.shape + (2,)), jnp.ones(b.ne))
            out.append(jnp.einsum("eai,eai->e", fe, U[b.conn]))
        return jnp.concatenate(out)

    hist = []
    t0 = time.time()
    xphys = (H @ x) / Hs
    xphys[frozen] = 1.0
    for it in range(1, iters + 1):
        if mon.should_stop():
            break
        Es = Emin + xphys ** penal * (1 - Emin)
        sc = [jnp.asarray(Es[offs[i]:offs[i + 1]]) for i in range(len(ne))]
        ops = ops_fn(sc)
        r = setup.free * setup.tr.apply_T(ops_matvec(ops, up, m.ndof) - fext)
        du = lin.solve(ops, -r, setup)
        u = up + setup.free * du
        u = setup.tr.apply(u) if setup.tr.active else u
        ce = np.asarray(energies(u))                       # u_e^T K0_e u_e
        c = float(np.sum(Es * ce))
        dc = -penal * xphys ** (penal - 1) * (1 - Emin) * ce
        dv = vol.copy()
        dc = H.T @ (dc / Hs)
        dv = H.T @ (dv / Hs)
        # optimality criteria with bisection on the Lagrange multiplier
        l1, l2 = 0.0, 1e9
        xold = x.copy()
        target = volfrac * vol.sum()
        while (l2 - l1) / (l1 + l2 + 1e-300) > 1e-4:
            lmid = 0.5 * (l1 + l2)
            Be = np.sqrt(np.maximum(-dc, 0) / (dv * lmid + 1e-300))
            xn = np.clip(np.clip(x * Be, x - move, x + move), 1e-3, 1.0)
            xn[frozen] = 1.0
            xp = (H @ xn) / Hs
            xp[frozen] = 1.0
            if np.sum(xp * vol) > target:
                l1 = lmid
            else:
                l2 = lmid
        x = xn
        xphys = xp
        change = float(np.abs(x - xold).max())
        vf = float(np.sum(xphys * vol) / vol.sum())
        hist.append((it, c, vf, change))
        mon.log(f"  topo it {it:3d}: compliance={c:.6g}  vol={vf:.3f}  change={change:.3f}")
        mon.increment(step=step.name, inc=it, t=float(it), iters=lin.last_iters, ok=True,
                      compliance=c, volume=vf, time_total=float(it))
        mon.frame(float(it), u, zero_s, step=step.name, density=xphys.copy(), compliance=c)
        if change < tol and it > 5:
            break
    mon.log(f"  topology optimisation done: {it} iterations, {time.time() - t0:.2f}s")
    return dict(density=xphys, history=np.array(hist), u=u)


def multi_load_cases(model, step, cases, cg_tol=1e-8):
    """Solve K u_i = f_i for many load vectors at once (linear elastic). `cases` is an array
    (ncase, ndof) of external force vectors; boundary conditions from `step`.
    Returns (ncase, ndof) displacements. All cases run in one vmapped PCG on the device."""
    from .linalg import pcg
    from .model import ops_diagonal
    m = model
    st = State(m)
    setup = _Setup(m, step, st, _ramp)
    ops = m.stiffness_ops(jnp.zeros(m.ndof), m.init_state())
    free = setup.free
    tp = setup.tr.params()
    from .constraints import t_apply, t_apply_T
    nd = m.ndof
    d = ops_diagonal(ops, nd)
    inv = jnp.where(free > 0, 1.0 / jnp.where(jnp.abs(d) > 0, d, 1.0), 1.0)

    def A(v):
        w = t_apply(tp, free * v, nd)
        return free * t_apply_T(tp, ops_matvec(ops, w, nd), nd) + (1 - free) * v

    @jax.jit
    def solve_all(F):
        return jax.vmap(lambda f: pcg(A, free * t_apply_T(tp, f, nd), lambda r: inv * r,
                                      tol=cg_tol, maxiter=20000)[0])(F)
    U = solve_all(jnp.asarray(cases))
    if setup.tr.active:
        U = jax.vmap(setup.tr.apply)(U)
    return U
