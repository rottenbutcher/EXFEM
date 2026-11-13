"""Analysis procedures.

* StaticSolver   : implicit quasi-static, Newton with tangent predictor, line search, automatic
                   incrementation with cutback (Abaqus *STATIC behaviour), contact, MPC, follower
                   pressure, thermal fields, amplitudes. Linear solver: direct / AMG-PCG / Jacobi-PCG,
                   with EbE element tangents or fully matrix-free JVP tangents.
* RiksSolver     : arc-length (Crisfield cylindrical) for snap-through / limit loads (*STATIC, RIKS).
* DynamicSolver  : implicit dynamics, HHT-alpha / Newmark (*DYNAMIC).
* ExplicitSolver : central difference, lumped mass, on-device lax.scan time loop (*DYNAMIC, EXPLICIT).
* modal(), buckling() : eigenvalue analyses (*FREQUENCY, *BUCKLE).
"""
import time
import numpy as np
import jax
import jax.numpy as jnp

from .model import ops_matvec, ops_diagonal, ops_assemble, csr_values, csr_matvec, gather_matvec
from .constraints import Transformation, t_apply, t_apply_T
from .loads import Amplitude, FollowerPressure, pressure_vector, smooth_step
from . import linalg


# ------------------------------------------------------------------ monitoring
class Monitor:
    """Receives solver progress. Subclass (GUI, logging) or pass verbose=True for printing."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.stop = False

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def iteration(self, **kw):
        pass

    def increment(self, **kw):
        pass

    def frame(self, t, u, states, **kw):
        pass

    def should_stop(self):
        return self.stop


class Step:
    """Analysis step.

    bcs     : [(nodes, dof(0..2), value[, Amplitude])]  prescribed displacement (totals at step end)
    loads   : [(nodes, dof, value[, Amplitude])]         concentrated forces
    pressures : [(Surface, p[, Amplitude[, follower]])]  uniform pressure, positive into surface
    dloads  : [(vector(ndof)[, Amplitude])]              precomputed load vectors (gravity, traction...)
    temperature : nodal temperatures at step end (thermal stress), linearly ramped
    kind    : static | riks | dynamic | explicit | frequency | buckle
    """

    def __init__(self, name="Step-1", kind="static", time_period=1.0, dt_init=None,
                 dt_min=None, dt_max=None, max_inc=1000, bcs=(), loads=(), dt=None, frames=20,
                 damping=0.0, pressures=(), dloads=(), temperature=None, amplitude=None,
                 nmodes=10, shift=0.0, riks=None, alpha=-0.05, output_every=1, mass_scaling=None,
                 nlgeom=None, eigensolver="auto"):
        self.name, self.kind, self.T = name, kind, float(time_period)
        self.dt_init = dt_init or self.T
        self.dt_min = dt_min if dt_min is not None else 1e-5 * self.T
        self.dt_max = dt_max or self.T
        self.max_inc = max_inc
        self.bcs, self.loads = list(bcs), list(loads)
        self.pressures, self.dloads = list(pressures), list(dloads)
        self.temperature = temperature
        self.dt, self.frames, self.damping = dt, frames, damping
        self.amplitude = amplitude
        self.nmodes, self.shift = nmodes, shift
        self.riks = riks or {}
        self.alpha = alpha
        self.output_every = output_every
        self.mass_scaling = mass_scaling
        self.nlgeom = nlgeom
        self.eigensolver = eigensolver


class State:
    """Solution state carried from step to step."""

    def __init__(self, model):
        self.u = jnp.zeros(model.ndof)
        self.v = jnp.zeros(model.ndof)
        self.a = jnp.zeros(model.ndof)
        self.states = model.init_state()
        self.fext = jnp.zeros(model.ndof)
        self.T = None
        self.followers = {}
        self.time = 0.0
        self.reactions = jnp.zeros(model.ndof)


def _amp_value(amp, t, T, default):
    if amp is None:
        return default(t, T)
    return amp(t)


class _Setup:
    """Host-side precomputation of BCs, loads, constraints for one step."""

    def __init__(self, model, step, state, default_amp):
        m = self.m = model
        self.step = step
        nd = m.ndof
        self.fixed = np.zeros(nd, bool)
        self.target = np.zeros(nd)
        amps = []
        self.bc_amp = np.full(nd, -1)

        def amp_id(a):
            if a is None:
                return -1
            for i, x in enumerate(amps):
                if x is a:
                    return i
            amps.append(a)
            return len(amps) - 1

        for bc in step.bcs:
            nodes, dof, val = bc[:3]
            d = m.dof(nodes, dof)
            self.fixed[d] = True
            self.target[d] = val
            self.bc_amp[d] = amp_id(bc[3] if len(bc) > 3 else None)
        # dead load groups by amplitude
        groups = {}
        X = np.asarray(m.X)
        followers = []
        for ld in step.loads:
            nodes, dof, val = ld[:3]
            k = amp_id(ld[3] if len(ld) > 3 else None)
            g = groups.setdefault(k, np.zeros(nd))
            np.add.at(g, m.dof(nodes, dof), val)
        for pr in step.pressures:
            surf, p = pr[:2]
            amp = pr[2] if len(pr) > 2 else None
            follower = pr[3] if len(pr) > 3 else False
            if follower:
                followers.append((FollowerPressure(surf, p), amp_id(amp), surf.name))
            else:
                g = groups.setdefault(amp_id(amp), np.zeros(nd))
                g += pressure_vector(X, surf, p)
        for dl in step.dloads:
            vec = np.asarray(dl[0], float)
            g = groups.setdefault(amp_id(dl[1] if len(dl) > 1 else None), np.zeros(nd))
            g += vec
        self.groups = {k: jnp.asarray(v) for k, v in groups.items()}
        self.amps = amps
        self.default = default_amp
        self.followers = followers
        self.fprev = state.fext
        self.ustart = np.asarray(state.u)
        T0 = jnp.zeros(m.nn) if m.T0 is None else jnp.asarray(m.T0)
        self.Tprev = state.T if state.T is not None else (
            jnp.asarray(step.temperature) if step.temperature is not None else T0)
        self.Tnew = jnp.asarray(step.temperature) if step.temperature is not None else self.Tprev
        self.fol_prev = dict(state.followers)
        self.tr = Transformation(nd, m.constraints, self.fixed)
        if self.tr.dropped:
            print(f"  warning: {self.tr.dropped} constrained dofs are also prescribed (constraint dropped)")
        self.free_np = (~self.fixed & ~self.tr.slave_mask).astype(float)
        self.free = jnp.asarray(self.free_np)
        self.fixed_j = jnp.asarray(self.fixed)
        fol_kernels = [f for f, _, _ in followers]
        Xj = m.X

        @jax.jit
        def ffol(u, scales):
            out = jnp.zeros(nd)
            for fp, s in zip(fol_kernels, scales):
                out = out + fp.force(Xj, u, s)
            return out
        self._ffol = ffol

    def amp_values(self, t):
        T = self.step.T
        return [a(t) for a in self.amps], self.default(t, T)

    def fext(self, t):
        av, r = self.amp_values(t)
        f = self.fprev * (1 - r)
        for k, g in self.groups.items():
            f = f + g * (r if k < 0 else av[k])
        return f

    def fext_final(self):
        return self.fext(self.step.T)

    def follower_scales(self, t):
        av, r = self.amp_values(t)
        out = []
        for fp, k, key in self.followers:
            a = r if k < 0 else av[k]
            prev = self.fol_prev.get(key, 0.0) / fp.p if fp.p else 0.0
            out.append(prev + (1 - prev) * a if k < 0 else a)
        return jnp.asarray(out) if out else jnp.zeros(0)

    def follower_force(self, u, t):
        if not self.followers:
            return 0.0
        return self._ffol(u, list(self.follower_scales(t)))

    def prescribed(self, t):
        av, r = self.amp_values(t)
        a = np.where(self.bc_amp < 0, r, np.take(np.asarray(av + [0.0]), self.bc_amp))
        tgt = np.where(self.bc_amp < 0, self.ustart + (self.target - self.ustart) * a, self.target * a)
        return jnp.asarray(np.where(self.fixed, tgt, 0.0))

    def temperature(self, t):
        _, r = self.amp_values(t)
        return self.Tprev + (self.Tnew - self.Tprev) * r


def _ramp(t, T):
    return min(max(t / T, 0.0), 1.0) if T > 0 else 1.0


def _diag_any(ops, nd):
    if isinstance(ops, dict) and "plan" in ops:
        return ops_diagonal(ops["blocks"] + list(ops["rest"]), nd)
    if isinstance(ops, dict):
        p = ops["pat"]
        d = jnp.sum(jnp.where(p["ell_cols"] == jnp.arange(nd)[:, None], ops["vals"], 0.0), axis=1)
        return d + ops_diagonal(ops["rest"], nd)
    return ops_diagonal(ops, nd)


# ------------------------------------------------------------------ linear system machinery
class _LinearSystem:
    """Solves (free T^T K T free + fixed I) x = b with the chosen back-end."""

    def __init__(self, model, method="auto", tangent="ebe", cg_tol=1e-9, cg_maxiter=20000,
                 monitor=None, multi_gpu=True, operator="auto"):
        self.m = model
        if method == "auto":
            gpu = jax.default_backend() != "cpu"
            method = "direct" if (model.ndof <= (20000 if gpu else 40000)) else "amg"
        self.method, self.tangent = method, tangent
        self.multi_gpu = multi_gpu
        # device CSR operator (atomic-free, less memory traffic) - default on GPUs
        self.operator = operator if operator != "auto" else "gather"
        self.async_amg = 150000          # dofs above which AMG setup runs in a background thread
        self.cg_tol, self.cg_maxiter = cg_tol, cg_maxiter
        self.amg = None
        self.amg_iters0 = None
        self.mon = monitor or Monitor(False)
        self._jit = {}
        self.last_iters = 0
        nd = model.ndof

        def opA(ops, free, tp, jv, v):
            w = t_apply(tp, free * v, nd)
            if isinstance(ops, dict) and "plan" in ops:      # EbE GEMV + atomic-free gather assembly
                y = gather_matvec(ops["plan"], ops["blocks"], w, nd) + ops_matvec(ops["rest"], w, nd)
            elif isinstance(ops, dict):                      # device CSR + remaining EbE ops
                y = csr_matvec(ops["pat"], ops["vals"], w, nd) + ops_matvec(ops["rest"], w, nd)
            else:
                y = ops_matvec(ops, w, nd)
            if jv is not None:
                y = y + jv(w)
            return free * t_apply_T(tp, y, nd) + (1 - free) * v
        self._opA = opA

    def _pcg_fn(self, kind, amg=None):
        key = (kind, id(amg))
        if key in self._jit:
            return self._jit[key]
        m, opA = self.m, self._opA
        pre = amg.make() if amg is not None else None
        tol, maxit = self.cg_tol, self.cg_maxiter

        nd = m.ndof

        def run(ops, b, free, tp, pp, jvp_args):
            if jvp_args is None:
                jv = None
            else:
                u, s, fields = jvp_args
                jv = lambda w: m.tangent_jvp(u, s, w, fields)
            A = lambda v: opA(ops, free, tp, jv, v)
            if kind == "jacobi":
                M = lambda r: pp * r
            elif kind == "cheby":
                # setup-free Chebyshev-Jacobi polynomial preconditioner (device only)
                dinv, lmax = pp["dinv"], pp["lmax"]

                def M(r):
                    lo, hi = lmax / 30.0, lmax
                    th, de = 0.5 * (hi + lo), 0.5 * (hi - lo)
                    sg = th / de
                    rh = 1.0 / sg
                    res = dinv * r
                    d = res / th
                    x = jnp.zeros_like(r)
                    for _ in range(6):
                        x = x + d
                        res = res - dinv * A(d)
                        rn = 1.0 / (2 * sg - rh)
                        d = rn * rh * d + 2 * rn / de * res
                        rh = rn
                    return x + d
            else:
                # fine level smoothed with the *current* tangent: the frozen hierarchy only
                # supplies the coarse spaces, so it stays effective as the Newton tangent evolves
                if jv is None:
                    dg = _diag_any(ops, nd)
                    dinv0 = jnp.where(free > 0, 1.0 / jnp.where(jnp.abs(dg) > 0, dg, 1.0), 1.0)
                    M = lambda r: free * pre(pp, free * r, A, dinv0) + (1 - free) * r
                else:
                    M = lambda r: free * pre(pp, free * r) + (1 - free) * r
            return linalg.pcg(A, b, M, tol=tol, maxiter=maxit)
        f = jax.jit(run)
        self._jit = {k: v for k, v in self._jit.items() if k[0] != kind} if kind == "amg" else self._jit
        self._jit[key] = f
        return f

    def assemble(self, ops, setup):
        A = ops_assemble(ops, self.m.ndof)
        A = setup.tr.project_matrix(A)
        return linalg.bc_matrix(A, setup.free_np)

    def _lmax(self, ops, setup, tp, dinv):
        """Largest eigenvalue of D^-1 A by power iteration on the device (Chebyshev bound)."""
        if not hasattr(self, "_jlmax"):
            opA = self._opA

            def f(ops, free, tp, dinv):
                v = jnp.sin(jnp.arange(free.shape[0]) * 0.37 + 0.1) * free + 1e-3
                lam = 1.0
                for _ in range(25):
                    w = dinv * opA(ops, free, tp, None, v)
                    lam = jnp.linalg.norm(w) / jnp.linalg.norm(v)
                    v = w / jnp.linalg.norm(w)
                return 1.1 * lam
            self._jlmax = jax.jit(f)
        return self._jlmax(ops, setup.free, tp, dinv)

    def _mixed(self, ops, b, setup, rebuild):
        """FP32 AMG-PCG inner solves + FP64 iterative refinement (defect correction).
        The FP64 residual keeps full accuracy while >95% of the flops run in FP32 - the fast
        path on GeForce / RTX / L4 / L40 class NVIDIA GPUs whose FP64 rate is 1/32-1/64."""
        f32 = jnp.float32
        tp = setup.tr.params()
        nd = self.m.ndof
        if self.amg is None or rebuild:
            t0 = time.time()
            self.amg = linalg.AMG(self.assemble(self._full_ops if isinstance(ops, dict) else ops, setup), np.asarray(self.m.X),
                                  rot_mask=self.m.rot_mask() if self.m.rot_of else None)
            self.mon.log(f"    {self.amg.info()} (mixed FP32/FP64)  setup {time.time() - t0:.2f}s")
            self._mixed_fn = None
        if getattr(self, "_mixed_fn", None) is None:
            pre = self.amg.make()
            opA = self._opA

            def inner(ops32, r32, free32, tp32, pp32):
                A = lambda v: opA(ops32, free32, tp32, None, v)
                M = lambda q: free32 * pre(pp32, free32 * q) + (1 - free32) * q
                return linalg.pcg(A, r32, M, tol=2e-5, maxiter=self.cg_maxiter)

            self._mixed_fn = (jax.jit(inner), jax.jit(lambda ops, free, tp, v: opA(ops, free, tp, None, v)))
        inner, A64 = self._mixed_fn
        cast = lambda t: jax.tree_util.tree_map(lambda a: a.astype(f32) if hasattr(a, "dtype") and jnp.issubdtype(a.dtype, jnp.floating) else a, t)
        ops32, pp32, tp32 = cast(ops), cast(self.amg.params()), cast(tp)
        free32 = setup.free.astype(f32)
        x = jnp.zeros_like(b)
        r = b
        bn = float(jnp.linalg.norm(b)) or 1.0
        its = 0
        for _ in range(12):
            sc = float(jnp.linalg.norm(r)) or 1.0
            d, k, _ = inner(ops32, (r / sc).astype(f32), free32, tp32, pp32)
            its += int(k)
            x = x + d.astype(b.dtype) * sc
            r = b - A64(ops, setup.free, tp, x)
            if float(jnp.linalg.norm(r)) <= self.cg_tol * bn:
                break
        self.last_iters = its
        return x

    def _gpu_direct(self, ops, b, setup):
        """Sparse direct solve on the device (cuSOLVER through jax.experimental.sparse)."""
        from jax.experimental.sparse.linalg import spsolve
        A = self.assemble(ops, setup).tocsr()
        A.sort_indices()
        x = spsolve(jnp.asarray(A.data), jnp.asarray(A.indices.astype(np.int32)),
                    jnp.asarray(A.indptr.astype(np.int32)), jnp.asarray(b), tol=0)
        self.last_iters = 1
        return x

    def solve(self, ops, b, setup, jvp_args=None, rebuild=False):
        tp = setup.tr.params()
        if self.multi_gpu and jax.device_count() > 1 and self.method not in ("direct", "gpu_direct"):
            from .parallel import shard_ops
            ops = shard_ops(ops)
        elif self.operator == "gather" and self.method in ("amg", "jacobi", "mixed", "cheby") and isinstance(ops, list) \
                and len(ops) >= len(self.m.blocks) and self.m.blocks:
            nb = len(self.m.blocks)
            self._full_ops = ops
            ops = dict(plan=self.m.gather(), blocks=list(ops[:nb]), rest=list(ops[nb:]))
        elif self.operator == "csr" and self.method in ("amg", "jacobi") and jvp_args is None:
            pat = self.m.csr()
            nb = len(self.m.blocks)
            if pat is not None and isinstance(ops, list) and len(ops) >= nb:
                if not hasattr(self, "_jcsr"):
                    self._jcsr = jax.jit(csr_values)
                full = ops
                ops = dict(pat=pat, vals=self._jcsr(pat, [K for _, K in ops[:nb]]), rest=list(ops[nb:]))
                self._full_ops = full
        if self.method == "mixed" and jvp_args is None:
            return self._mixed(ops, b, setup, rebuild)
        if self.method == "gpu_direct":
            return self._gpu_direct(ops, b, setup)
        if self.method == "direct":
            if jvp_args is not None:
                raise ValueError("direct solver requires tangent='ebe'")
            A = self.assemble(ops, setup)
            x = linalg.Direct(A).solve(np.asarray(b))
            self.last_iters = 1
            return jnp.asarray(x)
        if self.method == "cheby":
            d = _diag_any(ops, self.m.ndof)
            dinv = jnp.where(setup.free > 0, 1.0 / jnp.where(jnp.abs(d) > 0, d, 1.0), 1.0)
            x, k, rr = self._pcg_fn("cheby")(ops, b, setup.free, tp, dict(dinv=dinv, lmax=self._lmax(ops, setup, tp, dinv)), jvp_args)
            self.last_iters = int(k)
            return x
        if self.method == "jacobi":
            d = _diag_any(ops, self.m.ndof)
            pp = jnp.where(setup.free > 0, 1.0 / jnp.where(jnp.abs(d) > 0, d, 1.0), 1.0)
            x, k, rr = self._pcg_fn("jacobi")(ops, b, setup.free, tp, pp, jvp_args)
        else:
            fut = getattr(self, "_amg_future", None)
            if fut is not None and fut.done():
                self.amg, self._amg_future = fut.result(), None
                self.amg_iters0 = None
                self.mon.log(f"    {self.amg.info()}  (built in background)")
            if (self.amg is None or rebuild) and getattr(self, "_amg_future", None) is None:
                full = self._full_ops if isinstance(ops, dict) else ops
                rm = self.m.rot_mask() if self.m.rot_of else None
                if self.m.ndof >= self.async_amg:
                    # host AMG setup overlapped with device iterations (Chebyshev meanwhile)
                    import concurrent.futures as cf
                    A_host = self.assemble(full, setup)
                    Xh = np.asarray(self.m.X)
                    self._pool = getattr(self, "_pool", None) or cf.ThreadPoolExecutor(1)
                    self._amg_future = self._pool.submit(linalg.AMG, A_host, Xh, rot_mask=rm)
                    self.mon.log("    AMG setup started in background; Chebyshev-Jacobi PCG meanwhile")
                else:
                    t0 = time.time()
                    self.amg = linalg.AMG(self.assemble(full, setup), np.asarray(self.m.X), rot_mask=rm)
                    self.amg_iters0 = None
                    self.mon.log(f"    {self.amg.info()}  setup {time.time() - t0:.2f}s")
            if self.amg is None or (rebuild and getattr(self, "_amg_future", None) is not None):
                d = _diag_any(ops, self.m.ndof)
                dinv = jnp.where(setup.free > 0, 1.0 / jnp.where(jnp.abs(d) > 0, d, 1.0), 1.0)
                lmax = self._lmax(ops, setup, tp, dinv)
                x, k, rr = self._pcg_fn("cheby")(ops, b, setup.free, tp, dict(dinv=dinv, lmax=lmax), jvp_args)
                self.last_iters = int(k)
                return x
            x, k, rr = self._pcg_fn("amg", self.amg)(ops, b, setup.free, tp, self.amg.params(), jvp_args)
            k = int(k)
            if self.amg_iters0 is None:
                self.amg_iters0 = max(k, 5)
            elif k > 4 * self.amg_iters0 and not rebuild:
                return self.solve(ops, b, setup, jvp_args, rebuild=True)
        self.last_iters = int(k)
        if not np.isfinite(float(rr)) or float(rr) > 1e-3:
            self.mon.log(f"    warning: PCG not converged ({int(k)} it, rel.res {float(rr):.2e})")
        return x

    def reset(self):
        self.amg = None


# ------------------------------------------------------------------ static (Newton)
class StaticSolver:
    def __init__(self, model, rtol=5e-3, max_iter=16, linear_solver="auto", tangent="ebe",
                 cg_tol=1e-9, cg_maxiter=20000, verbose=True, monitor=None, line_search=True):
        self.m, self.rtol, self.max_iter = model, rtol, max_iter
        self.verbose = verbose
        self.mon = monitor or Monitor(verbose)
        self.line_search = line_search
        self.lin = _LinearSystem(model, linear_solver, tangent, cg_tol, cg_maxiter, self.mon)
        m = model
        inter = m.interactions
        X = m.X
        nd = m.ndof

        def fint_all(u, s_old, fields, cps):
            f, s_new = m.internal_force(u, s_old, fields)
            for c, p in zip(inter, cps):
                f = f + c.evaluate(m.X, u, p, nd)[0]
            return f, s_new

        def tangent_ops(u, s_old, fields, cps):
            ops = m.stiffness_ops(u, s_old, fields) if tangent == "ebe" else []
            for c, p in zip(inter, cps):
                ops.append(c.evaluate(m.X, u, p, nd)[1])
            return ops

        # model arrays are runtime arguments, not compile-time constants
        self._fint, self._ops = m.jit(fint_all), m.jit(tangent_ops)

    # ---- helpers
    def _cps(self, u, lam=0.0):
        X = self.m.X
        out = []
        for c in self.m.interactions:
            out.append(c.params(X, u, lam) if c.kind == "wall" else c.params(X, u))
        return out

    def _fields(self, T, scale=None):
        return dict(T=T, scale=scale)

    def _linsolve(self, u, s, fields, cps, b, setup):
        ops = self._ops(u, s, fields, cps)
        jv = (u, s, fields) if self.lin.tangent == "jvp" else None
        return self.lin.solve(ops, b, setup, jv)

    def solve(self, step, u0=None, states=None, fext0=None, callback=None, state=None,
              scale=None):
        """Returns (u, states, info). `state` (State) carries everything between steps."""
        m = self.m
        st = state or State(m)
        if u0 is not None:
            st.u = u0
        if states is not None:
            st.states = states
        if fext0 is not None:
            st.fext = fext0
        setup = _Setup(m, step, st, _ramp)
        self.setup = setup
        self.lin.reset()
        for c in m.interactions:
            c.nonsym = 1.0 if self.lin.method == "direct" else 0.0
        u, states = st.u, st.states
        t, dt, inc, history = 0.0, min(step.dt_init, step.T), 0, []
        t0 = time.time()
        total_iters = 0
        self.qref = 0.0
        while t < step.T * (1 - 1e-12):
            if self.mon.should_stop():
                self.mon.log("  stopped by user")
                break
            inc += 1
            if inc > step.max_inc:
                raise RuntimeError("max increments exceeded")
            dt = min(dt, step.T - t)
            tn = t + dt
            fields = self._fields(setup.temperature(tn), scale)
            fext = setup.fext(tn)
            ut = jnp.where(setup.fixed_j, setup.prescribed(tn), u)
            ut = setup.tr.apply(ut) if setup.tr.active else ut
            try:
                ut = self._predictor(u, states, ut, fext, tn, fields, setup)
                ok, it, ui, s_try, rh = self._newton(ut, states, fext, tn, fields, setup, inc)
            except FloatingPointError:
                ok, it, ui, s_try, rh = False, 0, u, states, []
            total_iters += it
            for _aug in range(4):                      # augmented-Lagrange passes
                if not ok:
                    break
                again = [c.augment() for c in m.interactions if hasattr(c, "augment")]
                if not any(again):
                    break
                ok, it2, ui, s_try, rh = self._newton(ui, states, fext, tn, fields, setup, inc)
                it += it2
            if ok:
                for c in m.interactions:
                    if hasattr(c, "commit"):
                        c.commit(m.X, ui)
                u, states, t = ui, s_try, tn
                history.append((t, it))
                self.mon.log(f"  inc {inc:4d}  t={t:.4g}  dt={dt:.3g}  iters={it}  lin={self.lin.last_iters}")
                self.mon.increment(step=step.name, inc=inc, t=t, dt=dt, iters=it, ok=True,
                                   time_total=st.time + t)
                if callback:
                    callback(t, u, states)
                if inc % step.output_every == 0 or t >= step.T * (1 - 1e-12):
                    self.mon.frame(st.time + t, u, states, step=step.name, fields=fields)
                if it <= 5:
                    dt = min(dt * 1.5, step.dt_max)
            else:
                dt *= 0.25
                self.mon.log(f"  inc {inc:4d}  cutback -> dt={dt:.3g}")
                self.mon.increment(step=step.name, inc=inc, t=t, dt=dt, iters=it, ok=False,
                                   time_total=st.time + t)
                if dt < step.dt_min:
                    raise RuntimeError("time increment smaller than minimum")
        self.mon.log(f"  step '{step.name}' done: {inc} incs, {total_iters} iterations, "
                     f"{time.time() - t0:.2f}s")
        fields = self._fields(setup.temperature(step.T), scale)
        fint, _ = self._fint(u, states, fields, self._cps(u, 1.0))
        reactions = (fint - setup.fext(step.T)) * setup.fixed_j
        fe = setup.fext_final()
        st.u, st.states, st.fext, st.T = u, states, fe, setup.temperature(step.T)
        st.followers = {key: fp.p for fp, _, key in setup.followers}
        st.time += step.T
        st.reactions = reactions
        return u, states, dict(history=history, reactions=reactions, fext=fe, state=st)

    def _predictor(self, uc, s_old, ut, fext, t, fields, setup):
        """Tangent predictor from the converged state: avoids the 'only the loaded nodes
        moved' shock that drives elements spuriously plastic."""
        delta = ut - uc
        cps = self._cps(uc, t / setup.step.T)
        fint, _ = self._fint(uc, s_old, fields, cps)
        ops = self._ops(uc, s_old, fields, cps)
        Kd = ops_matvec(ops, delta, self.m.ndof)
        if self.lin.tangent == "jvp":
            Kd = Kd + self.m.tangent_jvp(uc, s_old, delta, fields)
        r = fint + Kd - fext - setup.follower_force(uc, t)
        r = setup.free * setup.tr.apply_T(r)
        self.qref = max(self.qref, float(jnp.linalg.norm(r)))
        if float(jnp.linalg.norm(r)) == 0.0:
            return ut
        du = self._linsolve(uc, s_old, fields, cps, -r, setup)
        u = ut + setup.free * du
        return setup.tr.apply(u) if setup.tr.active else u

    def _residual(self, u, s_old, fext, t, fields, setup):
        cps = self._cps(u, t / setup.step.T)
        fint, s_new = self._fint(u, s_old, fields, cps)
        r = fint - fext - setup.follower_force(u, t)
        return setup.free * setup.tr.apply_T(r), fint, s_new, cps

    def _newton(self, u, s_old, fext, t, fields, setup, inc):
        rmask = getattr(setup, "rotdof", None)
        if rmask is None:
            rm = np.zeros(self.m.ndof, bool)
            rows = list(self.m.rot_of.values())
            for k in range(3):
                rm[np.asarray(rows, dtype=np.int64) * 3 + k] = True
            setup.rotdof = rmask = jnp.asarray(rm)
            setup.has_rot = bool(rm.any())
            self.qref_r = 0.0
        hist = []
        dut = dur = 0.0

        def norms(v):
            if not setup.has_rot:
                return float(jnp.linalg.norm(v)), 0.0
            return float(jnp.linalg.norm(jnp.where(rmask, 0.0, v))), float(jnp.linalg.norm(jnp.where(rmask, v, 0.0)))
        fn_t, fn_r = norms(fext)
        for it in range(1, self.max_iter + 1):
            r, fint, s_new, cps = self._residual(u, s_old, fext, t, fields, setup)
            rt, rr = norms(r)
            ft, fr = norms(fint)
            ref = max(fn_t, ft, 1e-3 * self.qref, 1e-30)
            self.qref = max(self.qref, fn_t, ft)
            self.qref_r = max(getattr(self, "qref_r", 0.0), fn_r, fr)
            if not hasattr(self, "_lc"):
                self._lc = self.m.char_length()
            ref_r = max(fn_r, fr, 1e-3 * self.qref_r, 1e-2 * ref * self._lc, 1e-30)
            ratio = max(rt / ref, rr / ref_r if setup.has_rot and self.qref_r > 0 else 0.0)
            hist.append(ratio)
            self.mon.iteration(inc=inc, it=it, residual=ratio, du=dut, t=t)
            if not np.isfinite(rt + rr):
                return False, it, u, s_old, hist
            ut_n, ur_n = norms(u)
            small_du = dut <= 1e-2 * max(ut_n, 1e-30) and (not setup.has_rot or dur <= 1e-2 * max(ur_n, 1e-30))
            if ratio <= self.rtol * 1e-2 or (it > 1 and ratio <= self.rtol and small_du):
                return True, it - 1, u, s_new, hist
            du = self._linsolve(u, s_old, fields, cps, -r, setup)
            alpha = 1.0
            if self.line_search:
                rn = float(jnp.linalg.norm(r))
                for _ in range(4):
                    ut = u + alpha * du
                    ut = setup.tr.apply(ut) if setup.tr.active else ut
                    r2 = self._residual(ut, s_old, fext, t, fields, setup)[0]
                    if float(jnp.linalg.norm(r2)) < rn or alpha < 0.2:
                        break
                    alpha *= 0.5
            u = u + alpha * du
            u = setup.tr.apply(u) if setup.tr.active else u
            dut, dur = norms(alpha * du)
        return False, self.max_iter, u, s_old, hist


# ------------------------------------------------------------------ Riks arc-length
class RiksSolver(StaticSolver):
    qref = 0.0
    """Crisfield cylindrical arc-length. The loads of the step are the reference load P;
    the load proportionality factor (LPF) multiplies them on top of the loads carried over
    from previous steps. Stops at LPF >= lpf_max, |u[monitor dof]| >= u_max, or max_inc."""

    def solve(self, step, u0=None, states=None, fext0=None, callback=None, state=None, scale=None):
        m = self.m
        st = state or State(m)
        if u0 is not None:
            st.u = u0
        if states is not None:
            st.states = states
        rk = dict(dlam0=0.1, lpf_max=1.0, u_max=None, dof=None, max_inc=step.max_inc, min_arc=1e-8)
        rk.update(step.riks)
        setup = _Setup(m, step, st, lambda t, T: 1.0)
        self.setup = setup
        self.lin.reset()
        f0 = st.fext
        P = sum(setup.groups.values()) if setup.groups else jnp.zeros(m.ndof)   # reference load
        P = setup.free * setup.tr.apply_T(P)
        fields = self._fields(setup.temperature(step.T), scale)
        u, states = st.u, st.states
        lam = 0.0
        cps = self._cps(u)
        dut = self._linsolve(u, states, fields, cps, P, setup)
        dl = rk["dlam0"] * float(jnp.linalg.norm(dut))
        du_prev = None
        curve = [(0.0, self._mon_value(u, rk))]
        t0 = time.time()
        inc = 0
        while inc < rk["max_inc"]:
            if self.mon.should_stop():
                break
            inc += 1
            cps = self._cps(u)
            dut = self._linsolve(u, states, fields, cps, P, setup)
            sgn = 1.0
            if du_prev is not None:
                sgn = 1.0 if float(jnp.vdot(du_prev, dut)) >= 0 else -1.0
            dlam = sgn * dl / max(float(jnp.linalg.norm(dut)), 1e-300)
            Du = dlam * dut
            ok = False
            for it in range(1, self.max_iter + 1):
                uc = u + Du
                cpi = self._cps(uc)
                fint, s_new = self._fint(uc, states, fields, cpi)
                R = setup.free * setup.tr.apply_T(fint - f0 - (lam + dlam) * P)
                rn = float(jnp.linalg.norm(R))
                ref = max(float(jnp.linalg.norm((lam + dlam) * P + f0)), float(jnp.linalg.norm(fint)), 1e-30)
                self.mon.iteration(inc=inc, it=it, residual=rn / ref, t=lam + dlam)
                if rn <= self.rtol * ref:
                    ok = True
                    break
                dur = self._linsolve(uc, states, fields, cpi, -R, setup)
                dut = self._linsolve(uc, states, fields, cpi, P, setup)
                a = float(jnp.vdot(dut, dut))
                w = Du + dur
                b = 2 * float(jnp.vdot(w, dut))
                c = float(jnp.vdot(w, w)) - dl ** 2
                disc = b * b - 4 * a * c
                if disc < 0:
                    break
                r1, r2 = (-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)
                c1 = float(jnp.vdot(Du, w + r1 * dut))
                c2 = float(jnp.vdot(Du, w + r2 * dut))
                dl_it = r1 if c1 >= c2 else r2
                Du = w + dl_it * dut
                dlam += dl_it
            if ok:
                u, states, lam = uc, s_new, lam + dlam
                du_prev = Du
                val = self._mon_value(u, rk)
                curve.append((lam, val))
                self.mon.log(f"  arc {inc:4d}  LPF={lam:.5g}  arc={dl:.3g}  iters={it}  u_mon={val:.5g}")
                self.mon.increment(step=step.name, inc=inc, t=lam, iters=it, ok=True, lpf=lam,
                                   u_mon=val, time_total=st.time + lam)
                self.mon.frame(st.time + abs(lam), u, states, step=step.name, lpf=lam, fields=fields)
                dl *= float(np.clip(np.sqrt(5.0 / max(it, 1)), 0.5, 2.0))
                if lam >= rk["lpf_max"] or (rk["u_max"] and abs(val) >= rk["u_max"]):
                    break
            else:
                dl *= 0.25
                self.mon.log(f"  arc {inc:4d}  cutback -> arc={dl:.3g}")
                if dl < rk["min_arc"]:
                    raise RuntimeError("arc length too small")
        self.mon.log(f"  Riks done: LPF={lam:.5g}, {inc} increments, {time.time() - t0:.2f}s")
        st.u, st.states, st.fext = u, states, f0 + lam * P
        st.time += step.T
        return u, states, dict(curve=np.array(curve), lpf=lam, state=st)

    @staticmethod
    def _mon_value(u, rk):
        if rk.get("dof") is None:
            return float(jnp.max(jnp.abs(u)))
        return float(u[rk["dof"]])


# ------------------------------------------------------------------ implicit dynamics
class DynamicSolver(StaticSolver):
    qref = 0.0
    """HHT-alpha implicit dynamics (alpha in [-1/3, 0]; 0 = trapezoidal Newmark) with lumped
    mass and optional Rayleigh mass damping `step.damping` (C = damping * M)."""

    def solve(self, step, u0=None, states=None, fext0=None, callback=None, state=None, scale=None):
        m = self.m
        st = state or State(m)
        setup = _Setup(m, step, st, _ramp if step.amplitude is None else (lambda t, T: step.amplitude(t)))
        self.setup = setup
        self.lin.reset()
        al = float(step.alpha)
        beta, gamma = (1 - al) ** 2 / 4, 0.5 - al
        M = m.lumped_mass()
        Mf = setup.tr.apply_T(M) if setup.tr.active else M        # approx. condensed mass
        c = step.damping
        u, v, a, states = st.u, st.v, st.a, st.states
        dt = step.dt or step.dt_init
        t, inc = 0.0, 0
        fields = self._fields(setup.temperature(0.0), scale)
        fint_n, _ = self._fint(u, states, fields, self._cps(u))
        fext_n = setup.fext(0.0)
        t0 = time.time()
        nd = m.ndof
        while t < step.T * (1 - 1e-12):
            if self.mon.should_stop():
                break
            inc += 1
            h = min(dt, step.T - t)
            tn = t + h
            fields = self._fields(setup.temperature(tn), scale)
            fext = setup.fext(tn)
            up = u + h * v + h * h * (0.5 - beta) * a
            vp = v + h * (1 - gamma) * a
            ut = jnp.where(setup.fixed_j, setup.prescribed(tn), up)
            ok = False
            for it in range(1, self.max_iter + 1):
                an = (ut - up) / (beta * h * h)
                vn = vp + gamma * h * an
                cps = self._cps(ut)
                fint, s_new = self._fint(ut, states, fields, cps)
                ff = setup.follower_force(ut, tn)
                r = Mf * an + c * Mf * vn + (1 + al) * (fint - fext - ff) - al * (fint_n - fext_n)
                r = setup.free * setup.tr.apply_T(r)
                rn = float(jnp.linalg.norm(r))
                ref = max(float(jnp.linalg.norm(fext)), float(jnp.linalg.norm(fint)),
                          float(jnp.linalg.norm(Mf * an)), 1e-30)
                self.mon.iteration(inc=inc, it=it, residual=rn / ref, t=tn)
                if rn <= self.rtol * ref * 1e-1 or (it > 1 and rn <= self.rtol * ref):
                    ok = True
                    break
                ops = [(o[0], o[1] * (1 + al)) for o in self._ops(ut, states, fields, cps)]
                mdiag = (1.0 / (beta * h * h) + c * gamma / (beta * h)) * Mf
                ops.append((jnp.arange(nd)[:, None], mdiag[:, None, None]))
                jv = (ut, states, fields) if self.lin.tangent == "jvp" else None
                du = self.lin.solve(ops, -r, setup, jv)
                ut = ut + du
                ut = setup.tr.apply(ut) if setup.tr.active else ut
            if not ok:
                dt *= 0.5
                inc -= 1
                self.mon.log(f"  dynamic: cutback dt={dt:.3g}")
                if dt < step.dt_min:
                    raise RuntimeError("time increment smaller than minimum")
                continue
            u, v, a, states, t = ut, vn, an, s_new, tn
            fint_n, fext_n = fint, fext + ff
            self.mon.increment(step=step.name, inc=inc, t=t, dt=h, iters=it, ok=True, time_total=st.time + t)
            if inc % step.output_every == 0 or t >= step.T * (1 - 1e-12):
                self.mon.frame(st.time + t, u, states, step=step.name, v=v, fields=fields)
            if callback:
                callback(t, u, states)
        self.mon.log(f"  dynamic done: {inc} increments, {time.time() - t0:.2f}s")
        st.u, st.v, st.a, st.states = u, v, a, states
        st.fext, st.T = setup.fext_final(), setup.temperature(step.T)
        st.time += step.T
        return u, states, dict(state=st)


# ------------------------------------------------------------------ explicit
class ExplicitSolver:
    def __init__(self, model, verbose=True, monitor=None):
        self.m, self.verbose = model, verbose
        self.mon = monitor or Monitor(verbose)

    def solve(self, step, u0=None, v0=None, states=None, callback=None, amplitude=None, state=None):
        """amplitude(t/T) multiplies loads and prescribed displacements (default: smooth step).
        Returns (u, v, states)."""
        m = self.m
        st = state or State(m)
        if u0 is not None:
            st.u = u0
        if v0 is not None:
            st.v = v0
        if states is not None:
            st.states = states
        if amplitude is not None:
            default = lambda t, T: amplitude(min(t / T, 1.0))
        elif step.amplitude is not None:
            default = lambda t, T: step.amplitude(t)
        else:
            default = smooth_step
        setup = _Setup(m, step, st, default)
        X = m.X
        nd = m.ndof
        M = m.lumped_mass()
        if m.rot_of:                                  # rotary inertia scaling (shells / beams)
            Mn = np.asarray(M).reshape(-1, 3).copy()
            nodes = np.array(list(m.rot_of.keys()))
            rows = np.array(list(m.rot_of.values()))
            lc = min(float(jnp.min(b.charlen(m.X[b.conn]))) for b in m.blocks if hasattr(b, "vis_conn"))
            Mn[rows] = np.maximum(Mn[rows], Mn[nodes] * lc * lc / 8.0)
            M = jnp.asarray(Mn.ravel())
        dt_stable = m.stable_dt()
        if step.mass_scaling:                         # target dt by fixed mass scaling
            fac = max(1.0, (step.mass_scaling / (0.8 * dt_stable)) ** 2)
            M = M * fac
            dt_stable *= np.sqrt(fac)
            self.mon.log(f"  mass scaling x{fac:.3g}")
        tr = setup.tr
        if tr.active:
            Mb = tr.apply_T(M)
        else:
            Mb = M
        Minv = jnp.where(Mb > 0, 1.0 / jnp.where(Mb > 0, Mb, 1.0), 0.0)
        dt = step.dt or 0.8 * dt_stable
        nsteps = int(np.ceil(step.T / dt))
        nfr = max(1, step.frames)
        per = int(np.ceil(nsteps / nfr))
        nsteps = per * nfr
        dt = step.T / nsteps
        c = step.damping
        u, v, s = st.u, st.v, st.states
        self.mon.log(f"  explicit: dt={dt:.3e} (stable {dt_stable:.3e}), {nsteps} steps, {nfr} frames")
        inter = m.interactions
        fixed = setup.fixed_j
        fol = setup.followers
        fol_k = [f for f, _, _ in fol]
        groups = list(setup.groups.items())
        T0 = setup.Tprev
        T1 = setup.Tnew
        tp = tr.params()

        bc_amp = jnp.asarray(setup.bc_amp)
        bc_tgt = jnp.asarray(setup.target)
        bc_u0 = jnp.asarray(setup.ustart)

        def one(carry, inp):
            u, v, s = carry
            t, a_def, amps, avals, cps, fs = inp
            # prescribed displacements evaluated on the device (no per-step host arrays)
            ampv = jnp.concatenate([avals, jnp.zeros(1)])[jnp.maximum(bc_amp, 0)]
            tgt = jnp.where(bc_amp < 0, bc_u0 + (bc_tgt - bc_u0) * a_def, bc_tgt * ampv)
            T = T0 + (T1 - T0) * a_def
            fint, sn = m.internal_force(u, s, dict(T=T))
            for ci, p in zip(inter, cps):
                fint = fint + ci.evaluate(m.X, u, p, nd)[0]
            fext = setup.fprev * (1 - a_def)
            for (k, g), a in zip(groups, amps):
                fext = fext + g * a
            for fp, sc in zip(fol_k, fs):
                fext = fext + fp.force(X, u, sc)
            ftot = t_apply_T(tp, fext - fint, nd) - c * Mb * v
            acc = ftot * Minv
            v = v + dt * acc
            un = u + dt * v
            vn = jnp.where(fixed, (tgt - u) / dt, v)
            un = jnp.where(fixed, tgt, un)
            un = t_apply(tp, un, nd) if tp is not None else un
            vn = t_apply(tp, vn, nd) if tp is not None else vn
            return (un, vn, sn), None

        def chunk(u, v, s, ts, a_def, amps, avals, cps, fs):
            def body(cc, x):
                t, ad, am, av_, f = x
                return one(cc, (t, ad, am, av_, cps, f))
            (u, v, s), _ = jax.lax.scan(body, (u, v, s), (ts, a_def, amps, avals, fs))
            return u, v, s

        chunk = m.jit(chunk)

        t0 = time.time()
        done = 0
        energy = []
        while done < nsteps:
            if self.mon.should_stop():
                break
            idx = np.arange(done, done + per)
            ts = (idx + 1) * dt
            a_def = np.array([default(t, step.T) for t in ts])
            av = [[a(t) for a in setup.amps] for t in ts]
            amps = np.array([[ad if k < 0 else avi[k] for k, _ in groups] for ad, avi in zip(a_def, av)]).reshape(per, len(groups))
            avals = np.array(av, float).reshape(per, len(setup.amps))
            fs = np.array([[ad if k < 0 else avi[k] for _, k, _ in fol] for ad, avi in zip(a_def, av)]).reshape(per, len(fol))
            cps = [ci.params(X, u, float(a_def[0])) if ci.kind == "wall" else ci.params(X, u) for ci in inter]
            u, v, s = chunk(u, v, s, jnp.asarray(ts), jnp.asarray(a_def), jnp.asarray(amps),
                            jnp.asarray(avals), cps, jnp.asarray(fs))
            done += per
            tcur = min(done, nsteps) * dt
            ke = 0.5 * float(jnp.sum(M * v * v))
            energy.append((tcur, ke))
            self.mon.increment(step=step.name, inc=done, t=tcur, dt=dt, iters=0, ok=True,
                               time_total=st.time + tcur, kinetic=ke)
            self.mon.frame(st.time + tcur, u, s, step=step.name, v=v, fields=dict(T=T0 + (T1 - T0) * a_def[-1]))
            if callback:
                callback(tcur, u, s)
        self.mon.log(f"  explicit done: {time.time() - t0:.2f}s")
        st.u, st.v, st.states = u, v, s
        st.fext, st.T = setup.fext_final(), T1
        st.time += step.T
        self.energy = np.array(energy)
        return u, v, s


# ------------------------------------------------------------------ eigen analyses
def _reduced(model, step, state, ops):
    """Assemble K restricted to independent dofs (fixed and MPC slaves removed)."""
    st = state or State(model)
    setup = _Setup(model, step, st, _ramp)
    A = ops_assemble(ops, model.ndof)
    A = setup.tr.project_matrix(A)
    keep = np.nonzero(setup.free_np > 0)[0]
    return A[keep][:, keep].tocsc(), keep, setup


def modal(model, step, state=None, monitor=None):
    """Natural frequencies (*FREQUENCY): K phi = w^2 M phi with lumped mass. Uses the tangent at
    the current state (pre-stress / spin softening included when `state` is given)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    mon = monitor or Monitor(False)
    st = state or State(model)
    t0 = time.time()
    method = getattr(step, "eigensolver", "auto") or "auto"
    if method == "auto":
        gpu = jax.default_backend() != "cpu"
        method = "lobpcg" if (model.ndof > (30000 if gpu else 400000)) else "lanczos"
    if method == "lobpcg":
        from .eigen import modal_lobpcg
        w2, modes = modal_lobpcg(model, step, st, mon)
        nm = len(w2)
    else:
        ops = model.stiffness_ops(st.u, st.states)
        K, keep, setup = _reduced(model, step, st, ops)
        M = np.asarray(model.lumped_mass())
        if setup.tr.active:
            M = (setup.tr.T.T @ sp.diags(M) @ setup.tr.T).diagonal()
        Mk = sp.diags(M[keep]).tocsc()
        nm = min(step.nmodes, K.shape[0] - 2)
        sigma = -abs(step.shift) if step.shift else -1e-6 * abs(K.diagonal()).max() / max(M.max(), 1e-300)
        mon.log(f"  modal: {K.shape[0]} free dofs, {nm} modes (shift-invert Lanczos)")
        w2, V = spla.eigsh(K, k=nm, M=Mk, sigma=sigma, which="LM")
        order = np.argsort(w2)
        w2, V = w2[order], V[:, order]
        w2 = np.maximum(w2, 0.0)
        modes = np.zeros((nm, model.ndof))
        modes[:, keep] = V.T
        if setup.tr.active:
            modes = np.asarray([setup.tr.T @ mm for mm in modes])
    freq = np.sqrt(w2) / (2 * np.pi)
    # participation factors and effective mass in x, y, z
    Mfull = np.asarray(model.lumped_mass())
    r = np.zeros((3, model.ndof))
    trans_rows = np.ones(model.nn, bool)
    trans_rows[list(model.rot_of.values())] = False
    for d in range(3):
        r[d, d::3] = trans_rows
    gm = np.einsum("mi,i,mi->m", modes, Mfull, modes)
    part = np.einsum("mi,i,di->md", modes, Mfull, r) / gm[:, None]
    meff = part ** 2 * gm[:, None]
    total = np.array([np.sum(Mfull[d::3] * trans_rows) for d in range(3)])
    for i, f in enumerate(freq):
        mon.log(f"    mode {i + 1:3d}: f = {f:.6g}  eff.mass ratio x/y/z = "
                + " ".join(f"{x:.3f}" for x in meff[i] / np.maximum(total, 1e-300)))
    mon.log(f"  modal done: {time.time() - t0:.2f}s")
    for i in range(nm):
        mon.frame(float(i + 1), jnp.asarray(modes[i] / max(np.abs(modes[i]).max(), 1e-300)),
                  st.states, step=step.name, mode=i + 1, freq=float(freq[i]))
    return dict(freq=freq, modes=modes, participation=part, effective_mass=meff, total_mass=total)


def buckling(model, step, state=None, monitor=None):
    """Linear (eigenvalue) buckling (*BUCKLE): (K0 + lambda KG(P)) phi = 0, where KG is the
    initial-stress stiffness of the linear response to the step's loads P."""
    import scipy.sparse.linalg as spla
    mon = monitor or Monitor(False)
    st = state or State(model)
    t0 = time.time()
    solver = StaticSolver(model, linear_solver=getattr(step, "linear_solver", "auto") or "auto", verbose=False)
    s1 = Step("base", bcs=[(b[0], b[1], 0.0) for b in step.bcs], loads=step.loads,
              pressures=[(p[0], p[1]) for p in step.pressures], dloads=step.dloads)
    u, states, _ = solver.solve(s1, state=State(model))
    stress = model.stresses(u, states)
    K0ops = model.stiffness_ops(jnp.zeros(model.ndof), model.init_state())
    KGops = model.geometric_ops(stress, u, states)
    method = getattr(step, "eigensolver", "auto") or "auto"
    if method == "auto":
        gpu = jax.default_backend() != "cpu"
        method = "lobpcg" if (model.ndof > (30000 if gpu else 400000)) else "lanczos"
    if method == "lobpcg":
        from .eigen import buckling_lobpcg
        setup = _Setup(model, step, st, _ramp)
        lam, modes = buckling_lobpcg(model, K0ops, KGops, setup, step.nmodes, mon)
        order = np.argsort(lam)
        lam, modes = lam[order], modes[order]
    else:
        K0, keep, setup = _reduced(model, step, st, K0ops)
        KG, _, _ = _reduced(model, step, st, KGops)
        nm = min(step.nmodes, K0.shape[0] - 2)
        lu = spla.splu(K0.tocsc())
        Kinv = spla.LinearOperator(K0.shape, matvec=lu.solve, dtype=float)
        mon.log(f"  buckling: {K0.shape[0]} free dofs, {nm} modes")
        mu, V = spla.eigsh(-KG, k=nm, M=K0, Minv=Kinv, which="LA")
        order = np.argsort(-mu)
        mu, V = mu[order], V[:, order]
        lam = np.where(mu > 0, 1.0 / np.where(mu > 0, mu, 1), np.inf)
        modes = np.zeros((nm, model.ndof))
        modes[:, keep] = V.T
        if setup.tr.active:
            modes = np.asarray([setup.tr.T @ mm for mm in modes])
    for i, l in enumerate(lam):
        mon.log(f"    mode {i + 1:3d}: load multiplier = {l:.6g}")
        mon.frame(float(i + 1), jnp.asarray(modes[i] / max(np.abs(modes[i]).max(), 1e-300)),
                  st.states, step=step.name, mode=i + 1, eigenvalue=float(l))
    mon.log(f"  buckling done: {time.time() - t0:.2f}s")
    return dict(load_factors=lam, modes=modes, base_u=u)


def apply_imperfection(model, modes, scales):
    """Geometric imperfection (Abaqus *IMPERFECTION): X += sum_i s_i phi_i, every mode normalised
    to a maximum translational amplitude of 1 (rotation rows untouched)."""
    X = np.array(model.X)
    nr = getattr(model, "n_real", model.nn)
    dX = np.zeros_like(X)
    for phi, s in zip(modes, scales):
        P = np.asarray(phi).reshape(-1, 3).copy()
        P[list(model.rot_of.values())] = 0.0
        P[nr:] = 0.0
        amp = np.abs(P[:nr]).max()
        dX += s * P / max(amp, 1e-300)
    for n, r in model.rot_of.items():
        dX[r] = dX[n]                      # rotation rows follow their node's coordinates
    model.X = jnp.asarray(X + dX)
    return dX


def postbuckling(model, step, modes=(1,), scale=None, monitor=None, riks=None, linear_solver="auto"):
    """Nonlinear (post-)buckling: eigenvalue buckling -> imperfection from the selected modes ->
    Riks arc-length along the equilibrium path. Returns dict(load_factors, curve, u)."""
    mon = monitor or Monitor(False)
    bk = buckling(model, Step(kind="buckle", bcs=step.bcs, loads=step.loads, pressures=step.pressures,
                              dloads=step.dloads, nmodes=max(modes), eigensolver=step.eigensolver), monitor=mon)
    if scale is None:
        scale = 1e-3 * model.char_length() * 10
    apply_imperfection(model, [bk["modes"][i - 1] for i in modes], [scale] * len(modes))
    mon.log(f"  imperfection: modes {list(modes)}, amplitude {scale:.4g}")
    lam1 = float(bk["load_factors"][0])
    rk = dict(dlam0=0.05 * lam1, lpf_max=1.5 * lam1, max_inc=60)
    rk.update(riks or {})
    st = Step("postbuckling", kind="riks", bcs=step.bcs, loads=step.loads, pressures=step.pressures,
              dloads=step.dloads, riks=rk)
    u, s, info = RiksSolver(model, monitor=mon, linear_solver=linear_solver).solve(st)
    return dict(load_factors=bk["load_factors"], curve=info["curve"], u=u, lpf=info["lpf"])


def harmonic(model, step, freqs, damping=0.02, nmodes=30, load=None, state=None, monitor=None):
    """Harmonic (frequency) response by mode superposition (ANSYS MSUP harmonic):
        q_j(w) = phi_j^T F / (w_j^2 - w^2 + 2 i zeta w_j w)
    All frequencies are evaluated in one vectorised device kernel. `load` is the force amplitude
    vector (default: the step's loads). Returns dict(freq, U (nf, ndof) complex, modes)."""
    mon = monitor or Monitor(False)
    st2 = Step(kind="frequency", bcs=step.bcs, nmodes=nmodes, eigensolver=step.eigensolver)
    res = modal(model, st2, state, mon)
    st = state or State(model)
    if load is None:
        setup = _Setup(model, step, st, _ramp)
        load = np.asarray(setup.fext(step.T))
    M = np.asarray(model.lumped_mass())
    Phi = jnp.asarray(res["modes"])                               # (nm, ndof)
    gm = jnp.einsum("mi,i,mi->m", Phi, M, Phi)
    Fm = (Phi @ jnp.asarray(load)) / gm
    wj = 2 * np.pi * jnp.asarray(res["freq"])
    w = 2 * np.pi * jnp.asarray(freqs, dtype=float)
    z = jnp.asarray(damping)
    H = 1.0 / (wj[None, :] ** 2 - w[:, None] ** 2 + 2j * z * wj[None, :] * w[:, None])   # (nf, nm)
    U = (H * Fm[None, :]) @ Phi.astype(H.dtype)
    mon.log(f"  harmonic: {len(freqs)} frequencies x {len(wj)} modes")
    return dict(freq=np.asarray(freqs), U=np.asarray(U), modal=res)
