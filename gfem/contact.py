"""Contact.

`PenaltyContact`: frictionless node-to-surface contact. The closest master facet of every slave
node is found on the host (k-d tree, updated every iteration); the gap, force and Gauss-Newton
tangent of each (slave, triangle) pair are evaluated as a batch of 4-node "contact elements" on
the device. The number of pairs equals the number of slave nodes and never changes, so nothing is
recompiled while the contact state evolves. Optional augmented-Lagrange updates remove the
penalty penetration.

`RigidWall`: analytical rigid plane with prescribed motion (indentation, crush, drop tests).
"""
import numpy as np
import jax
import jax.numpy as jnp

from .constraints import project_to_triangles


def _gap(xs, x0, x1, x2):
    e1, e2 = x1 - x0, x2 - x0
    n = jnp.cross(e1, e2)
    nh = n / jnp.sqrt(jnp.sum(n * n) + 1e-300)
    return jnp.dot(xs - x0, nh)


_dgap = jax.grad(lambda x: _gap(x[0], x[1], x[2], x[3]))


def _pair(x, k, lam, active):
    """x: (4,3) slave + triangle; returns force (4,3), GN tangent (12,12)."""
    g = _gap(x[0], x[1], x[2], x[3])
    dg = _dgap(x).reshape(-1)
    pn = jnp.maximum(lam - k * g, 0.0) * active            # contact pressure (force)
    on = (pn > 0).astype(x.dtype)
    f = -pn * dg
    K = k * on * jnp.outer(dg, dg)
    return f.reshape(4, 3), K


_pairs = jax.vmap(_pair)


def _bary(xs, x0, x1, x2):
    e1, e2, d = x1 - x0, x2 - x0, xs - x0
    d11, d12, d22 = e1 @ e1, e1 @ e2, e2 @ e2
    b1, b2 = d @ e1, d @ e2
    den = d11 * d22 - d12 * d12
    v = (d22 * b1 - d12 * b2) / den
    w = (d11 * b2 - d12 * b1) / den
    return jnp.array([1 - v - w, v, w])


def _fric_force(x, x0, k, kt, lam, active, t_old, mu):
    """Normal + Coulomb friction forces on (slave, tri) as residual contributions (4,3),
    and the tangential traction (3,) for the history update."""
    g = _gap(x[0], x[1], x[2], x[3])
    dg = _dgap(x).reshape(4, 3)
    pn = jnp.maximum(lam - k * g, 0.0) * active
    n = jnp.cross(x[2] - x[1], x[3] - x[1])
    n = n / jnp.sqrt(n @ n + 1e-300)
    P = jnp.eye(3) - jnp.outer(n, n)
    w = _bary(x[0], x[1], x[2], x[3])
    slip = (x[0] - x0[0]) - (w @ (x[1:] - x0[1:]))
    t_tr = P @ t_old - kt * (P @ slip)
    tn = jnp.sqrt(t_tr @ t_tr + 1e-300)
    lim = mu * pn
    t = jnp.where(tn > lim, t_tr * lim / tn, t_tr) * (pn > 0)
    f = -pn * dg
    f = f.at[0].add(-t)
    f = f.at[1:].add(w[:, None] * t[None, :])
    return f, t


def _fric_pair(x, x0, k, kt, lam, active, t_old, mu, nonsym):
    f, _ = _fric_force(x, x0, k, kt, lam, active, t_old, mu)
    K = jax.jacfwd(lambda y: _fric_force(y, x0, k, kt, lam, active, t_old, mu)[0])(x).reshape(12, 12)
    # exact (non-symmetric) tangent for direct solvers, symmetric part for CG / AMG
    return f, nonsym * K + (1 - nonsym) * 0.5 * (K + K.T)


_fric_pairs = jax.vmap(_fric_pair, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None))
_fric_t = jax.vmap(lambda *a: _fric_force(*a)[1], in_axes=(0, 0, 0, 0, 0, 0, 0, None))


class PenaltyContact:
    kind = "contact"

    def __init__(self, X, master_surface, slave, stiffness=None, fkn=1.0, E=None,
                 tol=None, name="contact", augmented=False, mu=0.0, ftkn=1.0):
        """`slave`: Surface (area-weighted penalty, recommended) or node array.
        Normal stiffness per slave node: k = fkn * E * A_node / h (ANSYS FKN convention)."""
        from .surfaces import Surface
        X = np.asarray(X)
        self.name = name
        self.tris = master_surface.triangles()
        mnodes = set(master_surface.nodes().tolist())
        snodes = slave.nodes() if isinstance(slave, Surface) else np.unique(slave)
        self.slave = np.array([n for n in snodes if n not in mnodes], dtype=np.int64)
        h = master_surface.char_length(X)
        self.h = h
        self.depth = tol if tol is not None else 0.5 * h     # max penetration to be captured
        E = 1.0 if E is None else E
        if stiffness is not None:
            k = np.full(len(self.slave), float(stiffness))
        elif isinstance(slave, Surface):
            w, _, _ = slave.integrate(X)
            k = fkn * E * np.abs(w[self.slave]) / h
        else:
            k = np.full(len(self.slave), fkn * E * h)
        self.k = k
        self.lam = np.zeros(len(self.slave))
        self.mu, self.ftkn = float(mu), float(ftkn)
        self.ncand = 8
        self.t_old = np.zeros((len(self.slave), 3))
        self.u_start = None
        self.augmented = augmented
        self.ns = len(self.slave)

    # ------------------------------------------------------------------ search
    def _search(self, x):
        """Host k-d tree: k candidate master triangles per slave (only when nodes moved)."""
        from scipy.spatial import cKDTree
        cen = x[self.tris].mean(1)
        k = min(self.ncand, len(self.tris))
        _, cand = cKDTree(cen).query(x[self.slave], k=k)
        cand = np.asarray(cand).reshape(len(self.slave), k)
        self._cand = jnp.asarray(self.tris[cand])                 # (ns, k, 3) node ids
        self._x_search = x.copy()
        self.searches = getattr(self, "searches", 0) + 1

    def params(self, X, u):
        X = np.asarray(X)
        x = X + np.asarray(u).reshape(-1, 3)
        moved = np.inf if getattr(self, "_x_search", None) is None else \
            float(np.abs(x[self.slave] - self._x_search[self.slave]).max())
        if moved > 0.25 * self.h:                                 # re-search only after real motion
            self._search(x)
        p = dict(slave=jnp.asarray(self.slave), cand=self._cand, lam=jnp.asarray(self.lam),
                 k=jnp.asarray(self.k), h=jnp.asarray(self.h), depth=jnp.asarray(self.depth))
        if self.mu > 0:
            if self.u_start is None:
                self.u_start = np.asarray(u).copy()
            p.update(x0all=jnp.asarray(X + self.u_start.reshape(-1, 3)), t_old=jnp.asarray(self.t_old),
                     mu=jnp.asarray(self.mu), kt=jnp.asarray(self.k * self.ftkn),
                     nonsym=jnp.asarray(float(getattr(self, "nonsym", 0.0))))
        self._last_p = p
        self._last_X, self._last_u = X, u
        return p

    @staticmethod
    def select(X, u, p):
        """Device: pick the closest of the candidate triangles, gap and active flag."""
        x = X + u.reshape(-1, 3)
        xs = x[p["slave"]]                                        # (ns, 3)
        T = x[p["cand"]]                                          # (ns, k, 3, 3)
        A, B, C = T[:, :, 0], T[:, :, 1], T[:, :, 2]
        ab, ac, ap = B - A, C - A, xs[:, None] - A
        d00, d01, d11 = (ab * ab).sum(-1), (ab * ac).sum(-1), (ac * ac).sum(-1)
        d20, d21 = (ap * ab).sum(-1), (ap * ac).sum(-1)
        den = jnp.maximum(d00 * d11 - d01 * d01, 1e-300)
        v = jnp.clip((d11 * d20 - d01 * d21) / den, 0, 1)
        w = jnp.clip((d00 * d21 - d01 * d20) / den, 0, 1)
        sw = v + w
        v, w = jnp.where(sw > 1, v / sw, v), jnp.where(sw > 1, w / sw, w)
        Q = A + v[..., None] * ab + w[..., None] * ac
        dist = jnp.linalg.norm(xs[:, None] - Q, axis=-1)
        j = jnp.argmin(dist, axis=1)
        ii = jnp.arange(xs.shape[0])
        tri = p["cand"][ii, j]                                    # (ns, 3)
        Ab, Bb, Cb, Qb = A[ii, j], B[ii, j], C[ii, j], Q[ii, j]
        n = jnp.cross(Bb - Ab, Cb - Ab)
        n = n / jnp.maximum(jnp.linalg.norm(n, axis=1, keepdims=True), 1e-300)
        g = ((xs - Ab) * n).sum(-1)
        lateral = jnp.linalg.norm((xs - Qb) - g[:, None] * n, axis=1)
        active = ((lateral < 0.05 * p["h"]) & (g > -p["depth"])).astype(x.dtype)
        nodes = jnp.concatenate([p["slave"][:, None], tri], 1)
        return nodes, active, g

    def _host_state(self):
        if not hasattr(self, "_jsel"):
            self._jsel = jax.jit(self.select)
        nodes, active, g = self._jsel(jnp.asarray(self._last_X), jnp.asarray(self._last_u), self._last_p)
        return nodes, np.asarray(active), np.asarray(g)

    @property
    def last(self):
        if getattr(self, "_last_p", None) is None:
            return None
        _, a, g = self._host_state()
        return dict(gap=g, active=a)

    def commit(self, X, u):
        """Accept the increment: store the tangential traction (stick/slip history)."""
        if self.mu > 0 and getattr(self, "_last_p", None) is not None:
            p = self.params(X, u)
            Xj, uj = jnp.asarray(X), jnp.asarray(u)
            nodes, active, _ = self.select(Xj, uj, p)
            x = (Xj + uj.reshape(-1, 3))[nodes]
            self.t_old = np.asarray(_fric_t(x, p["x0all"][nodes], p["k"], p["kt"], p["lam"], active, p["t_old"], p["mu"]))
            self.u_start = np.asarray(u).copy()

    @staticmethod
    def evaluate(X, u, p, ndof):
        nodes, active, _ = PenaltyContact.select(X, u, p)
        x = (X + u.reshape(-1, 3))[nodes]
        if "mu" in p:
            f, K = _fric_pairs(x, p["x0all"][nodes], p["k"], p["kt"], p["lam"], active, p["t_old"], p["mu"], p["nonsym"])
        else:
            f, K = _pairs(x, p["k"], p["lam"], active)
        fg = jnp.zeros((ndof // 3, 3)).at[nodes].add(f).reshape(-1)
        dofs = (nodes[:, :, None] * 3 + jnp.arange(3)).reshape(-1, 12)
        return fg, (dofs, K)

    def augment(self, pen_tol=0.03):
        """Augmented-Lagrange multiplier update. Returns True if another pass is needed
        (penetration above pen_tol * element size)."""
        if not self.augmented or self.last is None:
            return False
        g, a = self.last["gap"], self.last["active"]
        new = np.maximum(self.lam - self.k * g, 0.0) * a
        change = np.linalg.norm(new - self.lam) / max(np.linalg.norm(new), 1e-300)
        self.lam = new
        pen = float(np.max(np.where(a > 0, -g, 0.0), initial=0.0))
        return change > pen_tol and pen > 1e-8 * self.h

    def status(self):
        if self.last is None:
            return {}
        g, a = self.last["gap"], self.last["active"]
        pen = -(g * a)[a > 0].min() if (a > 0).any() else 0.0
        return dict(name=self.name, closed=int(((g < 0) & (a > 0)).sum()), max_penetration=float(max(pen, 0)))


class RigidWall:
    """Rigid plane through `point` with outward normal `normal`; slave nodes stay on the
    positive side. The plane moves by `motion` (vector) times the amplitude value."""
    kind = "wall"

    def __init__(self, nodes, point, normal, stiffness, motion=(0, 0, 0), name="wall"):
        self.slave = np.asarray(nodes, dtype=np.int64)
        n = np.asarray(normal, float)
        self.n = n / np.linalg.norm(n)
        self.p0 = np.asarray(point, float)
        self.motion = np.asarray(motion, float)
        self.k = float(stiffness)
        self.name = name
        self.ns = len(self.slave)
        self.reaction = 0.0

    def params(self, X, u, lam=0.0):
        return dict(nodes=jnp.asarray(self.slave), n=jnp.asarray(self.n),
                    p=jnp.asarray(self.p0 + lam * self.motion), k=jnp.asarray(self.k))

    @staticmethod
    def evaluate(X, u, p, ndof):
        x = (X + u.reshape(-1, 3))[p["nodes"]]
        g = (x - p["p"]) @ p["n"]
        pn = jnp.maximum(-p["k"] * g, 0.0)
        f = -pn[:, None] * p["n"][None]
        fg = jnp.zeros((ndof // 3, 3)).at[p["nodes"]].add(f).reshape(-1)
        on = (pn > 0).astype(x.dtype)
        K = p["k"] * on[:, None, None] * jnp.outer(p["n"], p["n"])[None]
        dofs = (p["nodes"][:, None] * 3 + jnp.arange(3)).reshape(-1, 3)
        return fg, (dofs, K)

    def status(self):
        return dict(name=self.name)
