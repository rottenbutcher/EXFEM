"""Connectors: remote points (rigid / deformable coupling) and bolt pretension.

* rigid_coupling      : kinematic (RBE2-like) coupling of a set of nodes to a 6-dof reference
                        point:  u_i = u_ref + theta_ref x (X_i - X_ref), eliminated as an MPC.
                        Remote displacement / rotation BCs and point loads act on the reference.
* remote_force_vector : deformable (RBE3-like) distribution of a force F acting at a point P
                        (plus an optional moment) over a surface - ANSYS "Remote Force".
* BoltPretension      : pretension of a bolt body by an axial eigenstrain that is iterated
                        (secant) until the bolt section force equals the target, then locked
                        for the following steps (ANSYS "Bolt Pretension: load / lock").
"""
import numpy as np
import jax.numpy as jnp

from .constraints import MPC
from . import loads as L


def rigid_coupling(model, nodes, point=None, name="remote"):
    X = np.asarray(model.X)
    nodes = np.unique(np.asarray(nodes, dtype=np.int64))
    P = X[nodes].mean(0) if point is None else np.asarray(point, float)
    ref, r = model.add_point(P)
    d = X[nodes] - P
    sl, ms, ws = [], [], []
    rr = [r * 3 + 0, r * 3 + 1, r * 3 + 2]
    for i, n in enumerate(nodes):
        dx, dy, dz = d[i]
        # theta x d = (ty dz - tz dy, tz dx - tx dz, tx dy - ty dx)
        coef = [[0.0, dz, -dy], [-dz, 0.0, dx], [dy, -dx, 0.0]]
        for k in range(3):
            sl.append(n * 3 + k)
            ms.append([ref * 3 + k] + rr)
            ws.append([1.0] + coef[k])
    model.add_constraint(MPC(sl, ms, ws, name))
    model.remotes = getattr(model, "remotes", {})
    model.remotes[name] = (ref, r, P)
    return ref, r


def remote_force_vector(X, surface, F, P, M=None):
    w, A, _ = surface.integrate(X)
    nodes = np.nonzero(w)[0]
    c = (w[nodes, None] * np.asarray(X)[nodes]).sum(0) / w[nodes].sum()
    F = np.asarray(F, float)
    Mt = np.cross(np.asarray(P, float) - c, F) + (np.zeros(3) if M is None else np.asarray(M, float))
    f = L.surface_force_vector(X, surface, F)
    if np.linalg.norm(Mt) > 0:
        f = f + L.surface_moment_vector(X, surface, Mt, center=c)
    return f


class BoltPretension:
    """bolt elements: list of (block index, local element ids); axis: bolt axis; force: target
    tensile force. `solve(run)` calls run() (one full step solve) repeatedly."""

    def __init__(self, model, elems, axis, force, name="bolt"):
        self.m, self.elems, self.F, self.name = model, elems, float(force), name
        a = np.asarray(axis, float)
        self.a = a / np.linalg.norm(a)
        X = np.asarray(model.X)
        nodes = np.unique(np.concatenate([np.asarray(model.blocks[bi].conn)[le].ravel() for bi, le in elems]))
        proj = X[nodes] @ self.a
        self.L = float(proj.max() - proj.min())
        vols = [np.asarray(model.blocks[bi].volume(model.X[model.blocks[bi].conn[le]])) for bi, le in elems]
        self.V = float(sum(v.sum() for v in vols))
        self.A = self.V / max(self.L, 1e-300)
        self.E = max(model.blocks[bi].material.modulus() for bi, _ in elems)
        self.eps = -self.F / (self.E * self.A)

    def apply(self):
        m = self.m
        eig = list(getattr(m, "eigenstrain", None) or [None] * len(m.blocks))
        aa = np.outer(self.a, self.a).ravel()
        for bi, le in self.elems:
            e = np.zeros((m.blocks[bi].ne, 9)) if eig[bi] is None else np.asarray(eig[bi]).copy()
            e[le] = self.eps * aa
            eig[bi] = jnp.asarray(e)
        m.eigenstrain = eig

    def axial_force(self, u, states):
        st = self.m.stresses(u, states)
        tot = 0.0
        for bi, le in self.elems:
            S = np.asarray(st[bi]["S"])[le]
            v = np.asarray(self.m.blocks[bi].volume(self.m.X[self.m.blocks[bi].conn[le]]))
            tot += float(np.einsum("eij,i,j,e->", S, self.a, self.a, v))
        return tot / max(self.L, 1e-300)


def solve_with_bolts(bolts, run, log=print, tol=1e-3, maxit=6):
    """Secant iteration on the bolt eigenstrains; `run()` -> (u, states, info)."""
    for b in bolts:
        b.apply()
    out = None
    for it in range(maxit):
        out = run()
        u, states = out[0], out[1]
        done = True
        for b in bolts:
            N = b.axial_force(u, states)
            log(f"  bolt {b.name}: pretension iteration {it + 1}: force {N:.6g} (target {b.F:.6g})")
            if abs(N - b.F) > tol * abs(b.F):
                done = False
                b.eps *= b.F / N if abs(N) > 1e-300 else 2.0
                b.apply()
        if done:
            return out
    return out
