"""Loads: amplitudes, dead load vectors (concentrated, surface pressure / traction / force /
moment, gravity, centrifugal) and the follower-pressure contributor (NLGEOM)."""
import numpy as np
import jax
import jax.numpy as jnp


class Amplitude:
    """Tabular amplitude a(t) (Abaqus *AMPLITUDE). t is step time unless total=True."""

    def __init__(self, table, name="amp", smooth=False):
        t = np.asarray(table, float).reshape(-1, 2)
        self.t, self.a, self.name, self.smooth = t[:, 0], t[:, 1], name, smooth

    def __call__(self, t):
        return float(np.interp(t, self.t, self.a))


def ramp(T):
    return Amplitude([[0, 0], [T, 1]], "RAMP")


def step_amp(T):
    return Amplitude([[0, 1], [T, 1]], "STEP")


def smooth_step(t, T):
    x = min(max(t / T, 0.0), 1.0)
    return x ** 3 * (10 - 15 * x + 6 * x ** 2)


# ------------------------------------------------------------------ dead load vectors (numpy)
def cload_vector(ndof, nodes, dof, value):
    f = np.zeros(ndof)
    np.add.at(f, np.asarray(nodes, dtype=np.int64) * 3 + dof, value)
    return f


def pressure_vector(X, surface, p):
    """Equivalent nodal forces of a uniform pressure p (positive into the surface) on the
    reference configuration: f_a = -p int N_a n dA."""
    X = np.asarray(X)
    f = np.zeros_like(X)
    for ft, fac in surface.groups:
        xe = X[fac]
        a1 = np.einsum("fai,ga->fgi", xe, ft.dN[:, :, 0])
        a2 = np.einsum("fai,ga->fgi", xe, ft.dN[:, :, 1])
        n = np.cross(a1, a2) * ft.gw[None, :, None]
        fe = -p * np.einsum("fgi,ga->fai", n, ft.N)
        np.add.at(f, fac, fe)
    return f.ravel()


def traction_vector(X, surface, t):
    """Uniform traction vector t (force/area) on a surface."""
    w, _, _ = surface.integrate(X)
    return (w[:, None] * np.asarray(t, float)[None]).ravel()


def surface_force_vector(X, surface, F):
    """Total force F distributed over a surface proportionally to area (ANSYS 'Force' on face)."""
    w, A, _ = surface.integrate(X)
    if A <= 0:
        return np.zeros(np.asarray(X).size)
    return (w[:, None] / A * np.asarray(F, float)[None]).ravel()


def surface_moment_vector(X, surface, M, center=None):
    """Moment distributed on a surface as a force couple (RBE3-like, area weighted):
    f_i = w_i (theta x r_i),  J theta = M,  J = sum w (|r|^2 I - r r^T)."""
    X = np.asarray(X)
    w, A, _ = surface.integrate(X)
    nodes = np.nonzero(w)[0]
    wi = w[nodes]
    c = (wi[:, None] * X[nodes]).sum(0) / wi.sum() if center is None else np.asarray(center)
    r = X[nodes] - c
    J = np.einsum("n,nk,nl->kl", wi, r, r)
    J = np.eye(3) * np.trace(J) - J
    theta = np.linalg.lstsq(J, np.asarray(M, float), rcond=None)[0]
    f = np.zeros_like(X)
    f[nodes] = wi[:, None] * np.cross(theta[None], r)
    return f.ravel()


def body_force_vector(model, density_times_g, blocks=None):
    """Gravity-like uniform body force per unit mass (vector g): f = int rho g N dV."""
    f = jnp.zeros((model.nn, 3))
    sel = range(len(model.blocks)) if blocks is None else blocks
    g = jnp.asarray(density_times_g, float)
    for i in sel:
        b = model.blocks[i]
        m = b.mass(model.X[b.conn])                     # lumped nodal mass (ne, nn)
        f = f.at[b.conn].add(m[:, :, None] * g[None, None, :])
    return np.asarray(f).ravel()


def centrifugal_vector(model, omega, axis_point, axis_dir, blocks=None):
    """Centrifugal load rho * omega^2 * r (dead, reference configuration, lumped)."""
    X = np.asarray(model.X)
    a = np.asarray(axis_dir, float)
    a = a / np.linalg.norm(a)
    d = X - np.asarray(axis_point, float)
    r = d - np.outer(d @ a, a)
    m = np.asarray(model.lumped_mass()).reshape(-1, 3)[:, 0] if blocks is None else None
    if m is None:
        mm = jnp.zeros(model.nn)
        for i in blocks:
            b = model.blocks[i]
            mm = mm.at[b.conn].add(b.mass(model.X[b.conn]))
        m = np.asarray(mm)
    return (m[:, None] * omega ** 2 * r).ravel()


# ------------------------------------------------------------------ follower pressure (NLGEOM)
class FollowerPressure:
    """Pressure acting on the deformed surface (contributes to the residual each iteration).
    The (non-symmetric) load stiffness is omitted from the tangent, which keeps the linear
    systems symmetric; Newton then converges linearly in this term, robustly for moderate
    rotations."""

    def __init__(self, surface, p, amplitude=None):
        self.surface, self.p, self.amp = surface, float(p), amplitude
        self.groups = [(ft, jnp.asarray(f)) for ft, f in surface.groups]
        self.kernels = []
        for ft, _ in self.groups:
            N, dN, gw = jnp.asarray(ft.N), jnp.asarray(ft.dN), jnp.asarray(ft.gw)

            def fe(xe, N=N, dN=dN, gw=gw):
                a1 = jnp.einsum("ai,ga->gi", xe, dN[:, :, 0])
                a2 = jnp.einsum("ai,ga->gi", xe, dN[:, :, 1])
                n = jnp.cross(a1, a2) * gw[:, None]
                return -jnp.einsum("gi,ga->ai", n, N)
            self.kernels.append(jax.vmap(fe))

    def force(self, X, u, scale):
        U = u.reshape(-1, 3)
        x = X + U
        f = jnp.zeros_like(U)
        for (ft, fac), k in zip(self.groups, self.kernels):
            f = f.at[fac].add(k(x[fac]) * self.p * scale)
        return f.reshape(-1)
