"""Discrete elements: springs, bushings, grounded springs and point masses.

They follow the element-block interface (force / stiff / mass / cauchy ...) so every solver,
the EbE/gather operators, AMG and the eigensolvers use them without special cases.

* SpringBlock (axial)    : 2-node spring along the current line of the nodes (large rotation),
                           energy k/2 (L - L0)^2              (Abaqus SPRINGA, Nastran CROD-like)
* SpringBlock (bushing)  : 2-node spring with stiffnesses (kx, ky, kz) in global axes
                           (Abaqus SPRING2 / CONN3D2 cartesian, Nastran CBUSH, ANSYS COMBIN14)
* GroundSpring           : node to ground, (kx, ky, kz)        (Nastran CELAS1 to ground)
* point masses           : Model.add_point_mass (Abaqus *MASS, Nastran CONM2)
"""
import numpy as np
import jax
import jax.numpy as jnp

from .structural import StructType

SPRING = StructType("SPRING2", 2, 2, 3, "spring")
GSPRING = StructType("SPRING1", 1, 1, 1, "spring")
I3 = jnp.eye(3)


class _Prop:
    """Minimal 'material' for discrete elements."""
    small_strain = True
    density = 0.0
    alpha = 0.0
    nstate = 1

    def __init__(self, k, name):
        self.k, self.name = k, name
        self.E, self.nu = float(np.max(k)), 0.0
        self.mu, self.lam = self.E / 2, 0.0

    def modulus(self):
        return self.E

    def wave_speed(self):
        return 1e30

    def describe(self):
        return f"spring(k={self.k})"


class SpringBlock:
    def __init__(self, conn, k, kind="axial", name="spring"):
        conn = np.asarray(conn, dtype=np.int64).reshape(-1, 2)
        self.conn = jnp.asarray(conn.astype(np.int32))
        self.vis_conn = self.conn
        self.ne = len(conn)
        self.kind = kind
        k = np.broadcast_to(np.asarray(k, float), (3,)) if kind == "bushing" else float(k)
        self.material = _Prop(k, name)
        self.etype, self.name = SPRING, name
        kk = jnp.asarray(k)

        def energy(Xe, ue):
            if kind == "axial":
                L0 = jnp.linalg.norm(Xe[1] - Xe[0])
                L = jnp.linalg.norm(Xe[1] + ue[1] - Xe[0] - ue[0])
                return 0.5 * kk * (L - L0) ** 2
            d = ue[1] - ue[0]
            return 0.5 * jnp.sum(kk * d * d)

        grad = jax.grad(energy, argnums=1)

        def force(Xe, ue, se, Te, sc):
            s0 = sc[0] if jnp.ndim(sc) else sc
            return s0 * grad(Xe, ue), se

        def stiff(Xe, ue, se, Te, sc):
            return jax.jacfwd(lambda u: force(Xe, u, se, Te, sc)[0])(ue).reshape(6, 6)

        def cauchy(Xe, ue, se, Te, scv=None):
            if kind == "axial":
                e = Xe[1] - Xe[0]
                L0 = jnp.linalg.norm(e)
                N = kk * (jnp.linalg.norm(Xe[1] + ue[1] - Xe[0] - ue[0]) - L0)
                t = e / L0
                return N * jnp.outer(t, t), 0.0, I3
            return jnp.diag(kk * (ue[1] - ue[0])), 0.0, I3

        self.force = jax.vmap(force)
        self.stiff = jax.vmap(stiff)
        self.mass = jax.vmap(lambda Xe: jnp.zeros(2))
        self.volume = jax.vmap(lambda Xe: 0.0)
        self.cauchy = jax.vmap(cauchy)
        self.charlen = jax.vmap(lambda Xe: jnp.asarray(1e30))

    def init_state(self):
        return jnp.zeros((self.ne, 1, 1))

    def dofs(self):
        return (self.conn[:, :, None] * 3 + jnp.arange(3)).reshape(self.ne, -1)


class GroundSpring(SpringBlock):
    def __init__(self, nodes, k, name="ground spring"):
        nodes = np.asarray(nodes, dtype=np.int64).reshape(-1, 1)
        self.conn = jnp.asarray(nodes.astype(np.int32))
        self.vis_conn = self.conn
        self.ne = len(nodes)
        self.kind = "ground"
        self.material = _Prop(np.broadcast_to(np.asarray(k, float), (3,)), name)
        self.etype, self.name = GSPRING, name
        kk = jnp.asarray(self.material.k)

        def force(Xe, ue, se, Te, sc):
            s0 = sc[0] if jnp.ndim(sc) else sc
            return s0 * kk[None] * ue, se

        def stiff(Xe, ue, se, Te, sc):
            s0 = sc[0] if jnp.ndim(sc) else sc
            return s0 * jnp.diag(kk)

        self.force = jax.vmap(force)
        self.stiff = jax.vmap(stiff)
        self.mass = jax.vmap(lambda Xe: jnp.zeros(1))
        self.volume = jax.vmap(lambda Xe: 0.0)
        self.cauchy = jax.vmap(lambda Xe, ue, se, Te, scv=None: (jnp.diag(kk * ue[0]), 0.0, I3))
        self.charlen = jax.vmap(lambda Xe: jnp.asarray(1e30))
