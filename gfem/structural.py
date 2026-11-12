"""Structural elements with rotational degrees of freedom: shells and beams.

Rotations live on *rotation nodes*: every shell/beam node n gets a companion row r(n) in the
model's node array, whose three "displacement" components are the total rotation vector of n.
All existing machinery (EbE operators, MPC, BCs, AMG, explicit, eigen solvers) therefore works
unchanged on a 3-dof-per-row layout; solids simply never reference rotation rows.

ShellBlock (S4/S4R/S3/S3R): geometrically exact degenerated-continuum shell.
  * director d_a = R(theta_a) D_a with the Rodrigues formula (finite rotations, any size)
  * Green-Lagrange strain, plane-stress St.Venant-Kirchhoff, 2x2 in-plane x 2 through-thickness
  * MITC4 assumed transverse shear (no shear locking); S3 = collapsed MITC4
  * nodal directors averaged over smoothly connected elements (folds keep element normals)
  * small drilling penalty
BeamBlock (B31/B33): geometrically exact (Simo-Reissner) 2-node Timoshenko beam with
  one-point (mid-point) integration: shear-locking free, large rotations.

Element force and tangent come from automatic differentiation of the element energy, so the
same kernels give the nonlinear tangent, the geometric stiffness for buckling and the mass.
"""
import numpy as np
import jax
import jax.numpy as jnp

I3 = jnp.eye(3)
KD_ON = 1.0
KD_FAC = 1e-3
ANS_MEMBRANE = False


def skew(v):
    return jnp.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rodrigues(th):
    """Rotation matrix of a rotation vector, AD-safe at zero."""
    t2 = jnp.dot(th, th)
    small = t2 < 1e-10
    t2s = jnp.where(small, 1.0, t2)
    t = jnp.sqrt(t2s)
    a = jnp.where(small, 1.0 - t2 / 6.0 + t2 * t2 / 120.0, jnp.sin(t) / t)
    b = jnp.where(small, 0.5 - t2 / 24.0 + t2 * t2 / 720.0, (1.0 - jnp.cos(t)) / t2s)
    K = skew(th)
    return I3 + a * K + b * (K @ K)


def elastic_constants(mat):
    if hasattr(mat, "E") and hasattr(mat, "nu") and not hasattr(mat, "C10"):
        return float(mat.E), float(mat.nu)
    mu, lam = float(mat.mu), float(mat.lam)
    return mu * (3 * lam + 2 * mu) / (lam + mu), lam / (2 * (lam + mu))


class StructType:
    """Element-type descriptor compatible with the solid ElementType interface."""

    def __init__(self, name, nnode, ncorner, vtk_id, kind):
        self.name, self.nnode, self.ncorner, self.vtk_id, self.kind = name, nnode, ncorner, vtk_id, kind
        self.faces, self.edges, self.order, self.fbar = [], [], 1, False
        self.ngp = 1


S4 = StructType("S4", 4, 4, 9, "shell")
S3 = StructType("S3", 3, 3, 5, "shell")
B31 = StructType("B31", 2, 2, 3, "beam")
STRUCT_ELEMENTS = {"S4": S4, "S4R": S4, "S4R5": S4, "S3": S3, "S3R": S3, "B31": B31, "B33": B31,
                   "B31H": B31, "B21": B31}


# ------------------------------------------------------------------ shell
_QN = jnp.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)


def _qN(x, y):
    return 0.25 * (1 + _QN[:, 0] * x) * (1 + _QN[:, 1] * y)


def _qdN(x, y):
    return 0.25 * jnp.stack([_QN[:, 0] * (1 + _QN[:, 1] * y), _QN[:, 1] * (1 + _QN[:, 0] * x)], 1)


_g = 1.0 / np.sqrt(3.0)
_GP = [(-_g, -_g), (_g, -_g), (_g, _g), (-_g, _g)]


def shell_directors(X, conn, angle=20.0):
    """Per element-node directors: average of the normals of elements meeting at the node whose
    normal is within `angle` of the element's own normal (folds/T-joints stay sharp)."""
    X = np.asarray(X)
    c = np.asarray(conn)
    q = c if c.shape[1] == 4 else np.hstack([c, c[:, 2:3]])
    n = np.cross(X[q[:, 2]] - X[q[:, 0]], X[q[:, 3]] - X[q[:, 1]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    acc = {}
    for e, row in enumerate(c):
        for a in row:
            acc.setdefault(int(a), []).append(e)
    cos = np.cos(np.radians(angle))
    D = np.zeros((len(c), 4, 3))
    for e, row in enumerate(q):
        for k, a in enumerate(row):
            els = acc[int(a)]
            nn = n[els]
            ok = nn @ n[e] > cos
            v = nn[ok].sum(0)
            D[e, k] = v / np.linalg.norm(v)
    return D


class ShellBlock:
    def __init__(self, conn, rconn, X, etype, material, thickness, name="shell", offset=0.0):
        conn = np.asarray(conn, dtype=np.int64)
        self.vis_conn = jnp.asarray(conn.astype(np.int32))
        self.tri = conn.shape[1] == 3
        q = conn if not self.tri else np.hstack([conn, conn[:, 2:3]])
        rq = np.asarray(rconn) if not self.tri else np.hstack([rconn, np.asarray(rconn)[:, 2:3]])
        self.conn = jnp.asarray(np.hstack([q, rq]).astype(np.int32))      # 4 nodes + 4 rotation nodes
        self.etype, self.material, self.name = etype, material, name
        self.h = float(thickness)
        self.ne = len(conn)
        self.directors = jnp.asarray(shell_directors(X, conn))
        self.E, self.nu = elastic_constants(material)
        self._build()

    def init_state(self):
        return self.directors                                          # constant per-element data

    def dofs(self):
        return (self.conn[:, :, None] * 3 + jnp.arange(3)).reshape(self.ne, -1)

    def _build(self):
        h, E, nu = self.h, self.E, self.nu
        G = E / (2 * (1 + nu))
        c1 = E / (1 - nu * nu)
        ks = 5.0 / 6.0
        rho = self.material.density
        alpha = self.material.alpha
        tri = self.tri

        def frame(xn, dn, xi, eta, zeta):
            N, dN = _qN(xi, eta), _qdN(xi, eta)
            g1 = dN[:, 0] @ xn + 0.5 * zeta * h * (dN[:, 0] @ dn)
            g2 = dN[:, 1] @ xn + 0.5 * zeta * h * (dN[:, 1] @ dn)
            g3 = 0.5 * h * (N @ dn)
            return g1, g2, g3

        def cov(Xn, D, xn, dn, xi, eta, zeta):
            G1, G2, G3 = frame(Xn, D, xi, eta, zeta)
            g1, g2, g3 = frame(xn, dn, xi, eta, zeta)
            Gm = jnp.stack([G1, G2, G3])
            gm = jnp.stack([g1, g2, g3])
            return 0.5 * (gm @ gm.T - Gm @ Gm.T), Gm

        def tying(Xn, D, xn, dn, zeta):
            return (cov(Xn, D, xn, dn, 0.0, -1.0, zeta)[0][0, 2], cov(Xn, D, xn, dn, 0.0, 1.0, zeta)[0][0, 2],
                    cov(Xn, D, xn, dn, -1.0, 0.0, zeta)[0][1, 2], cov(Xn, D, xn, dn, 1.0, 0.0, zeta)[0][1, 2])

        def strains(Xn, D, xn, dn, xi, eta, zeta, tie=None):
            E_c, Gm = cov(Xn, D, xn, dn, xi, eta, zeta)
            # MITC4 transverse shear tying (covariant 13 at (0,+-1), 23 at (+-1,0))
            eA, eC, eB, eD = tie if tie is not None else tying(Xn, D, xn, dn, zeta)
            e13 = 0.5 * (1 - eta) * eA + 0.5 * (1 + eta) * eC
            e23 = 0.5 * (1 - xi) * eB + 0.5 * (1 + xi) * eD
            E_c = E_c.at[0, 2].set(e13).at[2, 0].set(e13).at[1, 2].set(e23).at[2, 1].set(e23)
            if ANS_MEMBRANE:     # assumed membrane strains (removes membrane locking of curved shells)
                cA = cov(Xn, D, xn, dn, 0.0, -1.0, zeta)[0]
                cC = cov(Xn, D, xn, dn, 0.0, 1.0, zeta)[0]
                cB = cov(Xn, D, xn, dn, -1.0, 0.0, zeta)[0]
                cD = cov(Xn, D, xn, dn, 1.0, 0.0, zeta)[0]
                c0 = cov(Xn, D, xn, dn, 0.0, 0.0, zeta)[0]
                e11 = 0.5 * (1 - eta) * cA[0, 0] + 0.5 * (1 + eta) * cC[0, 0]
                e22 = 0.5 * (1 - xi) * cB[1, 1] + 0.5 * (1 + xi) * cD[1, 1]
                E_c = E_c.at[0, 0].set(e11).at[1, 1].set(e22).at[0, 1].set(c0[0, 1]).at[1, 0].set(c0[0, 1])
            Gcon = jnp.linalg.inv(Gm).T                                     # rows: contravariant G^i
            e3 = Gm[2] / jnp.linalg.norm(Gm[2])
            e1 = Gm[0] - (Gm[0] @ e3) * e3
            e1 = e1 / jnp.linalg.norm(e1)
            e2 = jnp.cross(e3, e1)
            Tf = jnp.stack([e1, e2, e3])                                    # local orthonormal frame
            T = Tf @ Gcon.T                                                 # T[k,i] = e_k . G^i
            El = T @ E_c @ T.T
            return El, jnp.abs(jnp.linalg.det(Gm)), Tf

        def stress_local(El, dT):
            e11, e22, e12 = El[0, 0] - alpha * dT, El[1, 1] - alpha * dT, El[0, 1]
            s11 = c1 * (e11 + nu * e22)
            s22 = c1 * (e22 + nu * e11)
            s12 = 2 * G * e12
            s13 = 2 * ks * G * El[0, 2]
            s23 = 2 * ks * G * El[1, 2]
            return jnp.array([[s11, s12, s13], [s12, s22, s23], [s13, s23, 0.0]])

        zgp = [(-_g, 1.0), (_g, 1.0)]

        def energy(Xe, ue, D, Te):
            Xn, xn = Xe[:4], Xe[:4] + ue[:4]
            th = ue[4:]
            dn = jax.vmap(lambda t, d0: rodrigues(t) @ d0)(th, D)
            dT = jnp.mean(Te[:4, 0] - Te[:4, 1])
            W = 0.0
            ties = [tying(Xn, D, xn, dn, z) for z, _ in zgp]
            for (xi, eta) in _GP:
                for (z, wz), tie in zip(zgp, ties):
                    El, dV, _ = strains(Xn, D, xn, dn, xi, eta, z, tie)
                    S = stress_local(El, dT)
                    W = W + 0.5 * jnp.sum(S * El) * dV * wz
            # drilling penalty (rotation about the director)
            area = jnp.linalg.norm(jnp.cross(Xn[2] - Xn[0], Xn[3] - Xn[1])) * 0.5
            kd = KD_FAC * G * h ** 3
            W = W + KD_ON * 0.5 * kd * jnp.sum(jnp.einsum("ai,ai->a", th, D) ** 2)
            return W

        self._energy = energy
        grad = jax.grad(energy, argnums=1)
        linear = bool(getattr(self.material, "small_strain", False))
        self.linear = linear

        def force_nl(Xe, ue, se, Te, sc):
            return sc * grad(Xe, ue, se, Te)

        def stiff_nl(Xe, ue, se, Te, sc):
            return jax.jacfwd(lambda u: force_nl(Xe, u, se, Te, sc))(ue).reshape(24, 24)

        def force(Xe, ue, se, Te, sc):
            if not linear:
                return force_nl(Xe, ue, se, Te, sc), se
            z = jnp.zeros_like(ue)                # exact linearisation (small displacement analysis)
            K0 = stiff_nl(Xe, z, se, Te, sc)
            return force_nl(Xe, z, se, Te, sc) + (K0 @ ue.reshape(-1)).reshape(ue.shape), se

        def stiff(Xe, ue, se, Te, sc):
            return stiff_nl(Xe, jnp.zeros_like(ue) if linear else ue, se, Te, sc)

        def mass(Xe):
            Xn = Xe[:4]
            area = 0.5 * jnp.linalg.norm(jnp.cross(Xn[2] - Xn[0], Xn[3] - Xn[1]))
            nreal = 3.0 if tri else 4.0
            m = rho * h * area / nreal
            mt = jnp.where(jnp.arange(4) < nreal, m, 0.0)
            mr = mt * h * h / 12.0
            return jnp.concatenate([mt, mr])

        def volume(Xe):
            Xn = Xe[:4]
            return 0.5 * jnp.linalg.norm(jnp.cross(Xn[2] - Xn[0], Xn[3] - Xn[1])) * h

        def cauchy(Xe, ue, se, Te):
            Xn, xn = Xe[:4], Xe[:4] + ue[:4]
            dn = jax.vmap(lambda t, d0: rodrigues(t) @ d0)(ue[4:], se)
            dT = jnp.mean(Te[:4, 0] - Te[:4, 1])
            out = []
            for z in (-1.0, 1.0):
                El, _, Tf = strains(Xn, se, xn, dn, 0.0, 0.0, z)
                S = stress_local(El, dT)
                S = S.at[0, 2].set(0.0).at[2, 0].set(0.0).at[1, 2].set(0.0).at[2, 1].set(0.0)
                out.append(Tf.T @ S @ Tf)
            mis = [jnp.sqrt(jnp.maximum(s[0, 0] ** 2 + s[1, 1] ** 2 + s[2, 2] ** 2 - s[0, 0] * s[1, 1]
                                        - s[1, 1] * s[2, 2] - s[0, 0] * s[2, 2] + 3 * (s[0, 1] ** 2 + s[1, 2] ** 2 + s[0, 2] ** 2), 0)) for s in out]
            S = jnp.where(mis[1] >= mis[0], out[1], out[0])
            return S, 0.0, I3

        def charlen(Xe):
            Xn = Xe[:4]
            d = jnp.linalg.norm(Xn[:, None] - Xn[None], axis=-1) + jnp.eye(4) * 1e30
            return jnp.minimum(d.min(), 1e30)

        self.force = jax.vmap(force)
        self.stiff = jax.vmap(stiff)
        self.mass = jax.vmap(mass)
        self.volume = jax.vmap(volume)
        self.cauchy = jax.vmap(cauchy)
        self.charlen = jax.vmap(charlen)
        self.geo_dir = jax.vmap(lambda Xe, ue0, se, Te: jax.jvp(
            lambda u: stiff_nl(Xe, u, se, Te, 1.0), (jnp.zeros_like(ue0),), (ue0,))[1])


# ------------------------------------------------------------------ beam
def beam_section(profile, dims):
    """Return dict(A, I2, I3, J, k, c2, c3) for RECT(b,h) CIRC(r) PIPE(r,t) BOX(b,h,t)
    I(h,b,tf,tw) GENERAL(A,I2,I3,J). Local axis 2 lies along `b`, axis 3 along `h`."""
    p = profile.upper()
    d = [float(x) for x in dims]
    if p == "RECT":
        b, h = d[:2]
        a, c = max(b, h), min(b, h)
        J = a * c ** 3 * (1 / 3 - 0.21 * c / a * (1 - c ** 4 / (12 * a ** 4)))
        return dict(A=b * h, I2=b * h ** 3 / 12, I3=h * b ** 3 / 12, J=J, k=5 / 6, c2=b / 2, c3=h / 2)
    if p == "CIRC":
        r = d[0]
        return dict(A=np.pi * r * r, I2=np.pi * r ** 4 / 4, I3=np.pi * r ** 4 / 4, J=np.pi * r ** 4 / 2,
                    k=0.9, c2=r, c3=r)
    if p == "PIPE":
        r, t = d[:2]
        ri = r - t
        I = np.pi * (r ** 4 - ri ** 4) / 4
        return dict(A=np.pi * (r * r - ri * ri), I2=I, I3=I, J=2 * I, k=0.5, c2=r, c3=r)
    if p == "BOX":
        b, h, t = d[:3]
        bi, hi = b - 2 * t, h - 2 * t
        J = 2 * t * t * (b - t) ** 2 * (h - t) ** 2 / (b * t + h * t - 2 * t * t)
        return dict(A=b * h - bi * hi, I2=(b * h ** 3 - bi * hi ** 3) / 12, I3=(h * b ** 3 - hi * bi ** 3) / 12,
                    J=J, k=0.44, c2=b / 2, c3=h / 2)
    if p == "I":
        h, b, tf, tw = d[:4]
        hw = h - 2 * tf
        A = 2 * b * tf + hw * tw
        I2 = (b * h ** 3 - (b - tw) * hw ** 3) / 12
        I3 = (2 * tf * b ** 3 + hw * tw ** 3) / 12
        J = (2 * b * tf ** 3 + hw * tw ** 3) / 3
        return dict(A=A, I2=I2, I3=I3, J=J, k=hw * tw / A, c2=b / 2, c3=h / 2)
    if p == "GENERAL":
        A, I2, I3, J = d[:4]
        return dict(A=A, I2=I2, I3=I3, J=J, k=5 / 6, c2=np.sqrt(I3 / A) * 1.7, c3=np.sqrt(I2 / A) * 1.7)
    raise ValueError(f"unknown beam profile {profile}")


def default_n1(X, conn):
    X = np.asarray(X)
    c = np.asarray(conn)
    t = X[c[:, 1]] - X[c[:, 0]]
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    ref = np.where(np.abs(t[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[0, 1.0, 0]]))
    n1 = ref - np.einsum("ij,ij->i", ref, t)[:, None] * t
    return n1 / np.linalg.norm(n1, axis=1, keepdims=True)


class BeamBlock:
    def __init__(self, conn, rconn, X, etype, material, section, name="beam", n1=None):
        conn = np.asarray(conn, dtype=np.int64)
        self.vis_conn = jnp.asarray(conn.astype(np.int32))
        self.conn = jnp.asarray(np.hstack([conn, np.asarray(rconn)]).astype(np.int32))
        self.etype, self.material, self.name = etype, material, name
        self.ne = len(conn)
        self.sec = section
        if n1 is None:
            n1 = default_n1(X, conn)
        else:
            n1 = np.broadcast_to(np.asarray(n1, float), (self.ne, 3)).copy()
        self.n1 = jnp.asarray(n1)[:, None, :]                       # state layout (ne, 1, 3)
        self.E, self.nu = elastic_constants(material)
        self._build()

    def init_state(self):
        return self.n1

    def dofs(self):
        return (self.conn[:, :, None] * 3 + jnp.arange(3)).reshape(self.ne, -1)

    def _build(self):
        s = self.sec
        E, nu = self.E, self.nu
        G = E / (2 * (1 + nu))
        EA, GAs, GJ, EI2, EI3 = E * s["A"], G * s["A"] * s["k"], G * s["J"], E * s["I2"], E * s["I3"]
        rho, A = self.material.density, s["A"]
        alpha = self.material.alpha

        def kin(Xe, ue, n1):
            X1, X2 = Xe[0], Xe[1]
            L0 = jnp.linalg.norm(X2 - X1)
            e1 = (X2 - X1) / L0
            e2 = n1 - (n1 @ e1) * e1
            e2 = e2 / jnp.linalg.norm(e2)
            R0 = jnp.stack([e1, e2, jnp.cross(e1, e2)], 1)
            x1, x2 = X1 + ue[0], X2 + ue[1]
            Rm = rodrigues(0.5 * (ue[2] + ue[3])) @ R0
            gam = Rm.T @ (x2 - x1) / L0 - jnp.array([1.0, 0.0, 0.0])
            kap = Rm.T @ (ue[3] - ue[2]) / L0
            return gam, kap, L0, Rm

        def energy(Xe, ue, se, Te):
            gam, kap, L0, _ = kin(Xe, ue, se[0])
            g0 = gam[0] - alpha * jnp.mean(Te[:2, 0] - Te[:2, 1])
            return 0.5 * L0 * (EA * g0 ** 2 + GAs * (gam[1] ** 2 + gam[2] ** 2) + GJ * kap[0] ** 2
                               + EI2 * kap[1] ** 2 + EI3 * kap[2] ** 2)

        grad = jax.grad(energy, argnums=1)
        linear = bool(getattr(self.material, "small_strain", False))
        self.linear = linear

        def force_nl(Xe, ue, se, Te, sc):
            return sc * grad(Xe, ue, se, Te)

        def stiff_nl(Xe, ue, se, Te, sc):
            return jax.jacfwd(lambda u: force_nl(Xe, u, se, Te, sc))(ue).reshape(12, 12)

        def force(Xe, ue, se, Te, sc):
            if not linear:
                return force_nl(Xe, ue, se, Te, sc), se
            z = jnp.zeros_like(ue)
            return force_nl(Xe, z, se, Te, sc) + (stiff_nl(Xe, z, se, Te, sc) @ ue.reshape(-1)).reshape(ue.shape), se

        def stiff(Xe, ue, se, Te, sc):
            return stiff_nl(Xe, jnp.zeros_like(ue) if linear else ue, se, Te, sc)

        def mass(Xe):
            L0 = jnp.linalg.norm(Xe[1] - Xe[0])
            m = rho * A * L0 / 2
            mr = rho * (s["I2"] + s["I3"]) * L0 / 2
            return jnp.array([m, m, mr, mr])

        def volume(Xe):
            return jnp.linalg.norm(Xe[1] - Xe[0]) * A

        def cauchy(Xe, ue, se, Te):
            gam, kap, L0, Rm = kin(Xe, ue, se[0])
            N = EA * (gam[0] - alpha * jnp.mean(Te[:2, 0] - Te[:2, 1]))
            sb = jnp.abs(EI2 * kap[1]) * s["c3"] / s["I2"] + jnp.abs(EI3 * kap[2]) * s["c2"] / s["I3"]
            sig = N / A + jnp.where(N >= 0, 1.0, -1.0) * sb
            e1 = Rm[:, 0]
            return sig * jnp.outer(e1, e1), 0.0, I3

        def charlen(Xe):
            return jnp.linalg.norm(Xe[1] - Xe[0])

        self.force = jax.vmap(force)
        self.stiff = jax.vmap(stiff)
        self.mass = jax.vmap(mass)
        self.volume = jax.vmap(volume)
        self.cauchy = jax.vmap(cauchy)
        self.charlen = jax.vmap(charlen)
        self.geo_dir = jax.vmap(lambda Xe, ue0, se, Te: jax.jvp(
            lambda u: stiff_nl(Xe, u, se, Te, 1.0), (jnp.zeros_like(ue0),), (ue0,))[1])
