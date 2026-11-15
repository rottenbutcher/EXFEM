"""Constitutive models.

Each model maps (F, state, T) -> (P, new_state) at one integration point, where P is the first
Piola-Kirchhoff stress (or the Cauchy stress for small-strain models, `small_strain = True`).
Consistent tangents are never hand-written: they come from automatic differentiation, so a new
material is one Python function (see `UserMaterial`).

All models also carry the thermal properties used by heat transfer and thermal-stress analysis:
density, alpha (expansion), tref (reference temperature), conductivity k and specific heat cp.
"""
import jax
import jax.numpy as jnp
import numpy as np

I3 = jnp.eye(3)


def lame(E, nu):
    return E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))


class Material:
    nstate = 1            # at least 1 so that arrays are never empty
    small_strain = False  # True: pk1 returns Cauchy stress directly
    name = "material"
    linear = False        # True: stress is linear in displacement gradient (one Newton iteration)

    def __init__(self, density=1.0, name=None, alpha=0.0, tref=0.0, k=1.0, cp=1.0):
        self.density = float(density)
        self.alpha, self.tref = float(alpha), float(tref)
        self.k, self.cp = float(k), float(cp)
        if name:
            self.name = name

    def pk1(self, F, s):
        raise NotImplementedError

    def pk1T(self, F, s, T):
        """Temperature-aware entry point used by the element kernels."""
        return self.pk1(F, s)

    def wave_speed(self):
        return float(np.sqrt((self.lam + 2 * self.mu) / self.density))

    def eqps(self, s):     # equivalent plastic strain for output
        return 0.0 * s[..., 0]

    def modulus(self):
        """Representative Young's modulus (penalty / contact stiffness scaling)."""
        return float(self.mu * (3 * self.lam + 2 * self.mu) / (self.lam + self.mu))

    def describe(self):
        return f"{type(self).__name__}({self.name})"


class LinearElastic(Material):
    """Small-strain isotropic linear elasticity."""
    small_strain = True
    linear = True

    def __init__(self, E, nu, **kw):
        super().__init__(**kw)
        self.E, self.nu = float(E), float(nu)
        self.mu, self.lam = lame(E, nu)

    def pk1(self, F, s):
        eps = 0.5 * (F + F.T) - I3
        return self.lam * jnp.trace(eps) * I3 + 2 * self.mu * eps, s


class StVenantKirchhoff(LinearElastic):
    """Large-rotation version of linear elasticity (Green-Lagrange strain)."""
    small_strain = False
    linear = False

    def pk1(self, F, s):
        Egl = 0.5 * (F.T @ F - I3)
        S = self.lam * jnp.trace(Egl) * I3 + 2 * self.mu * Egl
        return F @ S, s


# ---------------------------------------------------------------- orthotropic
def _voigt(e):
    return jnp.array([e[0, 0], e[1, 1], e[2, 2], 2 * e[0, 1], 2 * e[0, 2], 2 * e[1, 2]])


def _unvoigt(s):
    return jnp.array([[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]])


class OrthotropicElastic(Material):
    """Orthotropic elasticity from engineering constants (Abaqus *ELASTIC, TYPE=ENGINEERING CONSTANTS).
    `axes` is a 3x3 matrix whose columns are the material directions in global coordinates.
    nlgeom=True uses Green-Lagrange strain / PK2 (large rotation, small strain)."""

    def __init__(self, E1, E2, E3, nu12, nu13, nu23, G12, G13, G23, axes=None, nlgeom=False, **kw):
        super().__init__(**kw)
        S = np.zeros((6, 6))
        S[0, 0], S[1, 1], S[2, 2] = 1 / E1, 1 / E2, 1 / E3
        S[0, 1] = S[1, 0] = -nu12 / E1
        S[0, 2] = S[2, 0] = -nu13 / E1
        S[1, 2] = S[2, 1] = -nu23 / E2
        S[3, 3], S[4, 4], S[5, 5] = 1 / G12, 1 / G13, 1 / G23
        self.C = jnp.asarray(np.linalg.inv(S))
        self.R = jnp.asarray(np.eye(3) if axes is None else np.asarray(axes, float))
        self.small_strain = not nlgeom
        self.linear = not nlgeom
        Emax = max(E1, E2, E3)
        self.mu, self.lam = lame(Emax, 0.3)
        self.consts = (E1, E2, E3, nu12, nu13, nu23, G12, G13, G23)

    def _stress(self, e):
        el = self.R.T @ e @ self.R
        sl = _unvoigt(self.C @ _voigt(el))
        return self.R @ sl @ self.R.T

    def pk1(self, F, s):
        if self.small_strain:
            return self._stress(0.5 * (F + F.T) - I3), s
        return F @ self._stress(0.5 * (F.T @ F - I3)), s


# ---------------------------------------------------------------- hyperelastic
class Hyperelastic(Material):
    """Base for strain-energy based models: P = dW/dF by automatic differentiation."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._P = jax.grad(self.energy)

    def energy(self, F):
        raise NotImplementedError

    def pk1(self, F, s):
        return self._P(F), s

    @staticmethod
    def invariants(F):
        J = jnp.linalg.det(F)
        C = F.T @ F
        I1b = J ** (-2 / 3) * jnp.trace(C)
        I2b = J ** (-4 / 3) * 0.5 * (jnp.trace(C) ** 2 - jnp.sum(C * C))
        return J, I1b, I2b


class NeoHookean(Hyperelastic):
    """Abaqus form: U = C10 (I1bar - 3) + (J - 1)^2 / D1."""

    def __init__(self, C10=None, D1=None, E=None, nu=None, **kw):
        if C10 is None:
            mu, lam = lame(E, nu)
            C10, D1 = mu / 2, 2 / (lam + 2 * mu / 3)
        self.C10, self.D1 = float(C10), float(D1)
        self.mu = 2 * self.C10
        self.lam = 2 / self.D1 - 2 * self.mu / 3
        super().__init__(**kw)

    def energy(self, F):
        J, I1b, _ = self.invariants(F)
        return self.C10 * (I1b - 3) + (J - 1) ** 2 / self.D1


class MooneyRivlin(Hyperelastic):
    """U = C10 (I1b-3) + C01 (I2b-3) + (J-1)^2/D1."""

    def __init__(self, C10, C01, D1, **kw):
        self.C10, self.C01, self.D1 = float(C10), float(C01), float(D1)
        self.mu = 2 * (self.C10 + self.C01)
        self.lam = 2 / self.D1 - 2 * self.mu / 3
        super().__init__(**kw)

    def energy(self, F):
        J, I1b, I2b = self.invariants(F)
        return self.C10 * (I1b - 3) + self.C01 * (I2b - 3) + (J - 1) ** 2 / self.D1


class Yeoh(Hyperelastic):
    """U = sum Ci0 (I1b-3)^i + sum (J-1)^(2i)/Di  (Abaqus *HYPERELASTIC, YEOH)."""

    def __init__(self, C10, C20, C30, D1, D2=0.0, D3=0.0, **kw):
        self.C = [float(C10), float(C20), float(C30)]
        self.D = [float(D1), float(D2), float(D3)]
        self.mu = 2 * self.C[0]
        self.lam = 2 / self.D[0] - 2 * self.mu / 3
        super().__init__(**kw)

    def energy(self, F):
        J, I1b, _ = self.invariants(F)
        W = sum(c * (I1b - 3) ** (i + 1) for i, c in enumerate(self.C))
        W += sum((J - 1) ** (2 * (i + 1)) / d for i, d in enumerate(self.D) if d > 0)
        return W


class ArrudaBoyce(Hyperelastic):
    """Eight-chain model (Abaqus *HYPERELASTIC, ARRUDA-BOYCE): mu, locking stretch lam_m, D."""
    _C = (0.5, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750)

    def __init__(self, mu, lam_m, D, **kw):
        self.mu0, self.lm, self.D = float(mu), float(lam_m), float(D)
        self.mu = self.mu0
        self.lam = 2 / self.D - 2 * self.mu / 3
        super().__init__(**kw)

    def energy(self, F):
        J, I1b, _ = self.invariants(F)
        W = sum(c / self.lm ** (2 * i) * (I1b ** (i + 1) - 3 ** (i + 1)) for i, c in enumerate(self._C))
        return self.mu0 * W + ((J * J - 1) / 2 - jnp.log(J)) / self.D


class Gent(Hyperelastic):
    """Gent model: W = -mu Jm/2 ln(1 - (I1b-3)/Jm) + volumetric (limiting chain extensibility)."""

    def __init__(self, mu, Jm, D, **kw):
        self.mu, self.Jm, self.D = float(mu), float(Jm), float(D)
        self.lam = 2 / self.D - 2 * self.mu / 3
        super().__init__(**kw)

    def energy(self, F):
        J, I1b, _ = self.invariants(F)
        return -0.5 * self.mu * self.Jm * jnp.log(1 - (I1b - 3) / self.Jm) + ((J * J - 1) / 2 - jnp.log(J)) / self.D


# ---------------------------------------------------------------- plasticity
def _safe_norm(a):
    return jnp.sqrt(jnp.sum(a * a) + 1e-30)


class J2Plasticity(Material):
    """Small-strain von Mises plasticity with combined hardening.

    Isotropic part: (yield stress, plastic strain) table exactly like Abaqus *PLASTIC, or a
    Johnson-Cook law  sy = (A + B eps_p^n)(1 - T*^m).  Kinematic part: linear (Prager) with
    modulus Hk (Abaqus HARDENING=KINEMATIC / COMBINED). Radial return with a local Newton solve.
    State: [eps_p (9), alpha, backstress (9)]."""
    nstate = 19
    small_strain = True

    def __init__(self, E, nu, table=None, Hk=0.0, johnson_cook=None, **kw):
        super().__init__(**kw)
        self.E, self.nu = float(E), float(nu)
        self.mu, self.lam = lame(E, nu)
        self.K = self.lam + 2 * self.mu / 3
        self.Hk = float(Hk)
        self.jc = johnson_cook          # dict(A, B, n, m, Tmelt, Ttrans)
        if self.jc is None:
            t = np.asarray(table, dtype=np.float64).reshape(-1, 2)
            if t.shape[0] == 1:   # perfect plasticity
                t = np.array([[t[0, 0], 0.0], [t[0, 0], 1e3]])
            sy_tab, ep_tab = t[:, 0], t[:, 1]
            slope = (sy_tab[-1] - sy_tab[-2]) / max(ep_tab[-1] - ep_tab[-2], 1e-30)
            self.ep_tab = jnp.asarray(np.append(ep_tab, 1e6))
            self.sy_tab = jnp.asarray(np.append(sy_tab, sy_tab[-1] + slope * (1e6 - ep_tab[-1])))
            self.table = t
            self.sy0 = float(t[0, 0])
        else:
            self.sy0 = float(self.jc["A"])

    def sy(self, a, T=0.0):
        if self.jc is None:
            return jnp.interp(a, self.ep_tab, self.sy_tab)
        c = self.jc
        sy = c["A"] + c["B"] * (a + 1e-9) ** c["n"]
        Tm, Tt = c.get("Tmelt", 1e30), c.get("Ttrans", 0.0)
        Ts = jnp.clip((T - Tt) / (Tm - Tt), 0.0, 1.0)
        return sy * (1 - Ts ** c.get("m", 1.0))

    def eqps(self, s):
        return s[..., 9]

    def _return(self, eps, s, T):
        ep = s[:9].reshape(3, 3)
        a0 = s[9]
        beta = s[10:19].reshape(3, 3)
        ee = eps - ep
        tr = jnp.trace(ee)
        dev = ee - tr / 3 * I3
        s_tr = 2 * self.mu * dev
        xi = s_tr - beta
        q_tr = jnp.sqrt(1.5) * _safe_norm(xi)              # von Mises of relative stress
        f_tr = q_tr - self.sy(a0, T)
        H3 = 3 * self.mu + self.Hk
        dsy = jax.grad(self.sy)

        def body(_, dg):                                  # local Newton on dgamma
            g = q_tr - H3 * dg - self.sy(a0 + dg, T)
            return jnp.maximum(dg + g / (H3 + dsy(a0 + dg, T)), 0.0)

        dg = jax.lax.fori_loop(0, 25, body, 0.0)
        dg = jnp.where(f_tr > 0, dg, 0.0)
        n = 1.5 * xi / (q_tr + 1e-30)                     # flow direction
        s_dev = s_tr - 2 * self.mu * dg * n
        sig = s_dev + self.K * tr * I3
        ep_new = ep + dg * n
        beta_new = beta + (2.0 / 3.0) * self.Hk * dg * n
        s_new = jnp.concatenate([ep_new.reshape(-1), jnp.array([a0 + dg]), beta_new.reshape(-1)])
        return sig, s_new

    def pk1(self, F, s):
        return self.pk1T(F, s, 0.0)

    def pk1T(self, F, s, T):
        return self._return(0.5 * (F + F.T) - I3, s, T)


class J2FiniteStrain(J2Plasticity):
    """Finite-rotation J2: Green-Lagrange strain with the J2 return in the material frame
    (suitable for large rotation / moderate strain, objective, no hypoelastic drift)."""
    small_strain = False

    def pk1T(self, F, s, T):
        S, s_new = self._return(0.5 * (F.T @ F - I3), s, T)
        return F @ S, s_new


# ---------------------------------------------------------------- user defined
class UserMaterial(Material):
    """Python 'UMAT': pass any JAX-traceable function stress(F, state, props) -> (P, state').
    The consistent tangent is obtained automatically by AD."""

    def __init__(self, fn, props=(), nstate=1, small_strain=False, E=1.0, nu=0.3, **kw):
        super().__init__(**kw)
        self.fn, self.props = fn, jnp.asarray(props, dtype=jnp.float64)
        self.nstate, self.small_strain = int(max(nstate, 1)), small_strain
        self.mu, self.lam = lame(E, nu)

    def pk1(self, F, s):
        return self.fn(F, s, self.props)


MATERIALS = {"linear": LinearElastic, "svk": StVenantKirchhoff, "neohooke": NeoHookean,
             "mooney": MooneyRivlin, "yeoh": Yeoh, "ortho": OrthotropicElastic,
             "j2": J2Plasticity, "j2fs": J2FiniteStrain, "user": UserMaterial}


def from_props(p, nlgeom=False, name=None):
    """Create a material from a property dictionary (shared by the GUI project format and the
    input-file readers). Recognised keys: model, E, nu, density, alpha, tref, k, cp,
    plastic (table), Hk, johnson_cook, C10, C01, C20, C30, D1, ortho (9 constants), axes."""
    common = dict(density=p.get("density", 1.0), alpha=p.get("alpha", 0.0), tref=p.get("tref", 0.0),
                  k=p.get("k", 1.0), cp=p.get("cp", 1.0), name=name or p.get("name"))
    model = p.get("model")
    if model is None:
        if "neohooke" in p:
            model = "neohooke"
        elif "mooney" in p:
            model = "mooney"
        elif "yeoh" in p:
            model = "yeoh"
        elif "plastic" in p or "johnson_cook" in p:
            model = "plastic"
        elif "ortho" in p:
            model = "ortho"
        else:
            model = "linear"
    if model == "neohooke":
        if "neohooke" in p:
            C10, D1 = p["neohooke"][:2]
        else:
            C10, D1 = p.get("C10"), p.get("D1")
        if C10 is None:
            return NeoHookean(E=p["E"], nu=p["nu"], **common)
        if not D1:
            D1 = 1e-3 / C10
        return NeoHookean(C10=C10, D1=D1, **common)
    if model == "mooney":
        C10, C01, D1 = p.get("mooney", [p.get("C10"), p.get("C01"), p.get("D1")])[:3]
        return MooneyRivlin(C10, C01, D1 if D1 else 1e-3 / (C10 + C01), **common)
    if model == "yeoh":
        c = p.get("yeoh", [p.get("C10"), p.get("C20", 0), p.get("C30", 0), p.get("D1")])
        D1 = c[3] if len(c) > 3 and c[3] else 1e-3 / c[0]
        return Yeoh(c[0], c[1], c[2], D1, **common)
    if model == "arruda":
        D = p.get("D1") or 2 / (p.get("K", 1e3 * p["mu"]))
        return ArrudaBoyce(p["mu"], p.get("lam_m", 7.0), D, **common)
    if model == "gent":
        D = p.get("D1") or 2 / (p.get("K", 1e3 * p["mu"]))
        return Gent(p["mu"], p.get("Jm", 50.0), D, **common)
    if model == "ortho":
        return OrthotropicElastic(*p["ortho"], axes=p.get("axes"), nlgeom=nlgeom, **common)
    if model == "plastic":
        cls = J2FiniteStrain if nlgeom else J2Plasticity
        table = p.get("plastic")
        if table is None and "yield" in p and "johnson_cook" not in p:
            Et = p.get("tangent", 0.0)                    # bilinear (ANSYS BISO style)
            E = p["E"]
            Hp = E * Et / (E - Et) if Et > 0 else 0.0
            table = [[p["yield"], 0.0], [p["yield"] + Hp * 1.0, 1.0]]
        return cls(p["E"], p["nu"], table, Hk=p.get("Hk", 0.0),
                   johnson_cook=p.get("johnson_cook"), **common)
    cls = StVenantKirchhoff if nlgeom else LinearElastic
    return cls(p["E"], p["nu"], **common)
