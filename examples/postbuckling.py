"""Nonlinear post-buckling of an imperfect column (eigenmode imperfection + Riks) vs the elastica."""
import numpy as np
import gfem
from scipy.special import ellipk
from scipy.optimize import brentq

L, n = 400.0, 20
X = np.stack([np.zeros(n + 1), np.zeros(n + 1), np.linspace(0, L, n + 1)], 1)
m = gfem.Model(X)
m.add_beam(np.stack([np.arange(n), np.arange(1, n + 1)], 1), gfem.StVenantKirchhoff(200e3, 0.3), ("RECT", (20., 20.)), n1=[1, 0, 0])
st = gfem.Step(bcs=[([0], d, 0.) for d in range(6)], loads=[([n], 2, -1.0)])
r = gfem.postbuckling(m, st, modes=(1,), scale=0.2, riks=dict(max_inc=40, lpf_max=1.8 * 41135))
Pcr = r["load_factors"][0]
print(f"Euler load {Pcr:.1f} N")
for lpf, umax in r["curve"][::5]:
    print(f"  P/Pcr = {lpf / Pcr:6.3f}   max |u| = {umax:8.2f} mm")
ratio = r["lpf"] / Pcr
p = brentq(lambda p: (2 * ellipk(p * p) / np.pi) ** 2 - ratio, 1e-6, 0.999999)
U = np.asarray(r["u"]).reshape(-1, 3)
print(f"final: tip lateral {np.hypot(*U[n, :2]):.1f} mm, elastica {2 * p / ellipk(p * p) * L:.1f} mm")
