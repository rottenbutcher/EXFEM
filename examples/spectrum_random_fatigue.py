"""Response spectrum (CQC), random vibration (PSD, 1-sigma) and stress-life fatigue on a beam."""
import numpy as np
import gfem
from gfem.fatigue import stress_life

n, L = 20, 1000.0
X = np.stack([np.linspace(0, L, n + 1), np.zeros(n + 1), np.zeros(n + 1)], 1)
m = gfem.Model(X)
m.add_beam(np.stack([np.arange(n), np.arange(1, n + 1)], 1), gfem.LinearElastic(200e3, 0.3, density=7.85e-9),
           ("PIPE", (30.0, 3.0)), n1=[0, 1, 0])
m.add_point_mass([n], 0.005)                                     # 5 kg tip mass (t)
st = gfem.Step(bcs=[([0], d, 0.0) for d in range(6)])
spec = [[0.1, 980.0], [2.0, 6130.0], [10.0, 6130.0], [33.0, 2450.0], [100.0, 1960.0]]      # mm/s^2
rs = gfem.vibration.response_spectrum(m, st, spec, direction=(0, 0, 1), nmodes=8, combination="CQC")
print("frequencies [Hz]:", np.round(rs["freq"][:4], 2))
print(f"response spectrum (CQC): tip |u| = {np.linalg.norm(rs['U'][n]):.3f} mm, max Mises = {rs['MISES'].max():.2f} MPa")
rv = gfem.vibration.random_vibration(m, st, [[10.0, 1e5], [500.0, 1e5]], direction=(0, 0, 1), nmodes=8, damping=0.02)
print(f"random vibration: input {rv['grms']:.0f} mm/s^2 RMS, tip 1-sigma |u| = {np.linalg.norm(rv['U'][n]):.3f} mm")
f = stress_life(rv["S"] * 3.0, Su=460, Sf=750, b=-0.09, loading="fully_reversed", mean="none")   # 3-sigma
print(f"3-sigma fatigue: min life {f['life'].min():.3g} cycles, min safety factor {f['safety_factor'].min():.2f}")
