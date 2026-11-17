"""Shells and beams: Scordelis-Lo roof (shell benchmark) and a stiffened plate (shell + beam)."""
import numpy as np
import gfem

# --- Scordelis-Lo roof, quarter model, reference vertical displacement 0.3024
R, Lh, phi0, n = 25.0, 25.0, np.radians(40), 16
xs, ph = np.linspace(0, Lh, n + 1), np.linspace(0, phi0, n + 1)
X = np.array([[x, R * np.sin(p), R * np.cos(p)] for x in xs for p in ph])
nid = lambda i, j: i * (n + 1) + j
c = np.array([[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)] for i in range(n) for j in range(n)])
m = gfem.Model(X)
m.add_shell(c, gfem.LinearElastic(4.32e8, 0.0), 0.25)
sx, dia, cr = (np.nonzero(X[:, 0] < 1e-9)[0], np.nonzero(X[:, 0] > Lh - 1e-9)[0], np.nonzero(np.abs(X[:, 1]) < 1e-9)[0])
bcs = [(sx, 0, 0.), (sx, 4, 0.), (sx, 5, 0.), (dia, 1, 0.), (dia, 2, 0.), (cr, 1, 0.), (cr, 3, 0.), (cr, 5, 0.)]
w = np.zeros(len(X))
for row in c:
    w[row] += R * phi0 / n * Lh / n / 4
f = np.zeros(m.ndof)
f[np.arange(len(X)) * 3 + 2] = -90 * w
u, _, _ = gfem.StaticSolver(m, verbose=False).solve(gfem.Step(bcs=bcs, dloads=[(f,)]))
print(f"Scordelis-Lo: uz = {np.asarray(u).reshape(-1, 3)[nid(0, n), 2]:.4f}   (reference -0.3024)")

# --- stiffened plate: S4 skin + B31 stringer sharing nodes, cantilever under pressure
xs, ys = np.linspace(0, 200, 21), np.linspace(0, 100, 11)
X = np.array([[x, y, 0.0] for x in xs for y in ys])
nid = lambda i, j: i * 11 + j
c = np.array([[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)] for i in range(20) for j in range(10)])
s = [nid(i, 5) for i in range(21)]
for with_stringer in (False, True):
    m = gfem.Model(X)
    m.add_shell(c, gfem.LinearElastic(70e3, 0.33), 2.0)
    if with_stringer:
        m.add_beam(np.stack([s[:-1], s[1:]], 1), gfem.LinearElastic(70e3, 0.33), ("RECT", (4.0, 20.0)), n1=[0, 1, 0])
    root = np.nonzero(X[:, 0] < 1e-9)[0]
    from gfem.surfaces import Surface
    from gfem.elements import QUAD4
    p = gfem.loads.pressure_vector(X, Surface([(QUAD4, c)]), 0.01)
    full = np.zeros(m.ndof)
    full[:len(p)] = p
    u, _, _ = gfem.StaticSolver(m, verbose=False).solve(gfem.Step(bcs=[(root, d, 0.) for d in range(6)], dloads=[(full,)]))
    print(f"stiffened plate, stringer={with_stringer}: max |uz| = {np.abs(np.asarray(u).reshape(-1, 3)[:len(X), 2]).max():.4f} mm")
