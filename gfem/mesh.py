"""Simple structured mesh generators and converters."""
import numpy as np


def box_hex(L, H, W, nx, ny, nz, origin=(0, 0, 0)):
    xs = np.linspace(0, L, nx + 1) + origin[0]
    ys = np.linspace(0, H, ny + 1) + origin[1]
    zs = np.linspace(0, W, nz + 1) + origin[2]
    X = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    nid = lambda a, b, c: (a * (ny + 1) + b) * (nz + 1) + c
    conn = np.stack([nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                     nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1),
                     nid(i, j + 1, k + 1)], -1)
    return X, conn


_KUHN = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
                  [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]])


def hex_to_tet4(X, conn):
    tets = conn[:, _KUHN].reshape(-1, 4)
    v = np.einsum("ij,ij->i", np.cross(X[tets[:, 1]] - X[tets[:, 0]], X[tets[:, 2]] - X[tets[:, 0]]),
                  X[tets[:, 3]] - X[tets[:, 0]])
    neg = v < 0
    tets[neg, 1], tets[neg, 2] = tets[neg, 2].copy(), tets[neg, 1].copy()
    return X, tets


def tet4_to_tet10(X, conn):
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    e = np.concatenate([np.sort(conn[:, list(p)], 1) for p in edges])
    uniq, inv = np.unique(e, axis=0, return_inverse=True)
    mids = 0.5 * (X[uniq[:, 0]] + X[uniq[:, 1]])
    newid = inv.reshape(len(edges), -1).T + len(X)
    return np.vstack([X, mids]), np.hstack([conn, newid])


def nodes_where(X, fn):
    return np.nonzero(fn(X[:, 0], X[:, 1], X[:, 2]))[0]


def add_midside(X, conn, edges):
    """Generic linear -> quadratic conversion: one shared mid-node per edge (i, j)."""
    X, conn = np.asarray(X, float), np.asarray(conn)
    e = np.concatenate([np.sort(conn[:, list(p)], 1) for p in edges])
    uniq, inv = np.unique(e, axis=0, return_inverse=True)
    mids = 0.5 * (X[uniq[:, 0]] + X[uniq[:, 1]])
    newid = inv.reshape(len(edges), -1).T + len(X)
    return np.vstack([X, mids]), np.hstack([conn, newid])


def hex8_to_hex20(X, conn):
    from .elements import _H20_EDGES
    return add_midside(X, conn, _H20_EDGES)


def hex_to_wedge(X, conn):
    """Split every hexahedron into two 6-node wedges (prisms along local z)."""
    c = np.asarray(conn)
    w1 = c[:, [0, 1, 2, 4, 5, 6]]
    w2 = c[:, [0, 2, 3, 4, 6, 7]]
    return X, np.vstack([w1, w2])


def cylinder_hex(R, H, nr, nt, nz, Ri=0.0):
    """Structured hexahedral (hollow) cylinder along z: Ri inner radius (Ri > 0)."""
    r = np.linspace(Ri, R, nr + 1)
    t = np.linspace(0, 2 * np.pi, nt + 1)[:-1]
    z = np.linspace(0, H, nz + 1)
    rr, tt, zz = np.meshgrid(r, t, z, indexing="ij")
    X = np.stack([rr * np.cos(tt), rr * np.sin(tt), zz], -1).reshape(-1, 3)
    i, j, k = np.meshgrid(np.arange(nr), np.arange(nt), np.arange(nz), indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    nid = lambda a, b, c: (a * nt + b % nt) * (nz + 1) + c
    conn = np.stack([nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                     nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1),
                     nid(i, j + 1, k + 1)], -1)
    return X, conn
