"""ParaView output: binary (base64) .vtu files and .pvd time series."""
import base64
import numpy as np

_VTK_ORDER = {}     # Abaqus -> VTK node order (identical for the supported types, except C3D20)
# VTK quadratic hexahedron: 8 corners, 12 mid-edge nodes in order (0-1,1-2,2-3,3-0,4-5,5-6,6-7,7-4,0-4,1-5,2-6,3-7)
# which equals the Abaqus C3D20 order.


def _b64(arr, dtype):
    a = np.ascontiguousarray(arr, dtype=dtype).ravel()
    raw = a.tobytes()
    head = np.array([len(raw)], dtype=np.uint64).tobytes()
    return base64.b64encode(head + raw).decode()


def _da(name, arr, dtype="Float64", ncomp=None):
    np_t = {"Float64": np.float64, "Float32": np.float32, "Int64": np.int64, "UInt8": np.uint8}[dtype]
    nc = f' NumberOfComponents="{ncomp}"' if ncomp else ""
    return (f'<DataArray type="{dtype}" Name="{name}"{nc} format="binary">'
            f'{_b64(arr, np_t)}</DataArray>')


def write_vtu(path, model, frame=None, u=None, stresses=None):
    """Write mesh + result frame (dict from analysis.postprocess). Backwards compatible
    signature write_vtu(path, model, u, stresses) is accepted."""
    if frame is not None and not isinstance(frame, dict):
        u, stresses, frame = frame, u, None
    X = np.asarray(model.X)
    conns, types = [], []
    for b in model.blocks:
        c = np.asarray(b.conn)
        conns.append(c)
        types += [b.etype.vtk_id] * len(c)
    offsets = np.cumsum([c.shape[1] for c in conns for _ in range(len(c))])
    flat = np.concatenate([c.ravel() for c in conns]) if conns else np.zeros(0)
    pd, cd = [], []
    if frame is None:
        frame = {}
        if u is not None:
            frame["U"] = np.asarray(u).reshape(-1, 3)
        if stresses is not None:
            frame["MISES_elem"] = np.concatenate([np.asarray(s["MISES"]) for s in stresses])
    for k, v in frame.items():
        if not isinstance(v, np.ndarray):
            continue
        if len(v) == len(X):
            nc = v.shape[1] if v.ndim > 1 else None
            pd.append(_da(k, v, "Float64", nc))
        elif len(v) == len(types):
            nc = v.shape[1] if v.ndim > 1 else None
            cd.append(_da(k, v, "Float64", nc))
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="1.0" '
                'byte_order="LittleEndian" header_type="UInt64">\n<UnstructuredGrid>\n'
                f'<Piece NumberOfPoints="{len(X)}" NumberOfCells="{len(types)}">\n'
                f'<Points>{_da("Points", X, "Float64", 3)}</Points>\n'
                f'<Cells>{_da("connectivity", flat, "Int64")}{_da("offsets", offsets, "Int64")}'
                f'{_da("types", np.array(types), "UInt8")}</Cells>\n'
                f'<PointData>{"".join(pd)}</PointData>\n<CellData>{"".join(cd)}</CellData>\n'
                '</Piece>\n</UnstructuredGrid>\n</VTKFile>\n')


def write_pvd(path, entries):
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
        for t, fn in entries:
            f.write(f'<DataSet timestep="{t:.9g}" file="{fn}"/>\n')
        f.write('</Collection>\n</VTKFile>\n')
