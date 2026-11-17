"""Interchange formats. Solver decks: Abaqus .inp, Nastran .bdf/.nas/.dat, ANSYS .cdb, gmsh
.msh (via the CAD module). Results: ParaView .vtu/.pvd."""
import os


def read_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".inp":
        from .inp import read_inp_spec
        return read_inp_spec(path)
    if ext in (".bdf", ".nas", ".dat", ".blk", ".fem"):
        from .nastran import read_bdf
        return read_bdf(path)
    if ext in (".cdb", ".ans"):
        from .ansys_cdb import read_cdb
        return read_cdb(path)
    if ext in (".msh", ".unv", ".mesh", ".vtk", ".vtu", ".med", ".cgns"):
        from ..cad.mesher import read_mesh_file
        return read_mesh_file(path)
    raise ValueError(f"unsupported solver deck / mesh format: {ext}")


def write_any(spec, path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".inp":
        from .inp import write_inp
        return write_inp(spec, path)
    if ext in (".bdf", ".nas", ".dat"):
        from .nastran import write_bdf
        return write_bdf(spec, path)
    raise ValueError(f"unsupported output format: {ext}")


from .vtk import write_vtu, write_pvd  # noqa: E402,F401
