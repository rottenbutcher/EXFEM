"""CAD kernel (OpenCASCADE via gmsh): parametric features, STEP/IGES/BREP/STL/DXF import,
CAD-associative meshing."""
from .geometry import CadModel, GeometryError, match, GMSH_LOCK, dxf_loops  # noqa: F401
from .mesher import mesh_cad, read_mesh_file  # noqa: F401
