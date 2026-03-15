# Mode 1 Domain Initializers

This catalog lists the current fixed-topology domain initializers that can feed Mode 1 optimization.

## 1D

- `polyline_mesh`
  Build a line mesh from an ordered 2D or 3D point sequence.

- `unit_interval_line_mesh`
  Build a simple line mesh on the unit interval for baseline Mode 1 workflows.

## 2D

- `polygon_domain_tri_mesh`
  Build an unstructured triangle mesh for a polygonal region with optional polygonal holes.

- `polygon_domain_tri_mesh_tagged`
  Same 2D mesh initialization, with stable boundary tags for Gmsh export.

- `polygon_domain_quad_mesh`
  Build an unstructured conforming quad mesh for a polygonal region by subdividing a recovered triangle mesh into quads.

- `polygon_domain_quad_mesh_tagged`
  Same quad initialization, with stable boundary tags for Gmsh export.

## 3D

- `implicit_volume_tet_mesh`
  Build a tetrahedral mesh by filtering a structured background tet grid against an implicit level-set function.

- `box_volume_tet_mesh`
  Build a tetrahedral mesh for an axis-aligned box by filling the volume with a structured tet lattice.

- `box_volume_tet_mesh_tagged`
  Same box-volume initialization, with tagged boundary face groups for the six box sides.

- `implicit_volume_tet_mesh_tagged`
  Same implicit-volume initialization, with tagged boundary faces.

- `extruded_polygon_tet_mesh`
  Build a tetrahedral mesh by extruding an unstructured polygon-domain triangle mesh through layered prism-to-tet splitting.

## Import

- `load_gmsh_msh`
  Import a supported fixed mesh from a Gmsh MSH 2.2 file for use in Mode 1 workflows.
