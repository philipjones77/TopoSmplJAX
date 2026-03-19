# Mode 1 Domain Initializers Implementation

This note explains the discrete initialization paths that now support Mode 1.

## 1D

Line meshes are represented as 2-node elements with fixed connectivity. The Mode 1 quality objective minimizes segment-length variance, graph-Laplacian roughness, and short-segment collapse penalties.

## 2D Polygon Domains

`polygon_domain_tri_mesh` samples polygon loops, fills the interior with a point set, triangulates it, and explicitly recovers constrained loop edges. The result is a discrete initialization stage followed by a fixed triangle topology for JAX optimization.

The tagged variant emits Gmsh line blocks for the outer loop and each hole loop, along with stable physical names.

`polygon_domain_quad_mesh` reuses the same recovered polygon triangle mesh, inserts shared edge midpoints plus one centroid per triangle, and emits three conforming quads per triangle. This keeps the recovered polygon boundary exact while giving Mode 1 a practical arbitrary-domain quad path.

## 3D Implicit Volumes

`implicit_volume_tet_mesh` uses a structured tetrahedral background grid over a bounding box and retains tetrahedra whose centroids satisfy a user level-set condition. This is practical arbitrary-volume initialization, not CAD-level body-fitted meshing.

The tagged variant extracts boundary faces and exports them as triangle blocks.

`sphere_volume_tet_mesh` is the concrete convenience version of that pattern for spherical domains. It computes a sphere-aligned bounding box and reuses the same fixed-topology centroid filter, so it stays inside the existing Mode 1 contract instead of introducing a second 3D meshing strategy.

## 3D Box Volumes

`box_volume_tet_mesh` is the exact structured counterpart to the implicit path: it maps a unit cube lattice onto an axis-aligned box and keeps the full tetrahedral fill. The tagged variant groups extracted boundary faces into `xmin`, `xmax`, `ymin`, `ymax`, `zmin`, and `zmax` triangle blocks.

## 3D Polygon Extrusion

`extruded_polygon_tet_mesh` first creates an unstructured polygon-domain triangle mesh, then stacks it through a user-defined number of layers and splits each prism into tetrahedra. Tagged boundary blocks are generated for bottom, top, and wall groups.

## Import and Workflow

`load_gmsh_msh` imports supported line, triangle, quad, and tetrahedral blocks from Gmsh MSH 2.2 files. `initialize_mode1_domain` and `run_mode1_workflow` provide a single high-level entry path for create, optimize, export, and optional viewer launch.
