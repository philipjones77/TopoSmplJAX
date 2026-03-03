"""gmshjax: AD mesh creation for JAX."""

from .ad.compiled import build_quality_value_and_grad
from .ad.pipeline import (
    build_model_parametric_quality_value_and_grad,
    build_parametric_quality_value_and_grad,
    default_param_vector,
    default_params,
)
from .mesh.adaptive import AdaptiveHistory, adaptive_remesh_tri
from .mesh.adaptive_quad import QuadAdaptiveHistory, adaptive_remesh_quad, quad_area_magnitudes, quad_refinement_priority
from .mesh.adaptive_tet import TetAdaptiveHistory, adaptive_remesh_tet, tet_refinement_priority, tet_volume_magnitudes
from .mesh.boundary import (
    BoundaryCurves2D,
    boundary_constrained_points,
    line_segment,
    sinusoidal_top_boundary,
    smooth_boundary_constrained_points,
    transfinite_interpolation,
)
from .mesh.connectivity_opt import evaluate_edge_flip_candidates, evaluate_laplacian_smoothing_candidates
from .mesh.factory import make_unit_square_model
from .mesh.generators import project_cube_points_to_sphere, unit_square_points
from .mesh.manifold import DeformationParams, apply_deformation
from .mesh.mutation import TriMeshBuffer, active_elements, active_points, flip_diagonal, make_tri_mesh_buffer, split_triangle
from .mesh.mutation_qt import (
    QuadMeshBuffer,
    TetMeshBuffer,
    active_quad_elements,
    active_quad_points,
    active_tet_elements,
    active_tet_points,
    make_quad_mesh_buffer,
    make_tet_mesh_buffer,
    split_quad,
    split_tet,
)
from .mesh.diagnostics import quad_diagnostics, tet_diagnostics, tri_diagnostics
from .mesh.operators import (
    edge_lengths,
    graph_laplacian_step,
    quad_icn,
    quad_ige,
    tet_icn,
    tet_ige,
    triangle_icn,
    triangle_ige,
    triangle_signed_areas,
)
from .mesh.refine import (
    batched_refinement_step,
    refinement_midpoints,
    select_refinement_candidates,
    triangle_area_magnitudes,
    triangle_refinement_priority,
)
from .mesh.topology import MeshTopology, unit_cube_tet_mesh, unit_square_quad_mesh, unit_square_tri_mesh
from .model import GEntity, MeshModel, MeshState, update_points
from .numpy_impl import NumpyDeformationParams, NumpyMeshTopology
from .runtime import get_runtime_precision, jax_float_dtype, numpy_float_dtype, set_runtime_precision

__all__ = [
    "AdaptiveHistory",
    "BoundaryCurves2D",
    "DeformationParams",
    "GEntity",
    "MeshModel",
    "MeshState",
    "MeshTopology",
    "NumpyDeformationParams",
    "NumpyMeshTopology",
    "QuadAdaptiveHistory",
    "QuadMeshBuffer",
    "TetMeshBuffer",
    "TetAdaptiveHistory",
    "TriMeshBuffer",
    "active_elements",
    "active_points",
    "active_quad_elements",
    "active_quad_points",
    "active_tet_elements",
    "active_tet_points",
    "adaptive_remesh_quad",
    "adaptive_remesh_tet",
    "adaptive_remesh_tri",
    "apply_deformation",
    "batched_refinement_step",
    "boundary_constrained_points",
    "build_model_parametric_quality_value_and_grad",
    "build_parametric_quality_value_and_grad",
    "build_quality_value_and_grad",
    "default_param_vector",
    "default_params",
    "edge_lengths",
    "evaluate_edge_flip_candidates",
    "evaluate_laplacian_smoothing_candidates",
    "flip_diagonal",
    "graph_laplacian_step",
    "line_segment",
    "make_quad_mesh_buffer",
    "make_tet_mesh_buffer",
    "make_tri_mesh_buffer",
    "make_unit_square_model",
    "quad_area_magnitudes",
    "quad_icn",
    "quad_ige",
    "quad_refinement_priority",
    "quad_diagnostics",
    "refinement_midpoints",
    "project_cube_points_to_sphere",
    "select_refinement_candidates",
    "set_runtime_precision",
    "get_runtime_precision",
    "jax_float_dtype",
    "numpy_float_dtype",
    "sinusoidal_top_boundary",
    "smooth_boundary_constrained_points",
    "split_quad",
    "split_tet",
    "split_triangle",
    "tet_icn",
    "tet_ige",
    "tet_diagnostics",
    "tet_refinement_priority",
    "tet_volume_magnitudes",
    "transfinite_interpolation",
    "triangle_area_magnitudes",
    "triangle_icn",
    "triangle_ige",
    "triangle_refinement_priority",
    "triangle_signed_areas",
    "tri_diagnostics",
    "unit_cube_tet_mesh",
    "unit_square_points",
    "unit_square_quad_mesh",
    "unit_square_tri_mesh",
    "update_points",
]
