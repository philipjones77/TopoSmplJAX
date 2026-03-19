from topojax.mesh.generators import unit_square_points
from topojax.mesh.topology import unit_cube_tet_mesh, unit_square_quad_mesh


def test_unit_square_points_shape() -> None:
    points = unit_square_points(4, 3)
    assert points.shape == (12, 2)


def test_unit_square_quad_mesh_shape() -> None:
    topo, points = unit_square_quad_mesh(5, 4)
    assert points.shape == (20, 2)
    assert topo.elements.shape == (12, 4)


def test_unit_cube_tet_mesh_shape() -> None:
    topo, points = unit_cube_tet_mesh(4, 3, 2)
    assert points.shape == (24, 3)
    assert topo.elements.shape[1] == 4
