import numpy as np
import jax.numpy as jnp

from topojax.mesh.operators import mesh_quality_energy as jax_energy
from topojax.mesh.operators import quad_icn as jax_quad_icn
from topojax.mesh.operators import quad_ige as jax_quad_ige
from topojax.mesh.operators import tet_icn as jax_tet_icn
from topojax.mesh.operators import tet_ige as jax_tet_ige
from topojax.mesh.operators import triangle_icn as jax_icn
from topojax.mesh.operators import triangle_ige as jax_ige
from topojax.mesh.topology import (
    unit_cube_tet_mesh as jax_tet_mesh,
    unit_square_quad_mesh as jax_quad_mesh,
    unit_square_tri_mesh as jax_mesh,
)
from topojax.numpy_impl import (
    NumpyDeformationParams,
    apply_deformation as np_deform,
    mesh_quality_energy as np_energy,
    quad_icn as np_quad_icn,
    quad_ige as np_quad_ige,
    tet_icn as np_tet_icn,
    tet_ige as np_tet_ige,
    triangle_icn as np_icn,
    triangle_ige as np_ige,
    unit_cube_tet_mesh as np_tet_mesh,
    unit_square_quad_mesh as np_quad_mesh,
    unit_square_tri_mesh as np_mesh,
)


def test_numpy_jax_quality_parity() -> None:
    topo_jax, points_jax = jax_mesh(10, 7)
    topo_np, points_np = np_mesh(10, 7, dtype=np.float32)

    p_np = NumpyDeformationParams(
        translation=np.array([0.05, -0.02], dtype=np.float32),
        scale=np.array([1.03, 0.97], dtype=np.float32),
        shear=np.array([0.04, -0.01], dtype=np.float32),
        bend=np.array([0.02, 0.03], dtype=np.float32),
    )

    x_np = np_deform(points_np, p_np)
    x_jax = jnp.asarray(x_np)

    e_np = np_energy(x_np, topo_np)
    e_jax = jax_energy(x_jax, topo_jax)
    icn_np = np_icn(x_np, topo_np.elements)
    icn_jax = np.asarray(jax_icn(x_jax, topo_jax.elements))
    ige_np = np_ige(x_np, topo_np.elements)
    ige_jax = np.asarray(jax_ige(x_jax, topo_jax.elements))

    assert np.isfinite(e_np)
    assert jnp.isfinite(e_jax)
    assert np.all(np.isfinite(icn_np))
    assert np.all(np.isfinite(ige_np))
    assert np.allclose(float(e_np), float(e_jax), rtol=5e-5, atol=5e-6)
    assert np.allclose(icn_np, icn_jax, rtol=5e-5, atol=5e-6)
    assert np.allclose(ige_np, ige_jax, rtol=5e-5, atol=5e-6)


def test_numpy_jax_quad_metric_parity() -> None:
    topo_jax, points_jax = jax_quad_mesh(9, 6)
    topo_np, points_np = np_quad_mesh(9, 6, dtype=np.float32)

    p = NumpyDeformationParams(
        translation=np.array([0.03, 0.01], dtype=np.float32),
        scale=np.array([1.01, 0.95], dtype=np.float32),
        shear=np.array([0.02, 0.01], dtype=np.float32),
        bend=np.array([0.01, 0.02], dtype=np.float32),
    )
    x_np = np_deform(points_np, p)
    x_jax = jnp.asarray(x_np)

    icn_np = np_quad_icn(x_np, topo_np.elements)
    icn_jax = np.asarray(jax_quad_icn(x_jax, topo_jax.elements))
    ige_np = np_quad_ige(x_np, topo_np.elements)
    ige_jax = np.asarray(jax_quad_ige(x_jax, topo_jax.elements))
    assert np.allclose(icn_np, icn_jax, rtol=5e-5, atol=5e-6)
    assert np.allclose(ige_np, ige_jax, rtol=5e-5, atol=5e-6)


def test_numpy_jax_tet_metric_parity() -> None:
    topo_jax, points_jax = jax_tet_mesh(4, 4, 3)
    topo_np, points_np = np_tet_mesh(4, 4, 3, dtype=np.float32)

    points_np = points_np.copy()
    points_np[:, 0] = 1.02 * points_np[:, 0] + 0.01 * points_np[:, 1]
    points_np[:, 1] = 0.97 * points_np[:, 1] + 0.02 * np.sin(2.0 * np.pi * points_np[:, 0])
    points_np[:, 2] = 1.01 * points_np[:, 2] + 0.01 * points_np[:, 0]
    x_jax = jnp.asarray(points_np)

    icn_np = np_tet_icn(points_np, topo_np.elements)
    icn_jax = np.asarray(jax_tet_icn(x_jax, topo_jax.elements))
    ige_np = np_tet_ige(points_np, topo_np.elements)
    ige_jax = np.asarray(jax_tet_ige(x_jax, topo_jax.elements))
    assert np.allclose(icn_np, icn_jax, rtol=1e-4, atol=1e-5)
    assert np.allclose(ige_np, ige_jax, rtol=1e-4, atol=1e-5)
