import jax.numpy as jnp

from topojax.mesh.generators import unit_cube_points, unit_square_points
from topojax.numpy_impl import unit_square_points as np_unit_square_points
from topojax.runtime import get_runtime_precision, set_runtime_precision


def test_runtime_precision_float32() -> None:
    set_runtime_precision("float32")
    jp = unit_square_points(6, 5)
    jc = unit_cube_points(4, 4, 4)
    np_pts = np_unit_square_points(6, 5)
    assert jp.dtype == jnp.float32
    assert jc.dtype == jnp.float32
    assert str(np_pts.dtype) == "float32"


def test_runtime_precision_float64() -> None:
    set_runtime_precision("float64")
    jp = unit_square_points(6, 5)
    jc = unit_cube_points(4, 4, 4)
    np_pts = np_unit_square_points(6, 5)
    assert jp.dtype == jnp.float64
    assert jc.dtype == jnp.float64
    assert str(np_pts.dtype) == "float64"
    assert get_runtime_precision() == "float64"

    # Reset default for the rest of the suite.
    set_runtime_precision("float32")
