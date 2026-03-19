import numpy as np
import jax.numpy as jnp

from topojax.mesh.operators import triangle_icn, triangle_ige


def test_icn_ige_flip_sign_on_inverted_triangle() -> None:
    good = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.2, 0.8]], dtype=jnp.float32)
    bad = jnp.asarray([[0.0, 0.0], [0.2, 0.8], [1.0, 0.0]], dtype=jnp.float32)
    elems = jnp.asarray([[0, 1, 2]], dtype=jnp.int32)

    icn_good = float(triangle_icn(good, elems)[0])
    icn_bad = float(triangle_icn(bad, elems)[0])
    ige_good = float(triangle_ige(good, elems)[0])
    ige_bad = float(triangle_ige(bad, elems)[0])

    assert icn_good > 0.0
    assert icn_bad < 0.0
    assert ige_good > 0.0
    assert ige_bad < 0.0


def test_ige_is_near_one_for_equilateral_triangle() -> None:
    h = np.sqrt(3.0) / 2.0
    pts = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, h]], dtype=jnp.float32)
    elems = jnp.asarray([[0, 1, 2]], dtype=jnp.int32)
    val = float(triangle_ige(pts, elems)[0])
    assert abs(val - 1.0) < 1.0e-5
