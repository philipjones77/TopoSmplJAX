import jax.numpy as jnp

from smpljax.body_models import SMPLJAXModel
from smpljax.utils import SMPLModelData


def main() -> None:
    v = 8
    j = 4
    b = 3
    model = SMPLJAXModel(
        data=SMPLModelData(
            v_template=jnp.zeros((v, 3)),
            shapedirs=jnp.zeros((v, 3, b)),
            posedirs=jnp.zeros(((j - 1) * 9, v * 3)),
            j_regressor=jnp.ones((j, v)) / v,
            parents=jnp.array([-1, 0, 1, 2], dtype=jnp.int32),
            lbs_weights=jnp.ones((v, j)) / j,
        )
    )

    out = model(
        betas=jnp.zeros((1, b)),
        body_pose=jnp.zeros((1, j - 1, 3)),
        global_orient=jnp.zeros((1, 1, 3)),
    )
    print("vertices:", out.vertices.shape)
    print("joints:", out.joints.shape)


if __name__ == "__main__":
    main()
