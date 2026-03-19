import jax
import jax.numpy as jnp
import numpy as np

from smpljax import (
    OptimizedSMPLJAX,
    SMPLJAXModel,
    export_ct_mesh_payload_pose,
    export_ct_mesh_payload_template,
    export_posed_mesh,
    export_template_mesh,
    to_randomfields77_dynamic_mesh_state,
    to_randomfields77_static_domain_payload,
)
from smpljax.utils import SMPLModelData


def _mesh_model_data() -> SMPLModelData:
    num_joints = 2
    return SMPLModelData(
        v_template=jnp.zeros((4, 3), dtype=jnp.float32),
        shapedirs=jnp.zeros((4, 3, 1), dtype=jnp.float32),
        posedirs=jnp.zeros((9, 12), dtype=jnp.float32),
        j_regressor=jnp.ones((num_joints, 4), dtype=jnp.float32) / 4.0,
        parents=jnp.array([-1, 0], dtype=jnp.int32),
        lbs_weights=jnp.ones((4, num_joints), dtype=jnp.float32) / float(num_joints),
        num_body_joints=1,
        model_family="smpl",
        model_variant="demo",
        gender="neutral",
        faces_tensor=jnp.array([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32),
    )


def test_template_mesh_export_exposes_fixed_topology_metadata() -> None:
    model = SMPLJAXModel(data=_mesh_model_data())

    payload = model.export_template_mesh()

    assert isinstance(payload["nodes"], jax.Array)
    assert isinstance(payload["faces"], jax.Array)
    assert payload["nodes"].shape == (4, 3)
    assert payload["faces"].shape == (2, 3)
    assert payload["n_vertices"] == 4
    assert payload["n_faces"] == 2
    assert payload["metadata"]["model_family"] == "smpl"
    assert payload["metadata"]["model_variant"] == "demo"
    assert payload["metadata"]["output_kind"] == "template"
    assert payload["metadata"]["fixed_topology"] is True


def test_posed_mesh_export_is_shape_stable_and_topology_stable() -> None:
    model = SMPLJAXModel(data=_mesh_model_data())
    params0 = {
        "betas": jnp.zeros((1, 1), dtype=jnp.float32),
        "body_pose": jnp.zeros((1, 1, 3), dtype=jnp.float32),
        "transl": jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32),
    }
    params1 = {
        "betas": jnp.ones((1, 1), dtype=jnp.float32),
        "body_pose": jnp.zeros((1, 1, 3), dtype=jnp.float32),
        "transl": jnp.array([[4.0, 5.0, 6.0]], dtype=jnp.float32),
    }

    payload0 = export_posed_mesh(model, params0)
    payload1 = model.export_posed_mesh(params1)

    assert payload0["nodes"].shape == (1, 4, 3)
    assert payload1["nodes"].shape == (1, 4, 3)
    assert jnp.array_equal(payload0["faces"], payload1["faces"])
    assert payload0["topology_id"] == payload1["topology_id"]
    np.testing.assert_allclose(np.asarray(payload0["nodes"][0, 0]), np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(payload1["nodes"][0, 0]), np.array([4.0, 5.0, 6.0], dtype=np.float32))


def test_posed_mesh_export_is_ad_safe() -> None:
    model = SMPLJAXModel(data=_mesh_model_data())
    betas = jnp.zeros((1, 1), dtype=jnp.float32)
    body_pose = jnp.zeros((1, 1, 3), dtype=jnp.float32)

    def loss_fn(transl: jax.Array) -> jax.Array:
        payload = model.export_posed_mesh(
            {
                "betas": betas,
                "body_pose": body_pose,
                "transl": transl,
            }
        )
        return jnp.sum(payload["nodes"])

    grad = jax.grad(loss_fn)(jnp.zeros((1, 3), dtype=jnp.float32))
    np.testing.assert_allclose(np.asarray(grad), np.full((1, 3), 4.0, dtype=np.float32))


def test_optimized_runtime_mesh_export_supports_mapping_and_forward_inputs() -> None:
    runtime = OptimizedSMPLJAX(data=_mesh_model_data())
    params = {
        "betas": jnp.zeros((2, 1), dtype=jnp.float32),
        "body_pose": jnp.zeros((2, 1, 3), dtype=jnp.float32),
        "transl": jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=jnp.float32),
    }

    payload0 = runtime.export_posed_mesh(params)
    prepared = runtime.prepare_inputs(batch_size=2, **params)
    payload1 = export_posed_mesh(runtime, prepared)

    assert payload0["nodes"].shape == (2, 4, 3)
    assert payload1["nodes"].shape == (2, 4, 3)
    assert payload0["topology_id"] == payload1["topology_id"]
    assert payload0["metadata"]["jax_backend"] in {"cpu", "gpu", "tpu"}
    assert payload0["metadata"]["output_kind"] == "posed"


def test_ct_payload_helpers_match_randomfields77_bridge_aliases() -> None:
    model = SMPLJAXModel(data=_mesh_model_data())
    params = {
        "betas": jnp.zeros((1, 1), dtype=jnp.float32),
        "body_pose": jnp.zeros((1, 1, 3), dtype=jnp.float32),
    }

    template_payload = export_ct_mesh_payload_template(model)
    template_alias = to_randomfields77_static_domain_payload(model)
    posed_payload = export_ct_mesh_payload_pose(model, params)
    posed_alias = to_randomfields77_dynamic_mesh_state(model, params)

    assert jnp.array_equal(template_payload["cells"], template_payload["faces"])
    assert jnp.array_equal(template_payload["cells"], template_alias["cells"])
    assert jnp.array_equal(posed_payload["cells"], posed_alias["cells"])
    assert posed_payload["metadata"]["output_kind"] == "posed"
    assert template_payload["metadata"]["output_kind"] == "template"


def test_load_model_npz_preserves_mesh_export_metadata(tmp_path) -> None:
    from smpljax.io import load_model_npz

    out_path = tmp_path / "model.npz"
    np.savez(
        out_path,
        v_template=np.zeros((4, 3), dtype=np.float32),
        shapedirs=np.zeros((4, 3, 1), dtype=np.float32),
        posedirs=np.zeros((9, 12), dtype=np.float32),
        J_regressor=np.zeros((2, 4), dtype=np.float32),
        weights=np.ones((4, 2), dtype=np.float32) / 2.0,
        parents=np.array([-1, 0], dtype=np.int32),
        faces_tensor=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        model_family=np.array("smpl"),
        model_variant=np.array("neutral_demo"),
        gender=np.array("neutral"),
    )

    model_data = load_model_npz(out_path)

    assert model_data.model_family == "smpl"
    assert model_data.model_variant == "neutral_demo"
    assert model_data.gender == "neutral"
    assert export_template_mesh(SMPLJAXModel(data=model_data))["metadata"]["gender"] == "neutral"
