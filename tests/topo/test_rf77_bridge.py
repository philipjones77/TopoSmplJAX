import jax.numpy as jnp

from topojax.ad.modes import MeshMovementMode
from topojax.ad.workflow import initialize_mode1_domain, initialize_mode2_domain, run_mode2_restart_workflow
from topojax.mesh.topology import unit_square_tri_mesh
from topojax.rf77 import (
    build_mode1_randomfields77_bridge,
    build_mode2_randomfields77_bridge,
    build_mode3_randomfields77_bridge,
    build_mode4_randomfields77_bridge,
    build_mode5_randomfields77_bridge,
)


def test_mode1_rf77_payload_matches_expected_shape_contract() -> None:
    outer = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    domain = initialize_mode1_domain("polygon", outer_boundary=outer, target_edge_size=0.25)
    bridge = build_mode1_randomfields77_bridge(domain)

    payload = bridge.to_randomfields77_mesh_payload()
    report = bridge.runtime_report()

    assert payload["mesh_storage"]["canonical_format"] == "rf77.mesh_payload.v1"
    assert payload["metadata"]["topology_fixed"] is True
    assert payload["metadata"]["element_family"] == "triangle"
    assert payload["facets"] is not None
    assert payload["facet_tags"] is not None
    assert "outer_boundary" in payload["metadata"]["boundary_groups"]
    assert report["mode"] == MeshMovementMode.FIXED_TOPOLOGY.value
    assert report["capability_flags"]["supports_graph_export"] is True


def test_mode2_rf77_bridge_preserves_mode_distinction_and_phase_contract(tmp_path) -> None:
    outer = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    domain = initialize_mode2_domain("polygon", outer_boundary=outer, target_edge_size=0.25)
    run = run_mode2_restart_workflow(
        domain,
        output_dir=tmp_path,
        prefix="rf77_mode2",
        cycles=1,
        optimization_steps=4,
        optimization_step_size=0.02,
    )
    bridge = build_mode2_randomfields77_bridge(run)

    payload = bridge.to_randomfields77_mesh_payload()
    report = bridge.runtime_report()

    assert payload["metadata"]["topology_fixed"] is False
    assert payload["metadata"]["boundary_groups"] is not None
    assert report["mode"] == MeshMovementMode.REMESH_RESTART.value
    assert "restart_phases" in report["mode_contract"]["hook_points"][1] or report["mode_contract"]["mode"] == MeshMovementMode.REMESH_RESTART.value
    assert bridge.shape_signature()["element_family"] == "triangle"


def test_rf77_operator_exports_are_cached_and_applicable() -> None:
    topology, points = unit_square_tri_mesh(4, 4)
    bridge = build_mode3_randomfields77_bridge(points, topology)

    lap = bridge.graph_laplacian_dense(weight_mode="unit")
    sparse = bridge.graph_laplacian_sparse(weight_mode="unit")
    mass = bridge.mass_matrix()
    stiff = bridge.stiffness_matrix()
    x = jnp.ones((points.shape[0],), dtype=points.dtype)

    applied = bridge.operator_apply(x, operator="laplacian")
    cache = bridge.cached_operator_state()

    assert lap.shape == (points.shape[0], points.shape[0])
    assert sparse.shape == lap.shape
    assert mass.shape == lap.shape
    assert stiff.shape == lap.shape
    assert applied.shape == (points.shape[0],)
    assert "laplacian_dense:unit" in cache
    assert "mass" in cache
    assert "stiffness" in cache


def test_rf77_all_five_modes_are_hookable_and_reviewable() -> None:
    topology, points = unit_square_tri_mesh(3, 3)
    mode3 = build_mode3_randomfields77_bridge(points, topology, candidate_graph={"kind": "quad-diagonal-candidates"})
    mode4 = build_mode4_randomfields77_bridge(points, topology, forward_state={"hard_choices": []}, backward_surrogate={"kind": "softmax"})
    mode5 = build_mode5_randomfields77_bridge(
        points,
        topology,
        dynamic_remesh_fn="hook:dynamic_remesh",
        state_transfer_fn="hook:state_transfer",
        controller_fn="hook:controller",
    )

    for bridge, mode in (
        (mode3, MeshMovementMode.SOFT_CONNECTIVITY),
        (mode4, MeshMovementMode.STRAIGHT_THROUGH),
        (mode5, MeshMovementMode.FULLY_DYNAMIC),
    ):
        report = bridge.runtime_report()
        payload = bridge.to_randomfields77_mesh_payload()
        batched = bridge.batch_to_randomfields77_dynamic_mesh_state([{"step": 0}, {"step": 1}])
        assert report["mode"] == mode.value
        assert report["mode_contract"]["hook_points"]
        assert payload["nodes"].shape == points.shape
        assert batched["nodes"].shape == (2, points.shape[0], points.shape[1])
        assert payload["metadata"]["topology_fixed"] is False
