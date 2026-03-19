from topojax.ad.modes import MeshMovementMode, get_mesh_movement_mode, get_mesh_movement_modes


def test_mesh_movement_mode_catalog_has_five_modes() -> None:
    modes = get_mesh_movement_modes()
    assert len(modes) == 5
    assert [spec.mode for spec in modes] == [
        MeshMovementMode.FIXED_TOPOLOGY,
        MeshMovementMode.REMESH_RESTART,
        MeshMovementMode.SOFT_CONNECTIVITY,
        MeshMovementMode.STRAIGHT_THROUGH,
        MeshMovementMode.FULLY_DYNAMIC,
    ]


def test_fully_dynamic_mode_is_aspirational() -> None:
    spec = get_mesh_movement_mode(MeshMovementMode.FULLY_DYNAMIC)
    assert spec.geometry_motion
    assert spec.autodiff_status == "aspirational"
    assert spec.implementation_status == "planned"