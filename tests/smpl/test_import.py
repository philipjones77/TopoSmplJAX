from smpljax import (
    SMPLJAXModel,
    create,
    create_optimized,
    create_runtime,
    create_uncached,
    export_ct_mesh_payload_pose,
    export_ct_mesh_payload_template,
    export_posed_mesh,
    export_template_mesh,
    to_randomfields77_dynamic_mesh_state,
    to_randomfields77_static_domain_payload,
)
from smpljax import WarmupCoverage
from smpljax.visualization import (
    CameraPreset,
    ViewerConfig,
    ViewerPreset,
    ViewerState,
    available_presets,
    evaluate_model,
    load_viewer_state,
    preset_named,
    run_pyvista_viewer,
    run_viser_viewer,
    save_viewer_state,
)


def test_public_symbols() -> None:
    assert SMPLJAXModel is not None
    assert create is not None
    assert create_uncached is not None
    assert create_optimized is not None
    assert create_runtime is not None
    assert ViewerConfig is not None
    assert CameraPreset is not None
    assert ViewerPreset is not None
    assert ViewerState is not None
    assert available_presets is not None
    assert evaluate_model is not None
    assert preset_named is not None
    assert save_viewer_state is not None
    assert load_viewer_state is not None
    assert run_viser_viewer is not None
    assert run_pyvista_viewer is not None
    assert WarmupCoverage is not None
    assert export_template_mesh is not None
    assert export_posed_mesh is not None
    assert export_ct_mesh_payload_template is not None
    assert export_ct_mesh_payload_pose is not None
    assert to_randomfields77_static_domain_payload is not None
    assert to_randomfields77_dynamic_mesh_state is not None
