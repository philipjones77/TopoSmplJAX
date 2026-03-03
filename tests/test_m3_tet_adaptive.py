from pathlib import Path

from gmshjax.mesh.adaptive_tet import adaptive_remesh_tet
from gmshjax.mesh.generators import project_cube_points_to_sphere
from gmshjax.mesh.topology import unit_cube_tet_mesh


def test_adaptive_tet_loop_writes_snapshots(tmp_path: Path) -> None:
    topo, pts = unit_cube_tet_mesh(6, 6, 5)
    sph = project_cube_points_to_sphere(pts, radius=1.0)
    out, hist = adaptive_remesh_tet(
        sph,
        topo.elements,
        max_nodes=12000,
        max_elements=24000,
        max_iters=3,
        target_volume=0.05,
        target_mean_icn=0.0,
        snapshot_dir=tmp_path,
    )
    assert len(hist) >= 1
    assert out.node_count >= sph.shape[0]
    assert len(list(tmp_path.glob("iter_*.npz"))) >= 1
