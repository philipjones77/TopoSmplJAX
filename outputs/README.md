# Outputs

This folder stores retained exported artifacts that are needed later for reproducibility, inspection, promotion, or downstream workflows.

Governed structure:

- `common/`: shared outputs that are not owned by a single backend
- `topo/`: TopoJAX-owned artifacts
- `smpl/`: smplJAX-owned artifacts

Do not create backend-specific output folders directly under the repository root. Place them under this namespace.
