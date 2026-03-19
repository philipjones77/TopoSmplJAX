# Tools

This folder is the repository-wide tooling namespace.

Governed structure:

- `common/`: shared repository tooling
- `topo/`: TopoJAX-owned tooling
- `smpl/`: smplJAX-owned tooling

Root-level convenience scripts are also allowed in `tools/` when direct operator access is preferable.

Examples:

- repository packaging utilities
- one-step bootstrap scripts
- top-level maintenance entry points

Place scripts under the subtree that owns the workflow unless there is a clear reason to keep them directly under `tools/`.
