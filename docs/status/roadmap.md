# Roadmap

Status: active
Date: 2026-03-15

## Documentation Migration

- adopt the governed `docs/` layout and `contracts/` split
- keep existing root-level docs in place until they are materially revised
- move authored workflow and milestone notes into `docs/implementation/` over time

## Near-Term Priorities

- add formal mesh and object semantics under `docs/specs/`
- relocate legacy overview and milestone notes when their content is updated
- extend `contracts/` only when additional surfaces are intentionally treated as stable

## Mode 5 Planning

- define a prototype implementation plan for `fully-dynamic-remeshing`
- stage the work through transfer, surrogate expansion, unified control, and relaxed dynamic prototypes
- keep the mode-5 plan explicit about non-goals and approximation strategy

See `mode5_roadmap.md` for the concrete milestone plan and code targets.

## Mode 1 Closeout

- treat the current Mode 1 1D, 2D, and 3D fixed-topology surface as the practical baseline for create or import, optimize, export, and view workflows
- keep extending documentation and validation around the now-public domain initializers and workflow entry points
- prefer future investment in transfer, surrogate, and controller infrastructure over chasing broad parity inside Mode 1 itself

See `mode1_2d_3d_status.md` for the current-state summary and next-step implementation plan.
