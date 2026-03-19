# Documentation Governance

Status: active
Version: v1.0
Date: 2026-03-14

## Base Documents

- `docs/topo/architecture.md`
- `docs/topo/documentation_governance.md`
- `docs/topo/project_overview.md`

## Scope

This repository adopts the smplJAX documentation placement standard as the governing structure for TopoJAX.

This document defines:

- the intended repo-root folder layout
- the conceptual role of each `docs/` subfolder
- the authority split between `docs/` and top-level `contracts/`
- where new documents should be placed

It does not define mathematical semantics. Those belong in `docs/topo/specs/`.

## Repository Layout

Preferred top-level structure:

- `docs/common/`: shared repository documentation
- `docs/topo/`: TopoJAX documentation authority
- `docs/smpl/`: smplJAX documentation authority
- `contracts/topo/`: TopoJAX runtime and API guarantees
- `contracts/smpl/`: smplJAX runtime and API guarantees
- `src/`: executable implementation
- `tests/`: automated test namespace
  Expected subtree:
  - `tests/common/`
  - `tests/topo/`
  - `tests/smpl/`
- `examples/topo/`: TopoJAX runnable demos and templates
- `experiments/`: exploratory namespace
  Expected subtree:
  - `experiments/common/`
  - `experiments/topo/`
  - `experiments/smpl/`
- `tools/`: repository tooling namespace
  Expected subtree:
  - `tools/common/`
  - `tools/topo/`
  - `tools/smpl/`
  Root-level convenience scripts are also allowed in `tools/` when direct access is preferable for repository operators.
- `benchmarks/`: benchmark namespace
  Expected subtree:
  - `benchmarks/common/`
  - `benchmarks/topo/`
  - `benchmarks/smpl/`
- `outputs/`: retained exported artifacts needed later
  Expected subtree:
  - `outputs/common/`
  - `outputs/topo/`
  - `outputs/smpl/`
- `private_data/`: private or upstream-governed data assets not intended for source bundle packaging
- additional standard folders such as `configs/`, `stuff/`, and `papers/` may be added when the repository starts using them directly
- `_bundles/`: generated source zip bundles and similar packaging artifacts

## Docs Layout

The governed `docs/topo/` tree is organized as:

- `governance/`
- `notation/`
- `standards/`
- `specs/`
- `objects/`
- `theory/`
- `implementation/`
- `status/`
- `reports/`
- `practical/`

Top-level files inside `docs/topo/` are reserved for high-level entry documents such as:

- `architecture.md`
- `documentation_governance.md`
- `project_overview.md`

Root-level `docs/` should only contain `README.md`, `common/`, `gmsh/`, and `smpl/`.

## Authority Split

Use this order when documents overlap:

1. `docs/topo/specs/`
2. `contracts/topo/`
3. `docs/topo/objects/`
4. `docs/topo/theory/`
5. `docs/topo/implementation/`
6. `docs/topo/status/`
7. `docs/topo/reports/`
8. `docs/topo/practical/`

Operational guarantees belong in `contracts/topo/`, not under `docs/topo/`.

## Placement Rules

- semantic definitions and invariants go in `docs/topo/specs/`
- runtime and API obligations go in `contracts/topo/`
- named runtime catalogs go in `docs/topo/objects/`
- derivations and explanations go in `docs/topo/theory/`
- workflows, benchmarks, implementation mapping, and methodology notes go in `docs/topo/implementation/`
- roadmaps, current-state summaries, and active TODOs go in `docs/topo/status/`
- produced functionality reports, coverage summaries, release reports, benchmark result packages, and official validation reports go in `docs/topo/reports/`
- user-facing practical usage guides, operator playbooks, recipes, and how-to material go in `docs/topo/practical/`
- structural and process rules go in `docs/topo/governance/` or this file

## Current Repo Mapping

High-level entry documents:

- `docs/topo/architecture.md`
- `docs/topo/documentation_governance.md`
- `docs/topo/project_overview.md`

Current governance docs:

- `docs/topo/governance/README.md`

Current notation docs:

- `docs/topo/notation/README.md`

Current standards docs:

- `docs/topo/standards/README.md`

Current specification docs:

- `docs/topo/specs/README.md`

Current object docs:

- `docs/topo/objects/README.md`

Current theory docs:

- `docs/topo/theory/README.md`

Current implementation-oriented docs:

- `docs/topo/implementation/README.md`
- `docs/topo/implementation/milestone_m2_static_generator.md`

Current status docs:

- `docs/topo/status/README.md`
- `docs/topo/status/roadmap.md`
- `docs/topo/status/release_checklist.md`
- `docs/topo/status/overview.md`

Current reports docs:

- `docs/topo/reports/README.md`

Current practical docs:

- `docs/topo/practical/README.md`

Current contract docs:

- `contracts/topo/api_contract.md`
