# Documentation Governance

Status: active
Version: v1.0
Date: 2026-03-14

## Base Documents

- `docs/architecture.md`
- `docs/documentation_governance.md`
- `docs/project_overview.md`

## Scope

This repository adopts the smplJAX documentation placement standard as the governing structure for GmshJAX.

This document defines:

- the intended repo-root folder layout
- the conceptual role of each `docs/` subfolder
- the authority split between `docs/` and top-level `contracts/`
- where new documents should be placed

It does not define mathematical semantics. Those belong in `docs/specs/`.

## Repository Layout

Preferred top-level structure:

- `docs/`: documentation authority and explanation
- `contracts/`: binding runtime and API guarantees
- `src/`: executable implementation
- `tests/`: conformance and regression tests
- `examples/`: runnable demos and user-facing templates
- `experiments/`: exploratory work, typically with one subfolder per experiment
- `tools/`: repository tooling and maintenance scripts
- `benchmarks/`: pytest-collected benchmark harnesses
- `outputs/`: preferred top-level artifact directory
- `results/`: legacy artifact namespace retained for existing benchmark, validation, notebook, and pytest temp outputs until migration completes
- additional standard folders such as `configs/`, `data/`, `output`, `stuff/`, and `papers/` may be added when the repository starts using them directly

## Docs Layout

The governed `docs/` tree is organized as:

- `governance/`
- `notation/`
- `standards/`
- `specs/`
- `objects/`
- `theory/`
- `implementation/`
- `status/`

Root-level `docs/` files are reserved for high-level entry documents such as:

- `architecture.md`
- `documentation_governance.md`
- `project_overview.md`

During migration, existing root-level legacy documents outside the base set may remain in place temporarily. When materially updated, they should be relocated into the governed subtree that matches their role.

## Authority Split

Use this order when documents overlap:

1. `docs/specs/`
2. `contracts/`
3. `docs/objects/`
4. `docs/theory/`
5. `docs/implementation/`
6. `docs/status/`

Operational guarantees belong in `contracts/`, not under `docs/`.

## Placement Rules

- semantic definitions and invariants go in `docs/specs/`
- runtime and API obligations go in `contracts/`
- named runtime catalogs go in `docs/objects/`
- derivations and explanations go in `docs/theory/`
- workflows, benchmarks, implementation mapping, and methodology notes go in `docs/implementation/`
- roadmaps, current-state summaries, and active TODOs go in `docs/status/`
- structural and process rules go in `docs/governance/` or this file

## Current Repo Mapping

High-level entry documents:

- `docs/architecture.md`
- `docs/documentation_governance.md`
- `docs/project_overview.md`

Current governance docs:

- `docs/governance/README.md`

Current notation docs:

- `docs/notation/README.md`

Current standards docs:

- `docs/standards/README.md`

Current specification docs:

- `docs/specs/README.md`

Current object docs:

- `docs/objects/README.md`

Current theory docs:

- `docs/theory/README.md`

Current implementation-oriented docs:

- `docs/implementation/README.md`
- `docs/milestone_m2_static_generator.md` (legacy root implementation document pending relocation)

Current status docs:

- `docs/status/README.md`
- `docs/status/roadmap.md`
- `docs/status/release_checklist.md`
- `docs/overview.md` (legacy root overview document retained during migration)

Current contract docs:

- `contracts/api_contract.md`
