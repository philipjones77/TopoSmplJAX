# Release Checklist

Status: active
Date: 2026-03-14

## Documentation

- confirm `docs/topo/architecture.md`, `docs/topo/documentation_governance.md`, and `docs/topo/project_overview.md` are current
- update `contracts/topo/` when public API guarantees change
- place any new documents under the governed subtree instead of adding new root-level docs casually

## Validation

- run the standard pytest suite for repository changes
- refresh examples or notebooks when a user-facing workflow materially changes
- update changelog or release notes when behavior visible to users changes
