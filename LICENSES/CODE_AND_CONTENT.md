# Code And Content License Matrix

## Default Rule

Unless a more specific notice says otherwise, source code, examples, tests, tooling, contracts, and authored documentation in this repository are provided under the MIT license in the repository root.

Authority:
- [../LICENSE](/home/phili/projects/TopoSmplJAX/LICENSE)

## Path Matrix

MIT-licensed repository-authored material:

- `src/topojax/`
- `src/smpljax/`
- `src/toposmpljax/`
- `tests/topo/`
- `tests/smpl/`
- `tests/common/`
- `examples/topo/`
- `examples/smpl/`
- `tools/common/`
- `tools/topo/`
- `tools/smpl/`
- `docs/common/`
- `docs/topo/`
- `docs/smpl/`
- `contracts/topo/`
- `contracts/smpl/`
- `common/`
- top-level repository metadata and packaging files

Special boundary:

- `private_data/smpl/`
  Repository-authored README and helper metadata files are MIT unless noted otherwise.
  Third-party model files placed here are not relicensed by this repository and remain subject to their upstream terms.

## Domain Notes

`topojax`:
- Repository code and docs for the Topo backend are MIT under the root license.

`smpljax`:
- Repository code, wrappers, docs, tests, and conversion tools are MIT under the root license.
- Upstream SMPL-family model assets are not covered by the repository MIT grant unless the upstream license explicitly allows it.

## Operational Rule

Do not copy third-party model assets, scans, or vendor documentation into this repository unless:

1. their license permits redistribution, and
2. the imported path carries a clear notice describing the upstream source and governing terms.
