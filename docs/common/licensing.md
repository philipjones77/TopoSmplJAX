# Repository Licensing

This document defines the licensing structure for the combined `TopoSmplJAX` repository.

## Root Rule

Repository-authored code and documentation are MIT-licensed under the root [LICENSE](/home/phili/projects/TopoSmplJAX/LICENSE) unless a narrower notice says otherwise.

## Domain Mapping

Topo slice:
- `src/topojax/`
- `docs/topo/`
- `contracts/topo/`
- `examples/topo/`
- `tools/topo/`
- `tests/topo/`

SMPL slice:
- `src/smpljax/`
- `docs/smpl/`
- `contracts/smpl/`
- `examples/smpl/`
- `tools/smpl/`
- `tests/smpl/`

Shared slice:
- `src/toposmpljax/`
- `docs/common/`
- `tools/common/`
- `tests/common/`
- `common/`

The authoritative path matrix lives in [../../LICENSES/CODE_AND_CONTENT.md](/home/phili/projects/TopoSmplJAX/LICENSES/CODE_AND_CONTENT.md).

## Asset Boundary

`private_data/smpl/` is not a blanket MIT asset bucket. Repository-authored metadata and small synthetic fixtures may be MIT, but third-party SMPL-family model files remain governed by their upstream terms.

See [../../LICENSES/ASSET_BOUNDARY.md](/home/phili/projects/TopoSmplJAX/LICENSES/ASSET_BOUNDARY.md).
