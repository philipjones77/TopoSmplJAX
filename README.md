# TopoSmplJAX

`TopoSmplJAX` is a combined repository for two JAX backends:

- `topojax`: differentiable mesh generation, mesh movement, and mesh operators
- `smpljax`: fixed-topology SMPL-family body-model meshes and runtime export

The merge keeps the backend implementations separate while exposing shared mode and mesh-export concepts through the new `toposmpljax` namespace.

## Layout

- `src/topojax/`: existing TopoJAX backend
- `src/smpljax/`: imported smplJAX backend
- `src/toposmpljax/`: shared combined namespace and backend registry
- `tests/topo/`: TopoJAX and shared tests
- `tests/smpl/`: imported smplJAX tests
- `docs/common/`: shared repository docs
- `docs/topo/`: TopoJAX docs
- `docs/smpl/`: imported smplJAX docs
- `contracts/topo/`: TopoJAX contracts
- `contracts/smpl/`: imported smplJAX contracts
- `private_data/smpl/`: imported smplJAX model-data skeleton
- `examples/topo/`: TopoJAX examples
- `examples/smpl/`: imported smplJAX examples
- `tools/common/`, `tools/topo/`, `tools/smpl/`: shared and backend-owned tooling

## Install

```powershell
python -m pip install -e .[dev]
```

Visualization extras:

```powershell
python -m pip install -e .[dev,viz]
```

## Combined API

Backend registry:

```python
from toposmpljax import get_backends, build_mode_bridge

print(get_backends())
```

Topo backend mode bridge:

```python
from topojax import initialize_mode1_domain
from toposmpljax import build_mode_bridge

domain = initialize_mode1_domain("line", n=8)
bridge = build_mode_bridge("topo", domain, "fixed-topology-ad")
payload = bridge.to_randomfields77_mesh_payload()
```

SMPL backend mode bridge:

```python
from smpljax import create
from toposmpljax import build_mode_bridge

model = create("private_data/smpl/models/validated/smplx/MODEL_NAME.npz")
bridge = build_mode_bridge("smpl", model, "fixed-topology-ad")
payload = bridge.to_randomfields77_mesh_payload()
```

## Compatibility

- Existing `topojax` imports remain valid.
- Existing `smpljax` imports remain valid.
- New shared entry points live under `toposmpljax`.

## Licensing

- Repository-authored code, docs, tests, examples, tools, and contracts are MIT-licensed under `LICENSE`.
- Path-specific coverage is documented in `LICENSES/CODE_AND_CONTENT.md`.
- `private_data/smpl/` is a boundary folder: repository-authored metadata is MIT when present, but upstream SMPL-family model assets remain under their original terms and are not relicensed by this repository.

## Tests

Core combined and Topo slice:

```powershell
$env:PYTHONPATH='src'
python -m pytest -q tests/topo/test_rf77_bridge.py tests/topo/test_ad_modes.py tests/topo/test_mode1_workflow.py tests/topo/test_mode2_workflow.py
```

Imported smplJAX slice:

```powershell
$env:PYTHONPATH='src'
python -m pytest -q tests/smpl/test_import.py tests/smpl/test_api_factory.py
```
