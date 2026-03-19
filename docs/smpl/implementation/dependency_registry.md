# Dependency Registry (Version 1.0.0)

## Runtime
- `jax`
- `jaxlib`
- `numpy`

## Testing
- `pytest`

## Optional Visualization
- `viser`
- `pyvista`
- `trimesh`
- `ipywidgets` (notebook controls)

## Interop / Reference (local parity workflows)
- Local `smplx` checkout (PyTorch reference implementation).

## Notes
- Optional dependencies should not be required for core runtime tests.
- Visualization imports are lazy and should fail with actionable messages if missing.
