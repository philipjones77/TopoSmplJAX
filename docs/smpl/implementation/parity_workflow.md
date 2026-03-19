# Parity Workflow (Version 1.0.0)

This project keeps parity checks gated because they require:
- A local checkout of the reference `smplx` repository.
- Licensed SMPL-family model assets that cannot be stored in CI.

## Normal CI

Normal CI runs on both Windows and Linux and excludes parity:

```powershell
python -m pytest -q -m "not parity"
```

## Gated Parity Contract

Parity uses the same environment contract everywhere:

- `SMPLX_REPO_PATH`: absolute path to a local `smplx` checkout.
- `SMPLX_MODEL_PATH`: absolute path to the local validated SMPL-family model root or file.

The gated parity workflow lives in:
- `.github/workflows/parity-gated.yml`

It runs on a self-hosted runner because the runner must already have access to both dependencies above.

## Local Commands

Pytest parity:

```powershell
$env:PYTHONPATH='src'
$env:SMPLX_REPO_PATH='C:\path\to\smplx'
$env:SMPLX_MODEL_PATH='C:\path\to\repo\data\models\validated'
python -m pytest -q -m parity
```

Matrix parity script:

```powershell
$env:PYTHONPATH='src'
python tools/smpl/parity_smplx.py `
  --smplx-repo C:\path\to\smplx `
  --model-path C:\path\to\repo\data\models\validated `
  --model-types smpl smplh smplx `
  --genders neutral female male `
  --seeds 0 1 2
```

Linux shell form:

```bash
PYTHONPATH=src \
SMPLX_REPO_PATH=/path/to/smplx \
SMPLX_MODEL_PATH=/path/to/repo/private_data/smpl/models/validated \
python -m pytest -q -m parity
```

## Expectations

- Normal CI must stay green without parity assets.
- Gated parity should be reproducible on both Windows and Linux self-hosted runners.
- Failures should report the exact model type, gender, and seed that diverged.
