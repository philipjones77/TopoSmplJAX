# Model Assets

This repository uses the following local asset split:

- `validated/`
  canonical local SMPL-family model files that passed `tools/smpl/validate_model.py`
  and should be used by examples, parity checks, and manual workflows
- `quarantine/`
  raw, unvalidated, or manually dropped files that should not be treated as canonical

Recommended workflow:
1. convert official assets with `tools/smpl/convert_smplx_npz.py`
2. validate or promote with `tools/smpl/validate_model.py`
3. use files from `private_data/smpl/models/validated/`
4. inspect approved files with `tools/smpl/list_validated_models.py`

Licensing rule:

- do not assume files in this folder are MIT-licensed
- converted or copied SMPL-family assets remain subject to their upstream terms unless an individual file states otherwise
- this repository only licenses its own code, documentation, and metadata
