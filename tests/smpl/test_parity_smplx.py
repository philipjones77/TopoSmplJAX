from __future__ import annotations

import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from smpljax.body_models import SMPLJAXModel


def _maybe_import_torch():
    try:
        import torch  # type: ignore
    except Exception:
        return None
    return torch


def _build_ref_model(smplx_mod, model_path: str, model_type: str, gender: str):
    kwargs: dict[str, object] = {"gender": gender}
    if model_type == "smplx":
        kwargs.update({"num_betas": 10, "num_expression_coeffs": 10, "use_pca": False, "flat_hand_mean": True})
        return smplx_mod.SMPLXLayer(model_path=model_path, **kwargs)
    if model_type == "smplh":
        kwargs.update({"num_betas": 10, "use_pca": False, "flat_hand_mean": True})
        return smplx_mod.SMPLHLayer(model_path=model_path, **kwargs)
    if model_type == "smpl":
        kwargs.update({"num_betas": 10})
        return smplx_mod.SMPLLayer(model_path=model_path, **kwargs)
    raise ValueError(model_type)


@pytest.mark.parity
@pytest.mark.parametrize("model_type", ["smpl", "smplh", "smplx"])
@pytest.mark.parametrize("gender", ["neutral", "female", "male"])
def test_smpl_family_parity_local_repo(model_type: str, gender: str) -> None:
    model_path = os.getenv("SMPLX_MODEL_PATH")
    repo_path = os.getenv("SMPLX_REPO_PATH", r"C:\Users\phili\OneDrive\Documents\GitHub\smplx")
    if not model_path:
        pytest.skip("Set SMPLX_MODEL_PATH to run parity tests")
    if not Path(repo_path).exists():
        pytest.skip(f"SMPLX_REPO_PATH not found: {repo_path}")

    torch = _maybe_import_torch()
    if torch is None:
        pytest.skip("torch is required for parity tests")

    from smpljax.reference_smplx import (
        infer_spec,
        ref_forward_kwargs,
        sample_inputs,
        to_model_data,
    )

    sys.path.insert(0, repo_path)
    import smplx  # type: ignore

    try:
        ref = _build_ref_model(smplx, model_path=model_path, model_type=model_type, gender=gender)
    except FileNotFoundError:
        pytest.skip(f"Model assets missing for {model_type}/{gender}")
    ref.eval()
    spec = infer_spec(ref, model_type=model_type)
    model = SMPLJAXModel(data=to_model_data(ref_model=ref, spec=spec))
    sample = sample_inputs(spec=spec, batch_size=1, seed=7)

    with torch.no_grad():
        ref_out = ref(**ref_forward_kwargs(spec=spec, sample=sample))
    out = model(**{k: jnp.asarray(v) for k, v in sample.items()})

    np.testing.assert_allclose(np.asarray(out.vertices), ref_out.vertices.detach().cpu().numpy(), rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(np.asarray(out.joints), ref_out.joints.detach().cpu().numpy(), rtol=5e-5, atol=5e-5)
