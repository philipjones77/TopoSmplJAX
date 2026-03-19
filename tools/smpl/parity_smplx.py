from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch


def _load_local_smplx(repo_path: Path):
    sys.path.insert(0, str(repo_path))
    import smplx  # type: ignore

    return smplx


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
    raise ValueError(f"Unsupported model_type={model_type}")


def _run_case(
    smplx_mod,
    model_path: str,
    model_type: str,
    gender: str,
    seed: int,
    batch_size: int,
    atol: float,
    rtol: float,
) -> tuple[float, float]:
    from smpljax.body_models import SMPLJAXModel
    from smpljax.reference_smplx import infer_spec, ref_forward_kwargs, sample_inputs, to_model_data

    ref = _build_ref_model(smplx_mod, model_path=model_path, model_type=model_type, gender=gender)
    ref.eval()
    spec = infer_spec(ref, model_type=model_type)
    model = SMPLJAXModel(data=to_model_data(ref, spec))

    sample = sample_inputs(spec=spec, batch_size=batch_size, seed=seed)
    with torch.no_grad():
        ref_out = ref(**ref_forward_kwargs(spec=spec, sample=sample))

    out = model(**{k: jnp.asarray(v) for k, v in sample.items()})
    v_ref = ref_out.vertices.detach().cpu().numpy()
    j_ref = ref_out.joints.detach().cpu().numpy()
    v = np.asarray(out.vertices)
    j = np.asarray(out.joints)

    v_max = float(np.max(np.abs(v - v_ref)))
    j_max = float(np.max(np.abs(j - j_ref)))
    ok = np.allclose(v, v_ref, rtol=rtol, atol=atol) and np.allclose(j, j_ref, rtol=rtol, atol=atol)
    if not ok:
        raise AssertionError(
            f"Parity failed for model={model_type} gender={gender} seed={seed} (v_max={v_max:.6e}, j_max={j_max:.6e})"
        )
    return v_max, j_max


def main() -> None:
    parser = argparse.ArgumentParser(description="Matrix parity checks against local smplx.")
    parser.add_argument("--smplx-repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-types", nargs="+", default=["smpl", "smplh", "smplx"])
    parser.add_argument("--genders", nargs="+", default=["neutral", "female", "male"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--atol", type=float, default=5e-5)
    parser.add_argument("--rtol", type=float, default=5e-5)
    args = parser.parse_args()

    smplx_mod = _load_local_smplx(args.smplx_repo)
    torch.manual_seed(0)
    np.random.seed(0)

    worst_v = 0.0
    worst_j = 0.0
    ran = 0
    for model_type in args.model_types:
        for gender in args.genders:
            for seed in args.seeds:
                try:
                    v_max, j_max = _run_case(
                        smplx_mod=smplx_mod,
                        model_path=str(args.model_path),
                        model_type=model_type.lower(),
                        gender=gender.lower(),
                        seed=seed,
                        batch_size=args.batch_size,
                        atol=args.atol,
                        rtol=args.rtol,
                    )
                except FileNotFoundError:
                    print(f"skip: missing model files for {model_type}/{gender}")
                    continue
                ran += 1
                worst_v = max(worst_v, v_max)
                worst_j = max(worst_j, j_max)
                print(f"ok: {model_type}/{gender}/seed={seed} v_max={v_max:.3e} j_max={j_max:.3e}")

    if ran == 0:
        raise SystemExit("No parity cases were run. Check model-path content.")
    print(f"parity matrix passed ({ran} cases), worst vertex={worst_v:.6e}, worst joint={worst_j:.6e}")


if __name__ == "__main__":
    main()
