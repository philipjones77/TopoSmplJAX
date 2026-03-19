from __future__ import annotations

import argparse
from pathlib import Path

from smpljax.visualization import ViewerConfig, run_pyvista_viewer, run_viser_viewer
from smpljax.visualization.common import available_presets
from smpljax.visualization.pyvista_viewer import PyVistaViewerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an optional smplJAX visualization frontend.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--backend", choices=["viser", "pyvista"], default="viser")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-betas", type=int, default=10)
    parser.add_argument("--max-expression", type=int, default=10)
    parser.add_argument("--joint-limit", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--preset", choices=list(available_presets()), default="neutral")
    parser.add_argument("--state-json", type=Path, default=None)
    parser.add_argument("--export-state-json", type=Path, default=None)
    parser.add_argument("--diagnostics-json", type=Path, default=None)
    parser.add_argument("--diagnostics-jsonl", type=Path, default=None)
    parser.add_argument("--diagnostics-every-n-updates", type=int, default=1)
    parser.add_argument("--no-optimized-runtime", action="store_true")
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=900)
    args = parser.parse_args()

    common = dict(
        model_path=args.model_path,
        max_betas=args.max_betas,
        max_expression=args.max_expression,
        joint_limit=args.joint_limit,
        fps=args.fps,
        preset=args.preset,
        state_json=args.state_json,
        export_state_json=args.export_state_json,
        diagnostics_json=args.diagnostics_json,
        diagnostics_jsonl=args.diagnostics_jsonl,
        diagnostics_every_n_updates=args.diagnostics_every_n_updates,
        use_optimized_runtime=not args.no_optimized_runtime,
    )
    if args.backend == "pyvista":
        run_pyvista_viewer(
            PyVistaViewerConfig(
                **common,
                window_size=(args.window_width, args.window_height),
            )
        )
        return
    run_viser_viewer(
        ViewerConfig(
            **common,
            host=args.host,
            port=args.port,
        )
    )


if __name__ == "__main__":
    main()
