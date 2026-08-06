"""Matrix sweep over {inputs x envmaps x shading configs} with wandb logging.

Usage:
    python -m src.research.run_matrix \\
        --inputs examples/data/rgb_demo/pexels-masi.jpg examples/data/rgb_demo/bedroom.jpg \\
        --envmaps studio_warm sunset_side overcast_top \\
        --shading-preset small \\
        --frames 24 --project infinisplat-relight

Cells are named `{input_stem}__{envmap}__{shading}` and each writes an MP4
plus a metrics dict to wandb. The encoder is loaded once and Gaussians are
cached per input, so the cost per additional cell is only a rasterization
sweep plus the shading closed-form.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.demo.infer_single_image import _resolve_device
from src.research.envmaps import resolve_envmap
from src.research.metrics import summarize_cell
from src.research.pipeline import (
    RelightSession,
    ShadingConfig,
    build_envmap_pack,
    render_variant,
)
from src.utils.io import save_video


# ---------------------------------------------------------------------------
# Shading presets
# ---------------------------------------------------------------------------


def shading_preset_small() -> list[ShadingConfig]:
    """4 configs: diffuse | +albedo-demod | +spec | +spec+ao."""
    return [
        ShadingConfig(name="diffuse", baked_gray=None),
        ShadingConfig(name="diffuse_demod05", baked_gray=0.5),
        ShadingConfig(name="diffuse_spec", enable_specular=True, roughness=0.55),
        ShadingConfig(name="full", enable_specular=True, enable_ao=True,
                      roughness=0.55, ao_strength=2.0),
    ]


def shading_preset_ablate() -> list[ShadingConfig]:
    """Wider ablation: roughness + AO strength sweep on top of the small preset."""
    base = shading_preset_small()
    ablations: list[ShadingConfig] = []
    for r in (0.15, 0.35, 0.75):
        ablations.append(ShadingConfig(name=f"spec_r{int(r*100):02d}",
                                       enable_specular=True, roughness=r))
    for s in (1.0, 3.0, 5.0):
        ablations.append(ShadingConfig(name=f"ao_s{int(s*10):02d}",
                                       enable_ao=True, ao_strength=s))
    return base + ablations


SHADING_PRESETS = {
    "small": shading_preset_small,
    "ablate": shading_preset_ablate,
}


# ---------------------------------------------------------------------------
# Cell execution
# ---------------------------------------------------------------------------


def _frames_to_uint8_stack(frames: list[torch.Tensor]) -> np.ndarray:
    """[T x [3, H, W] cpu float in [0,1]] -> uint8 THWC np."""
    out = []
    for f in frames:
        arr = f.clamp(0.0, 1.0).mul(255.0).byte().permute(1, 2, 0).contiguous().numpy()
        out.append(arr)
    return np.stack(out, axis=0)


def _cell_key(input_path: Path, envmap_name: str, shading_name: str) -> str:
    return f"{input_path.stem}__{envmap_name}__{shading_name}"


def _write_local_artifacts(
    cell_dir: Path,
    cell_key: str,
    relit_frames: list[torch.Tensor],
    baked_frame: torch.Tensor,
    metrics: dict,
    fps: int,
) -> tuple[Path, Path]:
    cell_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = cell_dir / f"{cell_key}.mp4"
    save_video(relit_frames, mp4_path, fps=fps)
    baked_png = cell_dir / f"{cell_key}__baked.png"
    _save_tensor_png(baked_frame, baked_png)
    json_path = cell_dir / f"{cell_key}.json"
    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return mp4_path, baked_png


def _save_tensor_png(frame: torch.Tensor, path: Path) -> None:
    import imageio.v2 as imageio

    arr = frame.clamp(0.0, 1.0).mul(255.0).byte().permute(1, 2, 0).contiguous().numpy()
    imageio.imwrite(str(path), arr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep {inputs x envmaps x shading} for relighting.")
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--envmaps", nargs="+", type=str, required=True,
                   help="Names from SYNTHETIC_ENVMAPS or paths to HDRIs.")
    p.add_argument("--shading-preset", choices=tuple(SHADING_PRESETS), default="small")
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--max-long-edge", type=int, default=768)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/infinisplat_rgb.ckpt"))
    p.add_argument("--experiment", type=str, default="infinisplat_hypersim_rgb")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/research"))
    p.add_argument("--project", type=str, default="infinisplat-relight")
    p.add_argument("--entity", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--offline", action="store_true", help="Force wandb offline mode.")
    p.add_argument("--no-wandb", action="store_true", help="Skip wandb entirely.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    use_wandb = not args.no_wandb
    wandb = None
    run = None
    if use_wandb:
        import wandb as _wandb
        wandb = _wandb
        run_name = args.run_name or time.strftime("relight-%Y%m%d-%H%M%S")
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=run_name,
            config={
                "checkpoint": str(args.checkpoint),
                "experiment": args.experiment,
                "frames": args.frames,
                "max_long_edge": args.max_long_edge,
                "shading_preset": args.shading_preset,
                "envmaps": args.envmaps,
                "inputs": [str(p) for p in args.inputs],
            },
        )
        print(f"[matrix] wandb run: {run.url if hasattr(run, 'url') else '(offline)'}")

    device = _resolve_device(args.device)
    output_root = Path(args.output_dir) / time.strftime("run-%Y%m%d-%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[matrix] artifacts -> {output_root}")

    print(f"[matrix] loading session (checkpoint={args.checkpoint})")
    session = RelightSession(
        checkpoint=args.checkpoint,
        experiment=args.experiment,
        device=args.device,
    )

    shading_configs = SHADING_PRESETS[args.shading_preset]()
    envmap_specs = args.envmaps
    input_paths = [Path(p) for p in args.inputs]
    total = len(input_paths) * len(envmap_specs) * len(shading_configs)
    print(f"[matrix] cells: {len(input_paths)} x {len(envmap_specs)} x {len(shading_configs)} = {total}")

    # Pre-build envmap packs (shared across inputs).
    need_specular = any(s.enable_specular for s in shading_configs)
    envmap_packs = []
    for spec in envmap_specs:
        name, env = resolve_envmap(spec, device=device)
        envmap_packs.append(build_envmap_pack(
            name=name, envmap=env, device=device,
            specular_levels=4, need_specular=need_specular,
        ))
        print(f"[matrix]   envmap ready: {name}  shape={tuple(env.shape)}")

    aggregate_metrics: list[dict] = []
    cell_idx = 0
    t_start = time.time()

    for input_path in input_paths:
        print(f"[matrix] encoding {input_path}")
        scene = session.encode(input_path=input_path, max_long_edge=args.max_long_edge)
        print(f"[matrix]   gaussians={scene.num_gaussians}  "
              f"reliable_normals={scene.reliable_mask.float().mean().item():.1%}")

        # Cache a diffuse baseline per (input, envmap) so we can compute specular_delta.
        diffuse_cache: dict[str, np.ndarray] = {}

        for pack in envmap_packs:
            for shading in shading_configs:
                cell_idx += 1
                key = _cell_key(input_path, pack.name, shading.name)
                cell_start = time.time()
                relit, baked = render_variant(
                    session=session, scene=scene, envmap_pack=pack,
                    shading=shading, num_frames=args.frames,
                )

                relit_stack = _frames_to_uint8_stack(relit)
                baked_uint8 = baked.clamp(0, 1).mul(255).byte().permute(1, 2, 0).contiguous().numpy()

                diffuse_stack = None
                if shading.name == "diffuse":
                    diffuse_cache[pack.name] = relit_stack
                elif pack.name in diffuse_cache:
                    diffuse_stack = diffuse_cache[pack.name]

                metrics = summarize_cell(relit_stack, baked_uint8, diffuse_frames=diffuse_stack)
                metrics.update({
                    "cell": key,
                    "input": input_path.stem,
                    "envmap": pack.name,
                    "shading": shading.name,
                    "shading_cfg": asdict(shading),
                    "num_gaussians": scene.num_gaussians,
                    "reliable_normal_frac": float(scene.reliable_mask.float().mean().item()),
                    "seconds": time.time() - cell_start,
                })

                cell_dir = output_root / input_path.stem / pack.name
                mp4_path, baked_png = _write_local_artifacts(
                    cell_dir=cell_dir, cell_key=key,
                    relit_frames=relit, baked_frame=baked,
                    metrics=metrics, fps=args.fps,
                )

                print(f"[matrix] [{cell_idx}/{total}] {key}  "
                      f"tstd={metrics['temporal_std']:.3f}  "
                      f"eff={metrics['relight_effectiveness']:.3f}  "
                      f"time={metrics['seconds']:.1f}s")

                if wandb is not None:
                    log_payload = {k: v for k, v in metrics.items()
                                   if isinstance(v, (int, float, str)) and not isinstance(v, bool)}
                    log_payload[f"video/{key}"] = wandb.Video(str(mp4_path), fps=args.fps, format="mp4")
                    log_payload[f"baked/{key}"] = wandb.Image(str(baked_png))
                    wandb.log(log_payload, step=cell_idx)

                aggregate_metrics.append(metrics)

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(aggregate_metrics, indent=2, sort_keys=True))
    print(f"[matrix] wrote {summary_path}  elapsed={time.time() - t_start:.1f}s")

    if wandb is not None:
        # Also write a summary table for cross-cell comparison.
        columns = ["cell", "input", "envmap", "shading",
                   "temporal_std", "relight_effectiveness",
                   "brightness_mean", "silhouette_drift",
                   "specular_delta", "seconds"]
        data = []
        for m in aggregate_metrics:
            data.append([m.get(c) for c in columns])
        table = wandb.Table(columns=columns, data=data)
        wandb.log({"summary/table": table})
        wandb.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
