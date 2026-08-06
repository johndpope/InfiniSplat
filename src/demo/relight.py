"""Envmap-based relighting demo for a single InfiniSplat scene.

Encodes an input image once, then produces a video where the camera stays at
the original view but the environment map rotates around the world-up axis.
Diffuse-only (SH-9) IBL for phase 1. Optional albedo demodulation and
side-by-side baked-vs-relit comparison.

Usage:
    python -m src.demo.relight --input examples/data/rgb_demo/pexels-masi.jpg
    python -m src.demo.relight --input path/to.jpg --envmap path/to.exr
    python -m src.demo.relight --input path/to.jpg --compare
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from einops import rearrange

from src.demo.infer_single_image import (
    DEFAULT_FOCAL_35MM_MM,
    _resolve_checkpoint_path,
    _resolve_device,
    filter_final_gaussian_floaters,
    load_demo_config,
    load_demo_image_bundle,
    load_demo_model,
    normalize_intrinsics,
    run_single_image_inference,
    scale_intrinsics_px,
)
from src.model.decoder import Decoder
from src.model.decoder.decoder_gsplat import is_gsplat_available
from src.utils.gaussians import Gaussians3D
from src.utils.io import save_video
from src.utils.lighting import (
    build_specular_mip_chain,
    compute_density_ao,
    gaussian_normals,
    face_normals_toward,
    load_envmap,
    project_envmap_to_sh9,
    rotation_matrix_y,
    shade_gaussians,
    synthesize_directional_envmap,
)

MODE_EXPERIMENTS = {"rgb": "infinisplat_hypersim_rgb"}
MODE_CHECKPOINTS = {"rgb": Path("checkpoints/infinisplat_rgb.ckpt")}

# InfiniSplat runs the encoder in an OpenCV camera frame: +x right, +y down,
# +z forward. Physical "up" is therefore -y_world. SH-9 conventions assume
# +y = up, so we flip y and z to move world-space normals into light space.
WORLD_TO_LIGHT_OPENCV = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)


@dataclass(frozen=True)
class RenderTargets:
    """Static-view rendering parameters shared across every relight frame."""

    extrinsics: torch.Tensor       # [1, 1, 4, 4], world-to-camera
    intrinsics_norm: torch.Tensor  # [1, 1, 3, 3], normalized
    image_shape: tuple[int, int]   # (H, W)
    camera_position: torch.Tensor  # [3], world space


def _replace_colors(gaussians: Gaussians3D, colors: torch.Tensor) -> Gaussians3D:
    """Return a new Gaussians3D with the supplied per-Gaussian RGB colors."""
    return gaussians._replace(colors=colors)


def _prepare_render_targets(
    intrinsics_px: torch.Tensor,
    image_shape: tuple[int, int],
    device: torch.device,
) -> RenderTargets:
    """Build the static original-view extrinsics and normalized intrinsics."""
    height, width = image_shape
    intrinsics_norm = normalize_intrinsics(intrinsics_px, width, height)
    intrinsics_norm = intrinsics_norm.to(device=device, dtype=torch.float32)
    extrinsics = torch.eye(4, device=device, dtype=torch.float32)
    return RenderTargets(
        extrinsics=rearrange(extrinsics, "i j -> 1 1 i j"),
        intrinsics_norm=rearrange(intrinsics_norm, "i j -> 1 1 i j"),
        image_shape=image_shape,
        camera_position=torch.zeros(3, device=device, dtype=torch.float32),
    )


def _clamp_max_render_pixels(
    intrinsics_px: torch.Tensor,
    image_shape: tuple[int, int],
    max_long_edge: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Scale down oversized render targets to bound rasterizer memory."""
    height, width = image_shape
    long_edge = max(height, width)
    if long_edge <= max_long_edge:
        return intrinsics_px, image_shape
    scale = max_long_edge / float(long_edge)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    scaled = scale_intrinsics_px(intrinsics_px, width, height, new_w, new_h)
    return scaled, (new_h, new_w)


@torch.inference_mode()
def _render_one_frame(
    decoder: Decoder,
    gaussians: Gaussians3D,
    targets: RenderTargets,
) -> torch.Tensor:
    """Render one frame at the static original view."""
    rendered = decoder.forward(
        gaussians=gaussians,
        extrinsics=targets.extrinsics,
        intrinsics=targets.intrinsics_norm,
        image_shape=targets.image_shape,
    )
    return rendered[0, 0].detach().cpu()


@torch.inference_mode()
def render_relight_video(
    decoder: Decoder,
    gaussians: Gaussians3D,
    sh_coeffs: torch.Tensor,
    targets: RenderTargets,
    num_frames: int,
    baked_irradiance_gray: float | None,
    world_to_light: torch.Tensor,
    include_baked_frame: bool = False,
    specular_mips: list[torch.Tensor] | None = None,
    roughness: float = 0.55,
    ao_factor: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Return one rendered frame per envmap rotation step.

    Rotates the envmap around the light-space up axis by 2*pi over num_frames.
    The Gaussians and camera stay put; only per-Gaussian colors change.
    """
    frames: list[torch.Tensor] = []
    device = targets.extrinsics.device
    world_to_light = world_to_light.to(device=device, dtype=torch.float32)

    if include_baked_frame:
        frames.append(_render_one_frame(decoder, gaussians, targets))

    # Rotate specular mips by resampling in-place would be expensive; instead we
    # bake the rotation into the query direction (via world_to_light).
    for frame_idx in range(num_frames):
        angle = 2.0 * math.pi * frame_idx / max(1, num_frames)
        w2l = rotation_matrix_y(-angle, device=device) @ world_to_light
        shaded = shade_gaussians(
            gaussians=gaussians,
            sh_coeffs=sh_coeffs,
            camera_position_world=targets.camera_position,
            world_to_light=w2l,
            baked_irradiance_gray=baked_irradiance_gray,
            specular_mips=specular_mips,
            roughness=roughness,
            ao_factor=ao_factor,
        )
        relit = _replace_colors(gaussians, shaded)
        frames.append(_render_one_frame(decoder, relit, targets))

    return frames


def _side_by_side(*streams: list[torch.Tensor]) -> list[torch.Tensor]:
    """Concatenate N same-length frame streams horizontally."""
    if not streams:
        raise ValueError("At least one stream required.")
    length = len(streams[0])
    for s in streams:
        if len(s) != length:
            raise ValueError(f"Comparison streams must be equal length; got {[len(s) for s in streams]}")
    combined: list[torch.Tensor] = []
    for tup in zip(*streams):
        ref = tup[0]
        for f in tup:
            if f.shape != ref.shape:
                raise ValueError(f"Frame shape mismatch: {f.shape} vs {ref.shape}")
        combined.append(torch.cat(list(tup), dim=-1))
    return combined


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Envmap relighting demo (SH-9 diffuse).")
    parser.add_argument("--input", type=Path, required=True, help="Input image path.")
    parser.add_argument("--envmap", type=Path, default=None, help="HDR envmap (EXR/HDR/PNG). If omitted, a synthetic key+ambient env is used.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/relight"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mode", choices=tuple(MODE_EXPERIMENTS), default="rgb")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--frames", type=int, default=60, help="Number of envmap-rotation frames.")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-long-edge", type=int, default=1536, help="Cap the render long edge.")
    parser.add_argument("--baked-gray", type=float, default=None,
                        help="Assumed training-time gray irradiance for albedo demod (e.g. 0.5). "
                             "Skip if omitted.")
    parser.add_argument("--compare", action="store_true",
                        help="Render baked | diffuse | full (diff+spec+AO) side-by-side.")
    parser.add_argument("--no-floater-filter", action="store_true")
    parser.add_argument("--focal-px", type=float, default=None)
    parser.add_argument("--focal-mm", type=float, default=None)
    parser.add_argument("--enable-specular", action="store_true",
                        help="Add split-sum specular using a prefiltered envmap mip chain.")
    parser.add_argument("--enable-ao", action="store_true",
                        help="Multiply diffuse by a density-based AO factor.")
    parser.add_argument("--specular-levels", type=int, default=6)
    parser.add_argument("--roughness", type=float, default=0.55)
    parser.add_argument("--ao-neighbors", type=int, default=24)
    parser.add_argument("--ao-strength", type=float, default=6.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not is_gsplat_available():
        raise SystemExit(
            "gsplat is required for relighting. Install with:\n"
            "  uv pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation"
        )

    device = _resolve_device(args.device)
    experiment = MODE_EXPERIMENTS[args.mode]
    checkpoint = args.checkpoint or MODE_CHECKPOINTS[args.mode]
    checkpoint = _resolve_checkpoint_path(checkpoint)

    print(f"[relight] device={device} checkpoint={checkpoint}")
    cfg = load_demo_config(experiment)
    encoder, decoder = load_demo_model(cfg, checkpoint, device)

    bundle = load_demo_image_bundle(
        image_path=args.input,
        focal_length_px=args.focal_px,
        focal_length_mm=args.focal_mm,
        default_focal_35mm_mm=DEFAULT_FOCAL_35MM_MM,
    )

    print(f"[relight] running encoder on {args.input}")
    encoder_out = run_single_image_inference(
        encoder=encoder,
        image=bundle.inference_image,
        intrinsics_px=bundle.inference_intrinsics.intrinsics_px,
        device=device,
    )
    gaussians: Gaussians3D = encoder_out["gaussians"]

    if not args.no_floater_filter:
        gaussians = filter_final_gaussian_floaters(gaussians)

    print(f"[relight] gaussian count: {gaussians.mean_vectors.shape[1]}")

    render_intrinsics_px, render_shape = _clamp_max_render_pixels(
        intrinsics_px=bundle.original_intrinsics.intrinsics_px,
        image_shape=bundle.original_image_shape,
        max_long_edge=args.max_long_edge,
    )
    targets = _prepare_render_targets(
        intrinsics_px=render_intrinsics_px,
        image_shape=render_shape,
        device=device,
    )
    print(f"[relight] render resolution: {render_shape[0]}x{render_shape[1]}")

    if args.envmap is not None:
        envmap = load_envmap(args.envmap).to(device)
        env_label = args.envmap.name
    else:
        envmap = synthesize_directional_envmap(
            direction=(0.4, 0.9, 0.2),
            key_color=(5.0, 4.7, 4.2),
            ambient_color=(0.10, 0.12, 0.18),
            key_softness=8.0,
            device=device,
        )
        env_label = "synthetic-directional"
    sh_coeffs = project_envmap_to_sh9(envmap).to(device)

    specular_mips = None
    if args.enable_specular or args.compare:
        print(f"[relight] building specular mip chain (levels={args.specular_levels})")
        specular_mips = build_specular_mip_chain(envmap, num_levels=args.specular_levels)

    ao_factor = None
    if args.enable_ao or args.compare:
        print(f"[relight] computing density AO (k={args.ao_neighbors})")
        n, _ = gaussian_normals(gaussians.quaternions[0], gaussians.singular_values[0])
        view = torch.zeros(3, device=device) - gaussians.mean_vectors[0]
        n = face_normals_toward(n, view)
        ao_factor = compute_density_ao(
            positions=gaussians.mean_vectors[0],
            normals=n,
            k_neighbors=args.ao_neighbors,
            influence_strength=args.ao_strength,
        )
        print(f"[relight] AO stats: min={ao_factor.min().item():.3f} "
              f"mean={ao_factor.mean().item():.3f} max={ao_factor.max().item():.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    print(f"[relight] env='{env_label}', rendering {args.frames} rotation frames")
    if args.compare:
        # Three streams: baked (static), diffuse-only, full IBL.
        baked_frame = _render_one_frame(decoder, gaussians, targets)
        baked_stream = [baked_frame] * args.frames
        diffuse_stream = render_relight_video(
            decoder=decoder, gaussians=gaussians, sh_coeffs=sh_coeffs, targets=targets,
            num_frames=args.frames, baked_irradiance_gray=args.baked_gray,
            world_to_light=WORLD_TO_LIGHT_OPENCV,
        )
        full_stream = render_relight_video(
            decoder=decoder, gaussians=gaussians, sh_coeffs=sh_coeffs, targets=targets,
            num_frames=args.frames, baked_irradiance_gray=args.baked_gray,
            world_to_light=WORLD_TO_LIGHT_OPENCV,
            specular_mips=specular_mips, roughness=args.roughness, ao_factor=ao_factor,
        )
        sbs = _side_by_side(baked_stream, diffuse_stream, full_stream)
        out_path = args.output_dir / f"{stem}_relight_compare.mp4"
        save_video(sbs, out_path, fps=args.fps)
    else:
        relit_frames = render_relight_video(
            decoder=decoder, gaussians=gaussians, sh_coeffs=sh_coeffs, targets=targets,
            num_frames=args.frames, baked_irradiance_gray=args.baked_gray,
            world_to_light=WORLD_TO_LIGHT_OPENCV,
            specular_mips=specular_mips if args.enable_specular else None,
            roughness=args.roughness,
            ao_factor=ao_factor if args.enable_ao else None,
        )
        out_path = args.output_dir / f"{stem}_relight.mp4"
        save_video(relit_frames, out_path, fps=args.fps)

    print(f"[relight] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
