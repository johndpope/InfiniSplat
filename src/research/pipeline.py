"""Callable relighting pipeline for the research matrix runner.

Splits the CLI-only flow in `src/demo/relight.py` into three stages so the
matrix runner can amortize model loading across many cells:

    RelightSession(checkpoint, device) -> EncodedScene(...) -> render_variant(...)

Each stage is deterministic given its inputs; encoded scenes are cached in
memory across shading variants for the same input image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from einops import rearrange

from src.demo.infer_single_image import (
    DEFAULT_FOCAL_35MM_MM,
    _resolve_device,
    filter_final_gaussian_floaters,
    load_demo_config,
    load_demo_image_bundle,
    load_demo_model,
    normalize_intrinsics,
    run_single_image_inference,
    scale_intrinsics_px,
)
from src.demo.relight import WORLD_TO_LIGHT_OPENCV
from src.model.decoder import Decoder
from src.model.decoder.decoder_gsplat import is_gsplat_available
from src.model.encoder import Encoder
from src.utils.gaussians import Gaussians3D
from src.utils.lighting import (
    build_specular_mip_chain,
    compute_density_ao,
    face_normals_toward,
    gaussian_normals,
    project_envmap_to_sh9,
    rotation_matrix_y,
    shade_gaussians,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderTargets:
    extrinsics: torch.Tensor       # [1, 1, 4, 4]
    intrinsics_norm: torch.Tensor  # [1, 1, 3, 3]
    image_shape: tuple[int, int]   # (H, W)
    camera_position: torch.Tensor  # [3]


@dataclass
class EncodedScene:
    input_path: Path
    gaussians: Gaussians3D
    targets: RenderTargets
    normals_world: torch.Tensor        # [N, 3] view-facing
    reliable_mask: torch.Tensor        # [N] bool
    # AO is keyed by (k_neighbors, strength) so config sweeps recompute.
    ao_cache: dict[tuple[int, float], torch.Tensor] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.ao_cache is None:
            self.ao_cache = {}

    def get_or_compute_ao(self, k_neighbors: int, strength: float) -> torch.Tensor:
        key = (int(k_neighbors), float(strength))
        cached = self.ao_cache.get(key)
        if cached is None:
            cached = compute_density_ao(
                positions=self.gaussians.mean_vectors[0],
                normals=self.normals_world,
                k_neighbors=k_neighbors,
                influence_strength=strength,
            )
            self.ao_cache[key] = cached
        return cached

    @property
    def num_gaussians(self) -> int:
        return int(self.gaussians.mean_vectors.shape[1])


@dataclass(frozen=True)
class EnvmapPack:
    """Envmap-derived quantities that are reusable across shading variants."""
    name: str
    envmap: torch.Tensor           # [H, W, 3] linear RGB
    sh_coeffs: torch.Tensor        # [9, 3]
    specular_mips: list[torch.Tensor] | None


@dataclass(frozen=True)
class ShadingConfig:
    """Per-cell shading knobs."""
    name: str
    baked_gray: float | None = None
    enable_specular: bool = False
    enable_ao: bool = False
    roughness: float = 0.55
    ao_neighbors: int = 24
    ao_strength: float = 2.0


# ---------------------------------------------------------------------------
# Session (owns the loaded model)
# ---------------------------------------------------------------------------


class RelightSession:
    """Owns a loaded encoder+decoder pair and produces encoded scenes."""

    def __init__(
        self,
        checkpoint: Path,
        experiment: str = "infinisplat_hypersim_rgb",
        device: str = "auto",
    ) -> None:
        if not is_gsplat_available():
            raise RuntimeError(
                "gsplat is required. Install with: uv pip install "
                "git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation"
            )
        self.device = _resolve_device(device)
        self.checkpoint = Path(checkpoint)
        self.experiment = experiment
        cfg = load_demo_config(experiment)
        self.encoder, self.decoder = load_demo_model(cfg, self.checkpoint, self.device)
        # Cache exactly one scene at a time: 1.5M gaussians is ~150 MiB of state,
        # and stacking two scenes plus DINO/DepthPro activations OOMs a 24 GiB card.
        self._cached_scene: EncodedScene | None = None
        self._cached_key: tuple | None = None

    def encode(
        self,
        input_path: Path,
        max_long_edge: int = 1152,
        focal_length_px: float | None = None,
        focal_length_mm: float | None = None,
    ) -> EncodedScene:
        input_path = Path(input_path)
        cache_key = (input_path, max_long_edge, focal_length_px, focal_length_mm)
        if self._cached_key == cache_key and self._cached_scene is not None:
            return self._cached_scene

        # Evict the previous scene before allocating the next one.
        self._cached_scene = None
        self._cached_key = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        bundle = load_demo_image_bundle(
            image_path=input_path,
            focal_length_px=focal_length_px,
            focal_length_mm=focal_length_mm,
            default_focal_35mm_mm=DEFAULT_FOCAL_35MM_MM,
        )
        out = run_single_image_inference(
            encoder=self.encoder,
            image=bundle.inference_image,
            intrinsics_px=bundle.inference_intrinsics.intrinsics_px,
            device=self.device,
        )
        gaussians = filter_final_gaussian_floaters(out["gaussians"])

        render_intrinsics_px, render_shape = _clamp_max_render_pixels(
            intrinsics_px=bundle.original_intrinsics.intrinsics_px,
            image_shape=bundle.original_image_shape,
            max_long_edge=max_long_edge,
        )
        targets = _prepare_render_targets(
            intrinsics_px=render_intrinsics_px,
            image_shape=render_shape,
            device=self.device,
        )

        n, ok = gaussian_normals(gaussians.quaternions[0], gaussians.singular_values[0])
        view = torch.zeros(3, device=self.device) - gaussians.mean_vectors[0]
        n = face_normals_toward(n, view)

        scene = EncodedScene(
            input_path=input_path,
            gaussians=gaussians,
            targets=targets,
            normals_world=n,
            reliable_mask=ok,
        )
        self._cached_scene = scene
        self._cached_key = cache_key
        return scene


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@torch.inference_mode()
def render_variant(
    session: RelightSession,
    scene: EncodedScene,
    envmap_pack: EnvmapPack,
    shading: ShadingConfig,
    num_frames: int = 24,
    world_to_light: torch.Tensor = WORLD_TO_LIGHT_OPENCV,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Render one rotating-envmap sequence + a matching baked reference frame.

    Returns:
        (relit_frames, baked_frame)
        relit_frames: [num_frames] each [3, H, W] on CPU, values in [0, 1].
        baked_frame: [3, H, W] on CPU, values in [0, 1].
    """
    device = session.device
    w2l_world = world_to_light.to(device=device, dtype=torch.float32)

    ao_factor = (
        scene.get_or_compute_ao(shading.ao_neighbors, shading.ao_strength)
        if shading.enable_ao else None
    )

    baked_frame = _render_one_frame(session.decoder, scene.gaussians, scene.targets)

    relit_frames: list[torch.Tensor] = []
    for i in range(num_frames):
        angle = 2.0 * math.pi * i / max(1, num_frames)
        w2l = rotation_matrix_y(-angle, device=device) @ w2l_world
        shaded = shade_gaussians(
            gaussians=scene.gaussians,
            sh_coeffs=envmap_pack.sh_coeffs,
            camera_position_world=scene.targets.camera_position,
            world_to_light=w2l,
            baked_irradiance_gray=shading.baked_gray,
            specular_mips=envmap_pack.specular_mips if shading.enable_specular else None,
            roughness=shading.roughness,
            ao_factor=ao_factor,
        )
        relit = scene.gaussians._replace(colors=shaded)
        relit_frames.append(_render_one_frame(session.decoder, relit, scene.targets))
    return relit_frames, baked_frame


# ---------------------------------------------------------------------------
# Envmap prep
# ---------------------------------------------------------------------------


def build_envmap_pack(
    name: str,
    envmap: torch.Tensor,
    device: torch.device,
    specular_levels: int = 4,
    need_specular: bool = True,
) -> EnvmapPack:
    envmap = envmap.to(device)
    sh = project_envmap_to_sh9(envmap).to(device)
    mips = build_specular_mip_chain(envmap, num_levels=specular_levels) if need_specular else None
    return EnvmapPack(name=name, envmap=envmap, sh_coeffs=sh, specular_mips=mips)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare_render_targets(
    intrinsics_px: torch.Tensor,
    image_shape: tuple[int, int],
    device: torch.device,
) -> RenderTargets:
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
    rendered = decoder.forward(
        gaussians=gaussians,
        extrinsics=targets.extrinsics,
        intrinsics=targets.intrinsics_norm,
        image_shape=targets.image_shape,
    )
    return rendered[0, 0].detach().cpu()
