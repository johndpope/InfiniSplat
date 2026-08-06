"""Envmap-based image-based lighting for InfiniSplat Gaussians.

Order-2 spherical-harmonic (SH-9) irradiance approximation following
Ramamoorthi & Hanrahan 2001, plus a Gaussian-surface-normal proxy derived
from the shortest principal axis of each Gaussian.

Conventions:
- Directions are unit 3-vectors in "light space", with +y pointing up.
- A lat-long envmap has shape [H, W, 3] with row 0 at the north pole
  (v = 0 -> theta = 0 -> direction (0, +1, 0)) and column 0 at phi = 0
  (direction (+1, 0, 0)).
- To relight geometry in a different world-frame, pass a 3x3
  `world_to_light` rotation to `shade_gaussians_diffuse`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.gaussians import Gaussians3D
from src.utils.linalg import rotation_matrices_from_quaternions

# Physical constant for Lambertian normalization (divide E(n) by pi before
# multiplying by albedo). Kept as a module-level constant so callers can opt
# out (e.g. to match the pre-normalization Phase 1 look).
_INV_PI = 1.0 / math.pi

# ---------------------------------------------------------------------------
# Gaussian surface normals
# ---------------------------------------------------------------------------


def gaussian_normals(
    quaternions: torch.Tensor,
    singular_values: torch.Tensor,
    isotropy_threshold: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate per-Gaussian surface normals via the shortest principal axis.

    Args:
        quaternions: [N, 4] wxyz quaternions.
        singular_values: [N, 3] positive scales along the local axes.
        isotropy_threshold: If min_scale / median_scale > threshold, the
            Gaussian is too isotropic for a reliable normal.

    Returns:
        (normals, is_reliable) where normals has shape [N, 3] (unit) and
        is_reliable is a bool tensor of shape [N].
    """
    if quaternions.ndim != 2 or quaternions.shape[-1] != 4:
        raise ValueError(f"quaternions must be [N, 4], got {tuple(quaternions.shape)}")
    if singular_values.shape != (quaternions.shape[0], 3):
        raise ValueError(
            f"singular_values must be [N, 3] matching quaternions, got "
            f"{tuple(singular_values.shape)}"
        )

    R = rotation_matrices_from_quaternions(quaternions)  # [N, 3, 3]
    min_axis = singular_values.argmin(dim=-1)            # [N]
    idx = torch.arange(R.shape[0], device=R.device)
    normals = R[idx, :, min_axis]                        # [N, 3]
    normals = F.normalize(normals, dim=-1, eps=1e-8)

    sorted_scales, _ = singular_values.sort(dim=-1)
    ratio = sorted_scales[:, 0] / sorted_scales[:, 1].clamp_min(1e-8)
    is_reliable = ratio < isotropy_threshold

    return normals, is_reliable


def face_normals_toward(
    normals: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Flip normals so they face toward the reference direction.

    Args:
        normals: [N, 3] unit normals.
        reference: [N, 3] direction the normal should face (need not be unit).

    Returns:
        [N, 3] sign-flipped unit normals.
    """
    dot = (normals * reference).sum(dim=-1, keepdim=True)
    sign = torch.where(dot < 0, -torch.ones_like(dot), torch.ones_like(dot))
    return normals * sign


# ---------------------------------------------------------------------------
# SH-9 (order 2) spherical harmonics
# ---------------------------------------------------------------------------

# Basis constants (orthonormal real SH).
_SH_C0 = 0.282094791773878
_SH_C1 = 0.488602511902920
_SH_C2_1 = 1.092548430592079
_SH_C2_2 = 0.315391565252520
_SH_C2_3 = 0.546274215296039

# Irradiance-from-radiance coefficients (Ramamoorthi & Hanrahan).
_IRR_C1 = 0.429042572364886   # 2 * sqrt(15/pi) / 8 * pi/4
_IRR_C2 = 0.511663735956893   # sqrt(3/pi) / 2 * 2*pi/3
_IRR_C3 = 0.743124729909998   # sqrt(5/pi) * 3/4 * pi/4 (+ correction)
_IRR_C4 = 0.886226925452758   # sqrt(1/pi) / 2 * pi
_IRR_C5 = 0.247708320045221   # sqrt(5/pi) * 1/4 * pi/4


def sh9_basis(directions: torch.Tensor) -> torch.Tensor:
    """Evaluate the first nine real SH basis functions at unit directions.

    Args:
        directions: [..., 3] unit-length directions in light space (+y up).

    Returns:
        [..., 9] basis values in the order
        (Y_{0,0}, Y_{1,-1}, Y_{1,0}, Y_{1,1},
         Y_{2,-2}, Y_{2,-1}, Y_{2,0}, Y_{2,1}, Y_{2,2}).
    """
    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]
    basis = torch.stack(
        [
            torch.full_like(x, _SH_C0),
            _SH_C1 * y,
            _SH_C1 * z,
            _SH_C1 * x,
            _SH_C2_1 * x * y,
            _SH_C2_1 * y * z,
            _SH_C2_2 * (3.0 * z * z - 1.0),
            _SH_C2_1 * x * z,
            _SH_C2_3 * (x * x - y * y),
        ],
        dim=-1,
    )
    return basis


def project_envmap_to_sh9(envmap: torch.Tensor) -> torch.Tensor:
    """Project a lat-long HDR envmap into 9 SH coefficients per channel.

    Args:
        envmap: [H, W, 3] radiance in linear RGB. Any nonnegative float dtype.

    Returns:
        [9, 3] SH coefficients (order 2).
    """
    if envmap.ndim != 3 or envmap.shape[-1] != 3:
        raise ValueError(f"envmap must be [H, W, 3], got {tuple(envmap.shape)}")

    device = envmap.device
    dtype = torch.float32
    envmap = envmap.to(dtype)
    height, width = envmap.shape[:2]

    v = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    u = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    theta = v * math.pi                        # [0, pi]
    phi = u * 2.0 * math.pi                    # [0, 2pi)

    sin_theta = torch.sin(theta)               # [H]
    cos_theta = torch.cos(theta)               # [H]
    sin_phi = torch.sin(phi)                   # [W]
    cos_phi = torch.cos(phi)                   # [W]

    x = sin_theta[:, None] * cos_phi[None, :]  # [H, W]
    y = cos_theta[:, None].expand(-1, width)   # [H, W]
    z = sin_theta[:, None] * sin_phi[None, :]  # [H, W]

    dirs = torch.stack([x, y, z], dim=-1)      # [H, W, 3]
    basis = sh9_basis(dirs)                    # [H, W, 9]

    d_omega = (math.pi / height) * (2.0 * math.pi / width) * sin_theta
    weight = d_omega[:, None, None] * basis    # [H, W, 9]

    coeffs = torch.einsum("hwk,hwc->kc", weight, envmap)  # [9, 3]
    return coeffs


def sh9_irradiance(
    coeffs: torch.Tensor,
    normals: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the SH-9 cosine-convolved irradiance at surface normals.

    Args:
        coeffs: [9, 3] SH coefficients per channel.
        normals: [N, 3] unit normals in light space.

    Returns:
        [N, 3] linear-RGB irradiance (clamped to be nonnegative).
    """
    if coeffs.shape != (9, 3):
        raise ValueError(f"coeffs must be [9, 3], got {tuple(coeffs.shape)}")
    if normals.shape[-1] != 3:
        raise ValueError(f"normals must end in 3, got {tuple(normals.shape)}")

    coeffs = coeffs.to(normals)
    x = normals[..., 0]
    y = normals[..., 1]
    z = normals[..., 2]
    x2 = x * x
    y2 = y * y
    z2 = z * z

    L00 = coeffs[0]
    L1m1 = coeffs[1]
    L10 = coeffs[2]
    L11 = coeffs[3]
    L2m2 = coeffs[4]
    L2m1 = coeffs[5]
    L20 = coeffs[6]
    L21 = coeffs[7]
    L22 = coeffs[8]

    e = (
        _IRR_C1 * L22[None] * (x2 - y2)[..., None]
        + _IRR_C3 * L20[None] * z2[..., None]
        + _IRR_C4 * L00[None]
        - _IRR_C5 * L20[None]
        + 2.0 * _IRR_C1 * (
            L2m2[None] * (x * y)[..., None]
            + L21[None] * (x * z)[..., None]
            + L2m1[None] * (y * z)[..., None]
        )
        + 2.0 * _IRR_C2 * (
            L11[None] * x[..., None]
            + L1m1[None] * y[..., None]
            + L10[None] * z[..., None]
        )
    )
    return e.clamp_min(0.0)


def rotate_directions(
    directions: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Rotate a batch of directions by a 3x3 rotation matrix.

    Convenience for animating envmap orientation without re-projecting the SH:
    to render the scene as if the envmap were rotated by R, evaluate irradiance
    at R^T @ n. This function does the R^T application in one call.

    Args:
        directions: [..., 3] unit directions in world space.
        rotation: [3, 3] world-to-envmap rotation.

    Returns:
        [..., 3] rotated directions.
    """
    return directions @ rotation.to(directions).T


def rotation_matrix_y(angle_rad: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """Right-handed rotation around +y (up) axis."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return torch.tensor(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        device=device,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Envmap loading and synthetic generation
# ---------------------------------------------------------------------------


def load_envmap(path: str | Path) -> torch.Tensor:
    """Load a lat-long HDR envmap.

    Supports .hdr (Radiance, via OpenCV), .exr (via the OpenEXR Python
    binding if OpenCV wasn't built with EXR), and any LDR format OpenCV
    can decode.

    Args:
        path: File path.

    Returns:
        [H, W, 3] float32 tensor in linear RGB. NaN/inf are clamped to 0.
    """
    import os

    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

    import cv2

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Envmap not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".exr":
        try:
            rgb = _load_exr_openexr(path)
        except Exception:
            # Fall back to cv2, which may work if built with OpenEXR.
            rgb = _load_via_cv2(path)
    else:
        rgb = _load_via_cv2(path)

    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    rgb = np.clip(rgb, 0.0, None)
    return torch.from_numpy(rgb)


def _load_via_cv2(path: Path) -> np.ndarray:
    import cv2

    flags = cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR
    bgr = cv2.imread(str(path), flags)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read envmap: {path}")

    if bgr.dtype == np.uint8:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    elif bgr.dtype == np.uint16:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
    else:
        if bgr.shape[-1] == 4:
            bgr = bgr[..., :3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return rgb


def _load_exr_openexr(path: Path) -> np.ndarray:
    """Read a lat-long EXR envmap via the OpenEXR Python binding."""
    import Imath
    import OpenEXR

    ex = OpenEXR.InputFile(str(path))
    hdr = ex.header()
    dw = hdr["dataWindow"]
    h = dw.max.y - dw.min.y + 1
    w = dw.max.x - dw.min.x + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    channels = hdr["channels"]
    if not {"R", "G", "B"}.issubset(channels.keys()):
        raise RuntimeError(f"EXR file lacks RGB channels: {list(channels.keys())}")

    def read(ch: str) -> np.ndarray:
        raw = ex.channel(ch, pt)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(h, w).copy()
        return arr

    r, g, b = read("R"), read("G"), read("B")
    return np.stack([r, g, b], axis=-1)


def synthesize_directional_envmap(
    direction: tuple[float, float, float],
    key_color: tuple[float, float, float] = (5.0, 4.8, 4.5),
    ambient_color: tuple[float, float, float] = (0.10, 0.12, 0.16),
    key_softness: float = 8.0,
    height: int = 256,
    width: int = 512,
    device=None,
) -> torch.Tensor:
    """Create a synthetic HDR envmap with one soft key light plus sky ambient.

    Useful for smoke tests without a real HDRI. `key_softness` controls the
    exponent of a cosine lobe centered on `direction`.

    Args:
        direction: Unit direction the key light emits *from* (i.e. where the
            bright pixel sits in the envmap), in light space (+y up).
        key_color: RGB peak intensity of the key light.
        ambient_color: RGB of the constant sky term.
        key_softness: Exponent controlling how tight the key lobe is.
        height, width: Envmap resolution.
        device: Torch device.

    Returns:
        [H, W, 3] linear-RGB envmap.
    """
    dtype = torch.float32
    v = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    u = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    theta = v * math.pi
    phi = u * 2.0 * math.pi
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    sin_p = torch.sin(phi)
    cos_p = torch.cos(phi)
    x = sin_t[:, None] * cos_p[None, :]
    y = cos_t[:, None].expand(-1, width)
    z = sin_t[:, None] * sin_p[None, :]
    dirs = torch.stack([x, y, z], dim=-1)

    d = torch.tensor(direction, device=device, dtype=dtype)
    d = d / d.norm().clamp_min(1e-8)
    cosine = (dirs * d).sum(dim=-1).clamp_min(0.0)
    lobe = cosine.pow(key_softness)                          # [H, W]

    key = torch.tensor(key_color, device=device, dtype=dtype)
    amb = torch.tensor(ambient_color, device=device, dtype=dtype)
    envmap = lobe[..., None] * key[None, None] + amb[None, None]
    return envmap


# ---------------------------------------------------------------------------
# Specular IBL (split-sum approximation)
# ---------------------------------------------------------------------------


def sample_envmap_lonlat(
    envmap: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    """Bilinearly sample a lat-long envmap at the given unit directions.

    Args:
        envmap: [H, W, 3] or [C, H, W, 3] (mip pyramid). For a single mip,
            [H, W, 3] is expected.
        directions: [..., 3] unit directions in light space (+y up).

    Returns:
        [..., 3] sampled radiance.
    """
    if envmap.ndim != 3 or envmap.shape[-1] != 3:
        raise ValueError(f"envmap must be [H, W, 3], got {tuple(envmap.shape)}")

    orig_shape = directions.shape[:-1]
    dirs = directions.reshape(-1, 3)
    x, y, z = dirs.unbind(-1)
    theta = torch.acos(y.clamp(-1.0, 1.0))              # [0, pi]
    phi = torch.atan2(z, x)                              # [-pi, pi]
    u = (phi / (2.0 * math.pi)) % 1.0                    # [0, 1)
    v = theta / math.pi                                  # [0, 1]
    grid = torch.stack([u * 2.0 - 1.0, v * 2.0 - 1.0], dim=-1)  # [N, 2]
    grid = grid.view(1, 1, -1, 2)
    env = envmap.permute(2, 0, 1).unsqueeze(0).to(dirs)  # [1, 3, H, W]
    sampled = F.grid_sample(
        env, grid, mode="bilinear", padding_mode="border", align_corners=False,
    )[0, :, 0].T  # [N, 3]
    return sampled.reshape(*orig_shape, 3)


def build_specular_mip_chain(
    envmap: torch.Tensor,
    num_levels: int = 6,
    kernel_growth: float = 2.0,
) -> list[torch.Tensor]:
    """Build a coarse prefilter mip chain for split-sum specular IBL.

    Simple approximation: each level is a Gaussian-blurred + downsampled
    version of the previous. Not physically GGX-accurate, but visually
    "rougher = blurrier reflection", which is what we need for the demo.

    Args:
        envmap: [H, W, 3] linear RGB.
        num_levels: Number of mip levels (level 0 = original).
        kernel_growth: Sigma multiplier between levels.

    Returns:
        List of [H_i, W_i, 3] mips ordered from finest to coarsest.
    """
    if envmap.ndim != 3 or envmap.shape[-1] != 3:
        raise ValueError(f"envmap must be [H, W, 3], got {tuple(envmap.shape)}")

    current = envmap
    mips = [current]
    # Aggressive per-level blur so higher mips are genuinely rougher. Sigma
    # grows geometrically in *current-mip pixels*, capped so the kernel stays
    # under a quarter of the mip width (avoids circular-pad wraparound and
    # keeps compute bounded).
    base_sigma = 2.0
    for level in range(1, num_levels):
        h, w = current.shape[:2]
        target_sigma = base_sigma * (kernel_growth ** (level - 1))
        max_sigma = max(0.5, (w - 1) / 8.0)
        sigma = min(target_sigma, max_sigma)
        kernel = _make_gaussian_kernel_1d(sigma, device=current.device, dtype=current.dtype)
        blurred = _blur_lonlat(current, kernel)
        new_h, new_w = max(1, h // 2), max(1, w // 2)
        img = blurred.permute(2, 0, 1).unsqueeze(0)
        down = F.interpolate(img, size=(new_h, new_w), mode="area")[0].permute(1, 2, 0)
        current = down.contiguous()
        mips.append(current)
    return mips


def _make_gaussian_kernel_1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def _blur_lonlat(envmap: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Separable Gaussian blur with horizontal wrap and vertical clamp.

    Wrapping in phi respects the lat-long topology; theta uses replicate padding.
    """
    c = envmap.shape[-1]
    img = envmap.permute(2, 0, 1).unsqueeze(0)              # [1, C, H, W]
    kx = kernel.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    rx = kernel.numel() // 2
    ry = rx
    img_h = F.pad(img, (rx, rx, 0, 0), mode="circular")
    img_h = F.conv2d(img_h, kx, groups=c)
    img_v = F.pad(img_h, (0, 0, ry, ry), mode="replicate")
    img_v = F.conv2d(img_v, ky, groups=c)
    return img_v[0].permute(1, 2, 0).contiguous()


def sample_specular_mips(
    mips: list[torch.Tensor],
    directions: torch.Tensor,
    roughness: torch.Tensor,
) -> torch.Tensor:
    """Sample a mip chain at the reflection direction, interpolating by roughness.

    Args:
        mips: List of [H_i, W_i, 3] envmap mips, finest first.
        directions: [N, 3] unit reflection directions in light space.
        roughness: [N] or scalar in [0, 1]. 0 = mirror, 1 = fully diffuse.

    Returns:
        [N, 3] sampled specular radiance.
    """
    if isinstance(roughness, float):
        roughness = torch.full_like(directions[:, :1], roughness).squeeze(-1)
    num_levels = len(mips)
    level = roughness.clamp(0.0, 1.0) * (num_levels - 1)
    lo = level.floor().long().clamp(max=num_levels - 1)
    hi = (lo + 1).clamp(max=num_levels - 1)
    frac = (level - lo.float()).unsqueeze(-1)

    # Sample each mip once; select per-Gaussian output by lo/hi index.
    # For num_levels <=6 this is cheap.
    per_mip = [sample_envmap_lonlat(mips[i], directions) for i in range(num_levels)]
    stack = torch.stack(per_mip, dim=0)                   # [L, N, 3]
    idx_lo = lo.view(1, -1, 1).expand(1, -1, 3)
    idx_hi = hi.view(1, -1, 1).expand(1, -1, 3)
    lo_val = stack.gather(0, idx_lo)[0]
    hi_val = stack.gather(0, idx_hi)[0]
    return lo_val * (1.0 - frac) + hi_val * frac


def fresnel_schlick(cos_theta: torch.Tensor, F0: torch.Tensor) -> torch.Tensor:
    """Schlick's Fresnel approximation.

    Args:
        cos_theta: [N, 1] cosine of the view-normal angle, clamped to >= 0.
        F0: [N, 3] or [3] reflectance at normal incidence.

    Returns:
        [N, 3] Fresnel term.
    """
    cos_theta = cos_theta.clamp(0.0, 1.0)
    one_minus = (1.0 - cos_theta)
    return F0 + (1.0 - F0) * one_minus.pow(5.0)


# ---------------------------------------------------------------------------
# Ambient occlusion (density-based proxy)
# ---------------------------------------------------------------------------


def compute_density_ao(
    positions: torch.Tensor,
    normals: torch.Tensor,
    k_neighbors: int = 24,
    influence_strength: float = 6.0,
) -> torch.Tensor:
    """Density-based cheap ambient occlusion for a Gaussian point cloud.

    For each point, count K nearest neighbors and measure how many sit "above"
    the surface (positive dot product with the normal). High front-side
    neighbor density -> lower visibility -> smaller AO factor.

    Args:
        positions: [N, 3] world-space points.
        normals: [N, 3] unit normals.
        k_neighbors: Neighbor count for the local density estimate.
        influence_strength: Multiplier on the occlusion score before clipping.

    Returns:
        [N] AO factor in [0, 1] (1 = fully lit, 0 = fully occluded).

    Notes:
        Uses scipy.spatial.cKDTree on CPU. For 1.5M Gaussians this runs in a
        few seconds on a modern box. Results are cacheable per scene.
    """
    from scipy.spatial import cKDTree
    import numpy as np

    pts = positions.detach().float().cpu().numpy()
    n = normals.detach().float().cpu().numpy()
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=min(k_neighbors + 1, len(pts)), workers=-1)
    # Skip self (column 0).
    neighbors = pts[idx[:, 1:]]                            # [N, K, 3]
    d = neighbors - pts[:, None, :]                        # [N, K, 3]
    dist = np.linalg.norm(d, axis=-1, keepdims=True) + 1e-6
    d_unit = d / dist
    dots = (d_unit * n[:, None, :]).sum(axis=-1)           # [N, K]
    front_side = np.maximum(dots, 0.0).mean(axis=-1)       # [N]
    occlusion = np.clip(front_side * influence_strength, 0.0, 1.0)
    ao = 1.0 - occlusion
    return torch.as_tensor(ao, device=positions.device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Shading top-level
# ---------------------------------------------------------------------------


def shade_gaussians(
    gaussians: Gaussians3D,
    sh_coeffs: torch.Tensor,
    camera_position_world: torch.Tensor,
    *,
    world_to_light: torch.Tensor | None = None,
    baked_irradiance_gray: float | None = None,
    fallback_shade: float = 1.0,
    specular_mips: list[torch.Tensor] | None = None,
    roughness: float | torch.Tensor = 0.55,
    specular_F0: tuple[float, float, float] = (0.04, 0.04, 0.04),
    ao_factor: torch.Tensor | None = None,
    lambertian_normalize: bool = False,
) -> torch.Tensor:
    """Compute per-Gaussian relit RGB colors under an envmap.

    Diffuse term uses SH-9 irradiance; optional specular uses split-sum
    with a prefiltered mip chain and Schlick Fresnel.

    Args:
        gaussians: Container with [1, N, ...] tensors.
        sh_coeffs: [9, 3] envmap SH coefficients.
        camera_position_world: [3] camera position in world space.
        world_to_light: Optional [3, 3] rotation transforming world normals
            into the envmap frame.
        baked_irradiance_gray: If not None, divides colors by this scalar to
            recover an approximate albedo before shading.
        fallback_shade: Multiplier applied where the normal is unreliable.
        specular_mips: Optional prefiltered envmap mips (from
            build_specular_mip_chain). Enables specular when provided.
        roughness: Per-Gaussian roughness in [0, 1] or scalar. 0=mirror.
        specular_F0: Base reflectance at normal incidence (dielectric ~0.04).
        ao_factor: Optional [N] visibility factor in [0, 1].
        lambertian_normalize: If True, divide diffuse irradiance by pi
            (physically correct Lambertian). Off by default because it makes
            the demo darker than the baked reference.

    Returns:
        [1, N, 3] shaded linear-RGB colors, clipped to [0, 1].
    """
    if gaussians.mean_vectors.shape[0] != 1:
        raise ValueError("shade_gaussians currently expects batch size 1.")

    means = gaussians.mean_vectors[0]                       # [N, 3]
    quats = gaussians.quaternions[0]                        # [N, 4]
    scales = gaussians.singular_values[0]                   # [N, 3]
    colors = gaussians.colors[0]                            # [N, 3]

    normals, reliable = gaussian_normals(quats, scales)     # [N, 3], [N]
    view_vec = camera_position_world.to(means).view(1, 3) - means
    view_dir = F.normalize(view_vec, dim=-1, eps=1e-8)
    normals = face_normals_toward(normals, view_vec)

    if world_to_light is not None:
        w2l = world_to_light.to(normals)
        normals_light = normals @ w2l.T
        view_light = view_dir @ w2l.T
    else:
        normals_light = normals
        view_light = view_dir

    diffuse_irr = sh9_irradiance(sh_coeffs, normals_light)
    if lambertian_normalize:
        diffuse_irr = diffuse_irr * _INV_PI

    if baked_irradiance_gray is not None:
        albedo = colors / max(float(baked_irradiance_gray), 1e-6)
    else:
        albedo = colors

    shaded = albedo * diffuse_irr

    if specular_mips is not None:
        # Reflect view around normal in light space, then sample the mip chain.
        n = normals_light
        cos_nv = (n * view_light).sum(dim=-1, keepdim=True).clamp_min(0.0)
        reflect = 2.0 * cos_nv * n - view_light
        reflect = F.normalize(reflect, dim=-1, eps=1e-8)
        if isinstance(roughness, torch.Tensor):
            rough = roughness.to(means).view(-1)
        else:
            rough = torch.full((means.shape[0],), float(roughness),
                               device=means.device, dtype=means.dtype)
        spec_radiance = sample_specular_mips(specular_mips, reflect, rough)
        F0 = torch.tensor(specular_F0, device=means.device, dtype=means.dtype)
        F0 = F0.view(1, 3).expand(means.shape[0], 3)
        fresnel = fresnel_schlick(cos_nv, F0)
        shaded = shaded * (1.0 - fresnel) + spec_radiance * fresnel

    if ao_factor is not None:
        shaded = shaded * ao_factor.to(shaded).view(-1, 1)

    if not bool(reliable.all()):
        shaded = torch.where(
            reliable[:, None],
            shaded,
            colors * fallback_shade,
        )

    return shaded.clamp(0.0, 1.0).unsqueeze(0)


def shade_gaussians_diffuse(
    gaussians: Gaussians3D,
    sh_coeffs: torch.Tensor,
    camera_position_world: torch.Tensor,
    world_to_light: torch.Tensor | None = None,
    baked_irradiance_gray: float | None = None,
    fallback_shade: float = 1.0,
) -> torch.Tensor:
    """Backwards-compatible diffuse-only shading wrapper."""
    return shade_gaussians(
        gaussians=gaussians,
        sh_coeffs=sh_coeffs,
        camera_position_world=camera_position_world,
        world_to_light=world_to_light,
        baked_irradiance_gray=baked_irradiance_gray,
        fallback_shade=fallback_shade,
    )
