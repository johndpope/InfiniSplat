"""Per-cell metrics for the relighting research harness.

All functions take frame stacks as numpy arrays of shape [T, H, W, 3] in
[0, 1] float or uint8. They return either a scalar or a small dict of
scalars, easy to log to wandb or a spreadsheet cell.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def _to_float(frames: np.ndarray) -> np.ndarray:
    if frames.dtype == np.uint8:
        return frames.astype(np.float32) / 255.0
    if frames.dtype in (np.float32, np.float64):
        return frames.astype(np.float32)
    raise TypeError(f"Unsupported dtype: {frames.dtype}")


def temporal_std(frames: np.ndarray) -> float:
    """Mean per-pixel temporal standard deviation.

    Zero for a static stream. Values in [0, 1] since frames are in [0, 1].
    Interpretation: how much the envmap rotation actually moves pixels.
    """
    f = _to_float(frames)
    return float(f.std(axis=0).mean())


def spatial_mean_brightness(frames: np.ndarray) -> float:
    """Average pixel brightness across all frames and channels."""
    f = _to_float(frames)
    return float(f.mean())


def brightness_range(frames: np.ndarray, quantiles: tuple[float, float] = (0.02, 0.98)) -> dict[str, float]:
    """Robust brightness spread — 98th-percentile minus 2nd."""
    f = _to_float(frames).mean(axis=-1)  # grayscale
    lo = float(np.quantile(f, quantiles[0]))
    hi = float(np.quantile(f, quantiles[1]))
    return {"brightness_p02": lo, "brightness_p98": hi, "brightness_span": hi - lo}


def clip_saturation_fraction(
    frames: np.ndarray, low_thresh: float = 0.01, high_thresh: float = 0.99
) -> dict[str, float]:
    """Fraction of pixels clipped to black or white per channel-mean."""
    f = _to_float(frames).mean(axis=-1)
    dark = float((f <= low_thresh).mean())
    bright = float((f >= high_thresh).mean())
    return {"clip_dark_frac": dark, "clip_bright_frac": bright}


def specular_contribution_ratio(
    full_frames: np.ndarray, diffuse_frames: np.ndarray
) -> float:
    """How much the specular pass changes pixels, mean |full - diffuse|.

    Zero when specular is off or when Fresnel keeps it invisible; grows
    with rougher-vs-mirror configs.
    """
    a = _to_float(full_frames)
    b = _to_float(diffuse_frames)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return float(np.abs(a - b).mean())


def silhouette_delta_vs_baked(
    relit_frames: np.ndarray, baked_frame: np.ndarray, thresh: float = 0.02
) -> float:
    """IOU-style silhouette drift between baked scene and relit average.

    Non-background pixels in the baked frame that are missing in the mean
    relit frame indicate the shading pushed the subject into the background
    color (usually black), which is a warning sign for the current config.
    """
    b = _to_float(baked_frame)
    r = _to_float(relit_frames).mean(axis=0)
    bmask = (b.max(axis=-1) > thresh)
    rmask = (r.max(axis=-1) > thresh)
    if bmask.sum() == 0:
        return 0.0
    kept = float((bmask & rmask).sum()) / float(bmask.sum())
    return 1.0 - kept


def relight_effectiveness(relit_frames: np.ndarray, baked_frame: np.ndarray) -> float:
    """Ratio of relit temporal variation to overall brightness of the baked reference.

    Above ~0.05 usually means "the rotation is visibly changing things";
    below 0.01 means "the shading is barely doing anything".
    """
    tstd = temporal_std(relit_frames)
    base = spatial_mean_brightness(baked_frame[None])
    return tstd / max(base, 1e-6)


def rotation_frequency_peak(frames: np.ndarray, num_frames: int) -> dict[str, float]:
    """FFT peak location for the mean-brightness signal across frames.

    A well-behaved rotation of an anisotropic envmap should show a strong
    peak at frequency 1 (one cycle per sweep). Two dominant peaks or a flat
    spectrum both indicate something's off.
    """
    f = _to_float(frames).mean(axis=(1, 2, 3))  # [T]
    f = f - f.mean()
    if len(f) < 4:
        return {"peak_freq": 0.0, "peak_ratio": 0.0}
    spectrum = np.abs(np.fft.rfft(f))
    if len(spectrum) < 2:
        return {"peak_freq": 0.0, "peak_ratio": 0.0}
    spectrum[0] = 0.0
    peak = int(np.argmax(spectrum))
    total = float(spectrum.sum()) + 1e-6
    return {"peak_freq": float(peak), "peak_ratio": float(spectrum[peak] / total)}


@lru_cache(maxsize=1)
def _lpips_model():
    try:
        import lpips
        import torch
        m = lpips.LPIPS(net="alex", verbose=False)
        m.eval()
        if torch.cuda.is_available():
            m = m.cuda()
        return m
    except Exception:
        return None


def ssim_vs_baked(relit_frames: np.ndarray, baked_frame: np.ndarray) -> dict[str, float]:
    """Structural similarity of the mean relit frame vs baked reference.

    High SSIM = shading preserved the scene's structure; low SSIM often means
    the shading blew out or clipped a region. Uses grayscale to be robust to
    color shifts.
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        return {"ssim_mean_vs_baked": float("nan")}
    r = _to_float(relit_frames).mean(axis=0)
    b = _to_float(baked_frame)
    r_gray = r.mean(axis=-1)
    b_gray = b.mean(axis=-1)
    return {"ssim_mean_vs_baked": float(ssim(b_gray, r_gray, data_range=1.0))}


def lpips_vs_baked(relit_frames: np.ndarray, baked_frame: np.ndarray) -> dict[str, float]:
    """LPIPS perceptual distance between the mean relit frame and baked ref.

    Lower is closer to baked appearance. Ideal is in a middle range: 0 means
    shading did nothing, very high means it destroyed the scene.
    """
    model = _lpips_model()
    if model is None:
        return {"lpips_mean_vs_baked": float("nan")}
    import torch

    r = _to_float(relit_frames).mean(axis=0)      # [H, W, 3]
    b = _to_float(baked_frame)                    # [H, W, 3]
    # LPIPS wants [-1, 1] tensors of shape [B, 3, H, W].
    r_t = torch.from_numpy(r).permute(2, 0, 1)[None].mul(2.0).sub(1.0)
    b_t = torch.from_numpy(b).permute(2, 0, 1)[None].mul(2.0).sub(1.0)
    device = next(model.parameters()).device
    with torch.inference_mode():
        d = model(r_t.to(device), b_t.to(device)).item()
    return {"lpips_mean_vs_baked": float(d)}


def summarize_cell(
    relit_frames: np.ndarray,
    baked_frame: np.ndarray,
    diffuse_frames: np.ndarray | None = None,
    include_lpips: bool = True,
) -> dict[str, float]:
    """One-call metric bundle for a single sweep cell."""
    metrics: dict[str, float] = {
        "temporal_std": temporal_std(relit_frames),
        "brightness_mean": spatial_mean_brightness(relit_frames),
        "baked_brightness_mean": spatial_mean_brightness(baked_frame[None]),
        "silhouette_drift": silhouette_delta_vs_baked(relit_frames, baked_frame),
        "relight_effectiveness": relight_effectiveness(relit_frames, baked_frame),
    }
    metrics.update(brightness_range(relit_frames))
    metrics.update(clip_saturation_fraction(relit_frames))
    metrics.update(rotation_frequency_peak(relit_frames, num_frames=relit_frames.shape[0]))
    metrics.update(ssim_vs_baked(relit_frames, baked_frame))
    if include_lpips:
        metrics.update(lpips_vs_baked(relit_frames, baked_frame))
    if diffuse_frames is not None:
        metrics["specular_delta"] = specular_contribution_ratio(relit_frames, diffuse_frames)
    return metrics
