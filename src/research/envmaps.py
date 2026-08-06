"""Envmap sources used by the research matrix runner.

Provides a small set of named synthetic envmaps and helpers to load HDRI
files. Each entry returns a (name, [H, W, 3] linear-RGB tensor) pair.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.utils.lighting import load_envmap, synthesize_directional_envmap


def studio_warm(device=None) -> torch.Tensor:
    return synthesize_directional_envmap(
        direction=(0.6, 0.7, 0.4),
        key_color=(6.0, 5.4, 4.6),
        ambient_color=(0.10, 0.11, 0.14),
        key_softness=12.0,
        device=device,
    )


def sunset_side(device=None) -> torch.Tensor:
    return synthesize_directional_envmap(
        direction=(1.0, 0.15, 0.0),
        key_color=(8.0, 4.5, 2.0),
        ambient_color=(0.12, 0.10, 0.18),
        key_softness=6.0,
        device=device,
    )


def overcast_top(device=None) -> torch.Tensor:
    return synthesize_directional_envmap(
        direction=(0.0, 1.0, 0.0),
        key_color=(2.2, 2.3, 2.5),
        ambient_color=(0.8, 0.85, 0.95),
        key_softness=1.5,
        device=device,
    )


def night_back(device=None) -> torch.Tensor:
    return synthesize_directional_envmap(
        direction=(-0.4, 0.3, -1.0),
        key_color=(0.3, 0.4, 0.9),
        ambient_color=(0.02, 0.02, 0.04),
        key_softness=10.0,
        device=device,
    )


SYNTHETIC_ENVMAPS = {
    "studio_warm": studio_warm,
    "sunset_side": sunset_side,
    "overcast_top": overcast_top,
    "night_back": night_back,
}


def resolve_envmap(spec: str, device=None) -> tuple[str, torch.Tensor]:
    """Resolve a name from SYNTHETIC_ENVMAPS or an HDRI file path.

    Args:
        spec: Either a key of SYNTHETIC_ENVMAPS or a path to an .exr/.hdr/.png.
        device: Torch device.

    Returns:
        (name, envmap tensor).
    """
    if spec in SYNTHETIC_ENVMAPS:
        return spec, SYNTHETIC_ENVMAPS[spec](device=device)
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(
            f"Envmap '{spec}' is neither a known synthetic name nor an existing file."
        )
    return path.stem, load_envmap(path).to(device)
