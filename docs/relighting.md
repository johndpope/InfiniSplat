# Relighting Research Pipeline

Envmap-based IBL over InfiniSplat Gaussians, plus a wandb-logged sweep
harness for probing shading configurations across inputs and lights.

## What's in the box

- `src/utils/lighting.py` — SH-9 envmap projection, split-sum specular mips,
  Fresnel-Schlick, density-based AO, per-Gaussian normal proxy from the
  shortest principal axis, and the unified `shade_gaussians()` shader.
- `src/demo/relight.py` — single-scene CLI. Encodes once, rotates the
  envmap, renders a video. Best for eyeballing.
- `src/research/` — the sweep harness (see below).

## Single-scene relighting

```bash
# Use a bundled synthetic envmap (no file needed).
python -m src.demo.relight --input examples/data/rgb_demo/pexels-masi.jpg

# Use a real HDR file. Both .hdr (Radiance) and .exr are supported.
python -m src.demo.relight \
  --input photo.jpg \
  --envmap hdris/studio_small_09_2k.hdr \
  --enable-specular --enable-ao --compare

# Sweep just the roughness knob without a research run.
python -m src.demo.relight --input photo.jpg --enable-specular --roughness 0.2
```

Outputs land in `outputs/relight/<stem>_relight.mp4`. `--compare` writes a
3-panel `baked | diffuse | full-IBL` MP4.

## Research sweeps (recommended for iteration)

The matrix runner takes a set of inputs, a set of envmaps, and a shading
preset, and runs the cartesian product with the encoder loaded once. Each
cell writes an MP4, a baked-reference PNG, a JSON of metrics, and (if
online) a wandb log entry.

```bash
source ~/.secrets  # exposes WANDB_API_KEY
python -m src.research.run_matrix \
  --inputs examples/data/rgb_demo/pexels-masi.jpg examples/data/rgb_demo/bedroom.jpg \
  --envmaps studio_warm sunset_side overcast_top \
  --shading-preset small \
  --frames 16 --max-long-edge 640 \
  --project infinisplat-relight
```

Shading presets:

- `small` — 4 configs: diffuse, diffuse+demod0.5, diffuse+spec, full (spec+AO).
- `ablate` — the 4 above plus roughness sweep (0.15, 0.35, 0.75) and AO
  strength sweep (1.0, 3.0, 5.0). 10 configs.

Envmap specs are either:

- A key of `SYNTHETIC_ENVMAPS` in `src/research/envmaps.py`
  (`studio_warm`, `sunset_side`, `overcast_top`, `night_back`), or
- A path to a `.hdr` / `.exr` file. HDRIs from
  [Poly Haven](https://polyhaven.com/hdris) work well.

## Adding a real HDRI

Drop the file anywhere and reference it:

```bash
mkdir -p hdris/
# Grab a 2K HDRI from Poly Haven, then:
python -m src.research.run_matrix \
  --inputs examples/data/rgb_demo/pexels-masi.jpg \
  --envmaps studio_warm hdris/kloppenheim_06_puresky_2k.hdr \
  --shading-preset small --frames 24
```

Both `.hdr` (Radiance) and `.exr` are supported. `.exr` uses the `OpenEXR`
Python package as a fallback when OpenCV wasn't built with EXR (which is
the case on this box).

Envmap conventions:

- Lat-long layout, row 0 = north pole.
- Column 0 = azimuth 0 (image left = `+x` in light space).
- Rotating the envmap around `+y` (up) sweeps the light horizontally, which
  is what the matrix runner does. Symmetric envmaps around `+y` (e.g. pure
  overcast) will show `temporal_std = 0`, correctly.

## Metrics (per cell)

Written to `<cell>.json` and logged to wandb:

- **temporal_std** — mean per-pixel std over frames. Rotation-driven.
- **relight_effectiveness** — `temporal_std / baked_brightness_mean`.
  How much the shading is actually moving pixels. Sanity floor of ~0.05.
- **brightness_mean / span / p02 / p98** — level and dynamic range.
- **clip_dark_frac / clip_bright_frac** — saturated pixel fractions.
- **silhouette_drift** — 1 - IoU between baked and mean-relit foreground
  masks. Non-zero here often means shading pushed subject into background.
- **ssim_mean_vs_baked** — grayscale SSIM of mean relit vs baked ref.
- **lpips_mean_vs_baked** — LPIPS AlexNet perceptual distance vs baked.
- **specular_delta** — mean `|full - diffuse|`, populated when both cells
  exist for the same (input, envmap).
- **peak_freq / peak_ratio** — FFT peak of frame-average brightness. A
  rotation of an asymmetric envmap should give `peak_freq = 1`.

## HTML report

Scans a run directory and emits `index.html` with a sortable table and a
grid of playable cells:

```bash
python -m src.research.report --latest
# or
python -m src.research.report --run-dir outputs/research/run-YYYYMMDD-HHMMSS
open outputs/research/run-YYYYMMDD-HHMMSS/index.html
```

The report is fully offline and inlines nothing — it uses relative paths to
the MP4s and PNGs already in the run dir, so the whole directory can be
zipped and shared.

## Extending the harness

- **New envmap**: add a function in `src/research/envmaps.py` and register
  it in `SYNTHETIC_ENVMAPS`, or just pass a file path.
- **New shading knob**: add a field to `ShadingConfig` in
  `src/research/pipeline.py`, thread it into `shade_gaussians()`, then
  create configs in `run_matrix.py`'s preset functions.
- **New metric**: add a function in `src/research/metrics.py` and call it
  from `summarize_cell()`. All downstream reporting picks it up automatically.
