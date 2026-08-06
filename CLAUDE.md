# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

InfiniSplat is inference code for a SIGGRAPH Asia 2026 paper on single-image 3D Gaussian reconstruction. It has two modes:

- `rgb`: RGB image only. Uses `checkpoints/infinisplat_rgb.ckpt`.
- `lidar`: RGB + sparse prompt depth. Uses `checkpoints/infinisplat_lidar.ckpt`.

There is no training code in this repository — only inference and demo UIs.

## Common commands

Run everything from the repository root; the code is a package under `src/`, not an installed distribution.

```bash
# Batch / single-image CLI inference
python -m src.demo.infer_batch_images --input examples/data/rgb_demo
python -m src.demo.infer_batch_images --mode lidar --input examples/data/lidar_demo
python -m src.demo.infer_batch_images --help          # authoritative flag list

# Local Gradio demo (http://127.0.0.1:7860)
python demo.py

# Download both released checkpoints into checkpoints/
bash scripts/download_checkpoints.sh
```

Environment setup is documented in `INSTALL.md` (Python 3.10, PyTorch 2.9 + CUDA 12.8 via `uv pip`, then `requirements.txt`). Two optional deps are handled at runtime:

- `gsplat` (installed via `pip install git+…gsplat --no-build-isolation`) — needed for novel-view MP4 rendering. Without it, video export is skipped with a warning; PLY export still works. Availability is probed via `src.model.decoder.decoder_gsplat.is_gsplat_available()`.
- `@playcanvas/splat-transform` (Node CLI) — needed for standalone HTML viewer export. Without it, HTML export is skipped.

There are no tests, linters, or CI configured in this repo. Do not invent commands for them.

## Architecture

### Config composition (Hydra)

`config/inference.yaml` is the root composed for demo inference. It pulls in one encoder config (`config/model/encoder/*.yaml`) and one decoder config (`config/model/decoder/*.yaml`). Mode-specific overrides live in `config/experiment/infinisplat_hypersim_{rgb,lidar}.yaml` and are applied via `+experiment=<name>`. `src/demo/config.py` converts the resolved `DictConfig` into typed dataclasses (`RootCfg` → `ModelCfg` → `EncoderCfg | DecoderCfg`) using `dacite`.

The comment "SINGLE SOURCE OF TRUTH" at the top of `config/model/decoder/gsplat.yaml` is enforced: the Python dataclasses declare types only, YAML holds defaults. When adding a field, update both.

### Registry pattern

Encoders and decoders are selected by string name in `src/model/{encoder,decoder}/__init__.py`:

- `ENCODERS = {"infinisplat": ..., "infinisplat_infinidepth": ...}` — dispatched via `get_encoder(cfg)` where `cfg.name` picks the class.
- `DECODERS = {"gsplat": ...}` — dispatched via `get_decoder(cfg)`.

Each encoder/decoder is `Encoder[Cfg]` / `Decoder[Cfg]` generic. Adding a new one means: dataclass + `Literal["…"]` name field, registry entry, and matching YAML in `config/model/`.

### Encoder-decoder pipeline

- **RGB mode** (`EncoderInfiniSplat`): DINOv3 (ViT-L/16 loaded via `torch.hub.load` from bundled `src/model/encoder/blocks/torchhub/dinov3`) + `BasicEncoder` low-res features + Apple `DepthPro` for depth → `ImplicitGSHead` → `GaussianDecoder` produces `Gaussians3D`.
- **Lidar mode** (`EncoderInfiniDepthQuery`): same DINO/basic branch, but depth comes from `InfiniDepth` conditioned on a sparse prompt (disparity + mask). Prompt handling is in `src/demo/infer_single_image.py::load_prompt_depth_tensors`.
- **Decoder** (`DecoderGsplat`): calls `gsplat.rasterization` for novel-view synthesis. Extrinsics are OpenCV world-to-camera; intrinsics passed to the decoder are normalized (fx/fy scaled by W/H internally). Output is sRGB after `linearRGB2sRGB`.

`Gaussians3D` (`src/utils/gaussians.py`) is the shared data class carrying mean, opacity, quaternion, scale, color, and optional covariance tensors. `save_ply` and `prepare_gaussians_for_ply_export` handle the PLY output format used by all UIs.

### Checkpoint loading

Checkpoints are single-file state dicts with `encoder.` and `decoder.` prefixes. `src/demo/infer_single_image.py` provides `_extract_state_dict`, `_load_prefixed_state_dict`, `load_demo_model`, and `load_demo_encoder`. Loading is strict — any missing or unexpected key raises. When adding modules to an encoder/decoder, the checkpoint must be regenerated with matching keys.

### Inference entry points

- `src/demo/infer_single_image.py` — the core: image loading, intrinsics resolution (file → focal-px → focal-mm → EXIF → 30 mm fallback), prompt-depth loading, model construction, single-image forward, floater filtering, novel-view video, HTML patching.
- `src/demo/infer_batch_images.py` — wraps the above for batch runs, adds progress reporting, per-case output dirs, and resume behavior (skips cases whose artifacts all exist unless `--overwrite`).
- `src/demo/hf_runtime.py` + `src/demo/hf_ui.py` — Gradio app used by `demo.py` and the Hugging Face Space. `INFINISPLAT_CHECKPOINT` env var overrides the checkpoint path; otherwise the runtime downloads from `PLUS-WAVE/InfiniSplat` on first run.

Fixed inference resolution is `1152 × 1536` (see `INFERENCE_HEIGHT`/`INFERENCE_WIDTH`). Outputs default to `outputs/demo/<mode>/<image_stem>/`. Video render size is capped at 3840 long edge / 3840×2160 area.

Prompt-depth constraints (see `docs/inference.md`): shape `[H, W]`, aligned to the RGB image, larger = farther, finite values strictly in `(1, 100)` after decoding, ≤1500 valid samples used. Disparity/inverse depth must be converted before passing in.
