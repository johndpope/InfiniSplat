"""Static HTML report generator for a matrix-runner output directory.

Scans `outputs/research/run-*/` (or a directory passed via --run-dir),
gathers all per-cell JSON + MP4 + baked PNG artifacts, and emits an
index.html with:

- A sortable metrics table (one row per cell).
- Grid of cell cards: baked PNG thumbnail + inline MP4 + top metrics.
- Grouping by (input, envmap) so shading variants sit next to each other.

Standalone: works with just the files under the run dir. No wandb dependency.

Usage:
    python -m src.research.report --run-dir outputs/research/run-20260806-XXXXXX
    python -m src.research.report --latest
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


TABLE_COLUMNS = [
    "cell", "input", "envmap", "shading",
    "temporal_std", "relight_effectiveness",
    "brightness_mean", "brightness_span",
    "ssim_mean_vs_baked", "lpips_mean_vs_baked",
    "specular_delta", "silhouette_drift",
    "clip_bright_frac", "seconds",
]


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if v != v:  # NaN
            return ""
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:.1f}"
    if v is None:
        return ""
    return html.escape(str(v))


def _load_cell_metrics(run_dir: Path) -> list[dict]:
    """Collect all per-cell JSON files under run_dir, sorted by cell key."""
    cells = []
    for json_path in sorted(run_dir.rglob("*.json")):
        if json_path.name == "summary.json":
            continue
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            continue
        cell = data.get("cell") or json_path.stem
        # Relative paths for HTML rendering.
        mp4 = json_path.with_suffix(".mp4")
        baked = json_path.with_name(f"{cell}__baked.png")
        data["_mp4_rel"] = str(mp4.relative_to(run_dir)) if mp4.exists() else None
        data["_baked_rel"] = str(baked.relative_to(run_dir)) if baked.exists() else None
        cells.append(data)
    return cells


def _group_cells(cells: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for c in cells:
        key = (c.get("input", ""), c.get("envmap", ""))
        grouped.setdefault(key, []).append(c)
    return grouped


HTML_HEAD = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>InfiniSplat Relight — __TITLE__</title>
<style>
body { font-family: -apple-system, Segoe UI, sans-serif; margin: 24px; background: #0d1117; color: #c9d1d9; }
h1, h2 { color: #f0f6fc; }
h2 { border-bottom: 1px solid #30363d; padding-bottom: 4px; margin-top: 32px; }
table { border-collapse: collapse; margin: 16px 0; font-size: 12px; }
th, td { padding: 4px 8px; border: 1px solid #30363d; text-align: right; }
th { background: #161b22; cursor: pointer; user-select: none; }
th:first-child, td:first-child { text-align: left; }
tr:hover { background: #161b22; }
.card { display: inline-block; vertical-align: top; margin: 8px; padding: 8px;
        background: #161b22; border: 1px solid #30363d; border-radius: 6px;
        width: 320px; }
.card h3 { margin: 0 0 6px 0; font-size: 12px; color: #58a6ff; word-break: break-all; }
.card img, .card video { width: 100%; border-radius: 4px; }
.card .metrics { font-size: 11px; margin-top: 6px; }
.card .metrics span { display: inline-block; margin-right: 8px; color: #8b949e; }
.card .metrics strong { color: #f0f6fc; }
.small { color: #6e7681; font-size: 11px; }
</style>
<script>
function sortTable(tableId, colIdx) {
  const table = document.getElementById(tableId);
  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.rows);
  const dir = table.dataset.sortDir === 'asc' ? -1 : 1;
  rows.sort((a, b) => {
    let av = a.cells[colIdx].textContent.trim();
    let bv = b.cells[colIdx].textContent.trim();
    const an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return dir * (an - bn);
    return dir * av.localeCompare(bv);
  });
  rows.forEach(r => tbody.appendChild(r));
  table.dataset.sortDir = dir > 0 ? 'asc' : 'desc';
}
</script>
</head><body>
<h1>InfiniSplat Relight — __TITLE__</h1>
<div class=small>__SUBTITLE__</div>
"""

HTML_TAIL = "</body></html>\n"


def _render_table(cells: list[dict]) -> str:
    if not cells:
        return "<p>No cells found.</p>"
    header = "".join(
        f'<th onclick="sortTable(\'metrics-table\', {i})">{html.escape(c)}</th>'
        for i, c in enumerate(TABLE_COLUMNS)
    )
    rows = []
    for c in cells:
        cells_html = "".join(f"<td>{_fmt(c.get(k))}</td>" for k in TABLE_COLUMNS)
        rows.append(f"<tr>{cells_html}</tr>")
    return (
        f'<h2>Metrics ({len(cells)} cells) — click headers to sort</h2>'
        f'<table id="metrics-table" data-sort-dir="asc">'
        f'<thead><tr>{header}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def _render_group(cells: list[dict], input_name: str, envmap: str) -> str:
    parts = [f"<h2>{html.escape(input_name)} — {html.escape(envmap)}</h2>"]
    for c in sorted(cells, key=lambda x: x.get("shading", "")):
        mp4 = c.get("_mp4_rel")
        baked = c.get("_baked_rel")
        tstd = c.get("temporal_std", float("nan"))
        eff = c.get("relight_effectiveness", float("nan"))
        ssim = c.get("ssim_mean_vs_baked", float("nan"))
        lpips = c.get("lpips_mean_vs_baked", float("nan"))
        spec = c.get("specular_delta")
        shading = c.get("shading", "")
        video_tag = (
            f'<video src="{html.escape(mp4)}" controls loop muted autoplay playsinline></video>'
            if mp4 else '<div class=small>(no mp4)</div>'
        )
        baked_tag = (
            f'<img src="{html.escape(baked)}" alt="baked">'
            if baked else ''
        )
        spec_str = f'<span>specΔ <strong>{_fmt(spec)}</strong></span>' if spec is not None else ''
        parts.append(f"""
<div class="card">
  <h3>{html.escape(shading)}</h3>
  {video_tag}
  <div class=small>baked reference:</div>
  {baked_tag}
  <div class="metrics">
    <span>tstd <strong>{_fmt(tstd)}</strong></span>
    <span>eff <strong>{_fmt(eff)}</strong></span>
    <span>ssim <strong>{_fmt(ssim)}</strong></span>
    <span>lpips <strong>{_fmt(lpips)}</strong></span>
    {spec_str}
  </div>
</div>
""")
    return "".join(parts)


def build_report(run_dir: Path) -> Path:
    cells = _load_cell_metrics(run_dir)
    grouped = _group_cells(cells)

    title = run_dir.name
    subtitle = f"{len(cells)} cells across {len(grouped)} (input, envmap) groups"

    header = HTML_HEAD.replace("__TITLE__", html.escape(title)).replace("__SUBTITLE__", html.escape(subtitle))
    body_parts = [header]
    body_parts.append(_render_table(cells))
    for (input_name, envmap), group in sorted(grouped.items()):
        body_parts.append(_render_group(group, input_name, envmap))
    body_parts.append(HTML_TAIL)

    out = run_dir / "index.html"
    out.write_text("".join(body_parts))
    return out


def _find_latest_run(root: Path) -> Path | None:
    candidates = sorted(root.glob("run-*"))
    return candidates[-1] if candidates else None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Emit index.html for a matrix run directory.")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--latest", action="store_true",
                   help="Use the most recent run under outputs/research/")
    p.add_argument("--root", type=Path, default=Path("outputs/research"),
                   help="Search root when --latest is passed.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.latest:
        run_dir = _find_latest_run(args.root)
        if run_dir is None:
            raise SystemExit(f"No runs found under {args.root}")
    elif args.run_dir is not None:
        run_dir = args.run_dir
    else:
        raise SystemExit("Pass --run-dir or --latest")

    print(f"[report] scanning {run_dir}")
    out = build_report(run_dir)
    print(f"[report] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
