from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .geometry import Mesh


SEMANTIC_COLORS = {
    "forehead": [0.76, 0.61, 0.52],
    "brows": [0.28, 0.20, 0.17],
    "eyes": [0.83, 0.86, 0.86],
    "nose": [0.82, 0.62, 0.52],
    "lips": [0.63, 0.25, 0.30],
    "cheeks": [0.78, 0.56, 0.47],
    "jaw": [0.68, 0.49, 0.42],
    "ears_neck": [0.60, 0.45, 0.39],
}


def _mesh_payload(mesh: Mesh, reference: Mesh | None = None, max_faces: int = 50000) -> dict[str, object]:
    faces = mesh.faces
    if len(faces) > max_faces:
        step = int(np.ceil(len(faces) / max_faces))
        faces = faces[::step]
    colors = mesh.colors
    if colors is None:
        colors = np.full((mesh.vertex_count, 3), [0.72, 0.54, 0.46], dtype=np.float64)
    semantic_colors = colors
    if mesh.semantics is not None:
        semantic_colors = np.asarray([SEMANTIC_COLORS.get(str(s), [0.72, 0.54, 0.46]) for s in mesh.semantics], dtype=np.float64)
    heat = np.zeros(mesh.vertex_count, dtype=np.float64)
    if reference is not None and reference.vertex_count > 0:
        tree = cKDTree(reference.vertices)
        dist, _ = tree.query(mesh.vertices, k=1)
        scale = np.percentile(dist, 95) if len(dist) else 1.0
        heat = np.clip(dist / max(float(scale), 1e-9), 0.0, 1.0)
    return {
        "name": mesh.name,
        "vertices": np.round(mesh.vertices, 5).tolist(),
        "faces": faces.astype(int).tolist(),
        "colors": np.round(colors, 4).tolist(),
        "semanticColors": np.round(semantic_colors, 4).tolist(),
        "heat": np.round(heat, 4).tolist(),
    }


def image_data_uri(path: str | Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower().lstrip(".") or "png"
    if suffix == "jpg":
        suffix = "jpeg"
    return f"data:image/{suffix};base64,{encoded}"


def write_viewer(out_path: str | Path, title: str, meshes: list[Mesh], metrics: dict[str, object], photo_path: str | Path | None = None, ablation_report: dict[str, object] | None = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": title,
        "meshes": [_mesh_payload(m, meshes[0] if i else None) for i, m in enumerate(meshes)],
        "metrics": metrics,
        "ablation": ablation_report,
        "photo": image_data_uri(photo_path),
    }
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(payload))
    out_path.write_text(html, encoding="utf-8")
    return out_path


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face Semantic ICP Wrap</title>
<style>
:root { color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; }
body { margin: 0; background: #f6f7f9; color: #1d232b; }
header { height: 56px; display: flex; align-items: center; justify-content: space-between; padding: 0 18px; border-bottom: 1px solid #d9dee7; background: #ffffff; }
h1 { font-size: 18px; margin: 0; font-weight: 650; }
.controls { display: flex; gap: 12px; align-items: center; font-size: 13px; }
button { border: 1px solid #c5ccd8; background: #fff; border-radius: 6px; padding: 7px 10px; cursor: pointer; }
button:hover { background: #eef3f8; }
main { display: grid; grid-template-columns: minmax(0, 1fr) 320px; min-height: calc(100vh - 57px); }
.grid { display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 12px; padding: 12px; }
.panel { background: #fff; border: 1px solid #d9dee7; border-radius: 8px; overflow: hidden; min-height: 310px; display: flex; flex-direction: column; }
.panel-title { height: 34px; display: flex; align-items: center; padding: 0 10px; border-bottom: 1px solid #e4e8ef; font-size: 13px; font-weight: 650; }
canvas { width: 100%; height: 100%; min-height: 276px; display: block; background: linear-gradient(#fbfcfd, #edf1f5); }
aside { border-left: 1px solid #d9dee7; background: #ffffff; padding: 14px; overflow: auto; }
.photo { width: 100%; border-radius: 8px; border: 1px solid #d9dee7; margin-bottom: 12px; }
.metric { display: flex; justify-content: space-between; gap: 12px; padding: 7px 0; border-bottom: 1px solid #edf0f4; font-size: 13px; }
.metric span:first-child { color: #596473; }
.chart { width: 100%; height: 110px; border: 1px solid #dce2ea; border-radius: 8px; margin: 10px 0; background: #f8fafc; }
pre { white-space: pre-wrap; word-break: break-word; background: #f3f5f8; border: 1px solid #dce2ea; border-radius: 8px; padding: 10px; font-size: 12px; }
@media (max-width: 940px) { main { grid-template-columns: 1fr; } aside { border-left: 0; border-top: 1px solid #d9dee7; } .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1 id="title">Face Semantic ICP Wrap</h1>
  <div class="controls">
    <label>Rotate <input id="angle" type="range" min="-180" max="180" value="-18"></label>
    <select id="mode" title="Color mode">
      <option value="texture">Texture</option>
      <option value="semantic">Semantic</option>
      <option value="heat">Delta</option>
    </select>
    <button id="auto">Auto</button>
  </div>
</header>
<main>
  <section class="grid" id="grid"></section>
  <aside>
    <img class="photo" id="photo" alt="input photo" style="display:none">
    <div id="metrics"></div>
    <pre id="raw"></pre>
  </aside>
</main>
<script>
const DATA = __DATA__;
const grid = document.getElementById("grid");
document.getElementById("title").textContent = DATA.title;
if (DATA.photo) {
  const img = document.getElementById("photo");
  img.src = DATA.photo;
  img.style.display = "block";
}

const labels = ["Input / Proxy", "Initial FLAME", "Semantic ICP", "Detail + Texture"];
const canvases = DATA.meshes.map((mesh, i) => {
  const panel = document.createElement("div");
  panel.className = "panel";
  const title = document.createElement("div");
  title.className = "panel-title";
  title.textContent = labels[i] || mesh.name;
  const canvas = document.createElement("canvas");
  panel.appendChild(title);
  panel.appendChild(canvas);
  grid.appendChild(panel);
  return canvas;
});

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
}

function shade(color, amount) {
  return `rgb(${Math.round(255 * color[0] * amount)},${Math.round(255 * color[1] * amount)},${Math.round(255 * color[2] * amount)})`;
}

function heatColor(t) {
  t = Math.max(0, Math.min(1, t));
  return [0.15 + 0.80 * t, 0.75 * (1 - Math.abs(t - 0.5) * 1.5), 0.95 * (1 - t)];
}

function drawMesh(canvas, mesh, angleDeg) {
  resizeCanvas(canvas);
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const angle = angleDeg * Math.PI / 180;
  const ca = Math.cos(angle), sa = Math.sin(angle);
  const verts = mesh.vertices.map(p => {
    const x = p[0] * ca + p[2] * sa;
    const z = -p[0] * sa + p[2] * ca;
    const y = p[1];
    const scale = Math.min(w, h) * 0.43 / (1.8 + z * 0.18);
    return [w * 0.5 + x * scale, h * 0.55 - y * scale, z];
  });
  const tris = mesh.faces.map(f => {
    const a = verts[f[0]], b = verts[f[1]], c = verts[f[2]];
    const depth = (a[2] + b[2] + c[2]) / 3;
    return [depth, f, a, b, c];
  }).sort((a, b) => b[0] - a[0]);
  for (const tri of tris) {
    const f = tri[1], a = tri[2], b = tri[3], c = tri[4];
    const mode = document.getElementById("mode").value;
    let col = mesh.colors[f[0]] || [0.7, 0.52, 0.45];
    if (mode === "semantic") col = mesh.semanticColors[f[0]] || col;
    if (mode === "heat") col = heatColor(mesh.heat[f[0]] || 0);
    const light = 0.74 + 0.24 * Math.max(0, tri[0] + 0.3);
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.lineTo(c[0], c[1]);
    ctx.closePath();
    ctx.fillStyle = shade(col, Math.min(1.0, light));
    ctx.fill();
  }
}

function render() {
  const angle = Number(document.getElementById("angle").value);
  DATA.meshes.forEach((m, i) => drawMesh(canvases[i], m, angle));
}

let auto = false;
document.getElementById("auto").onclick = () => { auto = !auto; };
document.getElementById("angle").oninput = render;
document.getElementById("mode").onchange = render;
window.onresize = render;
function tick() {
  if (auto) {
    const slider = document.getElementById("angle");
    slider.value = ((Number(slider.value) + 1 + 180) % 360) - 180;
    render();
  }
  requestAnimationFrame(tick);
}

const metrics = document.getElementById("metrics");
for (const key of ["chamfer_initial_to_source", "chamfer_wrapped_to_source", "chamfer_detailed_to_source", "normal_consistency_detailed", "landmark_rmse"]) {
  if (DATA.metrics[key] === undefined || DATA.metrics[key] === null) continue;
  const row = document.createElement("div");
  row.className = "metric";
  const value = Number(DATA.metrics[key]);
  row.innerHTML = `<span>${key}</span><strong>${Number.isFinite(value) ? value.toFixed(5) : DATA.metrics[key]}</strong>`;
  metrics.appendChild(row);
}
const chart = document.createElement("canvas");
chart.className = "chart";
metrics.appendChild(chart);
function drawChart() {
  const logs = DATA.metrics.stage_logs || [];
  const rect = chart.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  chart.width = Math.max(1, Math.round(rect.width * dpr));
  chart.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = chart.getContext("2d");
  ctx.clearRect(0, 0, chart.width, chart.height);
  ctx.strokeStyle = "#b8c2d1";
  ctx.strokeRect(0.5, 0.5, chart.width - 1, chart.height - 1);
  if (!logs.length) return;
  ctx.beginPath();
  logs.forEach((row, i) => {
    const x = 14 + i * (chart.width - 28) / Math.max(1, logs.length - 1);
    const y = chart.height - 14 - (row.accepted_ratio || 0) * (chart.height - 28);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#2076d2";
  ctx.lineWidth = 3;
  ctx.stroke();
}
if (DATA.ablation) {
  const block = document.createElement("pre");
  block.textContent = "Ablation\n" + JSON.stringify(DATA.ablation.variants || [], null, 2);
  metrics.appendChild(block);
}
document.getElementById("raw").textContent = JSON.stringify(DATA.metrics, null, 2);
render();
drawChart();
tick();
</script>
</body>
</html>
"""
