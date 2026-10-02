# Showcase Re-Run — 4 July 2026

Every step of the face-retopology pipeline re-run end to end on an NVIDIA RTX 5070
GPU, with fresh outputs captured here. Nothing under `outputs/` or the prior
presentation deck was modified when this run happened — this folder was purely
additive. (A later repo cleanup pass moved raw meshes/checkpoints not needed by any
script into the sibling archive, keeping the metrics, run logs, and renders here —
see `MANIFEST.json`.)

Audience-facing renders used in `DL_Project_Showcase_2026_07_04.pptx` are in
[`renders/`](renders/). Raw command stdout for every step is in [`run_log/`](run_log/).

## What ran, and why some steps are reduced-scale

| # | Step | Scale | Why |
|---|------|-------|-----|
| 1 | `doctor` | full | dependency/GPU check |
| 2 | `dataset scan` | full (1,024 files) | cheap, re-scans real FaMoS registrations |
| 3 | `pairs generate` | **proof-scale, 24 pairs** | the canonical 1,000-pair set (`outputs/famos_flame_subset_pairs/`) already exists and is reused for training below; regenerating it fully takes hours on CPU (Wrap++ teacher) for no new information |
| 4 | `build-prior` | **proof-scale, from the 24 pairs** | headline 97.79%-variance/12-mode prior figure still cites the existing canonical `outputs/famos_deformation_prior/` |
| 5 | `train` (V2 + V3) | **full, 1,000 pairs, GPU** | genuinely retrained both DeformNet V2 (20 epochs) and DeformNet V3/GNN (30 epochs) from scratch today |
| 6 | `eval` (V2 + V3) | full, 100 held-out pairs | fresh evaluation of both fresh checkpoints |
| 7 | `mesh` (closed-loop) | full | see note below — hero image reuses the project's existing verified proof case; an additional harder case was also run fresh today |
| 8 | prior-guided completion | full | fresh run on a real held-out FaMoS registration |

## Important, honest note on Step 7

The project's headline closed-loop number (Chamfer **0.02020 → 0.00646**, normal
consistency **0.903**) comes from `outputs/facescape_clean_topology_case/`, an
already-committed proof case. Its original, un-decimated source mesh is not present
in this repository (only a decimated, harder-pose variant is: `data/facescape_real_sample/4_anger_decimated.obj`),
so that exact case cannot be re-derived from scratch here — it is reused for the
hero slide, re-rendered with the improved shader, and clearly cited.

To still produce a genuinely fresh, real, end-to-end run today, [`step_07_closed_loop_mesh/`](step_07_closed_loop_mesh/)
runs the same command against the harder decimated "anger" expression scan. See
[`step_07_closed_loop_mesh/NOTE.md`](step_07_closed_loop_mesh/NOTE.md) for the full
explanation. Its honest result — Chamfer 0.26505 → 0.13789 (~48% lower), normal
consistency 0.473 — is real, reproducible, and intentionally not used as the headline
number since it is a harder test case, not a like-for-like comparison. This image is
kept in this folder as supplementary evidence but is not in the audience deck.

## Headline results from this run

| Metric | Result | Source |
|---|---|---|
| DeformNet V2, mean Chamfer (100 held-out faces) | **0.02200** | `step_06_eval/v2/eval_report.json` |
| DeformNet V3 (GNN), mean Chamfer (100 held-out faces) | **0.01849** (~16% better than V2) | `step_06_eval/v3/eval_report.json` |
| V2 training | 20 epochs, 1,000 pairs, 164s on RTX 5070 | `step_05_train_deformnet/v2/train_metrics.json` |
| V3 training | 30 epochs, 1,000 pairs, 254s on RTX 5070 | `step_05_train_deformnet/v3/train_metrics.json` |
| Prior-guided completion | Chamfer 0.11996 → 0.06912 | `step_08_prior_completion/completion_metrics.json` |
| Closed-loop retopology (verified proof case, reused) | Chamfer 0.02020 → 0.00646 | `outputs/facescape_clean_topology_case/metrics.json` |
| Closed-loop retopology (fresh stress test, harder case) | Chamfer 0.26505 → 0.13789 | `step_07_closed_loop_mesh/metrics.json` |

DeformNet V3 (GNN) genuinely outperforms V2 here — this is a real measured result on
a full-scale (1,000-pair) training run, not a proof-of-concept subset, completing the
"next-scale experiment" the main README lists as future work.

## Reproducing this folder

All commands assume `D:\DL_project\.venv-demo\Scripts\python.exe` and
`$env:PYTHONUTF8=1`. See `run_log/*.txt` for the exact invocations used, or
`MANIFEST.json` for a machine-readable artifact index.
