# Fixed-Topology 3D Face Retopology — Architecture Comparison

**Deep Learning Mini Project.** We compare four deep-learning architecture families for
**fixed-topology face retopology**: turning an arbitrary face mesh (any vertex count,
ordering, connectivity) into a single shared template topology (FLAME-style,
5,023 vertices / 9,976 faces) while preserving identity, under one common protocol.

> Problem in one line: raw 3D face capture gives meshes with inconsistent topology, which
> blocks animation / avatar / tracking / statistical-model pipelines that all need one
> shared template. Classical nearest-point ICP mis-corresponds (brow↔eye, cheek↔lip) and
> breaks on partial or noisy input. We study which architecture handles this best.

## Team & model-to-member assignment

| Member | Model family (owned) | Where in code |
|---|---|---|
| **Anshul Shukla** | Geometric Semantic-ICP baseline (Wrap++ teacher) | `face_semantic_icp/semantic_icp.py`, `pipeline.py` |
| **Ayush Mishra** | PointNet-style DeformNet (point-encoder MLP) | `face_semantic_icp/train.py` (model v2), `deformation_prior.py` |
| **Kushal Bayaskar** | PCA statistical deformation prior + prior-guided completion | `face_semantic_icp/deformation_prior.py`, `scripts/learn_famos_deformation_prior.py`, `scripts/run_prior_guided_completion.py` |
| **Naman Mehta** | GNN DeformNet (graph convolution over template) | `face_semantic_icp/train.py` (model v3) |

Each member owns one model and defends it individually.

## The four architecture families

1. **Geometric Semantic-ICP baseline (Wrap++).** Non-learned, visibility-aware ICP
   constrained by facial semantics + landmarks + bidirectional checks. Also the **teacher**
   that generates supervised source→template pairs.
2. **PointNet-style DeformNet (v2).** Multi-scale PointNet over sampled source points →
   per-vertex template displacement + per-vertex uncertainty.
3. **PCA statistical deformation prior.** SVD deformation subspace learned from real FLAME
   registrations (12 modes); used for prior-guided completion of partial/noisy input and as
   an ICP regularizer.
4. **GNN DeformNet (v3).** PointNet source stream **+** graph-convolution stream over the
   template mesh adjacency → topology-aware displacement + uncertainty.

## Results (common protocol)

Learned models share one protocol: **100-pair FaMoS eval subset, identical split, same
metrics.** Chamfer distance lower is better; normal consistency higher is better.

| Model family | Task / eval set | Chamfer ↓ | Normal cons. ↑ | Notes |
|---|---|---:|---:|---|
| Semantic-ICP Wrap++ (teacher) | FaceScape clean case | **0.0065** | **0.90** | geometric upper bound; slow (~6 s/mesh) |
| PointNet DeformNet (v2) | FaMoS 100-pair | 0.0220 | 0.83 | 20 epochs, 1,000 train pairs, ~1 ms/mesh |
| GNN DeformNet (v3) | FaMoS 100-pair | **0.0185** | 0.84 | 30 epochs, graph conv; best learned model |
| PCA prior completion | FaMoS partial (34% kept) | 0.120 → **0.069** | 0.70 | reconstructs full mesh from partial obs |

**Why they differ:** the Semantic-ICP teacher is most accurate but iterative and slow; the
PointNet student learns a fast amortized map but ignores mesh structure; adding graph
convolution over the template (v3) lowers Chamfer ~16% vs the PointNet baseline by using
topology; the PCA prior is the only family that completes missing geometry. Numbers come
from the committed reports under `outputs/` and `showcase_2026_07_04/` — not hand-entered.

## Run it

```bash
pip install -r requirements.txt
# optional 3D render deps:
pip install -r requirements-visualization.txt

# Environment check
python -m face_semantic_icp doctor

# Wrap++ Semantic-ICP on a single mesh (baseline / teacher)
python -m face_semantic_icp mesh SOURCE.obj --template TEMPLATE.obj --out outputs/run_icp --quality balanced

# Train a DeformNet (v2 = PointNet, v3 = GNN) on generated pairs
python -m face_semantic_icp train --pairs PAIRS.jsonl --out outputs/train --epochs 20 --model-version v2
python -m face_semantic_icp train --pairs PAIRS.jsonl --out outputs/train_v3 --epochs 30 --model-version v3

# Evaluate a checkpoint
python -m face_semantic_icp eval --checkpoint outputs/train/deformnet.pt --pairs PAIRS.jsonl --out outputs/eval --baseline wrap++

# Learn the PCA deformation prior + run prior-guided completion
python scripts/learn_famos_deformation_prior.py
python scripts/run_prior_guided_completion.py --prior outputs/famos_deformation_prior/famos_deformation_prior.npz --pairs PAIRS.jsonl --out outputs/completion

# Plot per-mode / cumulative explained variance of the learned prior (reads prior_summary.json; no dataset needed)
python scripts/plot_prior_explained_variance.py

# Pair-by-pair comparison of two eval reports on the same pairs (default: PointNet v2 vs GNN v3 showcase evals)
python scripts/compare_eval_reports.py --out outputs/compare_v2_v3
```

See **[DATASET_ACCESS.md](DATASET_ACCESS.md)** for dataset commands and the synthetic
no-download fallback.

## Dataset

**Primary: FaMoS FLAME registrations** — 605,802 registered meshes at fixed FLAME topology
(5,023 v / 9,976 f), 95 subjects, 28 expression sequences. We use a controlled ~1,000-pair
subset. **Secondary: FaceScape sample** — raw multi-view mesh used as a before/after wrap
proof case.

> ⚠️ **Licensing.** FaMoS and FaceScape are license-gated. Raw dataset files and the
> 1.1 GB pair tensors are **not** committed — download them locally after accepting each
> dataset license (see `DATASET_ACCESS.md`). The repo ships only derived metrics, trained
> checkpoints, renders, and the small proof folders under `outputs/`.

## Repo map

```
face_semantic_icp/   core package: Wrap++ ICP, PCA prior, DeformNet (v2/v3), pipeline, metrics, CLI
scripts/             prior learning, prior-guided completion, render/visualization helpers
configs/             training + dataset configs
outputs/             committed proof folders: eval reports, trained checkpoint, PCA prior, completion
showcase_2026_07_04/ full end-to-end run: logs, metrics JSON, and step renders
tests/               pipeline unit tests
DATASET_ACCESS.md    dataset commands + license notes + synthetic fallback
plan_readme.md       team Git workflow
```

## Honesty notes (academic integrity)

- The building blocks are **established** methods — 3DMM/PCA priors, PointNet, graph
  convolution (GCN), and ICP. Our contribution is the **comparison and integration** for
  fixed-topology face retopology (semantic + visibility-aware ICP teacher, PCA-regularized
  completion, and a topology-aware GNN student), evaluated cross-subject.
- Development was **LLM-assisted** (code scaffolding and docs); all results in the tables
  come from the committed run artifacts, not hand-entered numbers.
- Established external methods referenced: FLAME, 3DMM (Blanz & Vetter), non-rigid ICP
  (Amberg et al.), PointNet (Qi et al.), GCN (Kipf & Welling), CoMA (Ranjan et al.).
