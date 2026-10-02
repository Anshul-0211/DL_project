# GNN DeformNet (v3) — model owner: Naman Mehta

This is the owner write-up for the **GNN DeformNet** family in the four-model comparison.
Like the PointNet DeformNet (v2), it maps an unordered source point set to the fixed FLAME
topology (5,023 vertices) in one forward pass. Unlike v2, it also runs a **graph
convolution over the template mesh**, so each vertex's prediction is informed by its
mesh neighbours.

Code: [`face_semantic_icp/train.py`](../face_semantic_icp/train.py) →
`build_model(..., version=3, template_faces=...)` (`PointDeformNetV3`), adjacency built by
`build_gcn_adj`.

## Architecture — two streams

Input: `points` `(B, N, 3)` sampled from the source mesh, and the template rest vertices
`(V, 3)`. Output: deformed template `(B, V, 3)`.

1. **Stream A — source encoder (PointNet, same as v2).** Three shared per-point MLP levels
   (`enc1`, `enc2`, `enc3`), each max-pooled over the N points, concatenated and
   aggregated into one permutation-invariant latent code `z ∈ R^latent` (`_encode_source`).
2. **Stream B — template encoder (GCN).** Three graph-convolution layers over the template
   mesh (`gcn1: 3→hidden`, `gcn2: hidden→hidden`, `gcn3: hidden→latent`), each computing
   `H' = Â · H · W` with the Kipf & Welling normalised adjacency

   ```
   Â = D^{-1/2} (A + I) D^{-1/2}
   ```

   where `A` is the template's edge adjacency (from the face list) and `I` adds self-loops.
   `Â` is built once with scipy as a sparse tensor (no PyTorch Geometric dependency) and
   registered as a buffer, so it moves with `.to(device)` and is saved in the checkpoint.
   Three layers give each vertex a 3-ring receptive field over the face surface.
3. **Decoder.** For every template vertex, the broadcast source code `z` is concatenated
   with that vertex's GCN feature `g_v` and passed through a 2-layer MLP trunk. Two heads
   predict a **displacement** (`pred = template + head_disp(trunk)`) and a per-vertex
   log-std for uncertainty (`σ = exp(logσ)`, clamped to `[0.01, 10]`).

**Why it helps:** in v2 every vertex is decoded independently from its rest position, so
neighbouring vertices only agree through the Laplacian loss. In v3 the per-vertex features
already carry their neighbourhood structure, so the decoder can produce locally coherent
deformations (e.g. around eyes and lips) directly.

Size at the default dims (`latent_dim=96`, `hidden_dim=128`): **134,084 trainable
parameters** vs **93,124** for v2 (from `count_parameters`). Parameter count does not depend
on the vertex count — `Â` is a buffer, not a parameter.

## Loss

Same objective as v2 (see [`pointnet_deformnet.md`](pointnet_deformnet.md)):
heteroscedastic NLL with the predicted σ, plus Laplacian smoothness on the displacement, a
small Chamfer term, and landmark consistency. Keeping the loss identical means the v2→v3
difference isolates the effect of the graph stream.

## Result (common protocol)

100-pair FaMoS eval subset, 1,000 training pairs, 30 epochs. Source:
`showcase_2026_07_04/step_05_train_deformnet/v3/train_metrics.json` and
`showcase_2026_07_04/step_06_eval/v3/eval_report.json`.

| Metric | v2 (PointNet) | v3 (GNN) |
|---|---:|---:|
| Mean Chamfer ↓ | 0.0220 | **0.0185** |
| Mean normal consistency ↑ | 0.818 | **0.839** |
| Median inference / mesh (eval run) | 1.4 ms | 1.3 ms |
| Training time (1,000 pairs, GPU) | 164 s / 20 epochs | 254 s / 30 epochs |

**Paired comparison** (same 100 pairs, `python scripts/compare_eval_reports.py`):

| Metric | Change of means | Median per-pair gain | v3 better on |
|---|---:|---:|---:|
| Chamfer | −15.9% | +14.6% | 96 / 100 pairs |
| Normal consistency | +2.5% | +2.2% | 84 / 100 pairs |

So the Chamfer improvement is consistent across faces, not driven by a few outliers.

**Caveat for a fair reading:** v3 was trained for 30 epochs and v2 for 20 in the committed
runs, so part of the gap may come from the longer schedule. An equal-epoch rerun would
separate the two effects.

Reproduce:

```bash
python -m face_semantic_icp train --pairs PAIRS.jsonl --out outputs/train_v3 \
  --epochs 30 --model-version 3
python -m face_semantic_icp eval --checkpoint outputs/train_v3/deformnet.pt \
  --pairs PAIRS.jsonl --out outputs/eval_v3
```

Unit tests for the v3 graph operator and checkpoint round-trip:
[`tests/test_gnn_deformnet.py`](../tests/test_gnn_deformnet.py).

## vs the other families

- **vs PointNet DeformNet (v2):** same source encoder, loss and inference speed; the added
  template GCN stream is the only architectural difference and gives the lower Chamfer
  above.
- **vs Semantic-ICP teacher:** still less accurate than the iterative teacher on the clean
  case (0.0185 vs 0.0065, different eval sets), but runs in ~1 ms instead of ~6 s per mesh.
- **vs PCA prior completion:** v3 assumes a reasonably complete source; it has no explicit
  shape prior for filling large missing regions, which is what the PCA family is for.

## Limitations

- The graph is the fixed template mesh, so the GCN stream only encodes the *template's*
  structure; the source is still seen as an unordered point cloud.
- `Â` is stored sparse, but its edge list is built with a Python loop over faces — fine for FLAME (9,976 faces),
  slower for much larger templates.
