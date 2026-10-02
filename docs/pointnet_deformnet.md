# PointNet-style DeformNet (v2) — model owner: Ayush Mishra

This is the owner write-up for the **PointNet DeformNet** family in the four-model
comparison. It predicts a fixed-topology FLAME mesh (5,023 vertices) directly from an
unordered source point set, as a fast amortized alternative to the iterative Semantic-ICP
teacher.

Code: [`face_semantic_icp/train.py`](../face_semantic_icp/train.py) →
`build_model(..., version=2)` (`PointDeformNetV2`).

## Architecture

Input: `points` `(B, N, 3)` sampled from the source mesh. Output: deformed template
`(B, V, 3)` with `V = 5023`.

1. **Multi-scale PointNet encoder.** Three shared per-point MLP levels
   (`enc1: 3→hidden`, `enc2: hidden→hidden`, `enc3: hidden→latent`), each followed by a
   **max-pool over the N points** to a permutation-invariant global feature. The three
   pooled vectors are concatenated and aggregated (`agg`) into one latent code
   `z ∈ R^latent`. Pooling at three depths captures coarse-to-fine source shape, unlike a
   single-level PointNet.
2. **Per-vertex decoder trunk.** For each template vertex we concatenate its rest position
   `(3)` with the broadcast latent `z` and pass it through a 2-layer MLP (`dec_trunk`). The
   network predicts a **displacement** from the template rest pose:
   `pred = template + head_disp(trunk)`. Predicting residuals (not absolute coordinates)
   keeps outputs near valid face geometry from epoch 0.
3. **Uncertainty head.** `head_sigma` predicts a per-vertex log-std; `σ = exp(logσ)` is
   clamped to `[0.01, 10]`. This lets the loss down-weight hard/occluded regions.

## Loss — heteroscedastic negative log-likelihood

Instead of plain L1/L2, v2 trains with a per-vertex uncertainty-weighted NLL:

```
NLL = 0.5 * mean( ||pred - target||^2 / σ^2  +  2·log σ )
```

plus Laplacian smoothness on the predicted displacement, a small Chamfer term, and a
landmark-consistency term. The `σ^2` denominator means confident vertices are penalized
hard while genuinely ambiguous vertices can widen σ instead of corrupting the fit — the
`log σ` term stops σ from exploding.

## Result (common protocol)

100-pair FaMoS eval subset, 1,000 training pairs, 20 epochs:

| Metric | Value |
|---|---:|
| Mean Chamfer ↓ | 0.0220 |
| Mean normal consistency ↑ | 0.83 |
| Inference | ~1 ms / mesh |

Reproduce:

```bash
python -m face_semantic_icp train --pairs PAIRS.jsonl --out outputs/train_v2 \
  --epochs 20 --model-version 2
python -m face_semantic_icp eval --checkpoint outputs/train_v2/deformnet.pt \
  --pairs PAIRS.jsonl --out outputs/eval_v2
```

See [`configs/train_deformnet_v2.json`](../configs/train_deformnet_v2.json) for the exact
hyperparameters.

## vs the other families

- **vs Semantic-ICP teacher:** ~6,000× faster at inference, but less accurate (0.0220 vs
  0.0065) — it has no iterative correspondence refinement.
- **vs GNN DeformNet (v3):** v2 ignores mesh connectivity; v3 adds a graph-convolution
  stream over the template adjacency and lowers Chamfer ~16% (0.0185). v2 is the baseline
  that isolates the value of that graph structure.
