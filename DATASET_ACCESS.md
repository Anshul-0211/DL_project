# Dataset Access And Training Pipeline Notes

This project is ready to use real 3D face datasets when their licensed files are available locally. It does not auto-download restricted assets, commit dataset files, or bundle model weights.

## Primary Real Datasets

### FaMoS

- Role: primary real dataset target.
- Why it fits: dynamic 3D head sequences with FLAME-topology registrations available for research use.
- Local hint: `D:/datasets/FaMoS`
- Access: sign up and accept the dataset license before downloading.
- Project path once available:

```bash
python -m face_semantic_icp dataset scan --dataset famos --root D:/datasets/FaMoS --out outputs/datasets/famos
python -m face_semantic_icp pairs generate --manifest outputs/datasets/famos/dataset_manifest.jsonl --out outputs/pairs_famos --teacher wrap++
```

If source scans and FLAME registrations are in separate folders, use target matching:

```bash
python -m face_semantic_icp dataset scan --dataset famos --root D:/datasets/FaMoS/scans --target-root D:/datasets/FaMoS/registrations --out outputs/datasets/famos
```

### D3DFACS FLAME Registrations

- Role: primary supervision target.
- Why it fits: FLAME ecosystem provides temporal registrations in FLAME topology.
- Local hint: `D:/datasets/D3DFACS_FLAME_registrations`
- Access: sign in through FLAME/MPI channels and accept the research data license.
- Project path once available:

```bash
python -m face_semantic_icp dataset scan --dataset d3dfacs --root D:/datasets/D3DFACS_FLAME_registrations --out outputs/datasets/d3dfacs
python -m face_semantic_icp pairs generate --manifest outputs/datasets/d3dfacs/dataset_manifest.jsonl --out outputs/pairs_d3dfacs --teacher wrap++
```

### HiFi3DFace / HIFI3D++

- Role: high-detail source geometry, not assumed FLAME topology.
- Why it fits: useful for high-frequency geometry/detail transfer experiments.
- Local hint: `D:/datasets/HiFi3DFace`
- Project path once available:

```bash
python -m face_semantic_icp dataset scan --dataset hifi3dface --root D:/datasets/HiFi3DFace --out outputs/datasets/hifi3dface
python -m face_semantic_icp pairs generate --manifest outputs/datasets/hifi3dface/dataset_manifest.jsonl --out outputs/pairs_hifi3dface --teacher wrap++
```

## Current No-Download Fallback

Until the licensed files are present, the project uses synthetic random-topology scan-like meshes to create controlled source/template pairs:

```bash
python -m face_semantic_icp pairs generate --out outputs/pairs_synthetic --synthetic-count 16 --point-count 1024
python -m face_semantic_icp train --pairs outputs/pairs_synthetic/pairs_manifest.jsonl --out outputs/train_deformnet --epochs 10 --device cuda
python -m face_semantic_icp eval --checkpoint outputs/train_deformnet/deformnet.pt --pairs outputs/pairs_synthetic/pairs_manifest.jsonl --out outputs/eval_deformnet --baseline wrap++
```

## Supported Dataset Commands

```bash
python -m face_semantic_icp dataset catalog
python -m face_semantic_icp dataset catalog --local-base D:/datasets/face_retopology
python -m face_semantic_icp dataset scan --dataset famos|d3dfacs|hifi3dface|generic --root PATH --out PATH
python -m face_semantic_icp pairs generate --manifest PATH --out PATH --teacher wrap++
python -m face_semantic_icp train --pairs PATH --out PATH --epochs 10 --device cuda
python -m face_semantic_icp eval --checkpoint PATH --pairs PATH --out PATH --baseline wrap++
```

## Meeting-Safe Claim

The implemented system is dataset-ready and training-ready:

- it can ingest local FaMoS/D3DFACS/HiFi3DFace-style mesh folders;
- it detects direct FLAME-topology supervision when available;
- it matches separate source and target registration folders by filename stem;
- it generates a consistent training-pair schema;
- it trains and evaluates a first PointNet-style template deformation model;
- it keeps Wrap++ Semantic ICP as both a strong teacher and an evaluation baseline.

The only missing piece is the actual licensed dataset files, which must be downloaded manually after license approval.
