from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import run_benchmark, run_hard_case_ablation
from .config import BenchmarkConfig, load_pipeline_config, to_jsonable
from .datasets import dataset_catalog, scan_dataset
from .deformation_prior import build_deformation_prior
from .doctor import run_doctor
from .evaluate import evaluate_deformnet
from .pairs import generate_pairs
from .pipeline import run_demo, run_mesh_case, run_photo_case
from .train import TrainConfig, train_deformnet


def add_shared_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quality", choices=["fast", "balanced", "best"], default=None, help="Wrapping quality preset.")
    parser.add_argument("--config", default=None, help="Optional JSON pipeline config.")
    parser.add_argument("--save-ablation", action="store_true", help="Write ablation_report.json and show ablations in the viewer.")
    parser.add_argument("--strict-flame", action="store_true", help="Require a valid official-size FLAME template.")
    parser.add_argument("--pca-prior", default=None, help="Path to deformation_prior.npz (built with build-prior command).")
    parser.add_argument("--deformnet", default=None, help="Path to deformnet.pt checkpoint for warm-start and uncertainty weighting.")


def config_from_args(args: argparse.Namespace):
    config = load_pipeline_config(
        getattr(args, "config", None),
        quality=getattr(args, "quality", None),
        save_ablation=True if getattr(args, "save_ablation", False) else None,
        strict_flame=True if getattr(args, "strict_flame", False) else None,
        template_path=getattr(args, "template", None),
    )
    pca_prior = getattr(args, "pca_prior", None)
    if pca_prior:
        config.semantic_icp.pca_prior_path = pca_prior
    deformnet = getattr(args, "deformnet", None)
    if deformnet:
        config.deformnet_checkpoint = deformnet
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Photo/mesh to fixed-topology face retopology demo.")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Generate synthetic photo/mesh inputs and run both pipelines.")
    demo.add_argument("--out", default="outputs/demo", help="Output directory.")
    demo.add_argument("--template", default=None, help="Optional official FLAME template OBJ.")
    add_shared_flags(demo)

    photo = sub.add_parser("photo", help="Run front-photo to fixed topology pipeline.")
    photo.add_argument("photo", help="Input front face photo.")
    photo.add_argument("--out", default="outputs/photo_case", help="Output directory.")
    photo.add_argument("--template", default=None, help="Optional official FLAME template OBJ.")
    add_shared_flags(photo)

    mesh = sub.add_parser("mesh", help="Run arbitrary mesh to fixed topology pipeline.")
    mesh.add_argument("mesh", help="Input OBJ/PLY mesh.")
    mesh.add_argument("--out", default="outputs/mesh_case", help="Output directory.")
    mesh.add_argument("--template", default=None, help="Optional official FLAME template OBJ.")
    add_shared_flags(mesh)

    doctor = sub.add_parser("doctor", help="Check dependencies, GPU, optional adapters, and FLAME assets.")
    doctor.add_argument("--template", default=None, help="Optional official FLAME template OBJ.")
    add_shared_flags(doctor)

    benchmark = sub.add_parser("benchmark", help="Run repeatable demo benchmarks and write JSON/Markdown reports.")
    benchmark.add_argument("--out", default="outputs/benchmark", help="Output directory.")
    benchmark.add_argument("--template", default=None, help="Optional official FLAME template OBJ.")
    benchmark.add_argument("--repeats", type=int, default=1, help="Repeats per seed.")
    benchmark.add_argument("--seeds", default="7,21,42", help="Comma-separated random seeds.")
    benchmark.add_argument("--hard-cases", action="store_true", help="Run hard-case ablation (expression/partial/noise variants).")
    add_shared_flags(benchmark)

    dataset = sub.add_parser("dataset", help="Dataset utilities for real 3D face corpora.")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_catalog_parser = dataset_sub.add_parser("catalog", help="Show supported datasets, access notes, and local path hints.")
    dataset_catalog_parser.add_argument("--local-base", default="D:/datasets", help="Base directory used for local dataset hints.")
    dataset_scan = dataset_sub.add_parser("scan", help="Scan a local dataset root and write dataset_manifest.jsonl.")
    dataset_scan.add_argument("--dataset", required=True, choices=["famos", "d3dfacs", "hifi3dface", "generic"], help="Dataset layout hint.")
    dataset_scan.add_argument("--root", required=True, help="Local dataset root.")
    dataset_scan.add_argument("--out", required=True, help="Output manifest directory.")
    dataset_scan.add_argument("--template", default=None, help="Optional official FLAME template OBJ.")
    dataset_scan.add_argument("--target-root", default=None, help="Optional folder of FLAME/registered target meshes matched by filename stem.")
    dataset_scan.add_argument("--source-glob", default=None, help="Optional glob under --root for source meshes, e.g. '**/*scan*.obj'.")
    dataset_scan.add_argument("--target-glob", default=None, help="Optional glob under --target-root for target meshes.")
    add_shared_flags(dataset_scan)

    pairs = sub.add_parser("pairs", help="Generate source/template training pairs.")
    pairs_sub = pairs.add_subparsers(dest="pairs_command", required=True)
    pairs_generate = pairs_sub.add_parser("generate", help="Generate training pairs from a manifest or synthetic source.")
    pairs_generate.add_argument("--manifest", default=None, help="dataset_manifest.jsonl from dataset scan.")
    pairs_generate.add_argument("--out", required=True, help="Output pair directory.")
    pairs_generate.add_argument("--teacher", default="wrap++", choices=["wrap++"], help="Teacher used for source-only meshes.")
    pairs_generate.add_argument("--template", default=None, help="Optional official FLAME template OBJ.")
    pairs_generate.add_argument("--point-count", type=int, default=1024, help="Sampled source points per pair.")
    pairs_generate.add_argument("--limit", type=int, default=None, help="Optional max manifest items.")
    pairs_generate.add_argument("--synthetic-count", type=int, default=0, help="Generate N synthetic random-topology pairs too.")
    pairs_generate.add_argument("--seed", type=int, default=17, help="Random seed.")
    pairs_generate.add_argument("--subject-split", default=None, choices=["train", "test", "all"], help="Filter by subject identity: train=first 80%% of subjects, test=last 20%%.")
    add_shared_flags(pairs_generate)

    train = sub.add_parser("train", help="Train a small deformation network on generated pairs.")
    train.add_argument("--pairs", required=True, help="pairs_manifest.jsonl.")
    train.add_argument("--out", default="outputs/train_deformnet", help="Training output directory.")
    train.add_argument("--template", default=None, help="Optional official FLAME template OBJ.")
    train.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    train.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    train.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    train.add_argument("--device", default="cuda", help="cuda or cpu.")
    train.add_argument("--latent-dim", type=int, default=96, help="PointNet latent dimension.")
    train.add_argument("--hidden-dim", type=int, default=128, help="MLP hidden dimension.")
    train.add_argument("--max-pairs", type=int, default=None, help="Optional max pairs.")
    train.add_argument("--model-version", type=int, default=2, choices=[1, 2, 3], help="DeformNet version: 1=V1, 2=V2 multi-scale+sigma, 3=GNN.")

    evaluate = sub.add_parser("eval", help="Evaluate a trained deformation network against Wrap++ baseline.")
    evaluate.add_argument("--checkpoint", required=True, help="deformnet.pt checkpoint.")
    evaluate.add_argument("--pairs", required=True, help="pairs_manifest.jsonl.")
    evaluate.add_argument("--out", default="outputs/eval_deformnet", help="Evaluation output directory.")
    evaluate.add_argument("--template", default=None, help="Optional official FLAME template OBJ.")
    evaluate.add_argument("--baseline", default="wrap++", choices=["wrap++", "none"], help="Baseline comparison.")
    evaluate.add_argument("--device", default="cuda", help="cuda or cpu.")
    evaluate.add_argument("--limit", type=int, default=None, help="Optional max eval pairs.")
    evaluate.add_argument("--subject-split", default=None, choices=["train", "test", "all"], help="Label the evaluation as a subject-split run.")

    build_prior = sub.add_parser("build-prior", help="Build a PCA deformation prior from FaMoS training pairs.")
    build_prior.add_argument("--pairs", required=True, help="Directory containing pair_XXXXXX/target_vertices.npy files.")
    build_prior.add_argument("--out", required=True, help="Output .npz path for the deformation prior.")
    build_prior.add_argument("--n-components", type=int, default=64, help="Number of PCA components to keep.")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    if args.command == "demo":
        result = run_demo(args.out, config=config)
    elif args.command == "photo":
        result = run_photo_case(args.photo, args.out, config=config)
    elif args.command == "mesh":
        result = run_mesh_case(args.mesh, args.out, config=config)
    elif args.command == "doctor":
        result = run_doctor(config)
    elif args.command == "benchmark":
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        config.benchmark = BenchmarkConfig(out=args.out, repeats=args.repeats, seeds=seeds)
        if getattr(args, "hard_cases", False):
            result = run_hard_case_ablation(config, Path(args.out) / "hard_cases", seeds)
        else:
            result = run_benchmark(config)
            result = to_jsonable(result)
    elif args.command == "dataset" and args.dataset_command == "catalog":
        result = dataset_catalog(args.local_base)
    elif args.command == "dataset" and args.dataset_command == "scan":
        result = scan_dataset(
            args.dataset,
            args.root,
            args.out,
            args.template,
            target_root=args.target_root,
            source_glob=args.source_glob,
            target_glob=args.target_glob,
        )
    elif args.command == "pairs" and args.pairs_command == "generate":
        result = generate_pairs(
            args.manifest,
            args.out,
            teacher=args.teacher,
            template_path=args.template,
            point_count=args.point_count,
            limit=args.limit,
            synthetic_count=args.synthetic_count,
            seed=args.seed,
            subject_split=getattr(args, "subject_split", None),
        )
    elif args.command == "train":
        result = train_deformnet(
            TrainConfig(
                pairs=args.pairs,
                out=args.out,
                template=args.template,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
                latent_dim=args.latent_dim,
                hidden_dim=args.hidden_dim,
                max_pairs=args.max_pairs,
                model_version=getattr(args, "model_version", 2),
            )
        )
    elif args.command == "eval":
        result = evaluate_deformnet(args.checkpoint, args.pairs, args.out, args.template, args.baseline, args.device, args.limit, subject_split=getattr(args, "subject_split", None))
    elif args.command == "build-prior":
        result = build_deformation_prior(args.pairs, args.out, n_components=args.n_components)
    else:
        raise SystemExit(f"Unknown command: {args.command}")
    print(json.dumps(result, indent=2))
    if args.command == "demo":
        print(f"\nOpen: {Path(args.out) / 'index.html'}")
    elif args.command in {"photo", "mesh"}:
        print(f"\nOpen: {Path(args.out) / 'viewer.html'}")
    elif args.command == "benchmark":
        if getattr(args, "hard_cases", False):
            print(f"\nReport: {Path(args.out) / 'hard_cases' / 'hard_case_ablation.md'}")
        else:
            print(f"\nOpen: {Path(args.out) / 'benchmark_report.md'}")
    elif args.command == "dataset" and args.dataset_command == "catalog":
        print(f"\nDataset root hint: {args.local_base}")
    elif args.command == "dataset" and args.dataset_command == "scan":
        print(f"\nManifest: {Path(args.out) / 'dataset_manifest.jsonl'}")
    elif args.command == "pairs" and args.pairs_command == "generate":
        print(f"\nPairs: {Path(args.out) / 'pairs_manifest.jsonl'}")
    elif args.command == "train":
        print(f"\nCheckpoint: {Path(args.out) / 'deformnet.pt'}")
    elif args.command == "eval":
        print(f"\nReport: {Path(args.out) / 'eval_report.md'}")
    elif args.command == "build-prior":
        print(f"\nPrior: {args.out}")
