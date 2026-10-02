from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from face_semantic_icp.benchmark import run_benchmark
from face_semantic_icp.config import BenchmarkConfig, PipelineConfig
from face_semantic_icp.datasets import dataset_catalog, scan_dataset
from face_semantic_icp.demo_data import make_random_topology_face
from face_semantic_icp.doctor import run_doctor
from face_semantic_icp.evaluate import _markdown, evaluate_deformnet
from face_semantic_icp.geometry import topology_signature
from face_semantic_icp.pairs import generate_pairs
from face_semantic_icp.pipeline import run_demo, run_mesh_case
from face_semantic_icp.template import load_template, validate_template
from face_semantic_icp.train import TrainConfig, build_model, train_deformnet
from face_semantic_icp.deformation_prior import DeformationPrior


class PipelineTests(unittest.TestCase):
    def test_template_topology_is_preserved(self) -> None:
        template = load_template()
        mesh = make_random_topology_face(seed=11, point_count=420)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source.obj"
            out = Path(tmp) / "out"
            from face_semantic_icp.geometry import write_obj, load_mesh

            write_obj(mesh, src)
            run_mesh_case(src, out)
            result = load_mesh(out / "result_flame.obj")
            self.assertEqual(topology_signature(template.mesh)["vertex_count"], topology_signature(result)["vertex_count"])
            self.assertEqual(topology_signature(template.mesh)["face_checksum"], topology_signature(result)["face_checksum"])

    def test_mesh_case_writes_core_outputs(self) -> None:
        mesh = make_random_topology_face(seed=21, point_count=360)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source.obj"
            out = Path(tmp) / "out"
            from face_semantic_icp.geometry import write_obj

            write_obj(mesh, src)
            metrics = run_mesh_case(src, out)
            self.assertTrue((out / "result_flame.obj").exists())
            self.assertTrue((out / "result_flame.ply").exists())
            self.assertTrue((out / "texture.png").exists())
            self.assertTrue((out / "viewer.html").exists())
            self.assertLess(metrics["chamfer_detailed_to_source"], metrics["chamfer_initial_to_source"])

    def test_demo_writes_manifest_overlay_and_residual(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "demo"
            run_demo(out, quality="fast", save_ablation=True)
            case = out / "mesh_case"
            self.assertTrue((case / "artifact_manifest.json").exists())
            self.assertTrue((case / "semantic_overlay.png").exists())
            self.assertTrue((case / "detail_residual.npy").exists())
            self.assertTrue((case / "ablation_report.json").exists())
            viewer = (case / "viewer.html").read_text(encoding="utf-8")
            self.assertIn("Delta", viewer)
            self.assertIn("Ablation", viewer)

    def test_doctor_reports_fallback_template(self) -> None:
        report = run_doctor(PipelineConfig())
        self.assertIn(report["status"], {"ok", "attention"})
        self.assertTrue(report["template"]["available"])
        self.assertIn("optional_adapters", report)
        self.assertIn("datasets", report)

    def test_benchmark_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = PipelineConfig(quality="fast")
            config.benchmark = BenchmarkConfig(out=str(Path(tmp) / "bench"), repeats=1, seeds=[3], include_photo=False, include_mesh=True)
            report = run_benchmark(config)
            self.assertEqual(report.summary["case_count"], 1)
            self.assertTrue(Path(report.json_path).exists())
            self.assertTrue(Path(report.markdown_path).exists())

    def test_strict_flame_validation_rejects_demo_topology(self) -> None:
        template = load_template()
        with self.assertRaises(ValueError):
            validate_template(template.mesh, strict_flame=True)

    def test_dataset_scan_pair_train_and_eval_smoke(self) -> None:
        from face_semantic_icp.geometry import write_obj

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "generic_dataset" / "subject_01"
            root.mkdir(parents=True)
            write_obj(make_random_topology_face(seed=31, point_count=240), root / "scan.obj")

            dataset_out = Path(tmp) / "dataset_out"
            summary = scan_dataset("generic", Path(tmp) / "generic_dataset", dataset_out)
            self.assertEqual(summary["ok_items"], 1)
            self.assertTrue((dataset_out / "dataset_manifest.jsonl").exists())

            pairs_out = Path(tmp) / "pairs"
            pair_summary = generate_pairs(dataset_out / "dataset_manifest.jsonl", pairs_out, point_count=64, limit=1)
            self.assertEqual(pair_summary["pair_count"], 1)
            self.assertTrue((pairs_out / "pair_000000" / "source_points.npz").exists())
            self.assertTrue((pairs_out / "pair_000000" / "target_vertices.npy").exists())

            train_out = Path(tmp) / "train"
            train_metrics = train_deformnet(
                TrainConfig(
                    pairs=str(pairs_out / "pairs_manifest.jsonl"),
                    out=str(train_out),
                    epochs=1,
                    batch_size=1,
                    device="cpu",
                    latent_dim=16,
                    hidden_dim=32,
                    max_pairs=1,
                )
            )
            self.assertTrue(Path(train_metrics["checkpoint"]).exists())

            eval_out = Path(tmp) / "eval"
            eval_report = evaluate_deformnet(train_metrics["checkpoint"], pairs_out / "pairs_manifest.jsonl", eval_out, baseline="wrap++", device="cpu", limit=1)
            self.assertEqual(eval_report["pair_count"], 1)
            self.assertIn("mean_learned_chamfer", eval_report)
            self.assertIn("mean_icp_chamfer", eval_report)
            self.assertTrue((eval_out / "eval_report.md").exists())

    def test_dataset_catalog_and_target_matching(self) -> None:
        from face_semantic_icp.geometry import write_obj

        with tempfile.TemporaryDirectory() as tmp:
            source_root = Path(tmp) / "famos_sources"
            target_root = Path(tmp) / "famos_targets"
            source_root.mkdir()
            target_root.mkdir()
            source = make_random_topology_face(seed=41, point_count=180)
            target = load_template().mesh
            write_obj(source, source_root / "subj01_seq02_frame0001_scan.obj")
            write_obj(target, target_root / "subj01_seq02_frame0001_flame.obj")

            catalog = dataset_catalog(Path(tmp))
            self.assertEqual(len(catalog["entries"]), 4)
            self.assertTrue(str(catalog["entries"][0]["local_path"]).startswith(str(Path(tmp))))

            out = Path(tmp) / "manifest"
            summary = scan_dataset("famos", source_root, out, target_root=target_root)
            self.assertEqual(summary["ok_items"], 1)
            self.assertEqual(summary["items_with_target"], 1)
            manifest_text = (out / "dataset_manifest.jsonl").read_text(encoding="utf-8")
            self.assertIn("subj01_seq02_frame0001_flame.obj", manifest_text)

    def test_deformation_prior_loads_showcase_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prior.npz"
            mean = np.zeros((4, 3), dtype=np.float32)
            modes = np.zeros((2, 4, 3), dtype=np.float32)
            modes[0, :, 0] = 0.5
            np.savez_compressed(path, mean_vertices=mean, modes=modes, explained_variance_ratio=np.array([0.8, 0.2], dtype=np.float32))
            prior = DeformationPrior.load(path)
            self.assertEqual(prior.vertex_count, 4)
            self.assertEqual(prior.components.shape, (2, 12))
            self.assertEqual(prior.project(mean).shape, (4, 3))

    def test_eval_markdown_marks_missing_baseline(self) -> None:
        text = _markdown([
            {
                "pair_id": "pair_000000",
                "learned_chamfer": 0.123,
                "learned_normal_consistency": 0.9,
                "learned_runtime_sec": 0.01,
                "icp_chamfer": None,
                "icp_normal_consistency": None,
                "icp_runtime_sec": None,
            }
        ])
        self.assertIn("| pair_000000 | 0.123000 | - | 0.900000 | - | 0.0100 | - |", text)

    def test_gnn_deformnet_v3_outputs_vertices_and_sigma(self) -> None:
        import torch

        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        model = build_model(4, latent_dim=8, hidden_dim=8, version=3, template_faces=faces)
        points = torch.zeros((1, 6, 3), dtype=torch.float32)
        template_vertices = torch.zeros((4, 3), dtype=torch.float32)
        pred, sigma = model.predict_with_uncertainty(points, template_vertices)
        self.assertEqual(tuple(pred.shape), (1, 4, 3))
        self.assertEqual(tuple(sigma.shape), (1, 4))


if __name__ == "__main__":
    unittest.main()
