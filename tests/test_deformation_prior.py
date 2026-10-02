from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from face_semantic_icp.deformation_prior import DeformationPrior

ROOT = Path(__file__).resolve().parents[1]


def _load_completion_script():
    spec = importlib.util.spec_from_file_location(
        "run_prior_guided_completion", ROOT / "scripts" / "run_prior_guided_completion.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_prior(vertex_count: int = 200, k: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random mean shape plus k orthonormal deformation modes of shape (k, V, 3)."""
    rng = np.random.default_rng(seed)
    mean = rng.uniform(-1.0, 1.0, size=(vertex_count, 3))
    q, _ = np.linalg.qr(rng.normal(size=(vertex_count * 3, k)))
    modes = q.T.reshape(k, vertex_count, 3)
    return mean, modes


class DeformationPriorTests(unittest.TestCase):
    def test_project_is_identity_inside_subspace(self) -> None:
        mean, modes = _synthetic_prior()
        prior = DeformationPrior(mean, modes.reshape(modes.shape[0], -1), np.array([3.0, 2.0, 1.0]))
        inside = mean + 0.4 * modes[0] - 0.2 * modes[2]
        np.testing.assert_allclose(prior.project(inside), inside, atol=1e-4)
        np.testing.assert_allclose(prior.project(mean), mean, atol=1e-5)

    def test_project_removes_out_of_subspace_component(self) -> None:
        mean, modes = _synthetic_prior()
        prior = DeformationPrior(mean, modes.reshape(modes.shape[0], -1), np.ones(3))
        inside = mean + 0.3 * modes[1]
        flat_modes = modes.reshape(modes.shape[0], -1)
        noise = np.random.default_rng(1).normal(size=flat_modes.shape[1])
        noise -= flat_modes.T @ (flat_modes @ noise)  # orthogonal to every mode
        noisy = inside + 0.05 * noise.reshape(mean.shape)
        np.testing.assert_allclose(prior.project(noisy), inside, atol=1e-4)

    def test_save_load_round_trip(self) -> None:
        mean, modes = _synthetic_prior()
        prior = DeformationPrior(mean, modes.reshape(modes.shape[0], -1), np.array([3.0, 2.0, 1.0]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prior.npz"
            prior.save(path)
            loaded = DeformationPrior.load(path)
        np.testing.assert_array_equal(loaded.mean, prior.mean)
        np.testing.assert_array_equal(loaded.components, prior.components)
        np.testing.assert_array_equal(loaded.singular_values, prior.singular_values)

    def test_build_spans_training_subspace_with_sorted_modes(self) -> None:
        mean, modes = _synthetic_prior(vertex_count=50, k=2)
        rng = np.random.default_rng(2)
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(20):
                pair = Path(tmp) / f"pair_{i:06d}"
                pair.mkdir()
                shape = mean + rng.normal(scale=2.0) * modes[0] + rng.normal(scale=0.5) * modes[1]
                np.save(pair / "target_vertices.npy", shape.astype(np.float32))
            prior = DeformationPrior.build(tmp, n_components=2)
        self.assertEqual(prior.n_components, 2)
        self.assertTrue(np.all(np.diff(prior.singular_values) <= 0))
        # Every training shape lies in the 2-mode subspace, so projection is exact.
        np.testing.assert_allclose(prior.project(mean + 1.5 * modes[0]), mean + 1.5 * modes[0], atol=1e-3)

    def test_rejects_mismatched_shapes(self) -> None:
        mean, modes = _synthetic_prior()
        with self.assertRaises(ValueError):
            DeformationPrior(mean, modes.reshape(modes.shape[0], -1)[:, :-3], np.ones(3))
        with self.assertRaises(ValueError):
            DeformationPrior(mean, modes.reshape(modes.shape[0], -1), np.ones(2))
        prior = DeformationPrior(mean, modes.reshape(modes.shape[0], -1), np.ones(3))
        with self.assertRaises(ValueError):
            prior.project(np.zeros((mean.shape[0] - 1, 3)))

    def test_build_rejects_mixed_topologies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for i, count in enumerate([10, 12]):
                pair = Path(tmp) / f"pair_{i:06d}"
                pair.mkdir()
                np.save(pair / "target_vertices.npy", np.zeros((count, 3), dtype=np.float32))
            with self.assertRaises(ValueError):
                DeformationPrior.build(tmp)


class PriorGuidedCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.completion = _load_completion_script()

    def test_fit_recovers_shape_from_partial_points(self) -> None:
        mean, modes = _synthetic_prior(vertex_count=400, k=3, seed=3)
        true_coeff = np.array([0.6, -0.4, 0.25])
        target = mean + np.tensordot(true_coeff, modes, axes=(0, 0))
        observed = np.random.default_rng(4).choice(len(target), size=200, replace=False)

        completed, correspondences, history = self.completion._fit_prior_to_points(
            target[observed],
            mean,
            modes,
            variance_ratio=np.array([0.5, 0.3, 0.2]),
            iterations=5,
            ridge=1e-6,
            distance_gate=0.18,
        )

        baseline_rmse = self.completion._vertex_rmse(mean, target)
        completed_rmse = self.completion._vertex_rmse(completed, target)
        self.assertEqual(completed.shape, target.shape)
        self.assertEqual(correspondences.shape, (len(observed),))
        self.assertEqual(len(history), 5)
        self.assertLess(completed_rmse, 0.1 * baseline_rmse)

    def test_partial_observation_respects_keep_ratio(self) -> None:
        mean, _ = _synthetic_prior(vertex_count=2000, seed=5)
        points, idx = self.completion._make_partial_observation(mean, keep_ratio=0.2, noise=0.0, seed=0)
        self.assertEqual(len(points), len(idx))
        self.assertEqual(len(np.unique(idx)), len(idx))
        self.assertLessEqual(len(idx), 400)
        np.testing.assert_allclose(points, mean[idx])


if __name__ == "__main__":
    unittest.main()
