from __future__ import annotations

from pathlib import Path
import importlib.util

import cv2
import numpy as np
from PIL import Image

from .config import LandmarkConfig, LandmarkResult
from .template import canonical_landmark_uv, face_surface


def read_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def detect_face_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    arr = np.asarray(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(64, 64))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: int(b[2]) * int(b[3]))
        pad_x = int(0.20 * w)
        pad_y = int(0.25 * h)
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(image.width - x, w + 2 * pad_x)
        h = min(image.height - y, h + 2 * pad_y)
        return int(x), int(y), int(w), int(h)

    side = int(min(image.width, image.height) * 0.86)
    x = (image.width - side) // 2
    y = (image.height - side) // 2
    return x, y, side, side


def heuristic_2d_landmarks(image: Image.Image, bbox: tuple[int, int, int, int] | None = None) -> np.ndarray:
    if bbox is None:
        bbox = detect_face_bbox(image)
    x, y, w, h = bbox
    uv = canonical_landmark_uv()
    px = x + (uv[:, 0] + 1.0) * 0.5 * w
    py = y + (1.0 - (uv[:, 1] + 1.0) * 0.5) * h
    return np.stack([px, py], axis=1)


def landmarks_2d_to_3d(landmarks: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    u = (landmarks[:, 0] - x) / max(w, 1) * 2.0 - 1.0
    v = ((y + h - landmarks[:, 1]) / max(h, 1)) * 2.0 - 1.0
    return face_surface(np.clip(u, -1.0, 1.0), np.clip(v, -1.0, 1.0))


def optional_landmark_backends() -> dict[str, bool]:
    return {
        "mediapipe": importlib.util.find_spec("mediapipe") is not None,
        "insightface": importlib.util.find_spec("insightface") is not None,
        "mica": importlib.util.find_spec("mica") is not None or importlib.util.find_spec("micalib") is not None,
        "deca": importlib.util.find_spec("decalib") is not None,
    }


def _to_landmark_result(points_2d: np.ndarray, points_3d: np.ndarray, bbox: tuple[int, int, int, int], source: str, notes: list[str] | None = None) -> LandmarkResult:
    confidence = np.ones(len(points_3d), dtype=np.float64)
    return LandmarkResult(
        points_2d=np.asarray(points_2d, dtype=np.float64).tolist(),
        points_3d=np.asarray(points_3d, dtype=np.float64).tolist(),
        confidences=confidence.tolist(),
        source=source,
        bbox=bbox,
        notes=notes or [],
    )


def detect_layered_landmarks(image: Image.Image, config: LandmarkConfig | None = None) -> LandmarkResult:
    config = config or LandmarkConfig()
    notes: list[str] = []
    backends = optional_landmark_backends()

    if config.use_mediapipe and backends["mediapipe"]:
        notes.append("MediaPipe is installed, but this lightweight adapter keeps model-bundle loading optional; using OpenCV fallback unless a bundle adapter is added.")
    if config.use_insightface and backends["insightface"]:
        notes.append("InsightFace is installed; use an external adapter/model path for production-grade 2D/3D landmarks.")
    if config.use_mica_deca_imports and (backends["mica"] or backends["deca"]):
        notes.append("MICA/DECA import detected; reconstructed meshes can be passed through the mesh path or future direct adapters.")

    bbox = detect_face_bbox(image)
    points_2d = heuristic_2d_landmarks(image, bbox)
    points_3d = landmarks_2d_to_3d(points_2d, bbox)
    notes.append("Used deterministic OpenCV/geometry fallback landmarks.")
    return _to_landmark_result(points_2d, points_3d, bbox, "opencv_heuristic_fallback", notes)


def landmark_result_points_3d(result: LandmarkResult) -> np.ndarray:
    return np.asarray(result.points_3d, dtype=np.float64)
