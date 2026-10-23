"""Local face embeddings and persistent demo identity clusters."""
from __future__ import annotations

import json
import importlib
import math
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence


@dataclass(frozen=True)
class FaceSample:
    embedding: tuple[float, ...]
    thumbnail: bytes
    detection_box: tuple[float, float, float, float] | None = None


class FaceClusterStore:
    MODEL_NAME = "ArcFace"
    DETECTOR_BACKEND = "yolov8n"
    COSINE_DISTANCE_THRESHOLD = 0.68
    MAX_SAMPLES_PER_CLUSTER = 24
    _ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")

    def __init__(
        self,
        storage_dir: Path,
        sample_extractor: Callable[[bytes], Sequence[FaceSample]] | None = None,
    ) -> None:
        self.storage_dir = storage_dir
        self.thumbnail_dir = storage_dir / "thumbnails"
        self.index_path = storage_dir / "clusters.json"
        self._sample_extractor = sample_extractor
        self._lock = threading.RLock()
        self._clusters: dict[str, dict] = {}
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict) or payload.get("model") != self.MODEL_NAME or payload.get("version") != 1:
            return
        for cluster in payload.get("clusters", []):
            identifier = cluster.get("id")
            samples = cluster.get("samples")
            if (
                not isinstance(identifier, str)
                or not self._ID_PATTERN.fullmatch(identifier)
                or not isinstance(samples, list)
                or not samples
                or not (self.thumbnail_dir / f"{identifier}.jpg").is_file()
            ):
                continue
            try:
                vectors = [tuple(float(value) for value in sample) for sample in samples]
            except (TypeError, ValueError):
                continue
            if any(not vector or not all(math.isfinite(value) for value in vector) for vector in vectors):
                continue
            self._clusters[identifier] = {
                "id": identifier,
                "name": str(cluster.get("name", "")).strip()[:40],
                "created_at": str(cluster.get("created_at", "")),
                "samples": vectors[-self.MAX_SAMPLES_PER_CLUSTER :],
            }

    def _save(self) -> None:
        payload = {
            "version": 1,
            "model": self.MODEL_NAME,
            "detector": self.DETECTOR_BACKEND,
            "clusters": list(self._clusters.values()),
        }
        temporary = self.index_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(self.index_path)

    @staticmethod
    def _distance(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            return 1.0
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 1.0
        cosine = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
        return 1.0 - max(-1.0, min(1.0, cosine))

    @staticmethod
    def _face_was_detected(face: dict) -> bool:
        confidence = face.get("face_confidence")
        return confidence is None or float(confidence) > 0.0

    @staticmethod
    def _square_face_crop(image, area):
        cv2 = importlib.import_module("cv2")
        x, y = int(area.get("x", 0)), int(area.get("y", 0))
        width, height = int(area.get("w", 0)), int(area.get("h", 0))
        if width <= 0 or height <= 0:
            return None

        side = max(48, int(round(max(width, height) * 1.2)))
        center_x, center_y = x + width / 2, y + height / 2
        left, top = int(round(center_x - side / 2)), int(round(center_y - side / 2))
        right, bottom = left + side, top + side
        image_height, image_width = image.shape[:2]
        clipped_left, clipped_top = max(0, left), max(0, top)
        clipped_right, clipped_bottom = min(image_width, right), min(image_height, bottom)
        if clipped_left >= clipped_right or clipped_top >= clipped_bottom:
            return None
        crop = image[clipped_top:clipped_bottom, clipped_left:clipped_right]
        return cv2.copyMakeBorder(
            crop,
            clipped_top - top,
            bottom - clipped_bottom,
            clipped_left - left,
            right - clipped_right,
            cv2.BORDER_REPLICATE,
        )

    def _extract_frame(self, jpeg: bytes) -> Sequence[FaceSample]:
        if self._sample_extractor is not None:
            return self._sample_extractor(jpeg)
        cv2 = importlib.import_module("cv2")
        np = importlib.import_module("numpy")
        deepface = importlib.import_module("deepface.DeepFace")

        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return ()
        faces = deepface.represent(
            img_path=image,
            model_name=self.MODEL_NAME,
            detector_backend=self.DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
        )
        samples = []
        for face in faces:
            if not self._face_was_detected(face):
                continue
            area = face.get("facial_area", {})
            crop = self._square_face_crop(image, area)
            if crop is None or crop.size == 0 or min(crop.shape[:2]) < 48:
                continue
            crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
            success, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if success:
                image_height, image_width = image.shape[:2]
                left = max(0, int(area["x"]))
                top = max(0, int(area["y"]))
                right = min(image_width, left + int(area["w"]))
                bottom = min(image_height, top + int(area["h"]))
                box = (
                    left / image_width,
                    top / image_height,
                    max(0, right - left) / image_width,
                    max(0, bottom - top) / image_height,
                )
                samples.append(
                    FaceSample(
                        tuple(float(v) for v in face["embedding"]),
                        encoded.tobytes(),
                        box,
                    )
                )
        return samples

    def extract_frame_samples(self, jpeg: bytes) -> list[FaceSample]:
        return list(self._extract_frame(jpeg))

    def process_frames(self, frames: Sequence[bytes]) -> dict:
        samples_by_frame = [self._extract_frame(frame) for frame in frames]
        samples = [sample for frame_samples in samples_by_frame for sample in frame_samples]
        if not samples:
            raise ValueError("No usable faces were found in the captured frames")
        preview_detections = [
            sample.detection_box
            for sample in (samples_by_frame[-1] if samples_by_frame else [])
            if sample.detection_box is not None
        ]
        created = updated = 0
        matched_ids = []
        with self._lock:
            for sample in samples:
                best_id = None
                best_distance = self.COSINE_DISTANCE_THRESHOLD
                for identifier, cluster in self._clusters.items():
                    center = self._centroid(cluster["samples"])
                    distance = self._distance(sample.embedding, center)
                    if distance < best_distance:
                        best_id, best_distance = identifier, distance
                if best_id is None:
                    best_id = uuid.uuid4().hex
                    self._clusters[best_id] = {
                        "id": best_id,
                        "name": "",
                        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "samples": [],
                    }
                    created += 1
                else:
                    updated += 1
                if best_id not in matched_ids:
                    matched_ids.append(best_id)
                cluster = self._clusters[best_id]
                if len(cluster["samples"]) < self.MAX_SAMPLES_PER_CLUSTER:
                    cluster["samples"].append(list(sample.embedding))
                (self.thumbnail_dir / f"{best_id}.jpg").write_bytes(sample.thumbnail)
            self._save()
            return {
                "created": created,
                "updated": updated,
                "detectedFaces": len(samples),
                "previewDetections": preview_detections,
                "matchedIds": matched_ids,
                "clusters": self.list_clusters(),
            }

    def recognize_frames(self, frames: Sequence[bytes]) -> dict:
        samples_by_frame = [self._extract_frame(frame) for frame in frames]
        samples = [sample for frame_samples in samples_by_frame for sample in frame_samples]
        preview_detections = [
            sample.detection_box
            for sample in (samples_by_frame[-1] if samples_by_frame else [])
            if sample.detection_box is not None
        ]
        distances_by_id: dict[str, list[float]] = {}
        with self._lock:
            for sample in samples:
                for identifier, cluster in self._clusters.items():
                    distance = self._distance(sample.embedding, self._centroid(cluster["samples"]))
                    if distance < self.COSINE_DISTANCE_THRESHOLD:
                        distances_by_id.setdefault(identifier, []).append(distance)
        ranked_distances = sorted(
            (
                (identifier, sum(distances) / len(distances))
                for identifier, distances in distances_by_id.items()
            ),
            key=lambda match: match[1],
        )
        ranked_matches = [
            {"id": identifier, "distance": distance}
            for identifier, distance in ranked_distances
        ]
        return {
            "faces_detected": len(samples),
            "matches": ranked_matches,
            "preview_detections": preview_detections,
        }

    @staticmethod
    def _centroid(samples: Sequence[Sequence[float]]) -> tuple[float, ...]:
        return tuple(sum(vector[index] for vector in samples) / len(samples) for index in range(len(samples[0])))

    def list_clusters(self) -> list[dict]:
        with self._lock:
            ordered = sorted(self._clusters.values(), key=lambda item: item["created_at"])
            return [
                {
                    "id": cluster["id"],
                    "label": cluster.get("name") or f"Person {index}",
                    "sample_count": len(cluster["samples"]),
                    "thumbnail": f"/face-clusters/{cluster['id']}.jpg",
                }
                for index, cluster in enumerate(ordered, 1)
            ]

    def thumbnail_path(self, identifier: str) -> Path | None:
        self._require_id(identifier)
        path = self.thumbnail_dir / f"{identifier}.jpg"
        return path if path.is_file() else None

    def contains_cluster(self, identifier: str) -> bool:
        self._require_id(identifier)
        with self._lock:
            return identifier in self._clusters

    def rename_cluster(self, identifier: str, name: str) -> list[dict]:
        self._require_id(identifier)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Enter a name for this person")
        with self._lock:
            cluster = self._clusters.get(identifier)
            if cluster is None:
                raise ValueError("Person cluster no longer exists")
            cluster["name"] = name.strip()[:40]
            self._save()
            return self.list_clusters()

    def delete_cluster(self, identifier: str) -> list[dict]:
        self._require_id(identifier)
        with self._lock:
            self._clusters.pop(identifier, None)
            (self.thumbnail_dir / f"{identifier}.jpg").unlink(missing_ok=True)
            self._save()
            return self.list_clusters()

    def merge_clusters(self, source_id: str, target_id: str) -> list[dict]:
        self._require_id(source_id)
        self._require_id(target_id)
        if source_id == target_id:
            raise ValueError("Choose two different people to merge")
        with self._lock:
            source, target = self._clusters.get(source_id), self._clusters.get(target_id)
            if source is None or target is None:
                raise ValueError("Person cluster no longer exists")
            target["samples"] = (target["samples"] + source["samples"])[-self.MAX_SAMPLES_PER_CLUSTER :]
            source_path = self.thumbnail_dir / f"{source_id}.jpg"
            if source_path.is_file():
                (self.thumbnail_dir / f"{target_id}.jpg").write_bytes(source_path.read_bytes())
            self._clusters.pop(source_id)
            source_path.unlink(missing_ok=True)
            self._save()
            return self.list_clusters()

    @classmethod
    def _require_id(cls, identifier: str) -> None:
        if not isinstance(identifier, str) or not cls._ID_PATTERN.fullmatch(identifier):
            raise ValueError("Invalid person cluster")