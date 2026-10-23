from ui.face_clusters import FaceClusterStore, FaceSample
from data.assistant_dataclasses import DriverEmotionState
from managers.database_manager import DatabaseManager
from managers.source_manager import SourceManager
from unittest.mock import Mock
import numpy as np


def test_face_clusters_create_match_persist_delete_and_merge(tmp_path):
    samples = {
        b"alice": FaceSample((1.0, 0.0), b"alice-photo"),
        b"alice-variant": FaceSample((0.99, 0.01), b"alice-new-photo"),
        b"bob": FaceSample((0.0, 1.0), b"bob-photo"),
    }
    store = FaceClusterStore(tmp_path, sample_extractor=lambda frame: [samples[frame]])

    first = store.process_frames([b"alice"])
    alice_id = first["clusters"][0]["id"]
    update = store.process_frames([b"alice-variant", b"bob"])

    assert update["created"] == 1
    assert update["updated"] == 1
    assert len(update["clusters"]) == 2
    assert update["clusters"][0]["id"] == alice_id
    assert (tmp_path / "thumbnails" / f"{alice_id}.jpg").read_bytes() == b"alice-new-photo"

    restored = FaceClusterStore(tmp_path, sample_extractor=lambda frame: [])
    clusters = restored.list_clusters()
    assert len(clusters) == 2
    bob_id = next(cluster["id"] for cluster in clusters if cluster["id"] != alice_id)
    merged = restored.merge_clusters(bob_id, alice_id)
    assert [cluster["id"] for cluster in merged] == [alice_id]
    assert restored.list_clusters()[0]["sample_count"] == 3

    assert restored.delete_cluster(alice_id) == []
    assert not (tmp_path / "thumbnails" / f"{alice_id}.jpg").exists()


def test_face_clusters_reject_invalid_ids_and_report_no_faces(tmp_path):
    store = FaceClusterStore(tmp_path, sample_extractor=lambda frame: [])

    try:
        store.process_frames([b"empty"])
    except ValueError as error:
        assert "No usable faces" in str(error)
    else:
        raise AssertionError("empty capture should be rejected")

    try:
        store.delete_cluster("../../index")
    except ValueError as error:
        assert "Invalid person" in str(error)
    else:
        raise AssertionError("invalid cluster identifiers should be rejected")


def test_startup_recognition_only_matches_existing_clusters(tmp_path):
    samples = {
        b"known": FaceSample((1.0, 0.0), b"known-photo", (0.1, 0.2, 0.3, 0.4)),
        b"unknown": FaceSample((0.0, 1.0), b"unknown-photo", (0.4, 0.1, 0.2, 0.5)),
    }
    store = FaceClusterStore(tmp_path, sample_extractor=lambda frame: [samples[frame]])
    registered = store.process_frames([b"known"])
    cluster_id = registered["clusters"][0]["id"]
    sample_count = registered["clusters"][0]["sample_count"]

    recognition = store.recognize_frames([b"known", b"unknown"])

    assert recognition["faces_detected"] == 2
    assert [match["id"] for match in recognition["matches"]] == [cluster_id]
    assert recognition["preview_detections"] == [(0.4, 0.1, 0.2, 0.5)]
    assert store.list_clusters()[0]["sample_count"] == sample_count


def test_face_thumbnail_square_crop_preserves_face_aspect_ratio():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[40:160, 100:180] = 255

    crop = FaceClusterStore._square_face_crop(
        image,
        {"x": 100, "y": 40, "w": 80, "h": 120},
    )

    assert crop is not None
    assert crop.shape[0] == crop.shape[1]
    ys, xs = np.where(crop[:, :, 0] == 255)
    observed_ratio = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
    assert abs(observed_ratio - (80 / 120)) < 0.02


def test_deepface_zero_confidence_fallback_is_not_a_detected_face():
    assert not FaceClusterStore._face_was_detected({"face_confidence": 0.0})
    assert FaceClusterStore._face_was_detected({"face_confidence": 0.94})


def test_face_cluster_name_persists_without_changing_id(tmp_path):
    store = FaceClusterStore(
        tmp_path,
        sample_extractor=lambda _: [FaceSample((1.0, 0.0), b"face")],
    )
    identifier = store.process_frames([b"frame"])["clusters"][0]["id"]

    renamed = store.rename_cluster(identifier, "  Alex  ")
    restored = FaceClusterStore(tmp_path).list_clusters()

    assert renamed[0]["id"] == identifier
    assert renamed[0]["label"] == "Alex"
    assert restored[0]["id"] == identifier
    assert restored[0]["label"] == "Alex"


def test_acquisition_returns_detection_boxes_for_last_preview_frame(tmp_path):
    samples = {
        b"first": FaceSample((1.0, 0.0), b"first-face", (0.1, 0.2, 0.3, 0.4)),
        b"last": FaceSample((0.99, 0.01), b"last-face", (0.4, 0.1, 0.2, 0.5)),
    }
    store = FaceClusterStore(tmp_path, sample_extractor=lambda frame: [samples[frame]])

    result = store.process_frames([b"first", b"last"])

    assert result["detectedFaces"] == 2
    assert result["previewDetections"] == [(0.4, 0.1, 0.2, 0.5)]


def test_database_manager_keeps_face_id_on_emotion_updates():
    identifier = "a" * 32
    database = object.__new__(DatabaseManager)
    database.current_state = {
        "DriverEmotionState": DriverEmotionState(face_cluster_id=identifier),
    }
    database.measurement_event_queue = None
    database.logger = Mock()
    database.write_point = Mock()
    incoming = DriverEmotionState(angry=0.7)

    database.write_dataclass(incoming)

    assert incoming.face_cluster_id == identifier


def test_vehicle_telemetry_cannot_assign_local_face_identity():
    emotion = SourceManager.deserialize({
        "name": "DriverEmotionState",
        "data": {"angry": 0.7, "face_cluster_id": "a" * 32},
    })

    assert emotion is not None
    assert emotion.angry == 0.7
    assert emotion.face_cluster_id is None