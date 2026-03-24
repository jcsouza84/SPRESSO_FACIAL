from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

import cv2
import numpy as np

from app.camera.service import camera_service
from app.detection.face_detector import face_detector, DetectionResult, DetectedFace
from app.recognition.matcher import face_matcher, RecognitionResult
from app.services.event_service import save_detection_event

router = APIRouter(prefix="/detection", tags=["detection"])


class FaceBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int
    confidence: float
    matched: bool = False
    person_id: int | None = None
    person_name: str | None = None
    category: str | None = None
    recognition_confidence: float = 0.0


class DetectionResponse(BaseModel):
    event_id: int | None = None
    timestamp: str
    face_count: int
    inference_ms: float
    frame_width: int
    frame_height: int
    faces: list[FaceBox]


@router.get("/status")
async def detection_status():
    return {
        "ready":           face_detector.is_ready,
        "persons_in_cache": face_matcher.persons_in_cache,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }


@router.get("/faces", response_model=DetectionResponse)
async def detect_faces() -> DetectionResponse:
    """Captura frame, detecta rostos, reconhece pessoas, persiste evento."""
    _check_ready()

    frame = camera_service.snapshot(save=False)
    result = face_detector.detect(frame.array)

    face_boxes = _recognize_faces(frame.array, result)
    event = await save_detection_event(result)

    return DetectionResponse(
        event_id=event.id,
        timestamp=event.timestamp.isoformat(),
        face_count=result.count,
        inference_ms=result.inference_ms,
        frame_width=result.frame_width,
        frame_height=result.frame_height,
        faces=face_boxes,
    )


@router.get(
    "/snapshot",
    responses={200: {"content": {"image/jpeg": {}}}},
    response_class=Response,
)
async def detection_snapshot() -> Response:
    """Captura frame, detecta rostos, reconhece e retorna JPEG anotado."""
    _check_ready()

    frame = camera_service.snapshot(save=True)
    result = face_detector.detect(frame.array)

    recognitions = _recognize_faces(frame.array, result)
    snapshot_path = _save_annotated(frame.array, result, recognitions)
    event = await save_detection_event(result, snapshot_path=snapshot_path)

    return Response(
        content=snapshot_path.read_bytes(),
        media_type="image/jpeg",
        headers={
            "X-Event-Id":     str(event.id),
            "X-Face-Count":   str(result.count),
            "X-Inference-Ms": str(result.inference_ms),
            "X-Captured-At":  event.timestamp.isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_ready() -> None:
    if not camera_service.is_ready:
        raise HTTPException(status_code=503, detail="Câmera não disponível")
    if not face_detector.is_ready:
        raise HTTPException(status_code=503, detail="Detector não disponível")


def _extract_roi(frame_rgb: np.ndarray, face: DetectedFace) -> np.ndarray:
    """Recorta o rosto do frame com margem mínima para o reconhecimento."""
    h, w = frame_rgb.shape[:2]
    pad = 10
    x1 = max(0, face.x1 - pad)
    y1 = max(0, face.y1 - pad)
    x2 = min(w, face.x2 + pad)
    y2 = min(h, face.y2 + pad)
    return frame_rgb[y1:y2, x1:x2]


def _recognize_faces(
    frame_rgb: np.ndarray,
    result: DetectionResult,
) -> list[FaceBox]:
    """Executa reconhecimento para cada rosto detectado."""
    boxes: list[FaceBox] = []
    for face in result.faces:
        base = face.to_dict()
        roi = _extract_roi(frame_rgb, face)
        rec = face_matcher.identify(roi)
        boxes.append(FaceBox(
            **base,
            matched=rec.matched,
            person_id=rec.person_id,
            person_name=rec.person_name,
            category=rec.category,
            recognition_confidence=rec.confidence,
        ))
    return boxes


def _save_annotated(
    frame_rgb: np.ndarray,
    result: DetectionResult,
    recognitions: list[FaceBox],
) -> Path:
    """Desenha bboxes com identidade e salva JPEG anotado."""
    img = frame_rgb.copy()

    for box in recognitions:
        color = (0, 200, 0) if box.matched else (0, 80, 255)
        cv2.rectangle(img, (box.x1, box.y1), (box.x2, box.y2), color=color, thickness=2)

        label = box.person_name if box.matched else f"{box.confidence:.0%}"
        cv2.putText(img, label,
                    (box.x1, max(box.y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        if box.matched:
            sub = f"{box.recognition_confidence:.0f}% {box.category or ''}"
            cv2.putText(img, sub,
                        (box.x1, max(box.y1 - 26, 26)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    total_label = f"{result.count} rosto(s) | {result.inference_ms:.0f}ms"
    cv2.putText(img, total_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = camera_service._snapshots_dir / f"annotated_{ts}.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path
