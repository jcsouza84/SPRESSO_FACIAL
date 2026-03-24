from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

import cv2
import numpy as np

from app.camera.service import camera_service
from app.detection.face_detector import face_detector, DetectionResult
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
        "ready": face_detector.is_ready,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/faces", response_model=DetectionResponse)
async def detect_faces() -> DetectionResponse:
    """Captura frame, detecta rostos, persiste evento e retorna JSON."""
    _check_ready()

    frame = camera_service.snapshot(save=False)
    result = face_detector.detect(frame.array)
    event = await save_detection_event(result)

    return DetectionResponse(
        event_id=event.id,
        timestamp=event.timestamp.isoformat(),
        face_count=result.count,
        inference_ms=result.inference_ms,
        frame_width=result.frame_width,
        frame_height=result.frame_height,
        faces=[FaceBox(**f.to_dict()) for f in result.faces],
    )


@router.get(
    "/snapshot",
    responses={200: {"content": {"image/jpeg": {}}}},
    response_class=Response,
)
async def detection_snapshot() -> Response:
    """Captura frame, detecta rostos, persiste evento e retorna JPEG anotado."""
    _check_ready()

    frame = camera_service.snapshot(save=True)
    result = face_detector.detect(frame.array)

    snapshot_path = _save_annotated(frame.array, result)
    event = await save_detection_event(result, snapshot_path=snapshot_path)

    annotated_bytes = snapshot_path.read_bytes()

    return Response(
        content=annotated_bytes,
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


def _save_annotated(frame_rgb: np.ndarray, result: DetectionResult) -> Path:
    """Desenha bounding boxes e salva a imagem anotada em disco."""
    img = frame_rgb.copy()
    for face in result.faces:
        cv2.rectangle(img, (face.x1, face.y1), (face.x2, face.y2),
                      color=(0, 255, 0), thickness=2)
        cv2.putText(img, f"{face.confidence:.0%}",
                    (face.x1, max(face.y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    label = (f"{result.count} rosto(s) | {result.inference_ms:.0f}ms"
             if result.count else "Nenhum rosto detectado")
    color = (0, 255, 0) if result.count else (0, 0, 255)
    cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2, cv2.LINE_AA)

    from datetime import timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = camera_service._snapshots_dir / f"annotated_{ts}.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path
