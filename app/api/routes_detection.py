from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

import cv2
import numpy as np

from app.camera.service import camera_service
from app.detection.face_detector import face_detector, DetectionResult

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
    """Captura frame, detecta rostos e retorna JSON com bounding boxes."""
    _check_ready()

    frame = camera_service.snapshot(save=False)
    result = face_detector.detect(frame.array)

    return DetectionResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
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
    """Captura frame e retorna JPEG com bounding boxes desenhados nos rostos."""
    _check_ready()

    frame = camera_service.snapshot(save=True)
    result = face_detector.detect(frame.array)

    annotated = _draw_detections(frame.array, result)
    _, encoded = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Face-Count":    str(result.count),
            "X-Inference-Ms":  str(result.inference_ms),
            "X-Captured-At":   datetime.now(timezone.utc).isoformat(),
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


def _draw_detections(frame_rgb: np.ndarray, result: DetectionResult) -> np.ndarray:
    img = frame_rgb.copy()
    for face in result.faces:
        cv2.rectangle(img, (face.x1, face.y1), (face.x2, face.y2),
                      color=(0, 255, 0), thickness=2)
        label = f"{face.confidence:.0%}"
        cv2.putText(img, label,
                    (face.x1, max(face.y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 2, cv2.LINE_AA)
    if result.count == 0:
        cv2.putText(img, "Nenhum rosto detectado",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        label = f"{result.count} rosto(s) | {result.inference_ms:.0f}ms"
        cv2.putText(img, label,
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 2, cv2.LINE_AA)
    return img
