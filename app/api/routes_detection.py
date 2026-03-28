from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

import cv2
import numpy as np

from app.camera.service import camera_service
from app.detection.face_detector import face_detector, DetectionResult, DetectedFace
from app.recognition.matcher import face_matcher, UNKNOWN
from app.recognition.embeddings import get_embeddings_from_frame, get_face_embedding
from app.services.event_service import save_detection_event
from app.config import settings
from app.logger import logger

router = APIRouter(prefix="/detection", tags=["detection"])

# Diretório para crops de faces detectadas
_CROPS_DIR = settings.data_dir / "face_crops"


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
        "ready":            face_detector.is_ready,
        "persons_in_cache": face_matcher.persons_in_cache,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }


@router.get("/faces", response_model=DetectionResponse)
async def detect_faces() -> DetectionResponse:
    """Captura frame, detecta rostos, reconhece pessoas, persiste evento."""
    _check_ready()

    frame = camera_service.snapshot(save=False)
    result = face_detector.detect(frame.array)

    recognitions, face_crops = _recognize_and_save_crops(frame.array, result)
    event = await save_detection_event(result, face_crops=face_crops)

    return DetectionResponse(
        event_id=event.id,
        timestamp=event.timestamp.isoformat(),
        face_count=result.count,
        inference_ms=result.inference_ms,
        frame_width=result.frame_width,
        frame_height=result.frame_height,
        faces=recognitions,
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

    recognitions, face_crops = _recognize_and_save_crops(frame.array, result)
    snapshot_path = _save_annotated(frame.array, result, recognitions)
    event = await save_detection_event(result, snapshot_path=snapshot_path, face_crops=face_crops)

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
    """Recorta o rosto com margem de 20% em cada lado."""
    h, w = frame_rgb.shape[:2]
    fw, fh = face.x2 - face.x1, face.y2 - face.y1
    pad_x, pad_y = int(fw * 0.2), int(fh * 0.2)
    x1 = max(0, face.x1 - pad_x)
    y1 = max(0, face.y1 - pad_y)
    x2 = min(w, face.x2 + pad_x)
    y2 = min(h, face.y2 + pad_y)
    return frame_rgb[y1:y2, x1:x2]


def _save_face_crop(roi_rgb: np.ndarray, event_ts: str, idx: int) -> Path:
    """Salva o crop do rosto em disco e retorna o path."""
    _CROPS_DIR.mkdir(parents=True, exist_ok=True)
    path = _CROPS_DIR / f"crop_{event_ts}_f{idx}.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR))
    return path


def _iou(b1: list[int], b2: np.ndarray) -> float:
    """IoU entre bbox SCRFD [x1,y1,x2,y2] e bbox InsightFace numpy."""
    ix1 = max(b1[0], float(b2[0]))
    iy1 = max(b1[1], float(b2[1]))
    ix2 = min(b1[2], float(b2[2]))
    iy2 = min(b1[3], float(b2[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (float(b2[2]) - float(b2[0])) * (float(b2[3]) - float(b2[1]))
    return inter / (a1 + a2 - inter + 1e-6)


# Rostos menores que este limiar (distância) usam upscale para melhor alinhamento
_SMALL_FACE_PX = 90


def _recognize_and_save_crops(
    frame_rgb: np.ndarray,
    result: DetectionResult,
) -> tuple[list[FaceBox], list[dict]]:
    """
    Pipeline unificado de reconhecimento:
      1. Uma única passagem InsightFace det_500m sobre o frame completo
         → bboxes alinhados + embeddings para todos os rostos
      2. Associa cada detecção SCRFD/Hailo com a detecção InsightFace
         mais próxima (IoU ≥ 0.3)
      3a. Rostos grandes (≥ _SMALL_FACE_PX): usa embedding do frame completo
          (alinhamento preciso, rápido)
      3b. Rostos pequenos / longe (< _SMALL_FACE_PX): usa crop com upscale
          antes do alinhamento para maximizar qualidade dos keypoints
      4. Fallback para ROI crop se InsightFace não encontrou o rosto

    Retorna (lista FaceBox, lista de dicts {crop_path, embedding}).
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    boxes: list[FaceBox] = []
    crops: list[dict] = []

    # Única passagem InsightFace sobre o frame inteiro
    frame_embeddings = get_embeddings_from_frame(frame_rgb)

    for i, face in enumerate(result.faces):
        scrfd_box = [face.x1, face.y1, face.x2, face.y2]
        face_is_small = face.width < _SMALL_FACE_PX or face.height < _SMALL_FACE_PX

        # Associa com a detecção InsightFace pelo maior IoU
        best_idx, best_iou = -1, 0.0
        for j, (ins_bbox, _, _) in enumerate(frame_embeddings):
            iou_val = _iou(scrfd_box, ins_bbox)
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = j

        roi = _extract_roi(frame_rgb, face)
        crop_path = _save_face_crop(roi, ts, i)

        if best_idx >= 0 and best_iou >= 0.3 and not face_is_small:
            # Rosto grande e próximo: embedding do frame completo (mais rápido)
            logger.info("Rosto #{}: {}x{}px — caminho FRAME (iou={:.2f})", i, face.width, face.height, best_iou)
            _, _, embedding = frame_embeddings[best_idx]
            emb_bytes = embedding.tobytes()
            rec = face_matcher.identify_from_embedding(embedding)
        else:
            # Rosto pequeno/distante ou não encontrado:
            motivo = f"pequeno({face.width}x{face.height}px)" if face_is_small else f"sem match InsightFace(iou={best_iou:.2f})"
            logger.info("Rosto #{}: {}x{}px — caminho CROP+UPSCALE ({})", i, face.width, face.height, motivo)
            embedding = get_face_embedding(roi)
            emb_bytes = embedding.tobytes() if embedding is not None else None
            rec = face_matcher.identify_from_embedding(embedding) if embedding is not None else UNKNOWN

        boxes.append(FaceBox(
            **face.to_dict(),
            matched=rec.matched,
            person_id=rec.person_id,
            person_name=rec.person_name,
            category=rec.category,
            recognition_confidence=rec.confidence,
        ))
        crops.append({"crop_path": str(crop_path), "embedding": emb_bytes})

    return boxes, crops


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
