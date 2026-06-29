import base64
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

import cv2
import numpy as np

from app.camera.service import camera_service
from app.detection.face_detector import face_detector, DetectionResult, DetectedFace
from app.detection.person_face_detector import person_face_detector
from app.recognition.matcher import face_matcher, UNKNOWN
from app.recognition.embeddings import get_embeddings_from_frame, get_face_embedding
from app.services.event_service import save_detection_event
from app.services.presence_service import presence_service
from app.services.alert_service import alert_service
from app.storage.models import PersonCategory
from app.config import settings
from app.logger import logger

router = APIRouter(prefix="/detection", tags=["detection"])

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
    recognition_distance: float = 1.0
    yaw: float = 0.0                        # ângulo de rotação horizontal estimado (graus)
    filter_reason: str | None = None        # motivo do descarte pelo filtro de qualidade


class AlertSummary(BaseModel):
    id: int
    person_id: int
    person_name: str
    category: str
    event_id: int | None = None
    face_id: int | None = None
    crop_path: str | None = None
    confidence: float
    message: str | None = None
    created_at: str


class DetectionResponse(BaseModel):
    event_id: int | None = None
    timestamp: str
    face_count: int
    inference_ms: float
    frame_width: int
    frame_height: int
    faces: list[FaceBox]
    deduplicated: bool = False
    alerts: list[AlertSummary] = []
    active_alert: AlertSummary | None = None
    threshold: float = 0.62
    frame_base64: str | None = None
    detection_stats: dict = {}              # diagnóstico para tela de calibração


@dataclass
class _PipelineOutput:
    result: DetectionResult
    recognitions: list[FaceBox]
    face_crops: list[dict]
    snapshot_path: Path | None
    annotated_jpeg: bytes | None
    deduplicated: bool
    event_id: int | None
    timestamp: datetime
    alerts: list[AlertSummary]
    active_alert: AlertSummary | None
    detection_stats: dict = None


@router.get("/status")
async def detection_status():
    active = await alert_service.get_active_alerts()
    stats = await alert_service.presence_stats_today()
    return {
        "ready":            face_detector.is_ready,
        "persons_in_cache": face_matcher.persons_in_cache,
        "active_alerts":    len(active),
        "presence_today":   stats,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }


@router.get("/faces", response_model=DetectionResponse)
async def detect_faces(
    dedup: bool = Query(default=True, description="Ignorar eventos repetidos"),
    persist: bool = Query(default=True, description="Salvar evento no banco"),
    include_frame: bool = Query(default=False, description="Incluir frame anotado em base64"),
    process_alerts: bool = Query(default=True, description="Processar alertas blacklist"),
    camera_id: Optional[str] = Query(default=None, description="ID da câmera (padrão: câmera primária configurada)"),
) -> DetectionResponse:
    """Captura frame, detecta rostos, reconhece pessoas."""
    resolved_id = camera_id or await _get_active_camera_id()
    _check_ready(camera_id=resolved_id)
    out = await _run_pipeline(
        dedup=dedup,
        persist=persist,
        process_alerts=process_alerts,
        camera_id=resolved_id,
    )
    return _to_response(out, include_frame=include_frame)


@router.get(
    "/snapshot",
    responses={200: {"content": {"image/jpeg": {}}}},
    response_class=Response,
)
async def detection_snapshot(
    dedup: bool = Query(default=True),
) -> Response:
    """Captura frame, detecta rostos, reconhece e retorna JPEG anotado."""
    _check_ready()
    out = await _run_pipeline(dedup=dedup, save_frame_snapshot=True, persist=True)

    if not out.snapshot_path or not out.snapshot_path.exists():
        raise HTTPException(status_code=500, detail="Falha ao gerar snapshot anotado")

    headers = {
        "X-Face-Count":   str(out.result.count),
        "X-Inference-Ms": str(out.result.inference_ms),
        "X-Deduplicated": "true" if out.deduplicated else "false",
    }
    if out.event_id:
        headers["X-Event-Id"] = str(out.event_id)
    if out.active_alert:
        headers["X-Active-Alert-Id"] = str(out.active_alert.id)

    return Response(
        content=out.annotated_jpeg or out.snapshot_path.read_bytes(),
        media_type="image/jpeg",
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def _get_active_camera_id() -> str:
    """Lê a câmera primária do banco de settings."""
    try:
        from app.services import settings_service
        return await settings_service.get_setting("active_camera_id") or "imx0"
    except Exception:
        return "imx0"


async def _run_pipeline(
    dedup: bool = True,
    save_frame_snapshot: bool = False,
    persist: bool = True,
    process_alerts: bool = True,
    save_crops: bool = True,
    camera_id: str | None = None,
) -> _PipelineOutput:
    if not persist:
        save_crops = False

    if camera_id is None:
        camera_id = await _get_active_camera_id()

    quality = await _load_quality_settings()

    from app.camera.registry import camera_registry
    if camera_registry.is_registered(camera_id):
        frame = camera_registry.snapshot(camera_id)
        if save_frame_snapshot and persist:
            camera_service._save(frame)
    else:
        frame = camera_service.snapshot(save=save_frame_snapshot and persist)

    result = face_detector.detect(
        frame.array,
        conf_override=quality.get("detection_confidence"),
    )

    # Filtro de tamanho mínimo para registro de evento
    min_detect_px = quality.get("min_face_px_detect", _QUALITY_DEFAULTS["min_face_px_detect"])
    scrfd_total = len(result.faces)
    result.faces = [
        f for f in result.faces
        if f.width >= min_detect_px and f.height >= min_detect_px
    ]
    n_filtered_size = scrfd_total - len(result.faces)

    # Filtro de sobreposição pessoa-rosto (requer PersonFaceDetector ativo)
    if quality.get("require_person_overlap") and person_face_detector.is_ready and result.faces:
        pf_result = person_face_detector.detect(
            frame.array,
            conf_override=quality.get("detection_confidence"),
        )
        if not pf_result.has_persons:
            logger.debug("Nenhuma pessoa detectada pelo YOLO — {} rosto(s) ignorados", len(result.faces))
            result.faces = []
        else:
            person_boxes = [[p.x1, p.y1, p.x2, p.y2] for p in pf_result.persons]
            result.faces = [
                f for f in result.faces
                if any(_iou([f.x1, f.y1, f.x2, f.y2], np.array(pb)) >= 0.1
                       for pb in person_boxes)
            ]
            logger.debug(
                "PersonFace: {} pessoa(s) → {} rosto(s) com sobreposição mantidos",
                len(pf_result.persons), len(result.faces),
            )

    recognitions, face_crops = _recognize_and_save_crops(
        frame.array, result, save_crops=save_crops, quality=quality,
    )

    annotated_jpeg = _encode_annotated_jpeg(frame.array, result, recognitions)

    if not persist:
        skip = False
    else:
        matched_ids = [
            r.person_id for r in recognitions
            if r.matched and r.person_id is not None
        ]
        skip = dedup and presence_service.should_skip_event(matched_ids, result.count)

    now = datetime.now(timezone.utc)
    event_id: int | None = None
    snapshot_path: Path | None = None
    alerts: list[AlertSummary] = []
    active_alert: AlertSummary | None = None

    if persist and not skip:
        matched_ids = [
            r.person_id for r in recognitions
            if r.matched and r.person_id is not None
        ]
        snapshot_path = _save_annotated_file(frame.array, result, recognitions)
        from app.camera.registry import camera_registry
        cam_label = None
        try:
            cam_label = camera_registry.get(camera_id).label
        except KeyError:
            pass
        event, face_ids = await save_detection_event(
            result,
            snapshot_path=snapshot_path,
            face_crops=face_crops,
            camera_id=camera_id,
            camera_label=cam_label,
        )
        event_id = event.id
        now = event.timestamp
        presence_service.record_event(matched_ids, result.count)

        if process_alerts:
            min_alert_px  = quality.get("min_face_px_alert",    _QUALITY_DEFAULTS["min_face_px_alert"])
            min_alert_conf = quality.get("alert_min_confidence", _QUALITY_DEFAULTS["alert_min_confidence"])

            for i, rec_box in enumerate(recognitions):
                if not rec_box.matched:
                    continue
                crop_path = face_crops[i].get("crop_path") if i < len(face_crops) else None
                face_id = face_ids[i] if i < len(face_ids) else None

                # Filtros de qualidade para disparo de alerta
                if rec_box.width < min_alert_px or rec_box.height < min_alert_px:
                    logger.debug(
                        "Alerta suprimido — rosto muito pequeno ({}x{}px < {}px)",
                        rec_box.width, rec_box.height, min_alert_px,
                    )
                    continue
                if rec_box.recognition_confidence < min_alert_conf:
                    logger.debug(
                        "Alerta suprimido — confiança insuficiente ({:.1f}% < {:.1f}%)",
                        rec_box.recognition_confidence, min_alert_conf,
                    )
                    continue

                if rec_box.category == PersonCategory.blacklist.value:
                    from app.recognition.matcher import RecognitionResult
                    rec_result = RecognitionResult(
                        matched=True,
                        person_id=rec_box.person_id,
                        person_name=rec_box.person_name,
                        category=rec_box.category,
                        confidence=rec_box.recognition_confidence,
                        distance=rec_box.recognition_distance,
                    )
                    alert = await alert_service.process_recognition(
                        rec_result,
                        event_id=event_id,
                        face_id=face_id,
                        crop_path=crop_path,
                    )
                    if alert:
                        summary = _alert_to_summary(alert)
                        alerts.append(summary)
                        if not alert.confirmed:
                            active_alert = summary
                elif rec_box.category == PersonCategory.vip.value:
                    from app.services.presence_service import presence_service as ps
                    await ps.log_presence(
                        person_id=rec_box.person_id,
                        person_name=rec_box.person_name,
                        category=rec_box.category,
                        event_id=event_id,
                    )
    elif persist and skip:
        logger.debug("Evento deduplicado — não persistido")

    if process_alerts and active_alert is None:
        active = await alert_service.get_active_alerts()
        if active:
            active_alert = _alert_to_summary(active[0])

    n_filtered_yaw = sum(1 for r in recognitions if r.filter_reason == "Rosto lateral")
    n_recognized   = sum(1 for r in recognitions if r.matched)
    stats = {
        "scrfd_total":      scrfd_total,
        "filtered_size":    n_filtered_size,
        "filtered_yaw":     n_filtered_yaw,
        "filtered_person":  0,
        "processed":        len([r for r in recognitions if r.filter_reason is None]),
        "recognized":       n_recognized,
    }

    return _PipelineOutput(
        result=result,
        recognitions=recognitions,
        face_crops=face_crops,
        snapshot_path=snapshot_path,
        annotated_jpeg=annotated_jpeg,
        deduplicated=skip if persist else False,
        event_id=event_id,
        timestamp=now,
        alerts=alerts,
        active_alert=active_alert,
        detection_stats=stats,
    )


def _alert_to_summary(alert) -> AlertSummary:
    return AlertSummary(**alert.to_dict())


def _to_response(out: _PipelineOutput, include_frame: bool = False) -> DetectionResponse:
    frame_b64 = None
    if include_frame and out.annotated_jpeg:
        frame_b64 = base64.b64encode(out.annotated_jpeg).decode("ascii")
    return DetectionResponse(
        event_id=out.event_id,
        timestamp=out.timestamp.isoformat(),
        face_count=out.result.count,
        inference_ms=out.result.inference_ms,
        frame_width=out.result.frame_width,
        frame_height=out.result.frame_height,
        faces=out.recognitions,
        deduplicated=out.deduplicated,
        alerts=out.alerts,
        active_alert=out.active_alert,
        threshold=settings.recognition_threshold,
        frame_base64=frame_b64,
        detection_stats=out.detection_stats or {},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_ready(camera_id: str = "imx0") -> None:
    from app.camera.registry import camera_registry
    if camera_registry.is_registered(camera_id):
        if not camera_registry.get(camera_id).is_ready:
            raise HTTPException(status_code=503, detail=f"Câmera '{camera_id}' não disponível")
    elif not camera_service.is_ready:
        raise HTTPException(status_code=503, detail="Câmera não disponível")
    if not face_detector.is_ready:
        raise HTTPException(status_code=503, detail="Detector não disponível")


def _extract_roi(frame_rgb: np.ndarray, face: DetectedFace) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    fw, fh = face.x2 - face.x1, face.y2 - face.y1
    pad_x, pad_y = int(fw * 0.2), int(fh * 0.2)
    x1 = max(0, face.x1 - pad_x)
    y1 = max(0, face.y1 - pad_y)
    x2 = min(w, face.x2 + pad_x)
    y2 = min(h, face.y2 + pad_y)
    return frame_rgb[y1:y2, x1:x2]


def _save_face_crop(roi_rgb: np.ndarray, event_ts: str, idx: int) -> Path:
    _CROPS_DIR.mkdir(parents=True, exist_ok=True)
    path = _CROPS_DIR / f"crop_{event_ts}_f{idx}.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR))
    return path


def _iou(b1: list[int], b2: np.ndarray) -> float:
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


_QUALITY_DEFAULTS: dict = {
    "detection_confidence":  0.65,
    "min_face_px_detect":    80,
    "min_face_px_recognize": 120,
    "min_face_px_alert":     150,
    "alert_min_confidence":  50.0,
    "require_frontal_face":  True,
    "max_face_yaw_degrees":  40,
    "require_person_overlap": False,
}


async def _load_quality_settings() -> dict:
    """Lê parâmetros de qualidade do banco. Usa defaults se ausentes."""
    try:
        from app.services import settings_service
        keys = list(_QUALITY_DEFAULTS.keys())
        values = {k: await settings_service.get_setting(k) for k in keys}
        result = {}
        for k, v in values.items():
            default = _QUALITY_DEFAULTS[k]
            if v is None or v == "":
                result[k] = default
            elif isinstance(default, bool):
                result[k] = v.lower() == "true"
            elif isinstance(default, int):
                result[k] = int(v)
            elif isinstance(default, float):
                result[k] = float(v)
            else:
                result[k] = v
        return result
    except Exception:
        return dict(_QUALITY_DEFAULTS)


def _estimate_yaw_from_kps(kps: np.ndarray) -> float:
    """Estima ângulo de rotação horizontal (yaw) em graus a partir de 5 keypoints.
    kps: (5, 2) — [olho_esq, olho_dir, nariz, boca_esq, boca_dir]
    Retorna grau absoluto de yaw: 0 = frontal, ~90 = perfil.
    """
    if kps is None or len(kps) < 3:
        return 0.0
    left_eye  = kps[0]
    right_eye = kps[1]
    nose      = kps[2]
    eye_width = abs(float(right_eye[0]) - float(left_eye[0]))
    if eye_width < 1.0:
        return 90.0
    eye_center_x = (float(left_eye[0]) + float(right_eye[0])) / 2.0
    nose_offset  = abs(float(nose[0]) - eye_center_x)
    ratio = nose_offset / (eye_width / 2.0)
    return min(ratio * 45.0, 90.0)


def _estimate_yaw_from_bbox(width: int, height: int) -> float:
    """Proxy de yaw via proporção da bbox quando não há keypoints.
    Rostos frontais têm aspect ≈ 0.75-0.95; perfis ficam abaixo de 0.5.
    """
    aspect = width / (height + 1e-6)
    if aspect >= 0.65:
        return 0.0
    # Mapeia aspect [0.65 → 0] para yaw [0 → 45]
    return min((0.65 - aspect) / 0.65 * 90.0, 90.0)


def _recognize_and_save_crops(
    frame_rgb: np.ndarray,
    result: DetectionResult,
    save_crops: bool = True,
    quality: dict | None = None,
) -> tuple[list[FaceBox], list[dict]]:
    q = quality or _QUALITY_DEFAULTS
    min_rec_px   = q.get("min_face_px_recognize", _QUALITY_DEFAULTS["min_face_px_recognize"])
    require_frontal = q.get("require_frontal_face", _QUALITY_DEFAULTS["require_frontal_face"])
    max_yaw      = q.get("max_face_yaw_degrees",  _QUALITY_DEFAULTS["max_face_yaw_degrees"])

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    boxes: list[FaceBox] = []
    crops: list[dict] = []

    frame_embeddings = get_embeddings_from_frame(frame_rgb)

    for i, face in enumerate(result.faces):
        scrfd_box = [face.x1, face.y1, face.x2, face.y2]
        face_is_small = face.width < min_rec_px or face.height < min_rec_px

        best_idx, best_iou, best_kps = -1, 0.0, None
        for j, (ins_bbox, kps, _) in enumerate(frame_embeddings):
            iou_val = _iou(scrfd_box, ins_bbox)
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = j
                best_kps = kps

        roi = _extract_roi(frame_rgb, face)
        if save_crops:
            crop_path = _save_face_crop(roi, ts, i)
            crop_path_str = str(crop_path)
        else:
            crop_path_str = None

        # Estimativa de yaw (sempre — para diagnóstico na calibração)
        if best_kps is not None and len(best_kps) >= 3:
            face_yaw = _estimate_yaw_from_kps(best_kps)
        else:
            face_yaw = _estimate_yaw_from_bbox(face.width, face.height)

        # Filtro de frontalidade
        if require_frontal and face_yaw > max_yaw:
            logger.debug(
                "Rosto #{} lateral (yaw≈{:.0f}° > {}°) — ignorado p/ reconhecimento",
                i, face_yaw, max_yaw,
            )
            boxes.append(FaceBox(**face.to_dict(), yaw=round(face_yaw, 1),
                                 filter_reason="Rosto lateral"))
            crops.append({"crop_path": crop_path_str, "embedding": None})
            continue

        if best_idx >= 0 and best_iou >= 0.3 and not face_is_small:
            logger.debug(
                "Rosto #{}: {}x{}px — caminho FRAME (iou={:.2f})",
                i, face.width, face.height, best_iou,
            )
            _, _, embedding = frame_embeddings[best_idx]
            emb_bytes = embedding.tobytes()
            rec = face_matcher.identify_from_embedding(embedding)
        else:
            motivo = (
                f"pequeno({face.width}x{face.height}px)" if face_is_small
                else f"sem match InsightFace(iou={best_iou:.2f})"
            )
            logger.debug(
                "Rosto #{}: {}x{}px — caminho CROP+UPSCALE ({})",
                i, face.width, face.height, motivo,
            )
            embedding = get_face_embedding(roi)
            emb_bytes = embedding.tobytes() if embedding is not None else None
            rec = (
                face_matcher.identify_from_embedding(embedding)
                if embedding is not None else UNKNOWN
            )

        boxes.append(FaceBox(
            **face.to_dict(),
            matched=rec.matched,
            person_id=rec.person_id,
            person_name=rec.person_name,
            category=rec.category,
            recognition_confidence=rec.confidence,
            recognition_distance=rec.distance,
            yaw=round(face_yaw, 1),
        ))
        crops.append({"crop_path": crop_path_str, "embedding": emb_bytes})

    return boxes, crops


def _draw_annotated(
    frame_rgb: np.ndarray,
    result: DetectionResult,
    recognitions: list[FaceBox],
) -> np.ndarray:
    img = frame_rgb.copy()

    for box in recognitions:
        if box.matched and box.category == PersonCategory.blacklist.value:
            color = (255, 60, 60)
        elif box.matched:
            color = (0, 200, 0)
        else:
            color = (0, 80, 255)

        cv2.rectangle(img, (box.x1, box.y1), (box.x2, box.y2), color=color, thickness=2)

        if box.matched:
            label = f"{box.person_name} {box.recognition_confidence:.0f}%"
        else:
            label = f"? {box.recognition_distance:.2f}"
        cv2.putText(img, label,
                    (box.x1, max(box.y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        if box.matched:
            sub = f"{box.category or ''} dist={box.recognition_distance:.2f}"
            cv2.putText(img, sub,
                        (box.x1, max(box.y1 - 24, 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        else:
            size_label = f"{box.width}x{box.height}px"
            cv2.putText(img, size_label,
                        (box.x1, max(box.y1 - 24, 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    total_label = f"{result.count} rosto(s) | {result.inference_ms:.0f}ms | thr={settings.recognition_threshold:.2f}"
    cv2.putText(img, total_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def _encode_annotated_jpeg(
    frame_rgb: np.ndarray,
    result: DetectionResult,
    recognitions: list[FaceBox],
) -> bytes:
    img = _draw_annotated(frame_rgb, result, recognitions)
    _, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    return buf.tobytes()


def _save_annotated_file(
    frame_rgb: np.ndarray,
    result: DetectionResult,
    recognitions: list[FaceBox],
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = camera_service._snapshots_dir / f"annotated_{ts}.jpg"
    path.write_bytes(_encode_annotated_jpeg(frame_rgb, result, recognitions))
    return path
