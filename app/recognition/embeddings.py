"""
Geração de embeddings faciais usando InsightFace (MobileFaceNet / ArcFace).

IMPORTANTE: O ArcFace exige faces alinhadas usando os 5 keypoints faciais
(olhos, nariz, cantos da boca). Sem o alinhamento, embeddings de ângulos
diferentes da mesma pessoa ficam muito distantes — tornando o reconhecimento inviável.

Fluxo correto:
  - Fotos de cadastro: detecta rosto + keypoints → align → embedding
  - Pipeline ao vivo: SCRFD/Hailo detecta rosto → align com det_500m → embedding
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.logger import logger

_MODEL_NAME = "buffalo_sc"

_recognizer = None   # ArcFaceONNX
_detector   = None   # det_500m — usado para cadastro E para alinhar no pipeline ao vivo


def _get_recognizer():
    global _recognizer
    if _recognizer is not None:
        return _recognizer

    from insightface.model_zoo import get_model
    model_path = Path.home() / ".insightface/models/buffalo_sc/w600k_mbf.onnx"
    if not model_path.exists():
        _ensure_models_downloaded()
    _recognizer = get_model(str(model_path), providers=["CPUExecutionProvider"])
    _recognizer.prepare(ctx_id=0)
    logger.info("Reconhecedor carregado: {}", model_path.name)
    return _recognizer


def _get_detector():
    global _detector
    if _detector is not None:
        return _detector

    from insightface.model_zoo import get_model
    model_path = Path.home() / ".insightface/models/buffalo_sc/det_500m.onnx"
    if not model_path.exists():
        _ensure_models_downloaded()
    _detector = get_model(str(model_path), providers=["CPUExecutionProvider"])
    _detector.prepare(ctx_id=0, input_size=(640, 640), det_thresh=0.4)
    logger.info("Detector carregado: {}", model_path.name)
    return _detector


def _ensure_models_downloaded():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name=_MODEL_NAME, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))


def _align_face(img_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Alinha o rosto para 112x112 usando os 5 keypoints (norm_crop)."""
    from insightface.utils import face_align
    return face_align.norm_crop(img_bgr, landmark=kps, image_size=112)


def _detect_align_crop(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Detecta o maior rosto, usa keypoints para alinhar e retorna crop BGR 112x112.
    Retorna None se nenhum rosto for encontrado ou muito pequeno.
    """
    det = _get_detector()

    h, w = img_bgr.shape[:2]
    max_side = 1280
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        h, w = img_bgr.shape[:2]

    try:
        bboxes, kpss = det.detect(img_bgr, input_size=(640, 640))
    except Exception as e:
        logger.warning("Erro na detecção: {}", e)
        return None

    if bboxes is None or len(bboxes) == 0:
        return None

    # Maior rosto
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    idx = int(np.argmax(areas))

    x1, y1, x2, y2 = [int(v) for v in bboxes[idx, :4]]
    face_w, face_h = x2 - x1, y2 - y1

    MIN_FACE_PX = 80
    if face_w < MIN_FACE_PX or face_h < MIN_FACE_PX:
        logger.warning(
            "Rosto muito pequeno ({}x{}px). Use foto mais próxima e frontal.", face_w, face_h
        )
        return None

    # Alinha usando keypoints se disponíveis
    if kpss is not None and len(kpss) > idx:
        kps = kpss[idx]
        try:
            aligned = _align_face(img_bgr, kps)
            logger.debug("Rosto alinhado com keypoints: {}x{}px → 112x112", face_w, face_h)
            return aligned
        except Exception as e:
            logger.warning("Falha no alinhamento, usando crop simples: {}", e)

    # Fallback: crop simples sem alinhamento
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = img_bgr[y1:y2, x1:x2]
    logger.debug("Crop simples (sem keypoints): {}x{}px", face_w, face_h)
    return cv2.resize(crop, (112, 112))


def _embedding_from_aligned(face_112_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Gera embedding normalizado a partir de crop alinhado BGR 112x112."""
    try:
        rec = _get_recognizer()
        embedding = rec.get_feat(face_112_bgr).flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)
    except Exception as e:
        logger.warning("Erro ao gerar embedding: {}", e)
        return None


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def get_face_embedding(face_roi_rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    Gera embedding a partir de um crop de rosto já extraído (RGB).
    Faz upscale se necessário e tenta re-detectar para obter alinhamento.
    """
    if face_roi_rgb is None or face_roi_rgb.size == 0:
        return None

    face_bgr = cv2.cvtColor(face_roi_rgb, cv2.COLOR_RGB2BGR)

    # Garante tamanho mínimo para o detector encontrar o rosto
    h, w = face_bgr.shape[:2]
    if min(h, w) < 160:
        scale = 160 / min(h, w)
        face_bgr = cv2.resize(face_bgr, (int(w * scale), int(h * scale)))

    aligned = _detect_align_crop(face_bgr)
    if aligned is not None:
        return _embedding_from_aligned(aligned)

    # Fallback: redimensiona direto sem alinhamento
    face_112 = cv2.resize(face_bgr, (112, 112))
    return _embedding_from_aligned(face_112)


def get_embeddings_from_frame(frame_rgb: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Detecta TODOS os rostos no frame completo e retorna lista de
    (bbox [x1,y1,x2,y2], keypoints, embedding) já alinhados.
    Usar no pipeline ao vivo para máxima precisão.
    """
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    det = _get_detector()
    rec = _get_recognizer()

    try:
        bboxes, kpss = det.detect(frame_bgr, input_size=(640, 640))
    except Exception as e:
        logger.warning("Erro na detecção do frame: {}", e)
        return []

    if bboxes is None or len(bboxes) == 0:
        return []

    results = []
    h, w = frame_bgr.shape[:2]
    from insightface.utils import face_align

    for i in range(len(bboxes)):
        x1, y1, x2, y2, conf = bboxes[i]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        face_w, face_h = x2 - x1, y2 - y1

        if face_w < 40 or face_h < 40:
            continue

        # Alinha com keypoints
        try:
            if kpss is not None and len(kpss) > i:
                aligned = face_align.norm_crop(frame_bgr, landmark=kpss[i], image_size=112)
            else:
                crop = frame_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                aligned = cv2.resize(crop, (112, 112))
        except Exception:
            crop = frame_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            aligned = cv2.resize(crop, (112, 112))

        emb = _embedding_from_aligned(aligned)
        if emb is not None:
            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
            kps = kpss[i] if kpss is not None and len(kpss) > i else np.array([])
            results.append((bbox, kps, emb))

    return results


def get_embedding_from_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Gera embedding a partir de bytes de uma foto de cadastro (JPEG/PNG).
    Detecta e alinha o rosto automaticamente.
    """
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        return _embedding_from_full_image(img_bgr)
    except Exception as e:
        logger.warning("Erro ao decodificar imagem: {}", e)
        return None


def get_embedding_from_file(path: Path) -> Optional[np.ndarray]:
    """
    Gera embedding a partir de arquivo de imagem no disco.
    Detecta e alinha o rosto automaticamente.
    """
    try:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            logger.warning("Não foi possível ler: {}", path)
            return None
        return _embedding_from_full_image(img_bgr)
    except Exception as e:
        logger.warning("Erro ao ler arquivo: {}", e)
        return None


def _embedding_from_full_image(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Detecta, alinha e gera embedding de uma imagem completa."""
    aligned = _detect_align_crop(img_bgr)
    if aligned is None:
        return None
    return _embedding_from_aligned(aligned)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distância coseno entre dois embeddings normalizados. Range: [0, 2]."""
    return float(1.0 - np.dot(a, b))
