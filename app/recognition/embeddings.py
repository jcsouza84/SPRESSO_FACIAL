"""
Geração de embeddings faciais usando InsightFace (MobileFaceNet / ArcFace).

Fluxo correto:
  - Fotos de cadastro (upload/arquivo): detecta o rosto na imagem completa
    usando o detector leve do buffalo_sc, depois gera o embedding do crop.
  - Pipeline ao vivo: recebe o crop já extraído pelo SCRFD no Hailo-8,
    gera o embedding diretamente (sem re-detectar).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.logger import logger

_MODEL_NAME = "buffalo_sc"

_recognizer = None   # ArcFaceONNX — apenas reconhecimento
_detector   = None   # RetinaFace leve — apenas para fotos de cadastro


def _get_recognizer():
    """Modelo de reconhecimento (MobileFaceNet). Lazy init."""
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
    """Detector de rosto leve (buffalo_sc/det_500m). Usado SOMENTE no cadastro."""
    global _detector
    if _detector is not None:
        return _detector

    from insightface.model_zoo import get_model

    model_path = Path.home() / ".insightface/models/buffalo_sc/det_500m.onnx"
    if not model_path.exists():
        _ensure_models_downloaded()

    _detector = get_model(str(model_path), providers=["CPUExecutionProvider"])
    _detector.prepare(ctx_id=0, input_size=(640, 640), det_thresh=0.4)
    logger.info("Detector de cadastro carregado: {}", model_path.name)
    return _detector


def _ensure_models_downloaded():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name=_MODEL_NAME, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))


def _detect_and_crop(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Detecta o maior rosto na imagem e retorna o crop BGR 112x112.
    Retorna None se nenhum rosto for encontrado ou se for muito pequeno.
    """
    det = _get_detector()

    # Redimensiona imagens grandes para não estourar memória
    h, w = img_bgr.shape[:2]
    max_side = 1280
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        h, w = img_bgr.shape[:2]

    try:
        bboxes, kpss = det.detect(img_bgr, input_size=(640, 640))
    except Exception as e:
        logger.warning("Erro na detecção para crop: {}", e)
        return None

    if bboxes is None or len(bboxes) == 0:
        return None

    # Seleciona o rosto com maior área
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    idx = int(np.argmax(areas))
    x1, y1, x2, y2 = [int(v) for v in bboxes[idx, :4]]

    face_w = x2 - x1
    face_h = y2 - y1

    # Rejeita rostos muito pequenos — embedding seria de baixa qualidade
    MIN_FACE_PX = 80
    if face_w < MIN_FACE_PX or face_h < MIN_FACE_PX:
        logger.warning(
            "Rosto detectado muito pequeno ({}x{}px < {}px mínimo). "
            "Use uma foto mais próxima e frontal.",
            face_w, face_h, MIN_FACE_PX,
        )
        return None

    # Garante limites
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    logger.debug("Rosto detectado para cadastro: {}x{}px", face_w, face_h)
    return cv2.resize(crop, (112, 112))


def _embedding_from_crop(face_112_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Gera embedding normalizado a partir de um crop BGR 112x112."""
    try:
        rec = _get_recognizer()
        embedding = rec.get_feat(face_112_bgr).flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)
    except Exception as e:
        logger.warning("Erro ao gerar embedding do crop: {}", e)
        return None


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def get_face_embedding(face_roi_rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    Gera embedding a partir de um crop de rosto já extraído (RGB).
    Usado no pipeline ao vivo — o SCRFD/Hailo já detectou e cortou o rosto.
    """
    if face_roi_rgb is None or face_roi_rgb.size == 0:
        return None
    face_bgr = cv2.cvtColor(face_roi_rgb, cv2.COLOR_RGB2BGR)
    face_112 = cv2.resize(face_bgr, (112, 112))
    return _embedding_from_crop(face_112)


def get_embedding_from_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Gera embedding a partir de bytes de uma foto de cadastro (JPEG/PNG).
    Detecta automaticamente o rosto na imagem antes de gerar o embedding.
    """
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        return get_embedding_from_file_bgr(img_bgr)
    except Exception as e:
        logger.warning("Erro ao decodificar imagem para embedding: {}", e)
        return None


def get_embedding_from_file(path: Path) -> Optional[np.ndarray]:
    """
    Gera embedding a partir de um arquivo de imagem no disco.
    Detecta automaticamente o rosto antes de gerar o embedding.
    """
    try:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            logger.warning("Não foi possível ler: {}", path)
            return None
        return get_embedding_from_file_bgr(img_bgr)
    except Exception as e:
        logger.warning("Erro ao ler arquivo para embedding: {}", e)
        return None


def get_embedding_from_file_bgr(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Detecta rosto em imagem BGR completa e gera embedding."""
    crop = _detect_and_crop(img_bgr)
    if crop is None:
        logger.warning("Nenhum rosto detectado na imagem de cadastro")
        return None
    return _embedding_from_crop(crop)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distância coseno entre dois embeddings normalizados. Range: [0, 2]."""
    return float(1.0 - np.dot(a, b))
