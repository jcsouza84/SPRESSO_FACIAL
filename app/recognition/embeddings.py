"""
Geração de embeddings faciais usando InsightFace (MobileFaceNet / ArcFace).

Usa APENAS o módulo de reconhecimento do InsightFace — a detecção
continua sendo feita pelo SCRFD no Hailo-8.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.logger import logger

# Modelo leve — MobileFaceNet treinado em WebFace600K
# Gera embeddings de 512 dimensões
_MODEL_NAME = "buffalo_sc"
_EMBED_SIZE = 512

_recognizer = None  # instância lazy do modelo de reconhecimento


def _get_recognizer():
    """Inicializa o modelo de reconhecimento na primeira chamada."""
    global _recognizer
    if _recognizer is not None:
        return _recognizer

    import onnxruntime as ort
    from insightface.model_zoo import get_model

    # Carrega apenas o modelo de reconhecimento (w600k_mbf.onnx)
    model_path = Path.home() / ".insightface/models/buffalo_sc/w600k_mbf.onnx"
    if not model_path.exists():
        # Baixa se não existir
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name=_MODEL_NAME, providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))

    _recognizer = get_model(str(model_path), providers=["CPUExecutionProvider"])
    _recognizer.prepare(ctx_id=0)
    logger.info("Modelo de reconhecimento carregado: {}", model_path.name)
    return _recognizer


def get_face_embedding(face_roi_rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    Gera embedding de 512 dimensões a partir do crop de um rosto (RGB).

    Args:
        face_roi_rgb: Imagem do rosto cortada, qualquer tamanho (será redimensionada).

    Returns:
        np.ndarray de shape (512,) normalizado, ou None se falhar.
    """
    if face_roi_rgb is None or face_roi_rgb.size == 0:
        return None

    try:
        rec = _get_recognizer()

        # get_feat espera BGR uint8 112x112 — ela faz normalização internamente
        face_bgr = cv2.cvtColor(face_roi_rgb, cv2.COLOR_RGB2BGR)
        face_112 = cv2.resize(face_bgr, (112, 112))

        embedding = rec.get_feat(face_112).flatten()

        # Normaliza para comprimento unitário (distância coseno)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    except Exception as e:
        logger.warning("Erro ao gerar embedding: {}", e)
        return None


def get_embedding_from_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """Gera embedding a partir de bytes de imagem JPEG/PNG."""
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return get_face_embedding(img_rgb)
    except Exception as e:
        logger.warning("Erro ao decodificar imagem para embedding: {}", e)
        return None


def get_embedding_from_file(path: Path) -> Optional[np.ndarray]:
    """Gera embedding a partir de um arquivo de imagem no disco."""
    try:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            logger.warning("Não foi possível ler imagem: {}", path)
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return get_face_embedding(img_rgb)
    except Exception as e:
        logger.warning("Erro ao ler arquivo para embedding: {}", e)
        return None


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distância coseno entre dois embeddings normalizados. Range: [0, 2]."""
    return float(1.0 - np.dot(a, b))
