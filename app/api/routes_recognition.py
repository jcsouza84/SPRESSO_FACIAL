"""
Endpoints de reconhecimento facial:
  GET  /recognition/threshold        — retorna threshold atual
  POST /recognition/threshold        — atualiza threshold em memória
  POST /recognition/test             — testa foto avulsa sem gravar no banco
  POST /recognition/regen-embeddings — regenera embeddings NULL sem subir novo processo
"""
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

import numpy as np

from app.recognition.matcher import face_matcher
from app.recognition.embeddings import get_embedding_from_image_bytes, get_embedding_from_file, cosine_distance
from app.storage.db import get_session
from app.storage.models import PersonPhoto
from app.config import settings
from app.logger import logger
from sqlalchemy import select

router = APIRouter(prefix="/recognition", tags=["recognition"])

# Threshold em memória (pode ser sobrescrito via API sem reiniciar)
_runtime_threshold: float | None = None


def current_threshold() -> float:
    return _runtime_threshold if _runtime_threshold is not None else settings.recognition_threshold


class ThresholdBody(BaseModel):
    threshold: float = Field(gt=0.0, lt=1.0)


@router.get("/threshold")
async def get_threshold():
    return {"threshold": current_threshold()}


@router.post("/threshold")
async def set_threshold(body: ThresholdBody):
    global _runtime_threshold
    _runtime_threshold = body.threshold
    # Garante que o matcher usa o novo valor na próxima chamada
    settings.__dict__["recognition_threshold"] = body.threshold
    logger.info("Threshold de reconhecimento atualizado: {}", body.threshold)
    return {"threshold": body.threshold, "message": "Threshold atualizado com sucesso"}


@router.post("/test")
async def test_photo(file: UploadFile = File(...)):
    """
    Testa o reconhecimento com uma foto avulsa sem criar eventos no banco.
    Retorna a melhor correspondência contra o cache atual.
    """
    if face_matcher.persons_in_cache == 0:
        raise HTTPException(
            status_code=400,
            detail="Nenhuma pessoa com embedding no cache. Cadastre pelo menos uma foto primeiro.",
        )

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="Arquivo deve ser uma imagem.")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="Arquivo muito grande (máximo 10MB).")

    embedding = get_embedding_from_image_bytes(image_bytes)
    if embedding is None:
        raise HTTPException(
            status_code=422,
            detail="Não foi possível extrair embedding da imagem. Verifique se o rosto está visível e bem iluminado.",
        )

    threshold = current_threshold()
    best_distance = float("inf")
    best_person_id = None

    for person_id, photo_embeddings in face_matcher._cache.items():
        for _, stored_emb in photo_embeddings:
            dist = cosine_distance(embedding, stored_emb)
            if dist < best_distance:
                best_distance = dist
                best_person_id = person_id

    if best_person_id is not None and best_distance <= threshold:
        person = face_matcher._persons[best_person_id]
        confidence = max(0.0, (1.0 - best_distance / threshold)) * 100
        return {
            "matched":      True,
            "person_id":    person.id,
            "person_name":  person.name,
            "category":     person.category.value,
            "confidence":   round(confidence, 2),
            "distance":     round(best_distance, 4),
            "threshold":    threshold,
        }

    return {
        "matched":     False,
        "person_id":   None,
        "person_name": None,
        "category":    None,
        "confidence":  0.0,
        "distance":    round(best_distance, 4),
        "threshold":   threshold,
    }


@router.post("/regen-embeddings")
async def regen_embeddings():
    """
    Regenera embeddings de todas as fotos com embedding NULL.
    Usa o modelo já carregado em memória — sem subir novo processo.
    """
    async with get_session() as session:
        result = await session.execute(
            select(PersonPhoto).where(PersonPhoto.embedding == None)  # noqa: E711
        )
        photos = result.scalars().all()

    if not photos:
        return {"message": "Nenhuma foto sem embedding. Tudo OK.", "updated": 0, "failed": 0}

    updated = 0
    failed = 0
    details = []

    for photo in photos:
        p = Path(photo.path)
        if not p.exists():
            details.append({"photo_id": photo.id, "status": "skip", "reason": "arquivo não encontrado"})
            failed += 1
            continue

        emb = get_embedding_from_file(p)
        if emb is None:
            details.append({"photo_id": photo.id, "status": "fail", "reason": "rosto não detectado"})
            failed += 1
            logger.warning("Embedding falhou para photo_id={}", photo.id)
            continue

        async with get_session() as session:
            result = await session.execute(
                select(PersonPhoto).where(PersonPhoto.id == photo.id)
            )
            db_photo = result.scalar_one_or_none()
            if db_photo:
                db_photo.embedding = emb.tobytes()
                await session.commit()

        details.append({"photo_id": photo.id, "status": "ok"})
        updated += 1
        logger.info("Embedding regenerado: photo_id={} person_id={}", photo.id, photo.person_id)

    # Recarrega cache com os novos embeddings
    await face_matcher.load_all()
    logger.info("Cache recarregado após regen: {} pessoa(s)", face_matcher.persons_in_cache)

    return {
        "message": f"{updated} embedding(s) gerado(s), {failed} falha(s).",
        "updated": updated,
        "failed": failed,
        "persons_in_cache": face_matcher.persons_in_cache,
        "details": details,
    }
