from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import Optional

from sqlalchemy import select
from app.storage.db import get_session
from app.storage.models import DetectedFaceRecord, Person, PersonCategory
from app.services.event_service import list_events, get_event_by_id, count_events
from app.services.person_service import (
    create_person, get_person, add_photo_from_crop,
)
from app.recognition.matcher import face_matcher
from app.logger import logger

router = APIRouter(prefix="/events", tags=["events"])


class FaceBox(BaseModel):
    id: int
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int
    confidence: float
    crop_path: str | None = None
    has_embedding: bool = False


class EventResponse(BaseModel):
    id: int
    timestamp: str
    face_count: int
    inference_ms: float
    frame_width: int
    frame_height: int
    snapshot_path: str | None
    faces: list[FaceBox]


class EventListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[EventResponse]


class AssignFaceBody(BaseModel):
    # Atribui a pessoa existente OU cria nova
    person_id: Optional[int] = None
    name: Optional[str] = None
    category: Optional[str] = "blacklist"
    phone: Optional[str] = None
    observation: Optional[str] = None


@router.get("", response_model=EventListResponse)
async def get_events(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    only_with_faces: bool = Query(default=False),
) -> EventListResponse:
    """Lista eventos de detecção paginados, do mais recente ao mais antigo."""
    events = await list_events(limit=limit, offset=offset, only_with_faces=only_with_faces)
    total = await count_events(only_with_faces=only_with_faces)
    return EventListResponse(
        total=total, limit=limit, offset=offset,
        items=[EventResponse(**e.to_dict()) for e in events],
    )


@router.get("/summary/stats")
async def get_stats():
    """Resumo geral — total de eventos e eventos com rostos."""
    total = await count_events()
    with_faces = await count_events(only_with_faces=True)
    return {
        "total_events":      total,
        "events_with_faces": with_faces,
        "events_empty":      total - with_faces,
    }


@router.get("/{event_id}", response_model=EventResponse)
async def get_event(event_id: int) -> EventResponse:
    """Retorna um evento específico com todos os rostos detectados."""
    event = await get_event_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail=f"Evento {event_id} não encontrado")
    return EventResponse(**event.to_dict())


@router.get(
    "/{event_id}/faces/{face_id}/crop",
    responses={200: {"content": {"image/jpeg": {}}}},
)
async def get_face_crop(event_id: int, face_id: int):
    """Retorna a imagem do crop de um rosto específico de um evento."""
    async with get_session() as session:
        result = await session.execute(
            select(DetectedFaceRecord).where(
                DetectedFaceRecord.id == face_id,
                DetectedFaceRecord.event_id == event_id,
            )
        )
        face = result.scalar_one_or_none()

    if not face:
        raise HTTPException(status_code=404, detail="Rosto não encontrado")
    if not face.crop_path:
        raise HTTPException(status_code=404, detail="Crop não disponível para este rosto")

    p = Path(face.crop_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Arquivo de crop não encontrado no disco")

    return FileResponse(str(p), media_type="image/jpeg")


@router.post("/{event_id}/faces/{face_id}/assign")
async def assign_face_to_person(event_id: int, face_id: int, body: AssignFaceBody):
    """
    Atribui o crop de um rosto detectado como foto de referência de uma pessoa.
    Pode criar uma nova pessoa ou atribuir a uma existente.

    O embedding já calculado no pipeline ao vivo é reutilizado —
    garantindo que referência e detecção usam exatamente o mesmo processo.
    """
    # Busca o rosto
    async with get_session() as session:
        result = await session.execute(
            select(DetectedFaceRecord).where(
                DetectedFaceRecord.id == face_id,
                DetectedFaceRecord.event_id == event_id,
            )
        )
        face = result.scalar_one_or_none()

    if not face:
        raise HTTPException(status_code=404, detail="Rosto não encontrado")
    if not face.crop_path or not face.embedding:
        raise HTTPException(
            status_code=400,
            detail="Este rosto não possui crop/embedding disponível. "
                   "Apenas rostos detectados após a atualização do sistema podem ser atribuídos.",
        )

    crop_path = Path(face.crop_path)
    if not crop_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo de crop não encontrado no disco")

    # Resolve a pessoa
    if body.person_id:
        person = await get_person(body.person_id)
        if not person:
            raise HTTPException(status_code=404, detail=f"Pessoa {body.person_id} não encontrada")
    elif body.name:
        cat = PersonCategory(body.category) if body.category else PersonCategory.blacklist
        person = await create_person(
            name=body.name,
            category=cat,
            phone=body.phone,
            observation=body.observation,
        )
        logger.info("Nova pessoa criada via atribuição de crop: {} ({})", person.name, person.category)
    else:
        raise HTTPException(
            status_code=422,
            detail="Informe person_id (pessoa existente) ou name (criar nova pessoa).",
        )

    # Adiciona o crop como foto de referência com embedding já pronto
    try:
        photo = await add_photo_from_crop(
            person_id=person.id,
            crop_source_path=crop_path,
            embedding_bytes=bytes(face.embedding),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Atualiza o cache de reconhecimento
    await face_matcher.refresh_person(person.id)

    return {
        "message":       "Rosto atribuído com sucesso",
        "person_id":     person.id,
        "person_name":   person.name,
        "category":      person.category.value,
        "photo_id":      photo.id,
        "persons_in_cache": face_matcher.persons_in_cache,
    }
