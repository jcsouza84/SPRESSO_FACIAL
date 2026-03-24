from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.event_service import (
    list_events,
    get_event_by_id,
    count_events,
)

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
        total=total,
        limit=limit,
        offset=offset,
        items=[EventResponse(**e.to_dict()) for e in events],
    )


@router.get("/{event_id}", response_model=EventResponse)
async def get_event(event_id: int) -> EventResponse:
    """Retorna um evento específico com todos os rostos detectados."""
    event = await get_event_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail=f"Evento {event_id} não encontrado")
    return EventResponse(**event.to_dict())


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
