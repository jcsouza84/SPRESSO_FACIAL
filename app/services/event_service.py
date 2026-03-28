"""
Serviço responsável por persistir eventos de detecção no banco de dados.
Recebe um DetectionResult + path do snapshot e grava no SQLite.
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import select, desc, func
from sqlalchemy.orm import selectinload

from app.detection.face_detector import DetectionResult
from app.storage.db import get_session
from app.storage.models import DetectionEvent, DetectedFaceRecord
from app.logger import logger


async def save_detection_event(
    result: DetectionResult,
    snapshot_path: Optional[Path] = None,
    face_crops: Optional[list[dict]] = None,
) -> DetectionEvent:
    """
    Persiste um evento de detecção e seus rostos no banco.

    face_crops: lista paralela a result.faces com dicts opcionais:
        {"crop_path": str, "embedding": bytes}
    """
    async with get_session() as session:
        event = DetectionEvent(
            timestamp=datetime.now(timezone.utc),
            face_count=result.count,
            inference_ms=result.inference_ms,
            frame_width=result.frame_width,
            frame_height=result.frame_height,
            snapshot_path=str(snapshot_path) if snapshot_path else None,
        )
        session.add(event)
        await session.flush()

        for i, face in enumerate(result.faces):
            crop_data = (face_crops[i] if face_crops and i < len(face_crops) else None) or {}
            session.add(DetectedFaceRecord(
                event_id=event.id,
                x1=face.x1, y1=face.y1,
                x2=face.x2, y2=face.y2,
                confidence=face.confidence,
                crop_path=crop_data.get("crop_path"),
                embedding=crop_data.get("embedding"),
            ))

        await session.commit()
        await session.refresh(event)

        logger.debug(
            "Evento #{} salvo — {} rosto(s) em {}ms",
            event.id, event.face_count, event.inference_ms,
        )
        return event


async def list_events(
    limit: int = 50,
    offset: int = 0,
    only_with_faces: bool = False,
) -> list[DetectionEvent]:
    """Retorna eventos paginados, do mais recente ao mais antigo."""
    async with get_session() as session:
        stmt = (
            select(DetectionEvent)
            .options(selectinload(DetectionEvent.faces))
            .order_by(desc(DetectionEvent.timestamp))
        )
        if only_with_faces:
            stmt = stmt.where(DetectionEvent.face_count > 0)
        stmt = stmt.offset(offset).limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())


async def get_event_by_id(event_id: int) -> Optional[DetectionEvent]:
    """Retorna um evento pelo ID, incluindo os rostos."""
    async with get_session() as session:
        stmt = (
            select(DetectionEvent)
            .options(selectinload(DetectionEvent.faces))
            .where(DetectionEvent.id == event_id)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


async def count_events(only_with_faces: bool = False) -> int:
    """Retorna o total de eventos registrados."""
    async with get_session() as session:
        stmt = select(func.count()).select_from(DetectionEvent)
        if only_with_faces:
            stmt = stmt.where(DetectionEvent.face_count > 0)
        result = await session.execute(stmt)
        return result.scalar_one()
