"""
Controle de deduplicação de eventos e log de presença (Fase 9).
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.config import settings
from app.logger import logger
from app.storage.db import get_session
from app.storage.models import PresenceRecord


class PresenceService:
    def __init__(self) -> None:
        self._last_by_person: dict[int, datetime] = {}
        self._last_empty: datetime | None = None

    def should_skip_event(
        self,
        matched_person_ids: list[int],
        face_count: int,
    ) -> bool:
        """Evita registrar eventos repetidos dentro da janela de dedup."""
        now = datetime.now(timezone.utc)
        window = settings.event_dedup_seconds

        if matched_person_ids:
            for pid in matched_person_ids:
                last = self._last_by_person.get(pid)
                if last and (now - last).total_seconds() < window:
                    logger.debug(
                        "Dedup: pessoa {} detectada há {:.0f}s — evento ignorado",
                        pid, (now - last).total_seconds(),
                    )
                    return True
            return False

        if face_count == 0:
            if self._last_empty and (now - self._last_empty).total_seconds() < window:
                logger.debug("Dedup: frame vazio recente — evento ignorado")
                return True

        return False

    def record_event(self, matched_person_ids: list[int], face_count: int) -> None:
        now = datetime.now(timezone.utc)
        for pid in matched_person_ids:
            self._last_by_person[pid] = now
        if face_count == 0:
            self._last_empty = now

    async def log_presence(
        self,
        person_id: int,
        person_name: str,
        category: str,
        event_id: int | None,
    ) -> PresenceRecord:
        async with get_session() as session:
            record = PresenceRecord(
                person_id=person_id,
                person_name=person_name,
                category=category,
                event_id=event_id,
                created_at=datetime.now(timezone.utc),
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            logger.info("Presença registrada: {} ({})", person_name, category)
            return record


presence_service = PresenceService()
