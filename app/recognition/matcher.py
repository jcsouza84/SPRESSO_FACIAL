"""
Comparação de embeddings para identificação de pessoas.

Fluxo:
  1. Ao cadastrar/atualizar foto → gera e armazena embedding no banco
  2. Em cada detecção → compara embedding do rosto ao vivo com banco
  3. Retorna a pessoa identificada + confiança, ou None se desconhecida
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.recognition.embeddings import get_face_embedding, cosine_distance
from app.storage.db import get_session
from app.storage.models import Person, PersonPhoto, PersonCategory
from app.config import settings
from app.logger import logger


@dataclass
class RecognitionResult:
    matched: bool
    person_id: Optional[int]
    person_name: Optional[str]
    category: Optional[str]
    confidence: float          # 0–100%
    distance: float            # distância coseno (menor = mais parecido)

    def to_dict(self) -> dict:
        return {
            "matched":     self.matched,
            "person_id":   self.person_id,
            "person_name": self.person_name,
            "category":    self.category,
            "confidence":  round(self.confidence, 2),
            "distance":    round(self.distance, 4),
        }


UNKNOWN = RecognitionResult(
    matched=False,
    person_id=None,
    person_name=None,
    category=None,
    confidence=0.0,
    distance=1.0,
)


class FaceMatcher:
    def __init__(self) -> None:
        # Cache em memória: person_id → list de embeddings numpy
        self._cache: dict[int, list[tuple[int, np.ndarray]]] = {}
        # person_id → Person (metadados)
        self._persons: dict[int, Person] = {}

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    async def load_all(self) -> None:
        """Carrega todos os embeddings ativos do banco para memória."""
        async with get_session() as session:
            stmt = (
                select(Person)
                .options(selectinload(Person.photos))
                .where(Person.active == True)
            )
            result = await session.execute(stmt)
            persons = result.scalars().all()

        self._cache.clear()
        self._persons.clear()

        loaded = 0
        for person in persons:
            embeddings = []
            for photo in person.photos:
                if photo.embedding:
                    emb = np.frombuffer(photo.embedding, dtype=np.float32)
                    embeddings.append((photo.id, emb))
                    loaded += 1
            if embeddings:
                self._cache[person.id] = embeddings
                self._persons[person.id] = person

        logger.info("Cache de reconhecimento: {} pessoa(s), {} embedding(s)", len(self._cache), loaded)

    async def refresh_person(self, person_id: int) -> None:
        """Atualiza o cache de uma pessoa específica."""
        async with get_session() as session:
            stmt = (
                select(Person)
                .options(selectinload(Person.photos))
                .where(Person.id == person_id)
            )
            result = await session.execute(stmt)
            person = result.scalar_one_or_none()

        if not person or not person.active:
            self._cache.pop(person_id, None)
            self._persons.pop(person_id, None)
            return

        embeddings = []
        for photo in person.photos:
            if photo.embedding:
                emb = np.frombuffer(photo.embedding, dtype=np.float32)
                embeddings.append((photo.id, emb))

        if embeddings:
            self._cache[person_id] = embeddings
            self._persons[person_id] = person
        else:
            self._cache.pop(person_id, None)
            self._persons.pop(person_id, None)

    # ------------------------------------------------------------------
    # Reconhecimento
    # ------------------------------------------------------------------

    def identify(self, face_roi_rgb: np.ndarray) -> RecognitionResult:
        """
        Identifica a pessoa a partir do crop de um rosto (RGB).
        Retorna RecognitionResult com a melhor correspondência ou UNKNOWN.
        """
        if not self._cache:
            return UNKNOWN

        embedding = get_face_embedding(face_roi_rgb)
        if embedding is None:
            return UNKNOWN

        best_distance = float("inf")
        best_person_id = None

        for person_id, photo_embeddings in self._cache.items():
            for _, stored_emb in photo_embeddings:
                dist = cosine_distance(embedding, stored_emb)
                if dist < best_distance:
                    best_distance = dist
                    best_person_id = person_id

        threshold = settings.recognition_threshold
        if best_distance <= threshold and best_person_id is not None:
            person = self._persons[best_person_id]
            confidence = max(0.0, (1.0 - best_distance / threshold)) * 100
            logger.debug(
                "Identificado: {} (dist={:.4f} threshold={:.4f} conf={:.1f}%)",
                person.name, best_distance, threshold, confidence,
            )
            return RecognitionResult(
                matched=True,
                person_id=person.id,
                person_name=person.name,
                category=person.category.value,
                confidence=confidence,
                distance=best_distance,
            )

        return RecognitionResult(
            matched=False,
            person_id=None,
            person_name=None,
            category=None,
            confidence=0.0,
            distance=best_distance,
        )

    @property
    def persons_in_cache(self) -> int:
        return len(self._cache)


face_matcher = FaceMatcher()
