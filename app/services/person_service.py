"""
CRUD de pessoas cadastradas e suas fotos de referência.
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.storage.db import get_session
from app.storage.models import Person, PersonCategory, PersonPhoto
from app.config import settings
from app.logger import logger

MAX_PHOTOS_PER_PERSON = 5
PHOTOS_DIR = settings.data_dir / "persons"


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Pessoas
# ---------------------------------------------------------------------------

async def create_person(
    name: str,
    category: PersonCategory,
    phone: Optional[str] = None,
    observation: Optional[str] = None,
) -> Person:
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

    async with get_session() as session:
        now = _now()
        person = Person(
            name=name,
            phone=phone,
            category=category,
            observation=observation,
            active=True,
            created_at=now,
            updated_at=now,
        )
        session.add(person)
        await session.commit()
        new_id = person.id

    # Busca novamente com eager load para evitar DetachedInstanceError
    created = await get_person(new_id)
    logger.info("Pessoa cadastrada: id={} nome='{}' categoria={}", created.id, created.name, created.category.value)
    return created


async def get_person(person_id: int) -> Optional[Person]:
    async with get_session() as session:
        stmt = (
            select(Person)
            .options(selectinload(Person.photos))
            .where(Person.id == person_id)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


async def list_persons(
    category: Optional[PersonCategory] = None,
    active_only: bool = True,
    limit: int = 100,
    offset: int = 0,
) -> list[Person]:
    async with get_session() as session:
        stmt = select(Person).options(selectinload(Person.photos))
        if active_only:
            stmt = stmt.where(Person.active == True)
        if category:
            stmt = stmt.where(Person.category == category)
        stmt = stmt.order_by(Person.created_at.desc()).offset(offset).limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())


async def update_person(
    person_id: int,
    name: Optional[str] = None,
    phone: Optional[str] = None,
    category: Optional[PersonCategory] = None,
    observation: Optional[str] = None,
    active: Optional[bool] = None,
) -> Optional[Person]:
    async with get_session() as session:
        stmt = (
            select(Person)
            .options(selectinload(Person.photos))
            .where(Person.id == person_id)
        )
        result = await session.execute(stmt)
        person = result.scalar_one_or_none()
        if not person:
            return None

        if name is not None:
            person.name = name
        if phone is not None:
            person.phone = phone
        if category is not None:
            person.category = category
        if observation is not None:
            person.observation = observation
        if active is not None:
            person.active = active
        person.updated_at = _now()

        await session.commit()
        await session.refresh(person)
        logger.info("Pessoa atualizada: id={}", person_id)
        return person


async def delete_person(person_id: int) -> bool:
    """Remove a pessoa e todas as suas fotos do banco e do disco."""
    async with get_session() as session:
        stmt = (
            select(Person)
            .options(selectinload(Person.photos))
            .where(Person.id == person_id)
        )
        result = await session.execute(stmt)
        person = result.scalar_one_or_none()
        if not person:
            return False

        for photo in person.photos:
            _delete_photo_file(Path(photo.path))

        await session.delete(person)
        await session.commit()
        logger.info("Pessoa removida: id={}", person_id)
        return True


# ---------------------------------------------------------------------------
# Fotos
# ---------------------------------------------------------------------------

async def add_photo(person_id: int, image_bytes: bytes, extension: str = "jpg") -> PersonPhoto:
    """Salva uma foto de referência para a pessoa (máx. 5)."""
    async with get_session() as session:
        stmt = (
            select(Person)
            .options(selectinload(Person.photos))
            .where(Person.id == person_id)
        )
        result = await session.execute(stmt)
        person = result.scalar_one_or_none()

        if not person:
            raise ValueError(f"Pessoa {person_id} não encontrada")
        if len(person.photos) >= MAX_PHOTOS_PER_PERSON:
            raise ValueError(
                f"Limite de {MAX_PHOTOS_PER_PERSON} fotos atingido para a pessoa {person_id}"
            )

        person_dir = PHOTOS_DIR / str(person_id)
        person_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        path = person_dir / f"photo_{ts}.{extension}"
        path.write_bytes(image_bytes)

        photo = PersonPhoto(
            person_id=person_id,
            path=str(path),
            created_at=_now(),
        )
        session.add(photo)
        await session.commit()
        photo_id = photo.id

    # Re-busca fora da sessão para evitar DetachedInstanceError
    async with get_session() as session2:
        stmt2 = select(PersonPhoto).where(PersonPhoto.id == photo_id)
        result2 = await session2.execute(stmt2)
        loaded_photo = result2.scalar_one()

    logger.info("Foto adicionada: person_id={} photo_id={} path={}", person_id, loaded_photo.id, path)
    return loaded_photo


async def delete_photo(photo_id: int) -> bool:
    """Remove uma foto específica do banco e do disco."""
    async with get_session() as session:
        stmt = select(PersonPhoto).where(PersonPhoto.id == photo_id)
        result = await session.execute(stmt)
        photo = result.scalar_one_or_none()
        if not photo:
            return False

        _delete_photo_file(Path(photo.path))
        await session.delete(photo)
        await session.commit()
        logger.info("Foto removida: id={}", photo_id)
        return True


def _delete_photo_file(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception as e:
        logger.warning("Não foi possível remover arquivo {}: {}", path, e)
