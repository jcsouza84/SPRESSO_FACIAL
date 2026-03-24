from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.camera.service import camera_service
from app.storage.models import PersonCategory
from app.recognition.matcher import face_matcher
from app.services.person_service import (
    create_person, get_person, list_persons,
    update_person, delete_person, add_photo, delete_photo,
)

router = APIRouter(prefix="/persons", tags=["persons"])

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PersonCreate(BaseModel):
    name: str
    category: PersonCategory
    phone: Optional[str] = None
    observation: Optional[str] = None


class PersonUpdate(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    category: Optional[PersonCategory] = None
    observation: Optional[str] = None
    active: Optional[bool] = None


class PhotoResponse(BaseModel):
    id: int
    person_id: int
    path: str
    created_at: str


class PersonResponse(BaseModel):
    id: int
    name: str
    phone: Optional[str]
    category: str
    observation: Optional[str]
    active: bool
    photo_count: int
    created_at: str
    updated_at: str
    photos: list[PhotoResponse] = []


# ---------------------------------------------------------------------------
# Endpoints — Pessoas
# ---------------------------------------------------------------------------

@router.post("", response_model=PersonResponse, status_code=201)
async def create(body: PersonCreate) -> PersonResponse:
    """Cadastra uma nova pessoa."""
    person = await create_person(
        name=body.name,
        category=body.category,
        phone=body.phone,
        observation=body.observation,
    )
    return PersonResponse(**person.to_dict())


@router.get("", response_model=list[PersonResponse])
async def list_all(
    category: Optional[PersonCategory] = Query(default=None),
    active_only: bool = Query(default=True),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[PersonResponse]:
    """Lista pessoas cadastradas com filtros opcionais."""
    persons = await list_persons(
        category=category, active_only=active_only, limit=limit, offset=offset
    )
    return [PersonResponse(**p.to_dict()) for p in persons]


@router.get("/{person_id}", response_model=PersonResponse)
async def get_one(person_id: int) -> PersonResponse:
    """Retorna uma pessoa pelo ID."""
    person = await get_person(person_id)
    if not person:
        raise HTTPException(status_code=404, detail=f"Pessoa {person_id} não encontrada")
    return PersonResponse(**person.to_dict())


@router.patch("/{person_id}", response_model=PersonResponse)
async def update(person_id: int, body: PersonUpdate) -> PersonResponse:
    """Atualiza dados de uma pessoa."""
    person = await update_person(
        person_id=person_id,
        name=body.name,
        phone=body.phone,
        category=body.category,
        observation=body.observation,
        active=body.active,
    )
    if not person:
        raise HTTPException(status_code=404, detail=f"Pessoa {person_id} não encontrada")
    return PersonResponse(**person.to_dict())


@router.delete("/{person_id}", status_code=204)
async def delete(person_id: int) -> None:
    """Remove uma pessoa e todas as suas fotos."""
    removed = await delete_person(person_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Pessoa {person_id} não encontrada")


# ---------------------------------------------------------------------------
# Endpoints — Fotos
# ---------------------------------------------------------------------------

@router.post("/{person_id}/photos/upload", response_model=PhotoResponse, status_code=201)
async def upload_photo(
    person_id: int,
    file: UploadFile = File(...),
) -> PhotoResponse:
    """Faz upload de uma foto de referência para a pessoa."""
    _check_person_exists(person_id)

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Tipo de arquivo inválido. Permitido: {', '.join(ALLOWED_IMAGE_TYPES)}"
        )

    ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename else "jpg"
    image_bytes = await file.read()

    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="Arquivo muito grande. Máximo: 5MB")

    try:
        photo = await add_photo(person_id, image_bytes, extension=ext)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    await face_matcher.refresh_person(person_id)
    return PhotoResponse(**photo.to_dict())


@router.post("/{person_id}/photos/capture", response_model=PhotoResponse, status_code=201)
async def capture_photo(person_id: int) -> PhotoResponse:
    """Captura um frame ao vivo da câmera e salva como foto de referência."""
    _check_person_exists(person_id)

    if not camera_service.is_ready:
        raise HTTPException(status_code=503, detail="Câmera não disponível")

    frame = camera_service.snapshot(save=False)

    import cv2
    _, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame.array, cv2.COLOR_RGB2BGR))
    image_bytes = encoded.tobytes()

    try:
        photo = await add_photo(person_id, image_bytes, extension="jpg")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    await face_matcher.refresh_person(person_id)
    return PhotoResponse(**photo.to_dict())


@router.get("/{person_id}/photos/{photo_id}/image")
async def get_photo_image(person_id: int, photo_id: int):
    """Retorna a imagem de uma foto de referência."""
    person = await get_person(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Pessoa não encontrada")

    photo = next((p for p in person.photos if p.id == photo_id), None)
    if not photo:
        raise HTTPException(status_code=404, detail="Foto não encontrada")

    from pathlib import Path
    path = Path(photo.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Arquivo de imagem não encontrado no disco")

    return FileResponse(str(path), media_type="image/jpeg")


@router.delete("/{person_id}/photos/{photo_id}", status_code=204)
async def remove_photo(person_id: int, photo_id: int) -> None:
    """Remove uma foto de referência específica."""
    removed = await delete_photo(photo_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Foto não encontrada")
    await face_matcher.refresh_person(person_id)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

async def _check_person_exists(person_id: int) -> None:
    person = await get_person(person_id)
    if not person:
        raise HTTPException(status_code=404, detail=f"Pessoa {person_id} não encontrada")
