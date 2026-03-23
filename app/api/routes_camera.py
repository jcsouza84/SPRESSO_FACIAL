from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

from app.camera.service import camera_service

router = APIRouter(prefix="/camera", tags=["camera"])


class CameraStatusResponse(BaseModel):
    ready: bool
    timestamp: str


@router.get("/status", response_model=CameraStatusResponse)
async def camera_status() -> CameraStatusResponse:
    return CameraStatusResponse(
        ready=camera_service.is_ready,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "/snapshot",
    responses={200: {"content": {"image/jpeg": {}}}},
    response_class=Response,
)
async def get_snapshot() -> Response:
    """Captura e retorna um frame da câmera como imagem JPEG."""
    if not camera_service.is_ready:
        raise HTTPException(status_code=503, detail="Câmera não disponível")

    try:
        frame = camera_service.snapshot(save=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao capturar frame: {exc}")

    return Response(
        content=frame.data,
        media_type="image/jpeg",
        headers={
            "X-Frame-Width": str(frame.width),
            "X-Frame-Height": str(frame.height),
            "X-Captured-At": datetime.now(timezone.utc).isoformat(),
        },
    )


@router.get(
    "/snapshot/last",
    responses={200: {"content": {"image/jpeg": {}}}},
    response_class=Response,
)
async def get_last_snapshot() -> Response:
    """Retorna o último frame capturado sem acionar a câmera novamente."""
    frame = camera_service.last_frame()
    if frame is None:
        raise HTTPException(
            status_code=404,
            detail="Nenhum snapshot capturado ainda. Use GET /camera/snapshot primeiro.",
        )

    return Response(
        content=frame.data,
        media_type="image/jpeg",
        headers={
            "X-Frame-Width": str(frame.width),
            "X-Frame-Height": str(frame.height),
        },
    )
