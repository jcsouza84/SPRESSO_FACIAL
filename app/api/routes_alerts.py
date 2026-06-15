from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from app.services.alert_service import alert_service
from app.logger import logger

router = APIRouter(prefix="/alerts", tags=["alerts"])


class AlertListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[dict]


@router.get("/active")
async def get_active_alerts():
    """Alertas de blacklist pendentes de confirmação."""
    alerts = await alert_service.get_active_alerts()
    return {"count": len(alerts), "items": [a.to_dict() for a in alerts]}


@router.get("", response_model=AlertListResponse)
async def list_alerts(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    only_unconfirmed: bool = Query(default=False),
) -> AlertListResponse:
    items, total = await alert_service.list_alerts(
        limit=limit, offset=offset, only_unconfirmed=only_unconfirmed,
    )
    return AlertListResponse(
        total=total, limit=limit, offset=offset,
        items=[a.to_dict() for a in items],
    )


@router.post("/{alert_id}/confirm")
async def confirm_alert(alert_id: int):
    """Operador confirma/dispensa um alerta de blacklist."""
    try:
        alert = await alert_service.confirm_alert(alert_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"message": "Alerta confirmado", "alert": alert.to_dict()}


@router.get("/presence/today")
async def presence_today():
    """Contagem de identificações no dia."""
    return await alert_service.presence_stats_today()


@router.get(
    "/{alert_id}/crop",
    responses={200: {"content": {"image/jpeg": {}}}},
)
async def get_alert_crop(alert_id: int):
    """Crop do rosto associado ao alerta."""
    from sqlalchemy import select
    from app.storage.db import get_session
    from app.storage.models import AlertRecord

    async with get_session() as session:
        result = await session.execute(
            select(AlertRecord).where(AlertRecord.id == alert_id)
        )
        alert = result.scalar_one_or_none()

    if not alert or not alert.crop_path:
        raise HTTPException(status_code=404, detail="Crop não disponível")
    p = Path(alert.crop_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    return FileResponse(str(p), media_type="image/jpeg")
