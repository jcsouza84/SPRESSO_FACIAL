"""
Serviço de configurações do sistema.
Persiste settings no SQLite (tabela system_settings) para permitir edição
pela UI em runtime, sem restart do serviço.
"""
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, delete

from app.storage.db import get_session
from app.storage.models import SystemSetting, RtspCameraConfig
from app.logger import logger

_MAX_RTSP_CAMERAS = 2

# Chaves de configuração válidas e seus defaults
_DEFAULTS: dict[str, str] = {
    # Telegram Bot
    "telegram_enabled":          "false",
    "telegram_bot_token":        "",
    "telegram_chat_ids":         "",   # múltiplos chat_ids separados por vírgula
    # WhatsApp
    "whatsapp_enabled":          "false",
    "whatsapp_api_url":          "",
    "whatsapp_api_key":          "",
    "whatsapp_instance":         "default",
    "whatsapp_notify_numbers":   "",   # múltiplos, separados por vírgula
    # Worker de detecção
    "detection_auto":            "false",
    "detection_fps":             "1.0",
    # Reconhecimento
    "recognition_threshold":     "",   # vazio = usa config.py default
    # Alertas
    "alert_cooldown_seconds":    "",   # vazio = usa config.py default (300s)
    # Qualidade de detecção e filtros
    "detection_confidence":      "0.65",  # confiança mínima SCRFD/YOLO (0-1)
    "min_face_px_detect":        "80",    # tamanho mínimo para registrar evento
    "min_face_px_recognize":     "120",   # tamanho mínimo para tentar reconhecer
    "min_face_px_alert":         "150",   # tamanho mínimo para disparar alerta
    "alert_min_confidence":      "50.0",  # % mínima de confiança para alertar
    "require_frontal_face":      "true",  # rejeitar rostos laterais
    "max_face_yaw_degrees":      "40",    # ângulo máximo de rotação horizontal
    "require_person_overlap":    "false", # exigir detecção de pessoa (YOLOv5)
    "event_dedup_seconds":       "",   # vazio = usa config.py default (30s)
    # Localização
    "app_timezone":              "America/Maceio",
    # Câmera primária para detecção e monitor
    "active_camera_id":          "imx0",
}


async def get_setting(key: str) -> Optional[str]:
    async with get_session() as session:
        result = await session.execute(select(SystemSetting).where(SystemSetting.key == key))
        row = result.scalar_one_or_none()
        if row is None:
            return _DEFAULTS.get(key)
        return row.value


async def get_all_settings() -> dict[str, str | None]:
    async with get_session() as session:
        result = await session.execute(select(SystemSetting))
        rows = {r.key: r.value for r in result.scalars().all()}
    merged = dict(_DEFAULTS)
    merged.update(rows)
    return merged


async def set_settings(data: dict[str, str]) -> None:
    now = datetime.now(timezone.utc)
    async with get_session() as session:
        for key, value in data.items():
            result = await session.execute(select(SystemSetting).where(SystemSetting.key == key))
            row = result.scalar_one_or_none()
            if row:
                row.value = value
                row.updated_at = now
            else:
                session.add(SystemSetting(key=key, value=value, updated_at=now))
        await session.commit()
    logger.info("Settings atualizadas: {}", list(data.keys()))


# ---------------------------------------------------------------------------
# RTSP cameras
# ---------------------------------------------------------------------------

async def list_rtsp_cameras(enabled_only: bool = False) -> list[RtspCameraConfig]:
    async with get_session() as session:
        stmt = select(RtspCameraConfig).order_by(RtspCameraConfig.order_index)
        if enabled_only:
            stmt = stmt.where(RtspCameraConfig.enabled == True)
        result = await session.execute(stmt)
        return list(result.scalars().all())


async def get_rtsp_camera(cam_id: int) -> Optional[RtspCameraConfig]:
    async with get_session() as session:
        result = await session.execute(
            select(RtspCameraConfig).where(RtspCameraConfig.id == cam_id)
        )
        return result.scalar_one_or_none()


async def get_rtsp_camera_by_camera_id(camera_id: str) -> Optional[RtspCameraConfig]:
    async with get_session() as session:
        result = await session.execute(
            select(RtspCameraConfig).where(RtspCameraConfig.camera_id == camera_id)
        )
        return result.scalar_one_or_none()


async def create_rtsp_camera(camera_id: str, label: str, url: str) -> RtspCameraConfig:
    cameras = await list_rtsp_cameras()
    if len(cameras) >= _MAX_RTSP_CAMERAS:
        raise ValueError(f"Máximo de {_MAX_RTSP_CAMERAS} câmeras RTSP atingido")

    # Verifica duplicata de camera_id
    existing = await get_rtsp_camera_by_camera_id(camera_id)
    if existing:
        raise ValueError(f"camera_id '{camera_id}' já está em uso")

    now = datetime.now(timezone.utc)
    cam = RtspCameraConfig(
        camera_id=camera_id,
        label=label,
        url=url,
        enabled=True,
        order_index=len(cameras),
        updated_at=now,
    )
    async with get_session() as session:
        session.add(cam)
        await session.commit()
        await session.refresh(cam)
    logger.info("Câmera RTSP criada: {} ({})", camera_id, label)
    return cam


async def update_rtsp_camera(
    cam_id: int, label: str | None = None, url: str | None = None,
    enabled: bool | None = None, camera_id: str | None = None,
) -> Optional[RtspCameraConfig]:
    async with get_session() as session:
        result = await session.execute(
            select(RtspCameraConfig).where(RtspCameraConfig.id == cam_id)
        )
        cam = result.scalar_one_or_none()
        if not cam:
            return None
        if label is not None:
            cam.label = label
        if url is not None:
            cam.url = url
        if enabled is not None:
            cam.enabled = enabled
        if camera_id is not None:
            cam.camera_id = camera_id
        cam.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(cam)
    logger.info("Câmera RTSP #{} atualizada", cam_id)
    return cam


async def delete_rtsp_camera(cam_id: int) -> bool:
    async with get_session() as session:
        result = await session.execute(
            select(RtspCameraConfig).where(RtspCameraConfig.id == cam_id)
        )
        cam = result.scalar_one_or_none()
        if not cam:
            return False
        await session.delete(cam)
        await session.commit()
    logger.info("Câmera RTSP #{} removida", cam_id)
    return True


async def toggle_rtsp_camera(cam_id: int) -> Optional[RtspCameraConfig]:
    async with get_session() as session:
        result = await session.execute(
            select(RtspCameraConfig).where(RtspCameraConfig.id == cam_id)
        )
        cam = result.scalar_one_or_none()
        if not cam:
            return None
        cam.enabled = not cam.enabled
        cam.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(cam)
    return cam
