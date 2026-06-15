"""
Alertas de blacklist, cooldown e notificação WhatsApp (Fase 9).
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import httpx
from sqlalchemy import desc, select, func

from app.config import settings
from app.logger import logger
from app.storage.db import get_session
from app.storage.models import AlertRecord, PersonCategory
from app.recognition.matcher import RecognitionResult


class AlertService:
    def __init__(self) -> None:
        self._last_alert_by_person: dict[int, datetime] = {}

    async def _get_cooldown_seconds(self) -> int:
        from app.services.settings_service import get_setting
        val = await get_setting("alert_cooldown_seconds")
        try:
            return int(val) if val else settings.alert_cooldown_seconds
        except (ValueError, TypeError):
            return settings.alert_cooldown_seconds

    def _cooldown_elapsed(self, person_id: int, cooldown: int) -> bool:
        last = self._last_alert_by_person.get(person_id)
        if not last:
            return True
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        return elapsed >= cooldown

    async def get_active_alerts(self) -> list[AlertRecord]:
        """Alertas de blacklist ainda não confirmados."""
        async with get_session() as session:
            stmt = (
                select(AlertRecord)
                .where(
                    AlertRecord.confirmed == False,
                    AlertRecord.category == PersonCategory.blacklist.value,
                )
                .order_by(desc(AlertRecord.created_at))
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_active_alert_for_person(self, person_id: int) -> AlertRecord | None:
        async with get_session() as session:
            stmt = (
                select(AlertRecord)
                .where(
                    AlertRecord.person_id == person_id,
                    AlertRecord.confirmed == False,
                    AlertRecord.category == PersonCategory.blacklist.value,
                )
                .order_by(desc(AlertRecord.created_at))
                .limit(1)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def process_recognition(
        self,
        rec: RecognitionResult,
        event_id: int | None,
        face_id: int | None,
        crop_path: str | None,
    ) -> AlertRecord | None:
        """
        Processa match de blacklist: registra presença, cria alerta se cooldown permitir.
        Retorna alerta ativo (novo ou existente) ou None.
        """
        if not rec.matched or rec.category != PersonCategory.blacklist.value:
            return None

        from app.services.presence_service import presence_service

        await presence_service.log_presence(
            person_id=rec.person_id,
            person_name=rec.person_name,
            category=rec.category,
            event_id=event_id,
        )

        existing = await self.get_active_alert_for_person(rec.person_id)
        if existing:
            return existing

        cooldown = await self._get_cooldown_seconds()
        if not self._cooldown_elapsed(rec.person_id, cooldown):
            logger.debug(
                "Cooldown ativo para {} — alerta não criado",
                rec.person_name,
            )
            return None

        now = datetime.now(timezone.utc)
        message = f"⚠️ ALERTA BLACKLIST: {rec.person_name} identificado(a)"

        async with get_session() as session:
            alert = AlertRecord(
                person_id=rec.person_id,
                person_name=rec.person_name,
                category=rec.category,
                event_id=event_id,
                face_id=face_id,
                crop_path=crop_path,
                confidence=rec.confidence,
                distance=rec.distance,
                message=message,
                confirmed=False,
                whatsapp_sent=False,
                created_at=now,
            )
            session.add(alert)
            await session.commit()
            await session.refresh(alert)

        self._last_alert_by_person[rec.person_id] = now
        logger.warning("Alerta blacklist criado: {} (#{})", rec.person_name, alert.id)

        await self._send_whatsapp(alert)
        return alert

    async def confirm_alert(self, alert_id: int) -> AlertRecord:
        async with get_session() as session:
            stmt = select(AlertRecord).where(AlertRecord.id == alert_id)
            result = await session.execute(stmt)
            alert = result.scalar_one_or_none()
            if not alert:
                raise ValueError(f"Alerta {alert_id} não encontrado")
            if alert.confirmed:
                return alert

            alert.confirmed = True
            alert.confirmed_at = datetime.now(timezone.utc)
            await session.commit()
            await session.refresh(alert)
            logger.info("Alerta #{} confirmado — {}", alert_id, alert.person_name)
            return alert

    async def list_alerts(
        self,
        limit: int = 50,
        offset: int = 0,
        only_unconfirmed: bool = False,
    ) -> tuple[list[AlertRecord], int]:
        async with get_session() as session:
            stmt = select(AlertRecord).order_by(desc(AlertRecord.created_at))
            count_stmt = select(func.count()).select_from(AlertRecord)
            if only_unconfirmed:
                stmt = stmt.where(AlertRecord.confirmed == False)
                count_stmt = count_stmt.where(AlertRecord.confirmed == False)
            total = (await session.execute(count_stmt)).scalar_one()
            stmt = stmt.offset(offset).limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all()), total

    async def presence_stats_today(self) -> dict:
        from app.storage.models import PresenceRecord

        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        async with get_session() as session:
            total = (await session.execute(
                select(func.count()).select_from(PresenceRecord)
                .where(PresenceRecord.created_at >= today_start)
            )).scalar_one()
            blacklist = (await session.execute(
                select(func.count()).select_from(PresenceRecord)
                .where(
                    PresenceRecord.created_at >= today_start,
                    PresenceRecord.category == PersonCategory.blacklist.value,
                )
            )).scalar_one()
            vip = (await session.execute(
                select(func.count()).select_from(PresenceRecord)
                .where(
                    PresenceRecord.created_at >= today_start,
                    PresenceRecord.category == PersonCategory.vip.value,
                )
            )).scalar_one()
        return {
            "date": today_start.date().isoformat(),
            "total": total,
            "blacklist": blacklist,
            "vip": vip,
        }

    async def _send_whatsapp(self, alert: AlertRecord) -> bool:
        # Lê configurações do banco (Sprint D) com fallback para .env
        from app.services.settings_service import get_setting
        wa_enabled = await get_setting("whatsapp_enabled") or ("true" if settings.whatsapp_enabled else "false")
        if wa_enabled != "true":
            return False

        api_url  = await get_setting("whatsapp_api_url")  or settings.whatsapp_api_url
        api_key  = await get_setting("whatsapp_api_key")  or settings.whatsapp_api_key
        instance = await get_setting("whatsapp_instance") or settings.whatsapp_instance
        # Suporte a múltiplos números (separados por vírgula)
        numbers_raw = await get_setting("whatsapp_notify_numbers") or settings.whatsapp_notify_number
        numbers = [n.strip() for n in (numbers_raw or "").split(",") if n.strip()]

        if not api_url or not numbers:
            logger.warning("WhatsApp habilitado mas URL/números não configurados")
            return False

        tz_name = await get_setting("app_timezone") or settings.app_timezone
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("America/Maceio")
        local_dt = alert.created_at.replace(tzinfo=timezone.utc).astimezone(tz)

        base    = api_url.rstrip("/")
        headers = {"apikey": api_key} if api_key else {}
        text    = (
            f"{alert.message}\n"
            f"Confiança: {alert.confidence:.0f}%\n"
            f"Hora: {local_dt.strftime('%d/%m %H:%M:%S')}"
        )

        sent_any = False
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                for number in numbers:
                    clean = number.replace("+", "").replace(" ", "").replace("-", "")
                    try:
                        if alert.crop_path and Path(alert.crop_path).exists():
                            media_b64 = base64.b64encode(Path(alert.crop_path).read_bytes()).decode()
                            resp = await client.post(
                                f"{base}/message/sendMedia/{instance}",
                                json={"number": clean, "mediatype": "image", "media": media_b64, "caption": text},
                                headers=headers,
                            )
                        else:
                            resp = await client.post(
                                f"{base}/message/sendText/{instance}",
                                json={"number": clean, "text": text},
                                headers=headers,
                            )
                        resp.raise_for_status()
                        sent_any = True
                        logger.info("WhatsApp enviado → {} (alerta #{})", clean, alert.id)
                    except Exception as e:
                        logger.warning("Falha ao enviar para {}: {}", clean, e)

            if sent_any:
                async with get_session() as session:
                    stmt = select(AlertRecord).where(AlertRecord.id == alert.id)
                    result = await session.execute(stmt)
                    db_alert = result.scalar_one()
                    db_alert.whatsapp_sent = True
                    await session.commit()

            return sent_any
        except Exception as exc:
            logger.error("Falha ao enviar WhatsApp: {}", exc)
            return False


alert_service = AlertService()
