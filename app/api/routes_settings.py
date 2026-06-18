"""
Endpoints de configuração do sistema — editáveis pela UI em runtime.
Inclui: settings gerais, câmeras RTSP (max 2), WhatsApp/Evolution API.
"""
import asyncio
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import settings_service
from app.camera.registry import camera_registry
from app.logger import logger

router = APIRouter(prefix="/settings", tags=["settings"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SettingsUpdate(BaseModel):
    whatsapp_enabled: Optional[bool] = None
    whatsapp_api_url: Optional[str] = None
    whatsapp_api_key: Optional[str] = None
    whatsapp_instance: Optional[str] = None
    whatsapp_notify_numbers: Optional[str] = None   # CSV: "+5511...,+5521..."
    detection_auto: Optional[bool] = None
    detection_fps: Optional[float] = None
    recognition_threshold: Optional[float] = None
    alert_cooldown_seconds: Optional[int] = None
    event_dedup_seconds: Optional[int] = None
    app_timezone: Optional[str] = None


class WhatsappTestBody(BaseModel):
    number: Optional[str] = None   # número avulso; usa lista salva se None


class RtspCameraCreate(BaseModel):
    camera_id: str
    label: str
    url: str


class RtspCameraUpdate(BaseModel):
    camera_id: Optional[str] = None
    label: Optional[str] = None
    url: Optional[str] = None
    enabled: Optional[bool] = None


# ---------------------------------------------------------------------------
# Settings gerais
# ---------------------------------------------------------------------------

@router.get("")
async def get_settings() -> dict:
    """Retorna todas as configurações editáveis (API key mascarada)."""
    from app.config import settings as cfg
    data = await settings_service.get_all_settings()
    # threshold: usa valor do banco se existir, senão usa config.py
    if not data.get("recognition_threshold"):
        data["recognition_threshold"] = str(cfg.recognition_threshold)
    # mascara a API key
    if data.get("whatsapp_api_key"):
        raw = data["whatsapp_api_key"]
        data["whatsapp_api_key"] = raw[:4] + "••••••••" + raw[-4:] if len(raw) > 8 else "••••••••"
        data["whatsapp_api_key_set"] = True
    else:
        data["whatsapp_api_key_set"] = False
    return data


@router.post("")
async def save_settings(body: SettingsUpdate) -> dict:
    """Salva configurações e aplica em runtime sem restart."""
    data = {}
    if body.whatsapp_enabled is not None:
        data["whatsapp_enabled"] = str(body.whatsapp_enabled).lower()
    if body.whatsapp_api_url is not None:
        data["whatsapp_api_url"] = body.whatsapp_api_url
    if body.whatsapp_api_key is not None and "••••" not in body.whatsapp_api_key:
        data["whatsapp_api_key"] = body.whatsapp_api_key
    if body.whatsapp_instance is not None:
        data["whatsapp_instance"] = body.whatsapp_instance
    if body.whatsapp_notify_numbers is not None:
        data["whatsapp_notify_numbers"] = body.whatsapp_notify_numbers
    if body.detection_auto is not None:
        data["detection_auto"] = str(body.detection_auto).lower()
        _apply_detection_auto(body.detection_auto)
    if body.detection_fps is not None:
        data["detection_fps"] = str(body.detection_fps)
    if body.recognition_threshold is not None:
        data["recognition_threshold"] = str(body.recognition_threshold)
        # aplica em runtime
        from app.config import settings as cfg
        cfg.recognition_threshold = body.recognition_threshold
        logger.info("Threshold atualizado para {}", body.recognition_threshold)
    if body.alert_cooldown_seconds is not None:
        data["alert_cooldown_seconds"] = str(body.alert_cooldown_seconds)
        logger.info("Cooldown de alerta atualizado para {}s", body.alert_cooldown_seconds)
    if body.event_dedup_seconds is not None:
        data["event_dedup_seconds"] = str(body.event_dedup_seconds)
        from app.config import settings as cfg
        cfg.event_dedup_seconds = body.event_dedup_seconds
        logger.info("Dedup de eventos atualizado para {}s", body.event_dedup_seconds)
    if body.app_timezone is not None:
        data["app_timezone"] = body.app_timezone
        from app.config import settings as cfg
        cfg.app_timezone = body.app_timezone
        logger.info("Timezone atualizado para {}", body.app_timezone)

    if data:
        await settings_service.set_settings(data)
    return {"ok": True, "updated": list(data.keys())}


def _apply_detection_auto(enabled: bool) -> None:
    """Ativa/desativa o detection worker em runtime."""
    import asyncio as _asyncio
    from app.config import settings
    if enabled and not settings.detection_auto:
        from app.workers import detection_worker
        _asyncio.create_task(detection_worker.run_forever())
        logger.info("Detection worker ativado pela UI")
    elif not enabled:
        logger.info("Detection worker será desativado no próximo restart (em runtime não é interrompível)")


# ---------------------------------------------------------------------------
# Câmeras RTSP
# ---------------------------------------------------------------------------

@router.get("/cameras")
async def list_rtsp_cameras() -> list[dict]:
    cameras = await settings_service.list_rtsp_cameras()
    result = []
    for cam in cameras:
        d = cam.to_dict()
        d["ready"] = camera_registry.is_registered(cam.camera_id) and camera_registry.get(cam.camera_id).is_ready
        result.append(d)
    return result


@router.post("/cameras")
async def add_rtsp_camera(body: RtspCameraCreate) -> dict:
    """Adiciona nova câmera RTSP (máximo 2)."""
    try:
        cam = await settings_service.create_rtsp_camera(
            camera_id=body.camera_id.strip(),
            label=body.label.strip(),
            url=body.url.strip(),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    status = await _register_rtsp_in_registry(cam.camera_id, cam.label, cam.url)
    d = cam.to_dict()
    d["registry_status"] = status
    return d


@router.put("/cameras/{cam_id}")
async def update_rtsp_camera(cam_id: int, body: RtspCameraUpdate) -> dict:
    """Atualiza câmera RTSP e re-registra no registry se URL/ID mudou."""
    cam = await settings_service.update_rtsp_camera(
        cam_id,
        label=body.label,
        url=body.url,
        enabled=body.enabled,
        camera_id=body.camera_id,
    )
    if not cam:
        raise HTTPException(status_code=404, detail="Câmera não encontrada")

    # Se URL ou camera_id mudou, re-registra
    status = "unchanged"
    if body.url is not None or body.camera_id is not None:
        _unregister_rtsp(cam.camera_id)
        if cam.enabled:
            status = await _register_rtsp_in_registry(cam.camera_id, cam.label, cam.url)

    if body.enabled is not None:
        if body.enabled:
            status = await _register_rtsp_in_registry(cam.camera_id, cam.label, cam.url)
        else:
            _unregister_rtsp(cam.camera_id)
            status = "disabled"

    d = cam.to_dict()
    d["registry_status"] = status
    return d


@router.delete("/cameras/{cam_id}")
async def delete_rtsp_camera(cam_id: int) -> dict:
    cam_record = await settings_service.get_rtsp_camera(cam_id)
    if cam_record:
        _unregister_rtsp(cam_record.camera_id)
    removed = await settings_service.delete_rtsp_camera(cam_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Câmera não encontrada")
    return {"ok": True}


@router.post("/cameras/{cam_id}/toggle")
async def toggle_rtsp_camera(cam_id: int) -> dict:
    cam = await settings_service.toggle_rtsp_camera(cam_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Câmera não encontrada")
    if cam.enabled:
        status = await _register_rtsp_in_registry(cam.camera_id, cam.label, cam.url)
    else:
        _unregister_rtsp(cam.camera_id)
        status = "disabled"
    d = cam.to_dict()
    d["registry_status"] = status
    return d


@router.post("/cameras/{cam_id}/test")
async def test_rtsp_camera(cam_id: int) -> dict:
    """Testa conectividade com a câmera RTSP (timeout 5s)."""
    cam = await settings_service.get_rtsp_camera(cam_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Câmera não encontrada")
    return await _test_rtsp_url(cam.url)


@router.post("/cameras/test-url")
async def test_rtsp_url_direct(body: dict) -> dict:
    """Testa uma URL RTSP antes de salvar."""
    url = body.get("url", "")
    if not url:
        raise HTTPException(status_code=400, detail="URL obrigatória")
    return await _test_rtsp_url(url)


# ---------------------------------------------------------------------------
# WhatsApp / Evolution API
# ---------------------------------------------------------------------------

@router.get("/whatsapp/status")
async def whatsapp_status() -> dict:
    """Verifica status da instância no Evolution API."""
    config = await _get_whatsapp_config()
    if not config["api_url"] or not config["api_key"]:
        return {"connected": False, "message": "Evolution API não configurada"}
    return await _check_evolution_status(config)


@router.post("/test/whatsapp")
async def test_whatsapp(body: WhatsappTestBody | None = None) -> dict:
    """Envia mensagem de teste. Usa número do body se informado, senão usa lista salva."""
    config = await _get_whatsapp_config()
    if not config["api_url"] or not config["api_key"]:
        raise HTTPException(status_code=400, detail="Evolution API não configurada (URL + Key obrigatórios)")
    test_number = (body.number if body else None) or ""
    if not test_number and not config["notify_numbers"]:
        raise HTTPException(status_code=400, detail="Informe um número de teste ou cadastre números na lista")
    return await _send_whatsapp_test(config, override_number=test_number or None)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _build_camera(url: str, camera_id: str, label: str):
    """Instancia a classe de câmera correta baseado no esquema da URL."""
    if url.lower().startswith("onvif://"):
        from app.camera.sources.onvif_cam import ONVIFCamera
        return ONVIFCamera(url=url, camera_id=camera_id, label=label)
    from app.camera.sources.rtsp import RTSPCamera
    return RTSPCamera(url=url, camera_id=camera_id, label=label)


async def _register_rtsp_in_registry(camera_id: str, label: str, url: str) -> str:
    """Registra câmera IP/ONVIF no registry."""
    try:
        _unregister_rtsp(camera_id)
        cam = _build_camera(url, camera_id, label)
        cam.open()
        camera_registry.register(cam)
        # ONVIF leva mais tempo para descobrir URI e abrir stream
        wait = 80 if url.lower().startswith("onvif://") else 30
        for _ in range(wait):
            await asyncio.sleep(0.1)
            if cam.is_ready:
                return "online"
        return "connecting"
    except Exception as e:
        logger.warning("Erro ao registrar câmera {}: {}", camera_id, e)
        return f"error: {e}"


def _unregister_rtsp(camera_id: str) -> None:
    """Fecha e remove câmera do registry se existir."""
    if camera_registry.is_registered(camera_id):
        try:
            cam = camera_registry.get(camera_id)
            cam.close()
        except Exception:
            pass
        camera_registry._cameras.pop(camera_id, None)


async def _test_rtsp_url(url: str) -> dict:
    """Testa conectividade RTSP/ONVIF em thread separada (cv2 é síncrono)."""
    if url.lower().startswith("onvif://"):
        return await _test_onvif_url(url)

    def _try_open():
        import cv2
        cap = cv2.VideoCapture(url)
        try:
            if not cap.isOpened():
                return False, "Não foi possível abrir o stream"
            ok, _ = cap.read()
            return ok, "OK" if ok else "Stream aberto mas sem frames"
        finally:
            cap.release()

    loop = asyncio.get_event_loop()
    try:
        ok, msg = await asyncio.wait_for(
            loop.run_in_executor(None, _try_open), timeout=6.0
        )
        return {"online": ok, "message": msg}
    except asyncio.TimeoutError:
        return {"online": False, "message": "Timeout — câmera não respondeu em 6s"}
    except Exception as e:
        return {"online": False, "message": str(e)}


async def _test_onvif_url(url: str) -> dict:
    """Testa câmera ONVIF: verifica autenticação e obtém URI RTSP."""
    def _try():
        from app.camera.sources.onvif_cam import discover_rtsp_uri, list_onvif_profiles
        from urllib.parse import urlparse
        p = urlparse(url)
        host = p.hostname or ""
        port = p.port or 2020
        user = p.username or ""
        passwd = p.password or ""
        profiles = list_onvif_profiles(host, port, user, passwd)
        if not profiles:
            return False, "ONVIF: autenticação falhou ou câmera não responde"
        profile = profiles[0]
        uri = discover_rtsp_uri(host, port, user, passwd, profile)
        if not uri:
            return False, f"ONVIF: perfis encontrados ({profiles[:3]}) mas sem URI RTSP"
        return True, f"ONVIF OK — URI: {uri} | Perfis: {profiles[:3]}"

    loop = asyncio.get_event_loop()
    try:
        ok, msg = await asyncio.wait_for(
            loop.run_in_executor(None, _try), timeout=12.0
        )
        return {"online": ok, "message": msg}
    except asyncio.TimeoutError:
        return {"online": False, "message": "Timeout ONVIF — câmera não respondeu em 12s"}
    except Exception as e:
        return {"online": False, "message": str(e)}


async def _get_whatsapp_config() -> dict:
    s = await settings_service.get_all_settings()
    return {
        "api_url":        s.get("whatsapp_api_url", ""),
        "api_key":        s.get("whatsapp_api_key", ""),
        "instance":       s.get("whatsapp_instance", "default"),
        "notify_numbers": s.get("whatsapp_notify_numbers", ""),
        "enabled":        s.get("whatsapp_enabled", "false") == "true",
    }


async def _check_evolution_status(config: dict) -> dict:
    url = f"{config['api_url'].rstrip('/')}/instance/connectionState/{config['instance']}"
    headers = {"apikey": config["api_key"]}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url, headers=headers)
        data = r.json()
        state = data.get("instance", {}).get("state") or data.get("state", "unknown")
        connected = state == "open"
        return {
            "connected": connected,
            "state": state,
            "instance": config["instance"],
            "message": "Conectado" if connected else f"Estado: {state}",
        }
    except httpx.ConnectError:
        return {"connected": False, "state": "unreachable", "message": "Evolution API inacessível"}
    except Exception as e:
        return {"connected": False, "state": "error", "message": str(e)}


async def _send_whatsapp_test(config: dict, override_number: str | None = None) -> dict:
    base     = config["api_url"].rstrip("/")
    instance = config["instance"]
    headers  = {"apikey": config["api_key"], "Content-Type": "application/json"}
    msg      = "✅ SPRESSO FACIAL — Teste de notificação. Sistema operacional."

    if override_number:
        targets = [override_number.strip()]
    else:
        targets = [n.strip() for n in config["notify_numbers"].split(",") if n.strip()]

    if not targets:
        return {"sent": False, "message": "Nenhum número alvo encontrado"}

    results = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        for number in targets:
            clean = number.replace("+", "").replace(" ", "").replace("-", "")
            try:
                r = await client.post(
                    f"{base}/message/sendText/{instance}",
                    json={"number": clean, "text": msg},
                    headers=headers,
                )
                ok = r.status_code in (200, 201)
                results.append({"number": number, "sent": ok, "status": r.status_code})
            except Exception as e:
                results.append({"number": number, "sent": False, "status": str(e)})

    all_ok = all(r["sent"] for r in results)
    return {
        "sent": all_ok,
        "message": f"Enviado para {sum(r['sent'] for r in results)}/{len(results)} número(s)",
        "details": results,
    }


# ---------------------------------------------------------------------------
# Função auxiliar usada pelo lifespan para carregar câmeras do banco
# ---------------------------------------------------------------------------

async def load_rtsp_cameras_from_db() -> None:
    """Carrega câmeras IP/ONVIF ativas do banco e registra no camera_registry."""
    cameras = await settings_service.list_rtsp_cameras(enabled_only=True)
    for cam in cameras:
        try:
            instance = _build_camera(cam.url, cam.camera_id, cam.label)
            instance.open()
            camera_registry.register(instance)
            scheme = "ONVIF" if cam.url.lower().startswith("onvif://") else "RTSP"
            logger.info("Câmera {} '{}' carregada do banco", scheme, cam.camera_id)
        except Exception as e:
            logger.warning("Erro ao carregar câmera '{}': {}", cam.camera_id, e)
