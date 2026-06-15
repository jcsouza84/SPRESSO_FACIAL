"""
Worker de detecção contínua — roda em background enquanto o serviço estiver ativo.
Captura frames da câmera e executa o pipeline completo (detect + recognize + persist).
Ativado pelo config.detection_auto = True no lifespan da aplicação.
"""
import asyncio

from app.config import settings
from app.logger import logger


async def run_forever() -> None:
    """Loop assíncrono de detecção contínua a settings.detection_fps frames/s."""
    interval = 1.0 / max(settings.detection_fps, 0.1)
    logger.info(
        "Detection worker iniciado — {fps} FPS (intervalo {interval:.1f}s)",
        fps=settings.detection_fps,
        interval=interval,
    )
    while True:
        try:
            from app.api.routes_detection import _run_pipeline
            await _run_pipeline(dedup=True, persist=True, process_alerts=True)
        except Exception as exc:
            logger.warning("Detection worker erro: {}", exc)
        await asyncio.sleep(interval)
