"""
Serviço de câmera — gerencia ciclo de vida e acesso ao snapshot mais recente.
É um singleton instanciado no lifespan da aplicação.
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.camera.capture import CameraCapture, Frame
from app.config import settings
from app.logger import logger


class CameraService:
    def __init__(self) -> None:
        self._capture = CameraCapture()
        self._last_frame: Optional[Frame] = None
        self._snapshots_dir = settings.data_dir / "snapshots"

    def start(self) -> None:
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._capture.open()
        logger.info("CameraService iniciado")

    def stop(self) -> None:
        self._capture.close()
        logger.info("CameraService encerrado")

    def snapshot(self, save: bool = True) -> Frame:
        """Captura um frame. Opcionalmente salva em disco."""
        frame = self._capture.capture()
        self._last_frame = frame

        if save:
            self._save(frame)

        return frame

    def last_frame(self) -> Optional[Frame]:
        return self._last_frame

    @property
    def is_ready(self) -> bool:
        return self._capture.is_open

    def _save(self, frame: Frame) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        path = self._snapshots_dir / f"snapshot_{ts}.jpg"
        path.write_bytes(frame.data)
        logger.debug("Snapshot salvo em {}", path)
        return path


camera_service = CameraService()
