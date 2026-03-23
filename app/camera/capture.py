"""
Interface de baixo nível com a câmera via picamera2.
Responsável exclusivamente por abrir, capturar e fechar a câmera.
"""
import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
from picamera2 import Picamera2

from app.config import settings
from app.logger import logger


@dataclass
class Frame:
    data: bytes          # JPEG comprimido
    array: np.ndarray    # Array RGB bruto
    width: int
    height: int


class CameraCapture:
    def __init__(self) -> None:
        self._cam: Optional[Picamera2] = None

    def open(self) -> None:
        if self._cam is not None:
            return

        logger.info("Abrindo câmera IMX500 (index={})", settings.camera_index)
        self._cam = Picamera2(settings.camera_index)

        config = self._cam.create_still_configuration(
            main={
                "size": (settings.camera_width, settings.camera_height),
                "format": "RGB888",
            },
            buffer_count=2,
        )
        self._cam.configure(config)
        self._cam.start()
        logger.info(
            "Câmera iniciada — resolução {}x{}",
            settings.camera_width,
            settings.camera_height,
        )

    def capture(self) -> Frame:
        if self._cam is None:
            raise RuntimeError("Câmera não está aberta. Chame open() primeiro.")

        array: np.ndarray = self._cam.capture_array("main")

        buf = io.BytesIO()
        import cv2
        _, encoded = cv2.imencode(".jpg", cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
        jpeg_bytes = encoded.tobytes()

        h, w = array.shape[:2]
        return Frame(data=jpeg_bytes, array=array, width=w, height=h)

    def close(self) -> None:
        if self._cam is None:
            return
        logger.info("Encerrando câmera")
        self._cam.stop()
        self._cam.close()
        self._cam = None

    @property
    def is_open(self) -> bool:
        return self._cam is not None
