"""
Detector de Pessoas e Rostos usando Hailo-8L + YOLOv5s PersonFace.

O modelo detecta 2 classes simultaneamente:
  - Classe 0: person  (corpo humano inteiro)
  - Classe 1: face    (rosto isolado)

Output pós-NMS já embutido no HEF:
  yolov5s_personface/yolov5_nms_postprocess — shape (2, 5, N_max)
    eixo 0: classe (0=person, 1=face)
    eixo 1: coordenadas [x1, y1, x2, y2, conf] normalizadas [0..1]
    eixo 2: até N_max detecções por classe (zero-padded)

Compartilha o VDevice Hailo com o FaceDetector (SCRFD) via hailo_device.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import hailo_platform as hp

from app.logger import logger

MODEL_PATH = Path("/usr/share/hailo-models/yolov5s_personface_h8l.hef")
INPUT_SIZE  = 640
OUTPUT_NAME = "yolov5s_personface/yolov5_nms_postprocess"

CLASS_PERSON = 0
CLASS_FACE   = 1


@dataclass
class PersonFaceDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str  # "person" | "face"

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class PersonFaceResult:
    persons: list[PersonFaceDetection] = field(default_factory=list)
    faces:   list[PersonFaceDetection] = field(default_factory=list)
    inference_ms: float = 0.0

    @property
    def has_persons(self) -> bool:
        return len(self.persons) > 0


class PersonFaceDetector:
    def __init__(
        self,
        confidence_threshold: float = 0.50,
        nms_iou_threshold: float = 0.45,
    ) -> None:
        self._conf_thresh = confidence_threshold
        self._nms_thresh  = nms_iou_threshold
        self._vdevice: Optional[hp.VDevice] = None
        self._network:  Optional[object]    = None
        self._in_params:  Optional[dict]    = None
        self._out_params: Optional[dict]    = None
        self._input_name: Optional[str]     = None

    def open(self) -> None:
        if self._vdevice is not None:
            return
        if not MODEL_PATH.exists():
            logger.warning("Modelo PersonFace não encontrado: {}", MODEL_PATH)
            return

        logger.info("Inicializando Hailo-8L + YOLOv5s PersonFace")
        from app.detection.hailo_device import get_vdevice
        self._vdevice = get_vdevice()
        hef = hp.HEF(str(MODEL_PATH))

        cfg = hp.ConfigureParams.create_from_hef(
            hef, interface=hp.HailoStreamInterface.PCIe
        )
        self._network = self._vdevice.configure(hef, cfg)[0]
        self._in_params  = hp.InputVStreamParams.make(
            self._network, format_type=hp.FormatType.UINT8
        )
        self._out_params = hp.OutputVStreamParams.make(
            self._network, format_type=hp.FormatType.FLOAT32
        )
        self._input_name = next(iter(self._in_params))
        logger.info("PersonFaceDetector pronto (conf={})", self._conf_thresh)

    def close(self) -> None:
        if self._vdevice is None:
            return
        logger.info("Encerrando PersonFaceDetector")
        # VDevice compartilhado — não libera aqui
        self._vdevice = None
        self._network  = None

    @property
    def is_ready(self) -> bool:
        return self._vdevice is not None and self._network is not None

    def detect(
        self,
        frame_rgb: np.ndarray,
        conf_override: float | None = None,
    ) -> PersonFaceResult:
        if not self.is_ready:
            raise RuntimeError("PersonFaceDetector não iniciado. Chame open() primeiro.")

        orig_h, orig_w = frame_rgb.shape[:2]
        blob = self._preprocess(frame_rgb)

        import time
        t0 = time.perf_counter()

        with self._network.activate():
            with hp.InferVStreams(self._network, self._in_params, self._out_params) as pipeline:
                raw_outputs = pipeline.infer({self._input_name: blob[np.newaxis]})

        inference_ms = (time.perf_counter() - t0) * 1000
        conf = conf_override if conf_override is not None else self._conf_thresh
        persons, faces = self._decode(raw_outputs, orig_w, orig_h, conf)

        logger.debug(
            "PersonFace: {} pessoa(s) / {} rosto(s) em {:.1f}ms",
            len(persons), len(faces), inference_ms,
        )
        return PersonFaceResult(
            persons=persons,
            faces=faces,
            inference_ms=round(inference_ms, 1),
        )

    def _preprocess(self, frame_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame_rgb, (INPUT_SIZE, INPUT_SIZE))
        return cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

    def _decode(
        self,
        outputs: dict,
        orig_w: int,
        orig_h: int,
        conf_thresh: float,
    ) -> tuple[list[PersonFaceDetection], list[PersonFaceDetection]]:
        """
        Decodifica output NMS do HEF.

        O tensor tem shape (2, 5, N) onde:
          - dim 0: classe (0=person, 1=face)
          - dim 1: valores [x1, y1, x2, y2, score] normalizados [0..1]
          - dim 2: até N detecções por classe (zero-padded)
        """
        raw = outputs[OUTPUT_NAME][0]   # shape (2, 5, N)

        persons: list[PersonFaceDetection] = []
        faces:   list[PersonFaceDetection] = []

        for class_id, label, target_list in [
            (CLASS_PERSON, "person", persons),
            (CLASS_FACE,   "face",   faces),
        ]:
            class_data = raw[class_id]   # shape (5, N)
            scores = class_data[4]        # conf scores para cada slot
            valid  = scores > conf_thresh
            if not valid.any():
                continue

            x1s = class_data[0][valid]
            y1s = class_data[1][valid]
            x2s = class_data[2][valid]
            y2s = class_data[3][valid]
            confs = scores[valid]

            for x1n, y1n, x2n, y2n, conf in zip(x1s, y1s, x2s, y2s, confs):
                # Desnormaliza para coordenadas do frame original
                x1 = int(np.clip(x1n * orig_w, 0, orig_w))
                y1 = int(np.clip(y1n * orig_h, 0, orig_h))
                x2 = int(np.clip(x2n * orig_w, 0, orig_w))
                y2 = int(np.clip(y2n * orig_h, 0, orig_h))

                if x2 <= x1 or y2 <= y1:
                    continue

                target_list.append(PersonFaceDetection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=float(conf),
                    label=label,
                ))

        return persons, faces


person_face_detector = PersonFaceDetector()
