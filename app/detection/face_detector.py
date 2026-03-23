"""
Detector de rostos usando Hailo-8 + modelo SCRFD 2.5G.

Pipeline:
  Frame (RGB) → resize 640x640 → Hailo inference → decode SCRFD → NMS → bboxes
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import hailo_platform as hp

from app.logger import logger

# ---------------------------------------------------------------------------
# Constantes do modelo SCRFD 2.5G
# ---------------------------------------------------------------------------
MODEL_PATH   = Path("/usr/share/hailo-models/scrfd_2.5g_h8l.hef")
INPUT_SIZE   = 640          # quadrado
NUM_ANCHORS  = 2            # anchors por posição
STRIDES      = [8, 16, 32]

# Mapeamento stride → (cls_layer, bbox_layer)  [kps ignorado por ora]
STRIDE_LAYERS = {
    8:  ("scrfd_2_5g/conv42", "scrfd_2_5g/conv43"),
    16: ("scrfd_2_5g/conv49", "scrfd_2_5g/conv50"),
    32: ("scrfd_2_5g/conv55", "scrfd_2_5g/conv56"),
}


# ---------------------------------------------------------------------------
# Estrutura de resultado
# ---------------------------------------------------------------------------
@dataclass
class DetectedFace:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_dict(self) -> dict:
        return {
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "width": self.width, "height": self.height,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class DetectionResult:
    faces: list[DetectedFace] = field(default_factory=list)
    inference_ms: float = 0.0
    frame_width: int = 0
    frame_height: int = 0

    @property
    def count(self) -> int:
        return len(self.faces)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class FaceDetector:
    def __init__(
        self,
        confidence_threshold: float = 0.45,
        nms_iou_threshold: float = 0.4,
        min_face_size: int = 30,
    ) -> None:
        self._conf_thresh    = confidence_threshold
        self._nms_thresh     = nms_iou_threshold
        self._min_face_size  = min_face_size
        self._vdevice: Optional[hp.VDevice] = None
        self._network:  Optional[object]    = None
        self._in_params:  Optional[dict]    = None
        self._out_params: Optional[dict]    = None
        self._input_name: Optional[str]     = None

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------
    def open(self) -> None:
        if self._vdevice is not None:
            return

        logger.info("Inicializando Hailo-8 + SCRFD 2.5G")
        self._vdevice = hp.VDevice()
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
        logger.info("FaceDetector pronto (conf={}, iou={})",
                    self._conf_thresh, self._nms_thresh)

    def close(self) -> None:
        if self._vdevice is None:
            return
        logger.info("Encerrando FaceDetector")
        self._vdevice.release()
        self._vdevice = None
        self._network  = None

    @property
    def is_ready(self) -> bool:
        return self._vdevice is not None

    # ------------------------------------------------------------------
    # Inferência pública
    # ------------------------------------------------------------------
    def detect(self, frame_rgb: np.ndarray) -> DetectionResult:
        if not self.is_ready:
            raise RuntimeError("FaceDetector não iniciado. Chame open() primeiro.")

        orig_h, orig_w = frame_rgb.shape[:2]
        resized, scale_x, scale_y = self._preprocess(frame_rgb)

        import time
        t0 = time.perf_counter()

        with self._network.activate():
            with hp.InferVStreams(self._network, self._in_params, self._out_params) as pipeline:
                input_batch = {self._input_name: resized[np.newaxis]}
                raw_outputs = pipeline.infer(input_batch)

        inference_ms = (time.perf_counter() - t0) * 1000

        faces = self._postprocess(raw_outputs, scale_x, scale_y, orig_w, orig_h)
        logger.debug("Detecção: {} rosto(s) em {:.1f}ms", len(faces), inference_ms)

        return DetectionResult(
            faces=faces,
            inference_ms=round(inference_ms, 1),
            frame_width=orig_w,
            frame_height=orig_h,
        )

    # ------------------------------------------------------------------
    # Pré-processamento
    # ------------------------------------------------------------------
    def _preprocess(
        self, frame_rgb: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        orig_h, orig_w = frame_rgb.shape[:2]
        resized = cv2.resize(frame_rgb, (INPUT_SIZE, INPUT_SIZE))
        # BGR esperado pelo modelo (treinado com OpenCV)
        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        scale_x = orig_w / INPUT_SIZE
        scale_y = orig_h / INPUT_SIZE
        return resized, scale_x, scale_y

    # ------------------------------------------------------------------
    # Pós-processamento SCRFD
    # ------------------------------------------------------------------
    def _postprocess(
        self,
        outputs: dict,
        scale_x: float,
        scale_y: float,
        orig_w: int,
        orig_h: int,
    ) -> list[DetectedFace]:
        all_boxes  : list[np.ndarray] = []
        all_scores : list[np.ndarray] = []

        for stride in STRIDES:
            cls_name, bbox_name = STRIDE_LAYERS[stride]
            cls_raw  = outputs[cls_name][0]   # (H, W, 2)
            bbox_raw = outputs[bbox_name][0]  # (H, W, 8)

            grid_h, grid_w = cls_raw.shape[:2]
            # Saídas cls já são probabilidades (HailoRT dequantiza + sigmoid)
            scores = cls_raw.reshape(-1)  # (H*W*NUM_ANCHORS,)

            anchor_centers = self._make_anchor_centers(grid_h, grid_w, stride)
            bboxes_raw = bbox_raw.reshape(-1, NUM_ANCHORS, 4)
            bboxes = self._decode_bboxes(anchor_centers, bboxes_raw, stride)

            mask = scores >= self._conf_thresh
            all_boxes.append(bboxes[mask])
            all_scores.append(scores[mask])

        if not any(len(b) for b in all_boxes):
            return []

        boxes  = np.concatenate(all_boxes,  axis=0)
        scores = np.concatenate(all_scores, axis=0)

        # Escala de volta para resolução original e clipa nos limites do frame
        boxes[:, 0] = np.clip(boxes[:, 0] * scale_x, 0, orig_w)
        boxes[:, 1] = np.clip(boxes[:, 1] * scale_y, 0, orig_h)
        boxes[:, 2] = np.clip(boxes[:, 2] * scale_x, 0, orig_w)
        boxes[:, 3] = np.clip(boxes[:, 3] * scale_y, 0, orig_h)

        keep = self._nms(boxes, scores)
        faces = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i].astype(int)
            w, h = x2 - x1, y2 - y1
            if w < self._min_face_size or h < self._min_face_size:
                continue
            faces.append(DetectedFace(
                x1=int(x1), y1=int(y1),
                x2=int(x2), y2=int(y2),
                confidence=float(scores[i]),
            ))
        return faces

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

    @staticmethod
    def _make_anchor_centers(
        grid_h: int, grid_w: int, stride: int
    ) -> np.ndarray:
        """Gera centros dos anchors: shape (H*W*num_anchors, 2)."""
        ys, xs = np.mgrid[0:grid_h, 0:grid_w]
        centers = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32)
        centers = (centers + 0.5) * stride
        return np.repeat(centers, NUM_ANCHORS, axis=0)  # duplica p/ 2 anchors

    @staticmethod
    def _decode_bboxes(
        anchor_centers: np.ndarray,
        bbox_raw: np.ndarray,
        stride: int,
    ) -> np.ndarray:
        """Decodifica offsets SCRFD → coordenadas absolutas (x1,y1,x2,y2)."""
        # bbox_raw: (H*W, num_anchors, 4)
        flat = bbox_raw.reshape(-1, 4) * stride   # (H*W*num_anchors, 4)
        x1 = anchor_centers[:, 0] - flat[:, 0]
        y1 = anchor_centers[:, 1] - flat[:, 1]
        x2 = anchor_centers[:, 0] + flat[:, 2]
        y2 = anchor_centers[:, 1] + flat[:, 3]
        return np.stack([x1, y1, x2, y2], axis=1)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        """NMS clássico, retorna índices dos boxes mantidos."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas  = (x2 - x1) * (y2 - y1)
        order  = scores.argsort()[::-1]
        keep   : list[int] = []

        while order.size:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            ix1 = np.maximum(x1[i], x1[order[1:]])
            iy1 = np.maximum(y1[i], y1[order[1:]])
            ix2 = np.minimum(x2[i], x2[order[1:]])
            iy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
            iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= self._nms_thresh]

        return keep


face_detector = FaceDetector()
