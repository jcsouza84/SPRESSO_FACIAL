"""
Modelos SQLAlchemy para o banco de dados local (SQLite).
"""
from datetime import datetime, timezone
from sqlalchemy import String, Integer, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class DetectionEvent(Base):
    """
    Registra cada vez que o sistema roda o detector e obtém resultado.
    Um evento = uma chamada ao pipeline de detecção.
    """
    __tablename__ = "detection_events"

    id:           Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp:    Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    face_count:   Mapped[int]   = mapped_column(Integer, nullable=False, default=0)
    inference_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    frame_width:  Mapped[int]   = mapped_column(Integer, nullable=False)
    frame_height: Mapped[int]   = mapped_column(Integer, nullable=False)
    snapshot_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    faces: Mapped[list["DetectedFaceRecord"]] = relationship(
        "DetectedFaceRecord",
        back_populates="event",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> dict:
        return {
            "id":            self.id,
            "timestamp":     self.timestamp.isoformat(),
            "face_count":    self.face_count,
            "inference_ms":  self.inference_ms,
            "frame_width":   self.frame_width,
            "frame_height":  self.frame_height,
            "snapshot_path": self.snapshot_path,
            "faces":         [f.to_dict() for f in self.faces],
        }


class DetectedFaceRecord(Base):
    """
    Coordenadas e confiança de cada rosto dentro de um evento.
    """
    __tablename__ = "detected_faces"

    id:         Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id:   Mapped[int]   = mapped_column(Integer, ForeignKey("detection_events.id"), nullable=False)
    x1:         Mapped[int]   = mapped_column(Integer, nullable=False)
    y1:         Mapped[int]   = mapped_column(Integer, nullable=False)
    x2:         Mapped[int]   = mapped_column(Integer, nullable=False)
    y2:         Mapped[int]   = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    event: Mapped["DetectionEvent"] = relationship("DetectionEvent", back_populates="faces")

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "x1":         self.x1, "y1": self.y1,
            "x2":         self.x2, "y2": self.y2,
            "width":      self.x2 - self.x1,
            "height":     self.y2 - self.y1,
            "confidence": round(self.confidence, 4),
        }
