"""
Modelos SQLAlchemy para o banco de dados local (SQLite).
"""
import enum
from datetime import datetime
from sqlalchemy import String, Integer, Float, Text, DateTime, ForeignKey, Enum, Boolean, LargeBinary
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Pessoas
# ---------------------------------------------------------------------------

class PersonCategory(str, enum.Enum):
    blacklist = "blacklist"
    vip       = "vip"


class Person(Base):
    """
    Pessoa cadastrada no sistema — pode ser blacklist ou VIP.
    """
    __tablename__ = "persons"

    id:         Mapped[int]    = mapped_column(Integer, primary_key=True, autoincrement=True)
    name:       Mapped[str]    = mapped_column(String(200), nullable=False)
    phone:      Mapped[str | None] = mapped_column(String(30), nullable=True)
    category:   Mapped[PersonCategory] = mapped_column(
        Enum(PersonCategory), nullable=False, default=PersonCategory.blacklist
    )
    observation: Mapped[str | None] = mapped_column(Text, nullable=True)
    active:     Mapped[bool]   = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    photos: Mapped[list["PersonPhoto"]] = relationship(
        "PersonPhoto",
        back_populates="person",
        cascade="all, delete-orphan",
    )

    def to_dict(self, include_photos: bool = True) -> dict:
        d = {
            "id":          self.id,
            "name":        self.name,
            "phone":       self.phone,
            "category":    self.category.value,
            "observation": self.observation,
            "active":      self.active,
            "photo_count": len(self.photos),
            "created_at":  self.created_at.isoformat(),
            "updated_at":  self.updated_at.isoformat(),
        }
        if include_photos:
            d["photos"] = [p.to_dict() for p in self.photos]
        return d


class PersonPhoto(Base):
    """
    Foto de referência de uma pessoa cadastrada (máx. 5 por pessoa).
    """
    __tablename__ = "person_photos"

    id:         Mapped[int]  = mapped_column(Integer, primary_key=True, autoincrement=True)
    person_id:  Mapped[int]  = mapped_column(Integer, ForeignKey("persons.id"), nullable=False)
    path:       Mapped[str]  = mapped_column(String(512), nullable=False)
    embedding:  Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    person: Mapped["Person"] = relationship("Person", back_populates="photos")

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "person_id":  self.person_id,
            "path":       self.path,
            "has_embedding": self.embedding is not None,
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Eventos de detecção
# ---------------------------------------------------------------------------

class DetectionEvent(Base):
    """
    Registra cada vez que o sistema roda o detector e obtém resultado.
    """
    __tablename__ = "detection_events"

    id:           Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp:    Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    face_count:   Mapped[int]   = mapped_column(Integer, nullable=False, default=0)
    inference_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    frame_width:  Mapped[int]   = mapped_column(Integer, nullable=False)
    frame_height: Mapped[int]   = mapped_column(Integer, nullable=False)
    snapshot_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # Sprint C — Fase 10: identificação da câmera de origem
    camera_id:    Mapped[str | None] = mapped_column(String(20), nullable=True)
    camera_label: Mapped[str | None] = mapped_column(String(100), nullable=True)

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
            "camera_id":     self.camera_id,
            "camera_label":  self.camera_label,
            "faces":         [f.to_dict() for f in self.faces],
        }


class DetectedFaceRecord(Base):
    """
    Coordenadas, confiança e crop de cada rosto dentro de um evento.
    """
    __tablename__ = "detected_faces"

    id:         Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id:   Mapped[int]   = mapped_column(Integer, ForeignKey("detection_events.id"), nullable=False)
    x1:         Mapped[int]   = mapped_column(Integer, nullable=False)
    y1:         Mapped[int]   = mapped_column(Integer, nullable=False)
    x2:         Mapped[int]   = mapped_column(Integer, nullable=False)
    y2:         Mapped[int]   = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    crop_path:  Mapped[str | None] = mapped_column(String(512), nullable=True)
    embedding:  Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    event: Mapped["DetectionEvent"] = relationship("DetectionEvent", back_populates="faces")

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "x1":         self.x1, "y1": self.y1,
            "x2":         self.x2, "y2": self.y2,
            "width":      self.x2 - self.x1,
            "height":     self.y2 - self.y1,
            "confidence": round(self.confidence, 4),
            "crop_path":  self.crop_path,
            "has_embedding": self.embedding is not None,
        }


# ---------------------------------------------------------------------------
# Alertas e presença (Fase 9)
# ---------------------------------------------------------------------------

class AlertRecord(Base):
    """Alerta de identificação — principalmente blacklist."""
    __tablename__ = "alert_records"

    id:          Mapped[int]  = mapped_column(Integer, primary_key=True, autoincrement=True)
    person_id:   Mapped[int]  = mapped_column(Integer, ForeignKey("persons.id"), nullable=False)
    person_name: Mapped[str]  = mapped_column(String(200), nullable=False)
    category:    Mapped[str]  = mapped_column(String(20), nullable=False)
    event_id:    Mapped[int | None] = mapped_column(Integer, ForeignKey("detection_events.id"), nullable=True)
    face_id:     Mapped[int | None] = mapped_column(Integer, ForeignKey("detected_faces.id"), nullable=True)
    crop_path:   Mapped[str | None] = mapped_column(String(512), nullable=True)
    confidence:  Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    distance:    Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    message:     Mapped[str | None] = mapped_column(Text, nullable=True)
    confirmed:   Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    whatsapp_sent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    telegram_sent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "person_id":    self.person_id,
            "person_name":  self.person_name,
            "category":     self.category,
            "event_id":     self.event_id,
            "face_id":      self.face_id,
            "crop_path":    self.crop_path,
            "confidence":   round(self.confidence, 2),
            "distance":     round(self.distance, 4),
            "message":      self.message,
            "confirmed":    self.confirmed,
            "confirmed_at": self.confirmed_at.isoformat() if self.confirmed_at else None,
            "whatsapp_sent": self.whatsapp_sent,
            "telegram_sent": self.telegram_sent,
            "created_at":   self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Sprint D — Configurações do sistema e câmeras RTSP
# ---------------------------------------------------------------------------

class SystemSetting(Base):
    """Par chave/valor para configurações editáveis pela UI sem restart."""
    __tablename__ = "system_settings"

    key:        Mapped[str] = mapped_column(String(100), primary_key=True)
    value:      Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    def to_dict(self) -> dict:
        return {"key": self.key, "value": self.value, "updated_at": self.updated_at.isoformat()}


class RtspCameraConfig(Base):
    """Câmeras IP/RTSP cadastradas e gerenciadas pela UI."""
    __tablename__ = "rtsp_cameras"

    id:          Mapped[int]  = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id:   Mapped[str]  = mapped_column(String(20), unique=True, nullable=False)
    label:       Mapped[str]  = mapped_column(String(100), nullable=False)
    url:         Mapped[str]  = mapped_column(Text, nullable=False)
    enabled:     Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    order_index: Mapped[int]  = mapped_column(Integer, nullable=False, default=0)
    updated_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "camera_id":   self.camera_id,
            "label":       self.label,
            "url":         self.url,
            "enabled":     self.enabled,
            "order_index": self.order_index,
            "updated_at":  self.updated_at.isoformat(),
        }


class PresenceRecord(Base):
    """Registro de presença — pessoa identificada na unidade."""
    __tablename__ = "presence_records"

    id:         Mapped[int]  = mapped_column(Integer, primary_key=True, autoincrement=True)
    person_id:  Mapped[int]  = mapped_column(Integer, ForeignKey("persons.id"), nullable=False)
    person_name: Mapped[str] = mapped_column(String(200), nullable=False)
    category:   Mapped[str]  = mapped_column(String(20), nullable=False)
    event_id:   Mapped[int | None] = mapped_column(Integer, ForeignKey("detection_events.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "person_id":     self.person_id,
            "person_name":   self.person_name,
            "category":      self.category,
            "event_id":      self.event_id,
            "created_at":    self.created_at.isoformat(),
        }
