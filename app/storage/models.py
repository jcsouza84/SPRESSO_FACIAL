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
