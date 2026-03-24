"""
Gerenciamento da conexão assíncrona com SQLite via SQLAlchemy + aiosqlite.
"""
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.logger import logger
from app.storage.models import Base

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_db() -> None:
    global _engine, _session_factory

    db_url = f"sqlite+aiosqlite:///{settings.db_path}"
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)

    _engine = create_async_engine(
        db_url,
        echo=settings.is_development,
        connect_args={"check_same_thread": False},
    )
    _session_factory = async_sessionmaker(
        _engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Banco de dados iniciado em {}", settings.db_path)


async def close_db() -> None:
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("Banco de dados encerrado")


def get_session() -> AsyncSession:
    if _session_factory is None:
        raise RuntimeError("Banco não inicializado. Chame init_db() primeiro.")
    return _session_factory()
