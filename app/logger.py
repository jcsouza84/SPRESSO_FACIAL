import sys
from pathlib import Path
from loguru import logger as _logger

from app.config import settings


def setup_logger() -> None:
    _logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Console
    _logger.add(
        sys.stdout,
        format=log_format,
        level=settings.app_log_level,
        colorize=True,
        backtrace=True,
        diagnose=settings.is_development,
    )

    # Arquivo rotativo (10 MB, 7 dias de retenção)
    log_file = settings.logs_dir / "spresso_{time:YYYY-MM-DD}.log"
    _logger.add(
        str(log_file),
        format=log_format,
        level=settings.app_log_level,
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )


logger = _logger
