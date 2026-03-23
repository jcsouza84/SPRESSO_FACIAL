"""
Ponto de entrada para execução direta.
Uso: .venv/bin/python run.py
"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.app_log_level.lower(),
        reload=settings.is_development,
    )
