from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.logger import logger, setup_logger
from app.camera.service import camera_service
from app.detection.face_detector import face_detector
from app.recognition.matcher import face_matcher
from app.storage.db import init_db, close_db
from app.api.routes_health import router as health_router
from app.api.routes_camera import router as camera_router
from app.api.routes_detection import router as detection_router
from app.api.routes_events import router as events_router
from app.api.routes_persons import router as persons_router
from app.api.routes_recognition import router as recognition_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logger()

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("SPRESSO FACIAL iniciando — ambiente: {env}", env=settings.app_env)
    logger.info("API disponível em http://{host}:{port}", host=settings.app_host, port=settings.app_port)

    await init_db()
    camera_service.start()
    face_detector.open()
    await face_matcher.load_all()

    yield

    face_detector.close()
    camera_service.stop()
    await close_db()
    logger.info("SPRESSO FACIAL encerrado")


app = FastAPI(
    title="SPRESSO FACIAL",
    description="Sistema de reconhecimento facial embarcado para unidades SPRESSO",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(camera_router)
app.include_router(detection_router)
app.include_router(events_router)
app.include_router(persons_router)
app.include_router(recognition_router)

# Serve a interface web
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(_static_dir / "index.html"))


@app.get("/ui", include_in_schema=False)
async def ui():
    return FileResponse(str(_static_dir / "index.html"))
