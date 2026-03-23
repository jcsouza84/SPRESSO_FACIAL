from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.logger import logger, setup_logger
from app.camera.service import camera_service
from app.detection.face_detector import face_detector
from app.api.routes_health import router as health_router
from app.api.routes_camera import router as camera_router
from app.api.routes_detection import router as detection_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logger()

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("SPRESSO FACIAL iniciando — ambiente: {env}", env=settings.app_env)
    logger.info("API disponível em http://{host}:{port}", host=settings.app_host, port=settings.app_port)

    camera_service.start()
    face_detector.open()

    yield

    face_detector.close()
    camera_service.stop()
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


@app.get("/", include_in_schema=False)
async def root():
    return {"service": "spresso-facial", "docs": "/docs"}
