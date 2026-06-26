import asyncio
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
from app.detection.person_face_detector import person_face_detector
from app.recognition.matcher import face_matcher
from app.storage.db import init_db, close_db
from app.api.routes_health import router as health_router
from app.api.routes_camera import router as camera_router
from app.api.routes_detection import router as detection_router
from app.api.routes_events import router as events_router
from app.api.routes_persons import router as persons_router
from app.api.routes_recognition import router as recognition_router
from app.api.routes_alerts import router as alerts_router
from app.api.routes_settings import router as settings_router, load_rtsp_cameras_from_db


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
    person_face_detector.open()
    await face_matcher.load_all()

    # Sprint D — carregar câmeras RTSP do banco (UI-configuradas têm prioridade)
    await load_rtsp_cameras_from_db()

    # Sprint E — aplicar threshold salvo no banco (sobrepõe .env)
    from app.services.settings_service import get_setting
    thr = await get_setting("recognition_threshold")
    if thr:
        try:
            settings.recognition_threshold = float(thr)
            logger.info("Threshold carregado do banco: {}", thr)
        except ValueError:
            pass

    # Sprint C fallback — câmera RTSP via .env (se não cadastrada no banco)
    if settings.camera_rtsp_url:
        from app.camera.registry import camera_registry
        if not camera_registry.is_registered(settings.camera_rtsp_id):
            from app.camera.sources.rtsp import RTSPCamera
            rtsp = RTSPCamera(
                url=settings.camera_rtsp_url,
                camera_id=settings.camera_rtsp_id,
                label=settings.camera_rtsp_label,
            )
            rtsp.open()
            camera_registry.register(rtsp)
            logger.info("Câmera RTSP do .env registrada: {}", settings.camera_rtsp_id)

    # Sprint B — worker de detecção automática
    # Lê do banco primeiro (salvo pela UI), com fallback para .env
    from app.services.settings_service import get_setting as _get_setting
    _auto_db = await _get_setting("detection_auto")
    _fps_db  = await _get_setting("detection_fps")
    if _auto_db is not None:
        settings.detection_auto = _auto_db == "true"
    if _fps_db is not None:
        try:
            settings.detection_fps = float(_fps_db)
        except ValueError:
            pass
    if settings.detection_auto:
        from app.workers import detection_worker
        asyncio.create_task(detection_worker.run_forever())
        logger.info("Detection worker ativado ({} FPS)", settings.detection_fps)

    yield

    person_face_detector.close()
    face_detector.close()
    camera_service.stop()
    # Libera VDevice compartilhado por último
    from app.detection.hailo_device import release_vdevice
    release_vdevice()
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
app.include_router(alerts_router)
app.include_router(settings_router)

# Serve a interface web
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(
        str(_static_dir / "index.html"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/ui", include_in_schema=False)
async def ui():
    return FileResponse(
        str(_static_dir / "index.html"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/display", include_in_schema=False)
async def display():
    """Tela HDMI da loja — monitoramento ao vivo sem interface de operador."""
    return FileResponse(
        str(_static_dir / "display.html"),
        headers={"Cache-Control": "no-store"},
    )
