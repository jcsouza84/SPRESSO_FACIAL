"""
Câmera IP via ONVIF (WS-Security Password Digest) + stream RTSP via OpenCV.

Usada para câmeras modernas (ex: Tapo C320WS firmware 1.6+) que exigem
autenticação ONVIF antes de abrir o stream RTSP.

URL de cadastro: onvif://usuario:senha@host:porta
Exemplo:         onvif://spresso:spresso123@192.168.84.202:2020
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import cv2
import numpy as np

from app.camera.capture import Frame
from app.logger import logger


# ---------------------------------------------------------------------------
# Helper ONVIF / WS-Security
# ---------------------------------------------------------------------------

def _ws_digest_header(username: str, password: str) -> str:
    """Gera o bloco WS-Security UsernameToken com PasswordDigest (SHA1)."""
    nonce_raw = os.urandom(16)
    nonce_b64 = base64.b64encode(nonce_raw).decode()
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    digest = base64.b64encode(
        hashlib.sha1(nonce_raw + created.encode() + password.encode()).digest()
    ).decode()
    return (
        f'<wsse:Security xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/'
        f'oasis-200401-wss-wssecurity-secext-1.0.xsd">'
        f'<wsse:UsernameToken>'
        f'<wsse:Username>{username}</wsse:Username>'
        f'<wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/'
        f'oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{digest}</wsse:Password>'
        f'<wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/'
        f'oasis-200401-wss-soap-message-security-1.0#Base64Binary">{nonce_b64}</wsse:Nonce>'
        f'<wsu:Created xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/'
        f'oasis-200401-wss-wssecurity-utility-1.0.xsd">{created}</wsu:Created>'
        f'</wsse:UsernameToken></wsse:Security>'
    )


def _soap_envelope(header: str, body: str) -> str:
    return (
        '<?xml version="1.0"?>'
        '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">'
        f'<s:Header>{header}</s:Header>'
        f'<s:Body>{body}</s:Body>'
        '</s:Envelope>'
    )


def discover_rtsp_uri(
    host: str,
    port: int,
    username: str,
    password: str,
    profile_token: str = "profile_1",
    timeout: float = 8.0,
) -> str | None:
    """
    Usa ONVIF GetStreamUri com WS-Security Digest para obter a URI RTSP real.
    Retorna a URI ou None em caso de falha.
    """
    import requests  # já no requirements

    endpoint = f"http://{host}:{port}/onvif/service"
    header = _ws_digest_header(username, password)
    body = (
        f'<GetStreamUri xmlns="http://www.onvif.org/ver10/media/wsdl">'
        f'<StreamSetup>'
        f'<Stream xmlns="http://www.onvif.org/ver10/schema">RTP-Unicast</Stream>'
        f'<Transport xmlns="http://www.onvif.org/ver10/schema"><Protocol>RTSP</Protocol></Transport>'
        f'</StreamSetup>'
        f'<ProfileToken>{profile_token}</ProfileToken>'
        f'</GetStreamUri>'
    )
    soap = _soap_envelope(header, body)
    try:
        resp = requests.post(
            endpoint,
            data=soap,
            headers={"Content-Type": "application/soap+xml"},
            timeout=timeout,
        )
        uris = re.findall(r"<tt:Uri>([^<]+)</tt:Uri>", resp.text)
        return uris[0] if uris else None
    except Exception as exc:
        logger.warning("ONVIFCamera: falha ao obter URI via ONVIF: {}", exc)
        return None


def list_onvif_profiles(
    host: str, port: int, username: str, password: str, timeout: float = 8.0
) -> list[str]:
    """Retorna lista de tokens de perfil disponíveis na câmera."""
    import requests

    endpoint = f"http://{host}:{port}/onvif/service"
    header = _ws_digest_header(username, password)
    body = '<GetProfiles xmlns="http://www.onvif.org/ver10/media/wsdl"/>'
    soap = _soap_envelope(header, body)
    try:
        resp = requests.post(
            endpoint,
            data=soap,
            headers={"Content-Type": "application/soap+xml"},
            timeout=timeout,
        )
        return re.findall(r'token="([^"]+)"', resp.text)
    except Exception as exc:
        logger.warning("ONVIFCamera: falha ao listar perfis: {}", exc)
        return []


# ---------------------------------------------------------------------------
# ONVIFCamera
# ---------------------------------------------------------------------------

class ONVIFCamera:
    """
    Câmera IP com autenticação ONVIF (WS-Security Password Digest).

    Fluxo:
    1. Conecta no endpoint ONVIF da câmera
    2. Obtém URI RTSP via GetStreamUri (autenticado com digest)
    3. Abre o stream RTSP com OpenCV (tenta com e sem credenciais na URL)
    4. Mantém leitura contínua em thread daemon

    URL de cadastro: onvif://user:pass@host:port[/profile]
    """

    def __init__(
        self,
        url: str,
        camera_id: str = "onvif1",
        label: str = "Câmera ONVIF",
    ) -> None:
        self.camera_id = camera_id
        self.label = label
        self._onvif_url = url

        parsed = urlparse(url)
        self._host = parsed.hostname or ""
        self._port = parsed.port or 2020
        self._username = parsed.username or ""
        self._password = parsed.password or ""
        # perfil pode vir no path: onvif://user:pass@host:port/profile_2
        path = (parsed.path or "").strip("/")
        self._profile = path if path else "profile_1"

        self._rtsp_uri: str | None = None
        self._cap: cv2.VideoCapture | None = None
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    def open(self) -> None:
        logger.info(
            "ONVIFCamera {}: conectando em {}:{} (perfil {})",
            self.camera_id, self._host, self._port, self._profile,
        )
        self._rtsp_uri = discover_rtsp_uri(
            self._host, self._port, self._username, self._password, self._profile
        )
        if not self._rtsp_uri:
            logger.warning(
                "ONVIFCamera {}: não foi possível obter URI RTSP via ONVIF",
                self.camera_id,
            )
        else:
            logger.info("ONVIFCamera {}: URI RTSP = {}", self.camera_id, self._rtsp_uri)
            self._cap = self._open_cap()

        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"onvif-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def _build_rtsp_candidates(self) -> list[str]:
        """Gera variações da URI RTSP para tentar (com/sem credenciais)."""
        if not self._rtsp_uri:
            return []
        parsed = urlparse(self._rtsp_uri)
        base = f"{parsed.scheme}://{parsed.hostname}"
        if parsed.port:
            base += f":{parsed.port}"
        path = parsed.path

        candidates = [
            # com credenciais
            f"{parsed.scheme}://{self._username}:{self._password}@{parsed.hostname}"
            + (f":{parsed.port}" if parsed.port else "")
            + path,
            # sem credenciais (algumas câmeras permitem LAN sem auth)
            f"{base}{path}",
        ]
        return candidates

    def _open_cap(self) -> cv2.VideoCapture | None:
        import os
        # Força backend FFmpeg com transporte TCP (necessário para Tapo/câmeras modernas)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        for uri in self._build_rtsp_candidates():
            cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
            if cap.isOpened():
                logger.info("ONVIFCamera {}: stream aberto: {}", self.camera_id, uri)
                return cap
            cap.release()
        logger.warning(
            "ONVIFCamera {}: todas as URIs RTSP falharam", self.camera_id
        )
        return None

    def _reader_loop(self) -> None:
        consecutive_failures = 0
        reconnect_delay = 10.0  # segundos entre tentativas de reconexão
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                logger.debug(
                    "ONVIFCamera {}: aguardando {}s antes de reconectar",
                    self.camera_id, reconnect_delay,
                )
                time.sleep(reconnect_delay)
                self._reconnect()
                # backoff progressivo: 10s → 20s → 30s (máx)
                reconnect_delay = min(reconnect_delay + 10.0, 30.0)
                continue
            reconnect_delay = 10.0  # reset após conexão bem-sucedida
            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._latest = frame
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    logger.warning(
                        "ONVIFCamera {}: {} falhas consecutivas, reconectando",
                        self.camera_id, consecutive_failures,
                    )
                    self._reconnect()
                    consecutive_failures = 0
                time.sleep(0.1)

    def _reconnect(self) -> None:
        if self._cap:
            self._cap.release()
        self._rtsp_uri = discover_rtsp_uri(
            self._host, self._port, self._username, self._password, self._profile
        )
        self._cap = self._open_cap()
        if self._cap and self._cap.isOpened():
            logger.info("ONVIFCamera {}: reconectada", self.camera_id)
        else:
            logger.warning("ONVIFCamera {}: falha na reconexão, tentará novamente", self.camera_id)

    # ------------------------------------------------------------------
    def capture(self) -> Frame:
        with self._lock:
            arr = self._latest
        if arr is None:
            raise RuntimeError(
                f"ONVIFCamera '{self.camera_id}' sem frame — aguarde inicialização"
            )
        h, w = arr.shape[:2]
        arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        _, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Frame(data=buf.tobytes(), array=arr_rgb, width=w, height=h)

    def close(self) -> None:
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("ONVIFCamera {} encerrada", self.camera_id)

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._latest is not None
