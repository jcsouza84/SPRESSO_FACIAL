"""
Gerenciador singleton do VDevice Hailo.

O Hailo-8L expõe apenas um VDevice físico. Todos os detectores devem
compartilhar esta instância para não conflitar; cada network group é
ativado exclusivamente durante sua própria janela de inferência.
"""
from __future__ import annotations

import hailo_platform as hp
from app.logger import logger

_vdevice: hp.VDevice | None = None


def get_vdevice() -> hp.VDevice:
    global _vdevice
    if _vdevice is None:
        logger.info("Criando VDevice Hailo-8L compartilhado")
        _vdevice = hp.VDevice()
    return _vdevice


def release_vdevice() -> None:
    global _vdevice
    if _vdevice is not None:
        logger.info("Liberando VDevice Hailo-8L")
        _vdevice.release()
        _vdevice = None
