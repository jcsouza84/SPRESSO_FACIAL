# SPRESSO FACIAL

Sistema de reconhecimento facial embarcado para unidades **SPRESSO** (minimercado autônomo).

Roda 100% localmente em edge computing — sem dependência de cloud para operação principal.

---

## Hardware alvo

| Componente | Especificação |
|---|---|
| Computador | Raspberry Pi 5 (8GB) |
| Câmera | Sony IMX500 (AI Camera) |
| Acelerador IA | Hailo-8 (PCIe) |
| Rede | Wi-Fi / Ethernet com IP fixo |
| Operação | Headless (sem monitor/teclado) |

---

## Stack tecnológica

- **Python 3.13**
- **FastAPI** + Uvicorn
- **HailoRT 4.23.0** — inferência no acelerador Hailo-8
- **SCRFD 2.5G** — modelo de detecção facial (compilado para Hailo)
- **picamera2** — captura via câmera IMX500
- **OpenCV** — pré/pós-processamento de imagens
- **SQLite** — persistência local (fase 4+)
- **loguru** — logs rotativos

---

## Estrutura do projeto

```
spresso-ai/
├── app/
│   ├── main.py              # FastAPI factory + lifespan
│   ├── config.py            # Settings via pydantic-settings + .env
│   ├── logger.py            # Logs console + arquivo rotativo
│   ├── api/
│   │   ├── routes_health.py     # GET /health
│   │   ├── routes_camera.py     # GET /camera/snapshot
│   │   └── routes_detection.py  # GET /detection/snapshot
│   ├── camera/
│   │   ├── capture.py       # Interface picamera2
│   │   └── service.py       # Singleton CameraService
│   ├── detection/
│   │   └── face_detector.py # Hailo-8 + SCRFD — detecção de rostos
│   ├── recognition/         # Fase 6 — embeddings + matching
│   ├── services/            # Fase 7+ — alertas, eventos
│   └── storage/             # Fase 4 — SQLite
├── data/                    # Snapshots e banco de dados
├── logs/                    # Logs rotativos
├── requirements.txt
├── run.py                   # Ponto de entrada
└── .env.example             # Variáveis de ambiente
```

---

## Instalação

```bash
# Clone o repositório
git clone https://github.com/jcsouza84/SPRESSO_FACIAL.git
cd SPRESSO_FACIAL

# Crie o ambiente virtual (com acesso a pacotes do sistema)
python3 -m venv .venv --system-site-packages

# Instale as dependências
.venv/bin/pip install -r requirements.txt

# Configure o ambiente
cp .env.example .env

# Execute
.venv/bin/python run.py
```

---

## Endpoints disponíveis

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Status da aplicação |
| GET | `/camera/status` | Status da câmera |
| GET | `/camera/snapshot` | Foto atual da câmera (JPEG) |
| GET | `/camera/snapshot/last` | Último frame capturado |
| GET | `/detection/status` | Status do detector Hailo |
| GET | `/detection/faces` | Detecção em JSON (bounding boxes) |
| GET | `/detection/snapshot` | Foto com rostos marcados (JPEG) |
| GET | `/docs` | Documentação interativa (Swagger) |

---

## Fases de desenvolvimento

- [x] **FASE 1** — Base: FastAPI + config + logs + systemd
- [x] **FASE 2** — Câmera: captura IMX500 via picamera2
- [x] **FASE 3** — Detecção: Hailo-8 + SCRFD (~13ms por frame)
- [ ] **FASE 4** — Persistência: SQLite + registro de eventos
- [ ] **FASE 5** — Cadastro de pessoas
- [ ] **FASE 6** — Reconhecimento facial (embeddings + matching)
- [ ] **FASE 7** — Regras de negócio (blacklist, cooldown)
- [ ] **FASE 8** — Integração (webhook, Evolution API)
- [ ] **FASE 9** — Serviço contínuo com systemd

---

## Performance

| Operação | Tempo |
|---|---|
| Captura de frame (IMX500) | ~1s (warm-up) |
| Inferência Hailo-8 + SCRFD | **~13–40ms** |
| Inferência CPU (estimado) | ~300–800ms |

---

## Licença

Proprietário — SPRESSO © 2026
