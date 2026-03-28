# SPRESSO FACIAL

Sistema de reconhecimento facial embarcado para unidades **SPRESSO** (minimercado autônomo).

Roda 100% localmente em edge computing — sem dependência de cloud para operação principal.

---

## Hardware alvo

| Componente | Especificação |
|---|---|
| Computador | Raspberry Pi 5 (8GB) |
| Câmera | Sony IMX500 (AI Camera) |
| Acelerador IA | Hailo-8 (PCIe, 26 TOPS) |
| Rede | Wi-Fi / Ethernet com IP fixo |
| Operação | Headless (sem monitor/teclado) |

---

## Stack tecnológica

| Camada | Tecnologia | Função |
|---|---|---|
| Runtime | Python 3.13 | — |
| API | FastAPI + Uvicorn | HTTP REST + interface web |
| Detecção facial | HailoRT 4.23 + SCRFD 2.5G | Inferência no Hailo-8 (~14ms) |
| Reconhecimento | InsightFace (buffalo_sc) + ONNX Runtime | Embeddings ArcFace no CPU |
| Captura | picamera2 | Frame via IMX500 |
| Processamento | OpenCV + NumPy | Pré/pós-processamento |
| Persistência | SQLite + SQLAlchemy async | Eventos, pessoas, embeddings |
| Logs | loguru | Rotativos por dia, 7 dias de retenção |
| Config | pydantic-settings + .env | Configuração por ambiente |

---

## Estrutura do projeto

```
spresso-ai/
├── app/
│   ├── main.py               # FastAPI factory + lifespan + UI estática
│   ├── config.py             # Settings via pydantic-settings + .env
│   ├── logger.py             # Logs console + arquivo rotativo
│   ├── api/
│   │   ├── routes_health.py      # GET /health
│   │   ├── routes_camera.py      # GET /camera/snapshot
│   │   ├── routes_detection.py   # GET /detection/snapshot + /faces
│   │   ├── routes_events.py      # GET /events + assign de crops
│   │   ├── routes_persons.py     # CRUD /persons + fotos
│   │   └── routes_recognition.py # threshold, regen-embeddings, test
│   ├── camera/
│   │   ├── capture.py        # Interface picamera2
│   │   └── service.py        # Singleton CameraService
│   ├── detection/
│   │   └── face_detector.py  # Hailo-8 + SCRFD — detecção de rostos
│   ├── recognition/
│   │   ├── embeddings.py     # InsightFace det_500m + ArcFace MobileFaceNet
│   │   └── matcher.py        # Cache de embeddings + comparação coseno
│   ├── services/
│   │   ├── event_service.py  # CRUD de eventos de detecção
│   │   └── person_service.py # CRUD de pessoas e fotos de referência
│   ├── static/
│   │   └── index.html        # Interface web SPA (Monitor, Pessoas, Eventos, Calibração)
│   └── storage/
│       ├── db.py             # Conexão async SQLite
│       └── models.py         # ORM: DetectionEvent, Person, PersonPhoto
├── data/
│   ├── spresso.db            # Banco de dados SQLite
│   ├── snapshots/            # Frames anotados
│   ├── face_crops/           # Crops individuais de rostos detectados
│   └── persons/              # Fotos de referência por pessoa
├── logs/                     # Logs rotativos por dia
├── scripts/
│   └── regen_embeddings.py   # Migração offline de embeddings
├── requirements.txt
├── run.py                    # Ponto de entrada
├── spresso-facial.service    # Systemd service unit
└── .env.example              # Variáveis de ambiente disponíveis
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
# Edite .env conforme necessário

# Execute em desenvolvimento
.venv/bin/python run.py

# Ou instale como serviço systemd
sudo cp spresso-facial.service /etc/systemd/system/
sudo systemctl enable --now spresso-facial
```

---

## Configuração (.env)

| Variável | Padrão | Descrição |
|---|---|---|
| `APP_ENV` | `development` | Ambiente (`development` / `production`) |
| `APP_HOST` | `0.0.0.0` | Endereço de bind |
| `APP_PORT` | `8000` | Porta HTTP |
| `APP_LOG_LEVEL` | `INFO` | Nível de log |
| `CAMERA_WIDTH` | `640` | Resolução horizontal |
| `CAMERA_HEIGHT` | `480` | Resolução vertical |
| `RECOGNITION_THRESHOLD` | `0.62` | Limiar de distância coseno para match (0.0–1.0) |
| `MAX_PHOTOS_PER_PERSON` | `20` | Máximo de fotos de referência por pessoa |

> **Threshold:** valores menores = mais restritivo (menos falsos positivos). Recomendado: 0.50–0.65 dependendo das condições de iluminação e distância.

---

## Endpoints principais

| Método | Rota | Descrição |
|---|---|---|
| GET | `/` ou `/ui` | Interface web |
| GET | `/health` | Status da aplicação |
| GET | `/detection/faces` | Detecção + reconhecimento em JSON |
| GET | `/detection/snapshot` | Frame anotado com rostos e identidades (JPEG) |
| GET | `/detection/status` | Status do detector e cache |
| GET | `/events` | Histórico de eventos de detecção |
| GET | `/events/{id}/faces/{fid}/crop` | Crop de rosto de evento |
| POST | `/events/{id}/faces/{fid}/assign` | Atribuir rosto detectado a uma pessoa |
| GET/POST | `/persons` | Listar / criar pessoa |
| POST | `/persons/{id}/photos` | Adicionar foto de referência |
| GET | `/recognition/threshold` | Consultar threshold atual |
| POST | `/recognition/threshold` | Atualizar threshold em runtime |
| POST | `/recognition/test` | Testar foto contra base |
| POST | `/recognition/regen-embeddings` | Regenerar embeddings no banco |
| GET | `/docs` | Swagger UI |

---

## Pipeline de reconhecimento

```
Frame (IMX500 640×480)
  │
  ├─► Hailo-8 / SCRFD 2.5G ──────► bboxes de rostos (~14ms)
  │
  └─► InsightFace det_500m (CPU) ──► bboxes + 5 keypoints
          │
          └─► norm_crop (alinhamento) ──► face 112×112 por rosto
                  │
                  └─► ArcFace MobileFaceNet ──► embedding 512-dim
                          │
                          └─► cosine distance vs. cache
                                  │
                                  ├─ dist ≤ threshold → MATCH (pessoa identificada)
                                  └─ dist > threshold → DESCONHECIDO
```

**Tratamento por distância:**
- Rostos ≥ 90px (perto): usa embedding direto do frame completo
- Rostos < 90px (longe, ~2m+): faz upscale do crop antes do alinhamento para melhorar qualidade dos keypoints

---

## Performance

| Operação | Tempo |
|---|---|
| Inferência Hailo-8 (SCRFD detecção) | **~13–15ms** |
| InsightFace det_500m + alinhamento (CPU) | ~50–80ms por frame |
| Geração de embedding ArcFace (CPU) | ~30–50ms por rosto |
| Total por scan (1 rosto) | ~120–180ms |
| Comparação de embedding vs. cache | < 1ms (NumPy) |

---

## Workflow de cadastro (fluxo recomendado)

1. **Detectar** — posicionar-se na frente da câmera e realizar scans em diferentes ângulos e distâncias
2. **Eventos** → "Ver rostos" → selecionar os melhores crops
3. **Atribuir** — associar o crop a uma pessoa nova ou existente
4. Repetir com **3–5 distâncias/ângulos diferentes** para cobertura ampla
5. Ajustar o **threshold** na aba Calibração se necessário

> O workflow via Eventos garante que referência e detecção ao vivo usam exatamente o mesmo pipeline de alinhamento, maximizando a acurácia.

---

## Fases de desenvolvimento

- [x] **FASE 1** — Base: FastAPI + config + logs + systemd
- [x] **FASE 2** — Câmera: captura IMX500 via picamera2
- [x] **FASE 3** — Detecção: Hailo-8 + SCRFD (~14ms por frame)
- [x] **FASE 4** — Persistência: SQLite + registro de eventos e crops
- [x] **FASE 5** — Cadastro de pessoas: API CRUD + fotos de referência
- [x] **FASE 6** — Reconhecimento facial: embeddings ArcFace + matching coseno
- [x] **FASE 7** — Interface web: Monitor ao vivo, Eventos, Pessoas, Calibração
- [x] **FASE 8** — Pipeline unificado: alinhamento por keypoints + upscale para rostos distantes
- [ ] **FASE 9** — Regras de negócio avançadas: cooldown, deduplicação, alertas externos
- [ ] **FASE 10** — Câmeras IP / RTSP simultâneas *(planejado — ver abaixo)*

---

## Fase 10 — Câmeras IP / RTSP (planejado)

**Objetivo:** suportar múltiplas câmeras (IMX500 + câmeras IP via RTSP) rodando em paralelo, permitindo cobertura de múltiplos ângulos e pontos de acesso simultâneos.

**Arquitetura prevista:**

```
CameraSource (interface comum)
├── IMX500Camera       → picamera2 (atual)
└── IPCamera           → cv2.VideoCapture("rtsp://...")

CameraRegistry         → gerencia N câmeras registradas

Pipeline por câmera:
  frame → Hailo SCRFD → InsightFace align → embedding → match

Eventos enriquecidos com camera_id e camera_label.
```

**Pontos de atenção:**
- O Hailo-8 suporta multi-stream nativo — precisará refatorar a ativação de rede de por-inferência para permanente com fila serializada
- Fotos de referência poderão ter `angle_hint` (frontal/perfil/distância) para matching direcionado por câmera
- Threshold configurável por câmera (câmera de entrada vs. câmera de corredor podem ter sensibilidades diferentes)

**Dependência:** estabilização completa do pipeline de câmera única antes de iniciar esta fase.

---

## Licença

Proprietário — SPRESSO © 2026
