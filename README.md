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
| Persistência | SQLite + SQLAlchemy async | Eventos, pessoas, embeddings, alertas |
| Notificações | Evolution API (Baileys/WhatsApp) | Alertas blacklist por WhatsApp |
| Logs | loguru | Rotativos por dia, 7 dias de retenção |
| Config | pydantic-settings + .env + SQLite | Configuração por ambiente + UI em runtime |

---

## Estrutura do projeto

```
spresso-ai/
├── app/
│   ├── main.py               # FastAPI factory + lifespan + worker de detecção
│   ├── config.py             # Settings via pydantic-settings + .env
│   ├── logger.py             # Logs console + arquivo rotativo
│   ├── api/
│   │   ├── routes_health.py      # GET /health
│   │   ├── routes_camera.py      # GET /camera/snapshot + /preview (RTSP + IMX500)
│   │   ├── routes_detection.py   # GET /detection/snapshot + /faces + pipeline
│   │   ├── routes_events.py      # GET /events + assign de crops
│   │   ├── routes_persons.py     # CRUD /persons + fotos
│   │   ├── routes_recognition.py # threshold, regen-embeddings, test
│   │   ├── routes_alerts.py      # GET /alerts/active + POST /confirm + presença
│   │   └── routes_settings.py    # GET/POST /settings — configs editáveis em runtime
│   ├── camera/
│   │   ├── capture.py        # Interface picamera2 (IMX500)
│   │   ├── service.py        # Singleton CameraService (IMX500)
│   │   ├── registry.py       # CameraRegistry — gerencia múltiplas câmeras
│   │   └── sources/          # CameraSource interface + IMX500Camera + IPCamera (RTSP)
│   ├── detection/
│   │   └── face_detector.py  # Hailo-8 + SCRFD — detecção de rostos
│   ├── recognition/
│   │   ├── embeddings.py     # InsightFace det_500m + ArcFace MobileFaceNet
│   │   └── matcher.py        # Cache de embeddings + comparação coseno
│   ├── services/
│   │   ├── alert_service.py    # Alertas blacklist, cooldown, envio WhatsApp
│   │   ├── event_service.py    # Persistência de eventos de detecção
│   │   ├── person_service.py   # CRUD de pessoas e fotos de referência
│   │   ├── presence_service.py # Deduplicação de eventos + log de presença
│   │   └── settings_service.py # Leitura/escrita de configs no banco (system_settings)
│   ├── workers/
│   │   └── detection_worker.py # Loop assíncrono de detecção automática (1 FPS)
│   ├── static/
│   │   └── index.html        # Interface web SPA (Monitor, Pessoas, Eventos, Calibração, Configurações)
│   └── storage/
│       ├── db.py             # Conexão async SQLite
│       └── models.py         # ORM: DetectionEvent, DetectedFace, Person, PersonPhoto,
│                             #       AlertRecord, PresenceRecord, SystemSetting
├── data/
│   ├── spresso.db            # Banco de dados SQLite
│   ├── snapshots/            # Frames anotados
│   ├── face_crops/           # Crops individuais de rostos detectados
│   └── persons/              # Fotos de referência por pessoa
├── docs/
│   └── evolution_api.md      # Credenciais, endpoints e operação da Evolution API
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
| `APP_TIMEZONE` | `America/Maceio` | Fuso horário local (IANA) — usado em mensagens WhatsApp |
| `CAMERA_WIDTH` | `640` | Resolução horizontal |
| `CAMERA_HEIGHT` | `480` | Resolução vertical |
| `RECOGNITION_THRESHOLD` | `0.62` | Limiar de distância coseno para match (0.0–1.0) |
| `MAX_PHOTOS_PER_PERSON` | `20` | Máximo de fotos de referência por pessoa |
| `ALERT_COOLDOWN_SECONDS` | `300` | Cooldown entre alertas da mesma pessoa (segundos) |
| `EVENT_DEDUP_SECONDS` | `30` | Janela de deduplicação de eventos (segundos) |
| `DETECTION_AUTO` | `false` | Ativar worker de detecção automática em background |
| `DETECTION_FPS` | `1.0` | FPS do worker de detecção automática |
| `WHATSAPP_ENABLED` | `false` | Enviar alertas blacklist via WhatsApp |
| `WHATSAPP_API_URL` | — | URL base da Evolution API |
| `WHATSAPP_API_KEY` | — | Token da instância Evolution API |
| `WHATSAPP_INSTANCE` | `default` | Nome da instância Evolution API |
| `WHATSAPP_NOTIFY_NUMBER` | — | Número destino padrão (ex: `5511999999999`) |

> Todas as configurações acima também são editáveis em tempo real pela interface web em **Configurações**, sem necessidade de restart. Os valores da UI são armazenados na tabela `system_settings` do SQLite e têm prioridade sobre o `.env`.

---

## Endpoints principais

| Método | Rota | Descrição |
|---|---|---|
| GET | `/` ou `/ui` | Interface web |
| GET | `/health` | Status da aplicação |
| GET | `/detection/faces` | Detecção + reconhecimento em JSON |
| GET | `/detection/snapshot` | Frame anotado com rostos e identidades (JPEG) |
| GET | `/detection/status` | Status do detector, cache e alertas ativos |
| GET | `/events` | Histórico de eventos de detecção |
| GET | `/events/{id}/faces/{fid}/crop` | Crop de rosto de evento |
| POST | `/events/{id}/faces/{fid}/assign` | Atribuir rosto detectado a uma pessoa |
| GET/POST | `/persons` | Listar / criar pessoa |
| POST | `/persons/{id}/photos` | Adicionar foto de referência |
| GET | `/recognition/threshold` | Consultar threshold atual |
| POST | `/recognition/threshold` | Atualizar threshold em runtime |
| POST | `/recognition/test` | Testar foto contra base |
| POST | `/recognition/regen-embeddings` | Regenerar embeddings no banco |
| GET | `/alerts/active` | Alertas blacklist pendentes de confirmação |
| POST | `/alerts/{id}/confirm` | Confirmar/dispensar alerta |
| GET | `/alerts/presence/today` | Estatísticas de presença do dia |
| GET | `/settings` | Ler todas as configurações editáveis |
| POST | `/settings` | Salvar configurações e aplicar em runtime |
| GET | `/settings/whatsapp/status` | Verificar conexão com a Evolution API |
| POST | `/settings/test/whatsapp` | Enviar mensagem WhatsApp de teste |
| GET | `/camera/preview` | Frame atual em JPEG (IMX500 ou RTSP) |
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

## Fluxo de alerta blacklist

```
detection_worker (1 FPS em background)
  └─► _run_pipeline(persist=True, process_alerts=True)
        └─► PresenceService.should_skip_event()   ← dedup (padrão 30s)
              └─► [não deduplicado] → salva DetectionEvent + DetectedFaces
                    └─► AlertService.process_recognition()
                          ├─► log_presence (PresenceRecord)
                          ├─► get_active_alert_for_person() → alerta já ativo?
                          │     SIM → retorna existente (sem reenvio)
                          │     NÃO →
                          ├─► _cooldown_elapsed() → cooldown (padrão 300s) passou?
                          │     NÃO → ignora
                          │     SIM →
                          └─► cria AlertRecord + _send_whatsapp()
                                └─► POST /message/sendMedia ou /sendText
                                      └─► Evolution API → WhatsApp do operador
```

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
- [x] **FASE 9** — Regras de negócio: cooldown, deduplicação, alertas blacklist, WhatsApp (Evolution API)
- [x] **FASE 10** — Câmeras IP / RTSP: CameraRegistry + suporte a múltiplas fontes simultâneas

---

## Fase 9 — Regras de negócio (concluída)

- **Cooldown de alertas** — mesma pessoa blacklist não gera novo alerta antes de `ALERT_COOLDOWN_SECONDS` (configurável pela UI)
- **Deduplicação** — modo auto não persiste eventos repetidos dentro de `EVENT_DEDUP_SECONDS` (configurável pela UI)
- **Alertas blacklist** — registro no banco + overlay persistente na UI até confirmar + auto-refresh a cada 30s
- **Log de presença** — identificações VIP e blacklist registradas em `presence_records`
- **WhatsApp via Evolution API** — envio de texto + foto do rosto; hora formatada no fuso horário configurado
- **Configurações em runtime** — todos os parâmetros editáveis via UI sem restart (persistidos em `system_settings`)

---

## Fase 10 — Câmeras IP / RTSP (concluída)

- **CameraRegistry** — gerencia N câmeras registradas (IMX500 + câmeras IP via RTSP)
- **CameraSource** — interface comum para IMX500Camera e IPCamera
- **Eventos enriquecidos** com `camera_id` e `camera_label`
- Threshold e deduplicação aplicados por câmera no mesmo pipeline

---

## Documentação adicional

- [`docs/evolution_api.md`](docs/evolution_api.md) — credenciais, endpoints e instruções de operação da Evolution API (instância Spresso)

---

## Licença

Proprietário — SPRESSO © 2026
